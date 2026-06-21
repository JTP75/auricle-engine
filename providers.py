import asyncio
import collections
import io
import json
import logging
import re
import struct
import wave
from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional, Tuple

import edge_tts

from consts import (
    F5_BUNDLED_REF_RELPATH,
    F5_DEFAULT_REF_TEXT,
    F5_SAMPLE_RATE,
    KOKORO_SAMPLE_RATE,
    SAMPLE_RATE,
    WHISPER_MIN_SPEECH_FRAMES,
    WHISPER_PADDING_MS,
    WHISPER_SILENCE_FRAMES,
    WHISPER_VAD_AGGRESSIVENESS,
    WHISPER_VAD_BLOCK_MS,
    WHISPER_FRAME_BYTES,
)

logger = logging.getLogger(__name__)

_MARKDOWN_RE = re.compile(r'[*_`#\[\]()]')


# ── Abstract interfaces ────────────────────────────────────────────────────

class STTProvider(ABC):
    @abstractmethod
    def load(self) -> None:
        """Load model into memory. Called once during engine boot."""

    @abstractmethod
    def feed(self, pcm_bytes: bytes) -> Tuple[Optional[str], Optional[str]]:
        """Feed a PCM chunk. Returns (final_text, partial_text); at most one is non-None."""

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state (call on wakeword detection and utterance completion)."""


class TTSProvider(ABC):
    @abstractmethod
    def stream_audio(self, sentence: str) -> AsyncIterator[bytes]:
        """Yield raw audio bytes for the sentence."""


# ── Vosk STT ───────────────────────────────────────────────────────────────

class VoskSTTProvider(STTProvider):
    def __init__(self, model_path: str) -> None:
        self._model_path = model_path
        self._model      = None
        self._rec        = None

    def load(self) -> None:
        from vosk import Model, KaldiRecognizer
        self._model = Model(self._model_path)
        self._rec   = KaldiRecognizer(self._model, SAMPLE_RATE)

    def feed(self, pcm_bytes: bytes) -> Tuple[Optional[str], Optional[str]]:
        assert self._rec is not None, "VoskSTTProvider not loaded"
        if self._rec.AcceptWaveform(pcm_bytes):
            text = json.loads(self._rec.Result()).get("text", "").strip()
            return (text or None), None
        partial = json.loads(self._rec.PartialResult()).get("partial", "").strip()
        return None, (partial or None)

    def reset(self) -> None:
        if self._rec is not None:
            self._rec.Reset()


# ── Whisper STT ────────────────────────────────────────────────────────────

class WhisperSTTProvider(STTProvider):
    """
    In-process Whisper STT provider.

    Loads a HuggingFace distil-whisper model directly into the engine venv.
    Uses webrtcvad for VAD framing: accumulates 1280-byte OWW chunks into
    30ms VAD frames, detects voiced regions, then runs inference when silence
    follows a voiced segment.

    Requires: torch, transformers, accelerate, webrtcvad (or webrtcvad-wheels)
    """

    def __init__(self, model_id: str) -> None:
        self._model_id       = model_id
        self._pipe           = None
        self._vad            = None
        self._is_multilingual = False
        # VAD state — populated by load(), reset by reset()
        self._remainder:     bytes                    = b""
        self._voiced_frames: list                     = []
        self._ring_buffer:   Optional[collections.deque] = None
        self._triggered:     bool                     = False
        self._silence_count: int                      = 0

    def load(self) -> None:
        import torch
        import webrtcvad
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline as hf_pipeline

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype  = torch.float16 if torch.cuda.is_available() else torch.float32

        logger.info("[auricle-engine] whisper: loading %r on %s", self._model_id, device)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self._model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        ).to(device)
        processor = AutoProcessor.from_pretrained(self._model_id)
        self._pipe = hf_pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            torch_dtype=dtype,
            device=device,
        )
        self._vad = webrtcvad.Vad(WHISPER_VAD_AGGRESSIVENESS)
        self._is_multilingual = getattr(model.config, "is_multilingual", False)
        padding_frames = WHISPER_PADDING_MS // WHISPER_VAD_BLOCK_MS
        self._ring_buffer = collections.deque(maxlen=padding_frames)
        logger.info("[auricle-engine] whisper: ready (multilingual=%s)", self._is_multilingual)

    def feed(self, pcm_bytes: bytes) -> Tuple[Optional[str], Optional[str]]:
        import numpy as np

        assert self._pipe is not None, "WhisperSTTProvider not loaded"

        # Re-slice 1280-byte OWW chunks into 30ms VAD frames (960 bytes each).
        data   = self._remainder + pcm_bytes
        offset = 0
        frames = []
        while offset + WHISPER_FRAME_BYTES <= len(data):
            frames.append(data[offset : offset + WHISPER_FRAME_BYTES])
            offset += WHISPER_FRAME_BYTES
        self._remainder = data[offset:]

        has_onset = False

        for pcm in frames:
            is_speech = self._vad.is_speech(pcm, SAMPLE_RATE)

            if not self._triggered:
                self._ring_buffer.append((pcm, is_speech))
                num_voiced = sum(1 for _, s in self._ring_buffer if s)
                if num_voiced > 0.8 * self._ring_buffer.maxlen:
                    self._triggered     = True
                    self._voiced_frames = [p for p, _ in self._ring_buffer]
                    self._ring_buffer.clear()
                    self._silence_count = 0
                    has_onset = True
            else:
                self._voiced_frames.append(pcm)
                if is_speech:
                    self._silence_count = 0
                else:
                    self._silence_count += 1

                if self._silence_count >= WHISPER_SILENCE_FRAMES:
                    self._triggered     = False
                    self._silence_count = 0

                    if len(self._voiced_frames) >= WHISPER_MIN_SPEECH_FRAMES:
                        audio_np  = np.frombuffer(b"".join(self._voiced_frames), dtype=np.int16)
                        audio_f32 = audio_np.astype(np.float32) / 32768.0
                        gen_kw    = {"language": "english"} if self._is_multilingual else {}
                        result    = self._pipe(
                            {"array": audio_f32, "sampling_rate": SAMPLE_RATE},
                            **({"generate_kwargs": gen_kw} if gen_kw else {}),
                        )
                        text = result["text"].strip()
                        if text:
                            self._voiced_frames = []
                            self._ring_buffer.clear()
                            return text, None

                    self._voiced_frames = []
                    self._ring_buffer.clear()

        if has_onset:
            return None, "..."  # partial sentinel — trips AWAITING_UTTERANCE → UTTERANCE
        return None, None

    def reset(self) -> None:
        self._remainder     = b""
        self._voiced_frames = []
        self._triggered     = False
        self._silence_count = 0
        if self._ring_buffer is not None:
            self._ring_buffer.clear()

    def terminate(self) -> None:
        self._pipe = None
        self._vad  = None


# ── Edge-TTS ───────────────────────────────────────────────────────────────

class EdgeTTSProvider(TTSProvider):
    def __init__(self, voice: str) -> None:
        self._voice = voice

    async def stream_audio(self, sentence: str) -> AsyncIterator[bytes]:
        clean = _MARKDOWN_RE.sub("", sentence).strip()
        if not clean:
            return
        communicate = edge_tts.Communicate(clean, self._voice)
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                yield chunk["data"]


# ── F5-TTS ─────────────────────────────────────────────────────────────────

class F5TTSProvider(TTSProvider):
    """
    In-process F5-TTS provider.

    Loads F5TTS directly into the engine venv. Synthesis runs in a thread-pool
    executor so it doesn't block the asyncio event loop.

    Requires: f5_tts, torch, torchaudio
    """

    def __init__(self, model: str, steps: int, speed: float,
                 ref_wav: str, ref_txt: str) -> None:
        self._model     = model
        self._steps     = steps
        self._speed     = speed
        self._ref_wav   = ref_wav
        self._ref_txt   = ref_txt
        self._tts       = None
        self._ref_audio = ""
        self._ref_text  = ""
        self._lock      = asyncio.Lock()

    def load(self) -> None:
        import f5_tts
        import torch
        from f5_tts.api import F5TTS

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("[auricle-engine] f5-tts: loading %r on %s", self._model, device)
        self._tts = F5TTS(model=self._model, device=device)
        self._ref_audio, self._ref_text = self._resolve_ref(f5_tts)
        logger.info("[auricle-engine] f5-tts: ready")

    def _resolve_ref(self, f5_tts_module) -> tuple:
        import os
        bundled = os.path.join(list(f5_tts_module.__path__)[0], F5_BUNDLED_REF_RELPATH)
        have_wav = bool(self._ref_wav)
        have_txt = bool(self._ref_txt)

        if have_wav and have_txt:
            if not os.path.exists(self._ref_wav):
                raise FileNotFoundError(f"F5 ref wav not found: {self._ref_wav}")
            if not os.path.exists(self._ref_txt):
                raise FileNotFoundError(f"F5 ref txt not found: {self._ref_txt}")
            with open(self._ref_txt, encoding="utf-8") as fh:
                text = fh.read().strip()
            logger.info("[auricle-engine] f5-tts: using clone ref: %s", self._ref_wav)
            return self._ref_wav, text

        if have_wav or have_txt:
            logger.warning(
                "[auricle-engine] f5-tts: only one of ref_wav/ref_txt is set; "
                "both required for cloning — falling back to bundled voice"
            )
        return bundled, F5_DEFAULT_REF_TEXT

    def _synth(self, text: str) -> bytes:
        """Blocking synthesis — called via run_in_executor."""
        import numpy as np

        wav, _sr, _ = self._tts.infer(
            ref_file=self._ref_audio,
            ref_text=self._ref_text,
            gen_text=text,
            nfe_step=self._steps,
            speed=self._speed,
            remove_silence=False,
        )
        wav = np.array(wav, dtype=np.float32)
        peak = np.abs(wav).max()
        if peak > 0:
            wav /= peak
        pcm = (np.clip(wav, -1.0, 1.0) * 32767).astype(np.int16)

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(F5_SAMPLE_RATE)
            wf.writeframes(pcm.tobytes())
        return buf.getvalue()

    async def stream_audio(self, sentence: str) -> AsyncIterator[bytes]:
        clean = _MARKDOWN_RE.sub("", sentence).strip()
        if not clean:
            return
        loop = asyncio.get_running_loop()
        async with self._lock:
            wav_bytes = await loop.run_in_executor(None, self._synth, clean)
        if wav_bytes:
            yield wav_bytes

    def terminate(self) -> None:
        self._tts = None


# ── Kokoro-TTS ─────────────────────────────────────────────────────────────

class KokoroTTSProvider(TTSProvider):
    """
    In-process Kokoro-TTS provider.

    Loads KPipeline directly into the engine venv. Synthesis runs in a
    thread-pool executor so it doesn't block the asyncio event loop.

    Requires: kokoro, soundfile; espeak-ng on PATH (phonemizer backend)
    """

    def __init__(self, voice: str) -> None:
        self._voice    = voice
        self._pipeline = None
        self._lock     = asyncio.Lock()

    def load(self) -> None:
        from kokoro import KPipeline
        lang = self._voice[0] if self._voice[:1] in ("a", "b") else "a"
        logger.info("[auricle-engine] kokoro: loading KPipeline (lang=%r, voice=%r)",
                    lang, self._voice)
        self._pipeline = KPipeline(lang_code=lang)
        logger.info("[auricle-engine] kokoro: ready")

    def _synth(self, text: str) -> bytes:
        """Blocking synthesis — called via run_in_executor."""
        import numpy as np

        chunks = []
        for _gs, _ps, audio in self._pipeline(text, voice=self._voice, speed=1.0):
            chunks.append(audio)

        if not chunks:
            return b""

        wav = np.concatenate(chunks).astype(np.float32)
        peak = np.abs(wav).max()
        if peak > 0:
            wav /= peak
        pcm = (np.clip(wav, -1.0, 1.0) * 32767).astype(np.int16)

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(KOKORO_SAMPLE_RATE)
            wf.writeframes(pcm.tobytes())
        return buf.getvalue()

    async def stream_audio(self, sentence: str) -> AsyncIterator[bytes]:
        clean = _MARKDOWN_RE.sub("", sentence).strip()
        if not clean:
            return
        loop = asyncio.get_running_loop()
        async with self._lock:
            wav_bytes = await loop.run_in_executor(None, self._synth, clean)
        if wav_bytes:
            yield wav_bytes

    def terminate(self) -> None:
        self._pipeline = None
