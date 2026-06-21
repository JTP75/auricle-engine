import asyncio
import json
import logging
import re
import struct
import subprocess
import sys
import threading
from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional, Tuple

import edge_tts

from consts import SAMPLE_RATE

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
        """Yield raw MP3 bytes for the sentence as they arrive from the TTS service."""


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


# ── Distil-Whisper STT ─────────────────────────────────────────────────────

class WhisperSTTProvider(STTProvider):
    """
    Subprocess-backed Whisper STT provider.

    Spawns whisper_worker.py in a separate Python 3.10 process that owns model
    loading, webrtcvad framing, and HuggingFace pipeline inference. The parent
    communicates via a binary/line protocol over stdin/stdout pipes.

    Wire protocol:
      IN  0x01 + 1280 bytes  → feed chunk  OUT: OK | P | F:<text>
      IN  0x02               → reset       OUT: OK
      IN  0x03               → shutdown    (no response)
    """

    def __init__(self, model_id: str, python_path: str, worker_path: str) -> None:
        self._model_id    = model_id
        self._python_path = python_path
        self._worker_path = worker_path
        self._proc: Optional[subprocess.Popen] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._ready_event = threading.Event()

    def load(self) -> None:
        if self._proc and self._proc.poll() is None:
            return
        self._ready_event.clear()
        self._proc = None
        self._proc = subprocess.Popen(
            [self._python_path, self._worker_path, "--model-id", self._model_id],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        self._start_stderr_forwarder()
        threading.Thread(target=self._wait_for_ready, daemon=True).start()

    def _wait_for_ready(self) -> None:
        line = self._proc.stdout.readline()
        if line and line.strip() == b"READY":
            self._ready_event.set()
            logger.info("[auricle-engine] Whisper STT worker ready")
        else:
            logger.error(
                "[auricle-engine] Whisper STT worker exited before READY (got %r). "
                "Check AURICLE_WHISPER_PYTHON and its installed dependencies.",
                line,
            )

    def _start_stderr_forwarder(self) -> None:
        proc = self._proc
        def _drain():
            for raw in proc.stderr:
                msg = raw.decode("utf-8", errors="replace").rstrip()
                if msg:
                    logger.debug("[auricle-engine/whisper_worker] %s", msg)
        t = threading.Thread(target=_drain, daemon=True)
        t.start()
        self._stderr_thread = t

    def feed(self, pcm_bytes: bytes) -> Tuple[Optional[str], Optional[str]]:
        if not self._ready_event.wait(timeout=60):
            return None, None
        if self._proc is None or self._proc.poll() is not None:
            raise RuntimeError("WhisperSTTProvider worker is not running")
        self._proc.stdin.write(b"\x01" + pcm_bytes)
        self._proc.stdin.flush()
        line = self._proc.stdout.readline().decode().strip()
        if line == "P":
            return None, "..."
        if line.startswith("F:"):
            return line[2:], None
        return None, None

    def reset(self) -> None:
        if not self._ready_event.is_set():
            return
        if self._proc is None or self._proc.poll() is not None:
            return
        self._proc.stdin.write(b"\x02")
        self._proc.stdin.flush()
        self._proc.stdout.readline()  # consume OK

    def terminate(self) -> None:
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.stdin.write(b"\x03")
                self._proc.stdin.flush()
            except OSError:
                pass
            self._proc.terminate()
        self._proc = None
        self._ready_event.clear()


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

def _read_exact_pipe(fp, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = fp.read(n - len(buf))
        if not chunk:
            return buf
        buf += chunk
    return buf


class F5TTSProvider(TTSProvider):
    """
    Subprocess-backed F5-TTS provider.

    Wire protocol:
      Handshake (stdout, line):  READY\\n
      Request  0x01 + uint32_be(len) + <len> text UTF-8  →  synth
      Request  0x03                                       →  shutdown
      Response 0x01 + uint32_be(n) + <n> WAV bytes       →  success
      Response 0x00 + uint32_be(0)                        →  blank input, no audio
      Response 0x02 + uint32_be(n) + <n> UTF-8 error      →  synthesis error
    """

    def __init__(self, model: str, python_path: str, worker_path: str,
                 steps: int, speed: float, ref_wav: str, ref_txt: str) -> None:
        self._model       = model
        self._python_path = python_path
        self._worker_path = worker_path
        self._steps       = steps
        self._speed       = speed
        self._ref_wav     = ref_wav
        self._ref_txt     = ref_txt
        self._proc: Optional[subprocess.Popen] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._lock = asyncio.Lock()
        self._ready_event = threading.Event()

    def load(self) -> None:
        if self._proc and self._proc.poll() is None:
            return
        self._ready_event.clear()
        self._proc = None
        self._proc = subprocess.Popen(
            [
                self._python_path, self._worker_path,
                "--model",   self._model,
                "--steps",   str(self._steps),
                "--speed",   str(self._speed),
                "--ref-wav", self._ref_wav,
                "--ref-txt", self._ref_txt,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        self._start_stderr_forwarder()
        threading.Thread(target=self._wait_for_ready, daemon=True).start()

    def _wait_for_ready(self) -> None:
        line = self._proc.stdout.readline()
        if line and line.strip() == b"READY":
            self._ready_event.set()
            logger.info("[auricle-engine] F5-TTS worker ready")
        else:
            logger.error(
                "[auricle-engine] F5-TTS worker exited before READY (got %r). "
                "Check AURICLE_F5_PYTHON, its installed dependencies, and ref file paths.",
                line,
            )

    def _start_stderr_forwarder(self) -> None:
        proc = self._proc
        def _drain():
            for raw in proc.stderr:
                msg = raw.decode("utf-8", errors="replace").rstrip()
                if msg:
                    logger.debug("[auricle-engine/f5_worker] %s", msg)
        t = threading.Thread(target=_drain, daemon=True)
        t.start()
        self._stderr_thread = t

    def _synth_request(self, text: str) -> bytes:
        if not self._ready_event.wait(timeout=120):
            raise RuntimeError("F5-TTS worker did not become ready within 120s")
        if self._proc is None or self._proc.poll() is not None:
            raise RuntimeError("F5TTSProvider worker is not running")
        encoded = text.encode("utf-8")
        self._proc.stdin.write(b"\x01" + struct.pack(">I", len(encoded)) + encoded)
        self._proc.stdin.flush()
        status = self._proc.stdout.read(1)
        if not status:
            raise RuntimeError("f5_worker closed stdout unexpectedly")
        n_bytes = _read_exact_pipe(self._proc.stdout, 4)
        if len(n_bytes) < 4:
            raise RuntimeError("f5_worker truncated length field")
        n = struct.unpack(">I", n_bytes)[0]
        payload = _read_exact_pipe(self._proc.stdout, n) if n > 0 else b""
        if status[0] == 0x01:
            return payload
        if status[0] == 0x00:
            return b""
        if status[0] == 0x02:
            logger.error("[auricle-engine/f5_worker] synthesis error: %s",
                         payload.decode("utf-8", errors="replace"))
            return b""
        logger.error("[auricle-engine/f5_worker] unknown response status: %#04x", status[0])
        return b""

    async def stream_audio(self, sentence: str) -> AsyncIterator[bytes]:
        clean = _MARKDOWN_RE.sub("", sentence).strip()
        if not clean:
            return
        loop = asyncio.get_running_loop()
        async with self._lock:
            wav_bytes = await loop.run_in_executor(None, self._synth_request, clean)
        if wav_bytes:
            yield wav_bytes

    def terminate(self) -> None:
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.stdin.write(b"\x03")
                self._proc.stdin.flush()
            except OSError:
                pass
            self._proc.terminate()
        self._proc = None
        self._ready_event.clear()


# ── Kokoro-TTS ─────────────────────────────────────────────────────────────

class KokoroTTSProvider(TTSProvider):
    """
    Subprocess-backed Kokoro-TTS provider. Wire protocol identical to F5TTSProvider.
    """

    def __init__(self, voice: str, python_path: str, worker_path: str) -> None:
        self._voice       = voice
        self._python_path = python_path
        self._worker_path = worker_path
        self._proc: Optional[subprocess.Popen] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._lock = asyncio.Lock()
        self._ready_event = threading.Event()

    def load(self) -> None:
        if self._proc and self._proc.poll() is None:
            return
        self._ready_event.clear()
        self._proc = None
        self._proc = subprocess.Popen(
            [self._python_path, self._worker_path, "--voice", self._voice],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        self._start_stderr_forwarder()
        threading.Thread(target=self._wait_for_ready, daemon=True).start()

    def _wait_for_ready(self) -> None:
        line = self._proc.stdout.readline()
        if line and line.strip() == b"READY":
            self._ready_event.set()
            logger.info("[auricle-engine] Kokoro-TTS worker ready")
        else:
            logger.error(
                "[auricle-engine] Kokoro-TTS worker exited before READY (got %r). "
                "Check AURICLE_KOKORO_PYTHON and its installed dependencies.",
                line,
            )

    def _start_stderr_forwarder(self) -> None:
        proc = self._proc
        def _drain():
            for raw in proc.stderr:
                msg = raw.decode("utf-8", errors="replace").rstrip()
                if msg:
                    logger.debug("[auricle-engine/kokoro_worker] %s", msg)
        t = threading.Thread(target=_drain, daemon=True)
        t.start()
        self._stderr_thread = t

    def _synth_request(self, text: str) -> bytes:
        if not self._ready_event.wait(timeout=120):
            raise RuntimeError("Kokoro-TTS worker did not become ready within 120s")
        if self._proc is None or self._proc.poll() is not None:
            raise RuntimeError("KokoroTTSProvider worker is not running")
        encoded = text.encode("utf-8")
        self._proc.stdin.write(b"\x01" + struct.pack(">I", len(encoded)) + encoded)
        self._proc.stdin.flush()
        status = self._proc.stdout.read(1)
        if not status:
            raise RuntimeError("kokoro_worker closed stdout unexpectedly")
        n_bytes = _read_exact_pipe(self._proc.stdout, 4)
        if len(n_bytes) < 4:
            raise RuntimeError("kokoro_worker truncated length field")
        n = struct.unpack(">I", n_bytes)[0]
        payload = _read_exact_pipe(self._proc.stdout, n) if n > 0 else b""
        if status[0] == 0x01:
            return payload
        if status[0] == 0x00:
            return b""
        if status[0] == 0x02:
            logger.error("[auricle-engine/kokoro_worker] synthesis error: %s",
                         payload.decode("utf-8", errors="replace"))
            return b""
        logger.error("[auricle-engine/kokoro_worker] unknown response status: %#04x", status[0])
        return b""

    async def stream_audio(self, sentence: str) -> AsyncIterator[bytes]:
        clean = _MARKDOWN_RE.sub("", sentence).strip()
        if not clean:
            return
        loop = asyncio.get_running_loop()
        async with self._lock:
            wav_bytes = await loop.run_in_executor(None, self._synth_request, clean)
        if wav_bytes:
            yield wav_bytes

    def terminate(self) -> None:
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.stdin.write(b"\x03")
                self._proc.stdin.flush()
            except OSError:
                pass
            self._proc.terminate()
        self._proc = None
        self._ready_event.clear()
