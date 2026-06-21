"""
auricle-engine entry point.

Usage:
    python __main__.py
    python -m auricle_engine   (if installed as a package)
"""

import asyncio
import logging
import os
import threading
from pathlib import Path

# Load .env from the engine directory before reading any env vars.
# Silent if python-dotenv is not installed or .env doesn't exist.
def _load_dotenv() -> None:
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=env_path, override=True, encoding="utf-8-sig")
    except ImportError:
        pass

_load_dotenv()

from consts import (
    AUDIO_CHUNK_BYTES,
    AUDIO_RING_BUFFER_CHUNKS,
    DEFAULT_ACTIVE_LISTEN_DURATION,
    DEFAULT_AUDIO_INPUT,
    DEFAULT_AUDIO_OUTPUT,
    DEFAULT_F5_MODEL,
    DEFAULT_F5_SPEED,
    DEFAULT_F5_STEPS,
    DEFAULT_KOKORO_VOICE,
    DEFAULT_MIC_DEVICE,
    DEFAULT_MUTE,
    DEFAULT_OWW_EMBEDDING_MODEL_PATH,
    DEFAULT_OWW_MELSPEC_MODEL_PATH,
    DEFAULT_OWW_WAKEWORD_MODEL_PATH,
    DEFAULT_SD_INPUT_DEVICE,
    DEFAULT_SD_OUTPUT_DEVICE,
    DEFAULT_SLEEP_FLUX_THRESHOLD,
    DEFAULT_SLEEP_TIMEOUT,
    DEFAULT_SLEEP_WAKE_SENSITIVITY,
    DEFAULT_SPEAKER_DEVICE,
    DEFAULT_STT_BACKEND,
    DEFAULT_TTS_BACKEND,
    DEFAULT_TTS_VOICE,
    DEFAULT_VOSK_MODEL_PATH,
    DEFAULT_WHISPER_MODEL_ID,
    ENV_ACTIVE_LISTEN_DURATION,
    ENV_AUDIO_INPUT,
    ENV_AUDIO_OUTPUT,
    ENV_F5_MODEL,
    ENV_F5_REF_TXT,
    ENV_F5_REF_WAV,
    ENV_F5_SPEED,
    ENV_F5_STEPS,
    ENV_KOKORO_VOICE,
    ENV_MIC_DEVICE,
    ENV_MUTE,
    ENV_OWW_EMBEDDING_MODEL_PATH,
    ENV_OWW_MELSPEC_MODEL_PATH,
    ENV_OWW_WAKEWORD_MODEL_PATH,
    ENV_SD_INPUT_DEVICE,
    ENV_SD_OUTPUT_DEVICE,
    ENV_SLEEP_FLUX_THRESHOLD,
    ENV_SLEEP_TIMEOUT,
    ENV_SLEEP_WAKE_SENSITIVITY,
    ENV_SPEAKER_DEVICE,
    ENV_STT_BACKEND,
    ENV_TTS_BACKEND,
    ENV_TTS_VOICE,
    ENV_VOSK_MODEL_PATH,
    ENV_WHISPER_MODEL_ID,
    OWW_THRESHOLD,
    SAMPLE_RATE,
    SLEEP_EMA_ALPHA,
    TTS_ECHO_TAIL_SECONDS,
)
from audio_buffer import AudioBuffer
from audio_io import (
    AplayOutput,
    ArecordInput,
    SounddeviceInput,
    SounddeviceOutput,
)
from egress import EgressController
from fsm import FSM, State
from ingress import run_ingress_loop
from providers import (
    EdgeTTSProvider,
    F5TTSProvider,
    KokoroTTSProvider,
    VoskSTTProvider,
    WhisperSTTProvider,
)
from client import AuricleClient
from sleep import SleepDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _parse_bool(value: str) -> bool:
    return str(value).lower() in ("1", "true", "yes", "on")


def _parse_sd_device(value: str | None):
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return value


def _make_audio_input():
    backend = os.getenv(ENV_AUDIO_INPUT, DEFAULT_AUDIO_INPUT).lower()
    if backend == "sounddevice":
        return SounddeviceInput(
            device=_parse_sd_device(os.getenv(ENV_SD_INPUT_DEVICE, DEFAULT_SD_INPUT_DEVICE))
        )
    return ArecordInput(device=os.getenv(ENV_MIC_DEVICE, DEFAULT_MIC_DEVICE))


def _make_audio_output():
    backend = os.getenv(ENV_AUDIO_OUTPUT, DEFAULT_AUDIO_OUTPUT).lower()
    if backend == "sounddevice":
        return SounddeviceOutput(
            device=_parse_sd_device(os.getenv(ENV_SD_OUTPUT_DEVICE, DEFAULT_SD_OUTPUT_DEVICE))
        )
    return AplayOutput(device=os.getenv(ENV_SPEAKER_DEVICE, DEFAULT_SPEAKER_DEVICE))


async def main() -> None:
    loop = asyncio.get_running_loop()

    # ── STT provider ───────────────────────────────────────────────────────
    stt_backend = os.getenv(ENV_STT_BACKEND, DEFAULT_STT_BACKEND).lower()
    if stt_backend == "whisper":
        stt = WhisperSTTProvider(
            model_id=os.getenv(ENV_WHISPER_MODEL_ID, DEFAULT_WHISPER_MODEL_ID),
        )
    else:
        stt = VoskSTTProvider(
            os.path.expanduser(os.getenv(ENV_VOSK_MODEL_PATH, DEFAULT_VOSK_MODEL_PATH))
        )

    # ── TTS provider ───────────────────────────────────────────────────────
    tts_backend = os.getenv(ENV_TTS_BACKEND, DEFAULT_TTS_BACKEND).lower()
    if tts_backend == "f5-tts":
        tts = F5TTSProvider(
            model=os.getenv(ENV_F5_MODEL, DEFAULT_F5_MODEL),
            steps=int(os.getenv(ENV_F5_STEPS, str(DEFAULT_F5_STEPS))),
            speed=float(os.getenv(ENV_F5_SPEED, str(DEFAULT_F5_SPEED))),
            ref_wav=os.path.expanduser(os.getenv(ENV_F5_REF_WAV, "")),
            ref_txt=os.path.expanduser(os.getenv(ENV_F5_REF_TXT, "")),
        )
    elif tts_backend == "kokoro-tts":
        tts = KokoroTTSProvider(
            voice=os.getenv(ENV_KOKORO_VOICE, DEFAULT_KOKORO_VOICE),
        )
    else:
        tts = EdgeTTSProvider(os.getenv(ENV_TTS_VOICE, DEFAULT_TTS_VOICE))

    # ── Audio I/O ──────────────────────────────────────────────────────────
    audio_input  = _make_audio_input()
    audio_output = _make_audio_output()

    # ── Core objects ───────────────────────────────────────────────────────
    fsm          = FSM()
    barge_in     = asyncio.Event()
    audio_buffer = AudioBuffer(AUDIO_RING_BUFFER_CHUNKS, tts_tail_seconds=TTS_ECHO_TAIL_SECONDS)
    egress       = EgressController(tts, barge_in, audio_buffer, audio_output)
    stop_event   = threading.Event()
    client       = AuricleClient(fsm, egress, audio_output, stop_event)
    client.set_loop(loop)

    # ── Load STT ───────────────────────────────────────────────────────────
    logger.info("[auricle-engine] loading STT (%s)", type(stt).__name__)
    stt.load()

    # ── Load TTS model ─────────────────────────────────────────────────────
    if isinstance(tts, (F5TTSProvider, KokoroTTSProvider)):
        logger.info("[auricle-engine] loading TTS model (%s)", type(tts).__name__)
        tts.load()

    # ── Load OWW ───────────────────────────────────────────────────────────
    ww_path  = os.path.expanduser(os.getenv(ENV_OWW_WAKEWORD_MODEL_PATH,  DEFAULT_OWW_WAKEWORD_MODEL_PATH))
    ms_path  = os.path.expanduser(os.getenv(ENV_OWW_MELSPEC_MODEL_PATH,   DEFAULT_OWW_MELSPEC_MODEL_PATH))
    emb_path = os.path.expanduser(os.getenv(ENV_OWW_EMBEDDING_MODEL_PATH, DEFAULT_OWW_EMBEDDING_MODEL_PATH))

    logger.info("[auricle-engine] loading OWW model: %s", os.path.basename(ww_path))
    from openwakeword.model import Model as OWWModel
    oww = OWWModel(
        wakeword_models=[ww_path],
        melspec_model_path=ms_path,
        embedding_model_path=emb_path,
        inference_framework="onnx",
    )
    wakeword_key = os.path.splitext(os.path.basename(ww_path))[0]

    # ── Open audio input ───────────────────────────────────────────────────
    audio_input.open()

    # ── Sleep detector ─────────────────────────────────────────────────────
    sleep_detector = SleepDetector(
        timeout_seconds=float(os.getenv(ENV_SLEEP_TIMEOUT, str(DEFAULT_SLEEP_TIMEOUT))),
        sample_rate=SAMPLE_RATE,
        chunk_bytes=AUDIO_CHUNK_BYTES,
        flux_threshold=float(os.getenv(ENV_SLEEP_FLUX_THRESHOLD, str(DEFAULT_SLEEP_FLUX_THRESHOLD))),
        wake_multiplier=float(os.getenv(ENV_SLEEP_WAKE_SENSITIVITY, str(DEFAULT_SLEEP_WAKE_SENSITIVITY))),
        ema_alpha=SLEEP_EMA_ALPHA,
    )

    fsm.muted = _parse_bool(os.getenv(ENV_MUTE, str(DEFAULT_MUTE)))

    # ── Start ingress thread ───────────────────────────────────────────────
    ingress_thread = threading.Thread(
        target=run_ingress_loop,
        name="auricle-ingress",
        daemon=True,
        kwargs=dict(
            audio_input=audio_input,
            audio_output=audio_output,
            oww=oww,
            wakeword_key=wakeword_key,
            stt_provider=stt,
            egress=egress,
            audio_buffer=audio_buffer,
            fsm=fsm,
            loop=loop,
            dispatch_fn=client.get_dispatch_fn(),
            stop_event=stop_event,
            active_listen_duration=float(
                os.getenv(ENV_ACTIVE_LISTEN_DURATION, str(DEFAULT_ACTIVE_LISTEN_DURATION))
            ),
            oww_threshold=OWW_THRESHOLD,
            sleep_detector=sleep_detector,
        ),
    )
    fsm.transition(State.IDLE)
    ingress_thread.start()
    logger.info("[auricle-engine] ingress started — listening for wakeword")

    # ── Connect to connector and run (blocks until cancelled) ──────────────
    try:
        await client.run()
    finally:
        stop_event.set()
        if isinstance(stt, WhisperSTTProvider):
            stt.terminate()
        if isinstance(tts, (F5TTSProvider, KokoroTTSProvider)):
            tts.terminate()
        audio_input.close()
        ingress_thread.join(timeout=3)
        logger.info("[auricle-engine] shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
