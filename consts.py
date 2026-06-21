from pathlib import Path
from typing import Tuple

# ── Paths ──────────────────────────────────────────────────────────────────
_ENGINE_DIR = Path(__file__).parent
ASSETS_DIR  = _ENGINE_DIR / "assets"
_MODELS_DIR = _ENGINE_DIR / "models"

ASSET_WAKEUP   = ASSETS_DIR / "auricle-wakeup.wav"
ASSET_TOSLEEP  = ASSETS_DIR / "auricle-tosleep.wav"
ASSET_NOTIFY   = ASSETS_DIR / "auricle-notify.wav"
ASSET_CONFUSED = ASSETS_DIR / "auricle-confused.wav"
ALL_ASSETS: Tuple[Path, ...] = (ASSET_WAKEUP, ASSET_TOSLEEP, ASSET_NOTIFY, ASSET_CONFUSED)

TTS_CLEARED   = "Session cleared."
TTS_STOPPED   = "Run stopped."
TTS_ERROR     = "Something went wrong."
TTS_MAX_CHARS = 3000  # hard cap on characters spoken per response

# ── Misinput guard ─────────────────────────────────────────────────────────
MISINPUT_MAX_CONSECUTIVE = 2

MISINPUT_PHRASES: frozenset = frozenset({
    # Articles
    "the", "a", "an",
    # Possessive determiners
    "my", "your", "his", "her", "its", "their", "our",
    # Prepositions
    "of", "in", "on", "at", "to", "for", "from", "with", "by", "about",
    # Conjunctions
    "and", "or", "but", "if", "so",
    # Bare subject pronouns
    "i", "he", "she", "it", "we", "they",
    # Dangling contractions
    "it's", "he's", "she's", "that's", "there's", "what's", "who's",
    "they're", "we're", "i'm", "i've", "i'd", "i'll", "you're",
    # Two-word mid-sentence fragments (conjunction/preposition + article)
    "and the", "or the", "but the",
    "in the", "on the", "of the", "to the", "for the",
})

# ── Env var names ──────────────────────────────────────────────────────────
ENV_MIC_DEVICE               = "AURICLE_MIC_DEVICE"
ENV_SPEAKER_DEVICE           = "AURICLE_SPEAKER_DEVICE"
ENV_TTS_VOICE                = "AURICLE_TTS_VOICE"
ENV_TTS_BACKEND              = "AURICLE_TTS_BACKEND"
ENV_ACTIVE_LISTEN_DURATION   = "AURICLE_ACTIVE_LISTEN_DURATION"
ENV_MUTE                     = "AURICLE_MUTE"
ENV_STT_BACKEND              = "AURICLE_STT_BACKEND"
ENV_VOSK_MODEL_PATH          = "AURICLE_VOSK_MODEL_PATH"
ENV_WHISPER_MODEL_ID         = "AURICLE_WHISPER_MODEL_ID"
ENV_WHISPER_PYTHON           = "AURICLE_WHISPER_PYTHON"
ENV_F5_PYTHON                = "AURICLE_F5_PYTHON"
ENV_F5_MODEL                 = "AURICLE_F5_MODEL"
ENV_F5_STEPS                 = "AURICLE_F5_STEPS"
ENV_F5_SPEED                 = "AURICLE_F5_SPEED"
ENV_F5_REF_WAV               = "AURICLE_F5_REF_WAV"
ENV_F5_REF_TXT               = "AURICLE_F5_REF_TXT"
ENV_KOKORO_PYTHON            = "AURICLE_KOKORO_PYTHON"
ENV_KOKORO_VOICE             = "AURICLE_KOKORO_VOICE"
ENV_OWW_WAKEWORD_MODEL_PATH  = "AURICLE_OWW_WAKEWORD_MODEL_PATH"
ENV_OWW_MELSPEC_MODEL_PATH   = "AURICLE_OWW_MELSPEC_MODEL_PATH"
ENV_OWW_EMBEDDING_MODEL_PATH = "AURICLE_OWW_EMBEDDING_MODEL_PATH"
ENV_AUDIO_INPUT              = "AURICLE_AUDIO_INPUT"
ENV_AUDIO_OUTPUT             = "AURICLE_AUDIO_OUTPUT"
ENV_SD_INPUT_DEVICE          = "AURICLE_SD_INPUT_DEVICE"
ENV_SD_OUTPUT_DEVICE         = "AURICLE_SD_OUTPUT_DEVICE"
ENV_ENGINE_PORT              = "AURICLE_ENGINE_PORT"
ENV_ENGINE_HOST              = "AURICLE_ENGINE_HOST"

# ── Auto-sleep ─────────────────────────────────────────────────────────────
ENV_SLEEP_TIMEOUT          = "AURICLE_SLEEP_TIMEOUT"
ENV_SLEEP_WAKE_SENSITIVITY = "AURICLE_SLEEP_WAKE_SENSITIVITY"
ENV_SLEEP_FLUX_THRESHOLD   = "AURICLE_SLEEP_FLUX_THRESHOLD"

# ── Defaults ───────────────────────────────────────────────────────────────
DEFAULT_MIC_DEVICE               = "plughw:0,0"
DEFAULT_SPEAKER_DEVICE           = "plughw:0,0"
DEFAULT_TTS_VOICE                = "en-GB-LibbyNeural"
DEFAULT_TTS_BACKEND              = "edge-tts"
DEFAULT_F5_MODEL                 = "F5TTS_v1_Base"
DEFAULT_F5_STEPS                 = 5
DEFAULT_F5_SPEED                 = 1.0
DEFAULT_KOKORO_VOICE             = "af_heart"
DEFAULT_ACTIVE_LISTEN_DURATION   = 5       # seconds
DEFAULT_MUTE                     = False
DEFAULT_STT_BACKEND              = "vosk"
DEFAULT_VOSK_MODEL_PATH          = str(_MODELS_DIR / "vosk-model")
DEFAULT_WHISPER_MODEL_ID         = "distil-whisper/distil-large-v3"
DEFAULT_OWW_WAKEWORD_MODEL_PATH  = str(_MODELS_DIR / "wakeword.onnx")
DEFAULT_OWW_MELSPEC_MODEL_PATH   = str(_MODELS_DIR / "melspectrogram.onnx")
DEFAULT_OWW_EMBEDDING_MODEL_PATH = str(_MODELS_DIR / "embedding_model.onnx")
DEFAULT_AUDIO_INPUT              = "arecord"
DEFAULT_AUDIO_OUTPUT             = "aplay"
DEFAULT_SD_INPUT_DEVICE          = ""
DEFAULT_SD_OUTPUT_DEVICE         = ""
DEFAULT_SLEEP_TIMEOUT            = 60      # seconds of IDLE silence before sleep
DEFAULT_SLEEP_WAKE_SENSITIVITY   = 3.0     # × sleep_baseline → wake threshold
DEFAULT_SLEEP_FLUX_THRESHOLD     = 0.02    # normalized flux EMA "quiet" cutoff

# ── WebSocket server ────────────────────────────────────────────────────────
ENGINE_HOST = "localhost"  # set AURICLE_ENGINE_HOST=0.0.0.0 for remote clients
ENGINE_PORT = 57310

# ── F5 TTS ─────────────────────────────────────────────────────────────────
F5_SAMPLE_RATE         = 24000
F5_BUNDLED_REF_RELPATH = "infer/examples/basic/basic_ref_en.wav"
F5_DEFAULT_REF_TEXT    = "Some call me nature, others call me mother nature."

# ── Kokoro TTS ─────────────────────────────────────────────────────────────
KOKORO_SAMPLE_RATE = 24000

# ── Whisper STT (VAD tuning) ───────────────────────────────────────────────
WHISPER_VAD_AGGRESSIVENESS = 2    # webrtcvad 0 (permissive) – 3 (strict)
WHISPER_VAD_BLOCK_MS       = 30   # VAD frame size; must be 10, 20, or 30
WHISPER_SILENCE_FRAMES     = 25   # consecutive silent frames → end of utterance (~750 ms)
WHISPER_MIN_SPEECH_FRAMES  = 10   # minimum voiced frames before transcribing (~300 ms)
WHISPER_PADDING_MS         = 300  # ring buffer context to avoid clipping word starts

# ── Audio ──────────────────────────────────────────────────────────────────
AUDIO_CHUNK_BYTES         = 1280   # OWW hard requirement: 40ms at 16kHz 16-bit mono
SAMPLE_RATE               = 16000
WHISPER_FRAME_BYTES       = int(SAMPLE_RATE * WHISPER_VAD_BLOCK_MS / 1000) * 2  # 960 bytes
OWW_THRESHOLD             = 0.5
AUDIO_RING_BUFFER_SECONDS = 2.0
AUDIO_RING_BUFFER_CHUNKS  = int(AUDIO_RING_BUFFER_SECONDS * SAMPLE_RATE * 2 / AUDIO_CHUNK_BYTES)  # 50
TTS_ECHO_TAIL_SECONDS     = 0.15

# ── Auto-sleep ─────────────────────────────────────────────────────────────
SLEEP_EMA_ALPHA = 0.01    # ~4-second smoothing at 40ms/chunk

# ── Doctor (doctor.py only) ────────────────────────────────────────────────
DOCTOR_MIC_SILENCE_THRESHOLD = 500  # int16 peak below which mic is warned as silent/muted

# ── Timing ─────────────────────────────────────────────────────────────────
RETRY_DELAY_SECONDS        = 30
PROACTIVE_PRE_SPEECH_PAUSE = 1.0   # seconds of silence after notify chime before TTS

# ── Binaries ───────────────────────────────────────────────────────────────
EDGE_TTS_BIN = "edge-tts"
APLAY_BIN    = "aplay"
FFMPEG_BIN   = "ffmpeg"

# ── Voice commands (exact whole-transcript match, case-insensitive) ─────────
CLEAR_COMMANDS: Tuple[str, ...] = ("clear", "reset", "it's clear", "its clear", "the clear", "clear.", "Clear.")
STOP_COMMANDS:  Tuple[str, ...] = ("stop", "top", "the stop", "its stop", "it's stop", "stop.", "Stop.")

# ── Internal dispatch sentinels ────────────────────────────────────────────
_CMD_CLEAR = "__AURICLE_CLEAR__"
_CMD_STOP  = "__AURICLE_STOP__"
