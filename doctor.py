#!/usr/bin/env python3
"""Auricle engine doctor — standalone diagnostic.

Run from the engine directory with the engine venv active:
    python doctor.py

Stop any running engine instance first — audio device tests need exclusive
ALSA access and the port-availability check will flag a false positive if
the engine is already listening.
"""
from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

_ENGINE_DIR = Path(__file__).parent
sys.path.insert(0, str(_ENGINE_DIR))

from consts import (
    APLAY_BIN,
    ALL_ASSETS,
    ASSET_NOTIFY,
    DEFAULT_AUDIO_INPUT,
    DEFAULT_AUDIO_OUTPUT,
    DEFAULT_MIC_DEVICE,
    DEFAULT_OWW_EMBEDDING_MODEL_PATH,
    DEFAULT_OWW_MELSPEC_MODEL_PATH,
    DEFAULT_OWW_WAKEWORD_MODEL_PATH,
    DEFAULT_SD_INPUT_DEVICE,
    DEFAULT_SD_OUTPUT_DEVICE,
    DEFAULT_SPEAKER_DEVICE,
    DEFAULT_STT_BACKEND,
    DEFAULT_TTS_BACKEND,
    DEFAULT_VOSK_MODEL_PATH,
    DOCTOR_MIC_SILENCE_THRESHOLD,
    ENGINE_HOST,
    ENGINE_PORT,
    ENV_AUDIO_INPUT,
    ENV_AUDIO_OUTPUT,
    ENV_ENGINE_HOST,
    ENV_ENGINE_PORT,
    ENV_F5_REF_TXT,
    ENV_F5_REF_WAV,
    ENV_MIC_DEVICE,
    ENV_OWW_EMBEDDING_MODEL_PATH,
    ENV_OWW_MELSPEC_MODEL_PATH,
    ENV_OWW_WAKEWORD_MODEL_PATH,
    ENV_SD_INPUT_DEVICE,
    ENV_SD_OUTPUT_DEVICE,
    ENV_SPEAKER_DEVICE,
    ENV_STT_BACKEND,
    ENV_TTS_BACKEND,
    ENV_VOSK_MODEL_PATH,
    FFMPEG_BIN,
    SAMPLE_RATE,
)

# ── ANSI output ────────────────────────────────────────────────────────────────

_G, _Y, _R, _C, _B, _D, _X = (
    "\033[32m", "\033[33m", "\033[31m", "\033[36m",
    "\033[1m",  "\033[2m",  "\033[0m",
)


def _c(s: str, *codes: str) -> str:
    return ("".join(codes) + s + _X) if sys.stdout.isatty() else s


def _ok(t: str, d: str = "") -> None:
    print(f"  {_c('✓', _G)} {t}" + (f" {_c(d, _D)}" if d else ""))


def _warn(t: str, d: str = "") -> None:
    print(f"  {_c('⚠', _Y)} {t}" + (f" {_c(d, _D)}" if d else ""))


def _fail(t: str, d: str = "", issues: list[str] | None = None) -> None:
    print(f"  {_c('✗', _R)} {t}" + (f" {_c(d, _D)}" if d else ""))
    if issues is not None:
        issues.append(t + (f" — {d}" if d else ""))


def _info(t: str) -> None:
    print(f"    {_c('→', _C)} {t}")


def _sec(title: str) -> None:
    print()
    print(_c(f"◆ {title}", _C, _B))


# ── port availability check ────────────────────────────────────────────────────

def _engine_port_in_use(host: str, port: int) -> bool:
    """Return True if the engine's WebSocket port is already bound."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
            return False
        except OSError:
            return True


# ── sounddevice device resolution ─────────────────────────────────────────────

def _sd_device(env_var: str, default: str) -> int | str | None:
    val = os.getenv(env_var, default).strip()
    if not val:
        return None
    try:
        return int(val)
    except ValueError:
        return val


# ── peak amplitude helper ──────────────────────────────────────────────────────

def _report_peak(peak: int, issues: list[str]) -> None:
    if peak == 0:
        _fail("Mic device opened", "all-zero samples — hardware mute or driver issue", issues)
    elif peak < DOCTOR_MIC_SILENCE_THRESHOLD:
        _warn("Mic device opened, low signal",
              f"peak {peak}/32767 — may be muted or room is silent")
    else:
        _ok("Mic device OK", f"peak {peak}/32767")


# ── section A: environment ─────────────────────────────────────────────────────

def _check_env() -> None:
    _sec("Environment")

    # Look for .env next to doctor.py (the engine repo root).
    env_path = _ENGINE_DIR / ".env"
    if not env_path.exists():
        _warn(".env not found", f"({env_path}) — vars must already be set in shell")
        return

    try:
        from dotenv import load_dotenv
    except ImportError:
        _warn(".env found but python-dotenv not installed",
              "pip install python-dotenv  (optional — env vars already in shell work fine)")
        return

    try:
        load_dotenv(dotenv_path=env_path, override=True, encoding="utf-8")
    except UnicodeDecodeError:
        try:
            load_dotenv(dotenv_path=env_path, override=True, encoding="latin-1")
        except Exception as e:
            _warn(".env load failed", str(e))
            return
    except Exception as e:
        _warn(".env load failed", str(e))
        return

    _ok(".env loaded", f"({env_path})")


# ── section B: active configuration ───────────────────────────────────────────

def _check_config(issues: list[str]) -> tuple[str, str, str, str, str, str]:
    """Validate and display active config.
    Returns (stt_backend, tts_backend, audio_in, audio_out, mic_device, spk_device)."""
    _sec("Active Configuration")

    stt_backend = os.getenv(ENV_STT_BACKEND, DEFAULT_STT_BACKEND).lower()
    tts_backend = os.getenv(ENV_TTS_BACKEND, DEFAULT_TTS_BACKEND).lower()
    audio_in    = os.getenv(ENV_AUDIO_INPUT,  DEFAULT_AUDIO_INPUT).lower()
    audio_out   = os.getenv(ENV_AUDIO_OUTPUT, DEFAULT_AUDIO_OUTPUT).lower()
    mic_device  = os.getenv(ENV_MIC_DEVICE,   DEFAULT_MIC_DEVICE)
    spk_device  = os.getenv(ENV_SPEAKER_DEVICE, DEFAULT_SPEAKER_DEVICE)
    ws_host     = os.getenv(ENV_ENGINE_HOST, ENGINE_HOST)
    ws_port     = int(os.getenv(ENV_ENGINE_PORT, str(ENGINE_PORT)))

    if stt_backend in ("vosk", "whisper"):
        _ok(f"STT backend:  {stt_backend}")
    else:
        _fail(f"STT backend: {stt_backend!r}", "must be 'vosk' or 'whisper'", issues)
        stt_backend = DEFAULT_STT_BACKEND

    if tts_backend in ("edge-tts", "f5-tts", "kokoro-tts"):
        _ok(f"TTS backend:  {tts_backend}")
    else:
        _fail(f"TTS backend: {tts_backend!r}",
              "must be 'edge-tts', 'f5-tts', or 'kokoro-tts'", issues)
        tts_backend = DEFAULT_TTS_BACKEND

    if audio_in in ("arecord", "sounddevice"):
        _ok(f"Audio input:  {audio_in}")
    else:
        _fail(f"Audio input: {audio_in!r}", "must be 'arecord' or 'sounddevice'", issues)
        audio_in = DEFAULT_AUDIO_INPUT

    if audio_out in ("aplay", "sounddevice"):
        _ok(f"Audio output: {audio_out}")
    else:
        _fail(f"Audio output: {audio_out!r}", "must be 'aplay' or 'sounddevice'", issues)
        audio_out = DEFAULT_AUDIO_OUTPUT

    mic_note = " (default)" if mic_device == DEFAULT_MIC_DEVICE else ""
    spk_note = " (default)" if spk_device == DEFAULT_SPEAKER_DEVICE else ""
    _info(f"Mic device:     {mic_device}{mic_note}")
    _info(f"Speaker device: {spk_device}{spk_note}")
    _info(f"WebSocket:      ws://{ws_host}:{ws_port}")

    if _engine_port_in_use(ws_host if ws_host != "0.0.0.0" else "127.0.0.1", ws_port):
        _warn(
            f"Port {ws_port} is already in use",
            "audio device tests may fail — stop the running engine first",
        )

    return stt_backend, tts_backend, audio_in, audio_out, mic_device, spk_device


# ── section C: python dependencies ────────────────────────────────────────────

def _check_python_deps(issues: list[str], stt_backend: str, tts_backend: str,
                       audio_in: str, audio_out: str) -> None:
    _sec("Python Dependencies")

    for pkg, label, pip_hint in [
        ("openwakeword", "openwakeword", "openwakeword"),
        ("numpy",        "numpy",        "numpy"),
        ("websockets",   "websockets",   "websockets"),
    ]:
        try:
            __import__(pkg)
            _ok(label)
        except ImportError:
            _fail(label, f"pip install {pip_hint}", issues)

    if tts_backend == "edge-tts":
        try:
            __import__("edge_tts")
            _ok("edge-tts")
        except ImportError:
            _fail("edge-tts", "pip install edge-tts", issues)
    else:
        _info("edge-tts: skipped (non-edge backend)")

    if stt_backend == "vosk":
        try:
            __import__("vosk")
            _ok("vosk")
        except ImportError:
            _fail("vosk", "pip install vosk", issues)
    else:
        _info("vosk: skipped (whisper backend)")

    # whisper, f5-tts, and kokoro deps are checked in their dedicated sections below

    if audio_in == "sounddevice" or audio_out == "sounddevice":
        try:
            __import__("sounddevice")
            _ok("sounddevice")
        except ImportError:
            _fail("sounddevice", "pip install sounddevice", issues)
    else:
        _info("sounddevice: skipped (arecord/aplay backend)")


# ── section D: system binaries ─────────────────────────────────────────────────

def _check_binaries(issues: list[str], audio_in: str, audio_out: str) -> dict[str, bool]:
    _sec("System Binaries")
    found: dict[str, bool] = {}

    if audio_in == "arecord":
        if shutil.which("arecord"):
            _ok("arecord")
            found["arecord"] = True
        else:
            _fail("arecord", "not found on PATH — install alsa-utils", issues)
            found["arecord"] = False

    if audio_out == "aplay":
        for name, hint in [(APLAY_BIN, "alsa-utils"), (FFMPEG_BIN, "ffmpeg")]:
            if shutil.which(name):
                _ok(name)
                found[name] = True
            else:
                _fail(name, f"not found on PATH — install {hint}", issues)
                found[name] = False

    if not found:
        _info("No ALSA binaries required by current backend")

    return found


# ── section E: model & asset files ────────────────────────────────────────────

def _check_files(issues: list[str], stt_backend: str) -> None:
    _sec("Model & Asset Files")

    for env, default, label in [
        (ENV_OWW_WAKEWORD_MODEL_PATH,  DEFAULT_OWW_WAKEWORD_MODEL_PATH,  "OWW wakeword model"),
        (ENV_OWW_MELSPEC_MODEL_PATH,   DEFAULT_OWW_MELSPEC_MODEL_PATH,   "OWW melspec model"),
        (ENV_OWW_EMBEDDING_MODEL_PATH, DEFAULT_OWW_EMBEDDING_MODEL_PATH, "OWW embedding model"),
    ]:
        p = Path(os.path.expanduser(os.getenv(env, default)))
        if p.exists():
            _ok(label, f"({p.name})")
        else:
            _fail(label, f"not found: {p}", issues)

    if stt_backend == "vosk":
        vosk_path = Path(os.path.expanduser(os.getenv(ENV_VOSK_MODEL_PATH, DEFAULT_VOSK_MODEL_PATH)))
        if not vosk_path.exists():
            _fail("Vosk model", f"not found: {vosk_path}", issues)
        elif not vosk_path.is_dir():
            _fail("Vosk model", f"not a directory: {vosk_path}", issues)
        elif (vosk_path / "conf").is_dir() and (vosk_path / "am").is_dir():
            _ok("Vosk model", f"({vosk_path.name})")
        else:
            _warn("Vosk model directory exists but looks incomplete",
                  "missing conf/ or am/ — download may be partial")
    else:
        _info("Vosk model: skipped (whisper backend)")

    for asset in ALL_ASSETS:
        if asset.exists():
            _ok(f"Asset: {asset.name}")
        else:
            _fail(f"Asset: {asset.name}", f"not found: {asset}", issues)


# ── section F: whisper dependencies ───────────────────────────────────────────

def _check_whisper_deps(issues: list[str]) -> None:
    _sec("Whisper STT Dependencies")

    for pkg, pip_hint in [
        ("torch",        "torch"),
        ("transformers", "transformers accelerate"),
        ("webrtcvad",    "webrtcvad-wheels"),
    ]:
        try:
            __import__(pkg)
            _ok(pkg)
        except ImportError:
            _fail(pkg, f"pip install {pip_hint}", issues)

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _info(f"torch device: {device}")
    except Exception:
        pass


# ── section G: F5-TTS dependencies ────────────────────────────────────────────

def _check_f5_deps(issues: list[str]) -> None:
    _sec("F5-TTS Dependencies")

    for pkg, pip_hint in [
        ("f5_tts",    "f5-tts"),
        ("torch",     "torch"),
        ("torchaudio","torchaudio"),
    ]:
        try:
            __import__(pkg)
            _ok(pkg)
        except ImportError:
            _fail(pkg, f"pip install {pip_hint}", issues)

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _info(f"torch device: {device}")
    except Exception:
        pass

    ref_wav = os.path.expanduser(os.getenv(ENV_F5_REF_WAV, "").strip())
    ref_txt = os.path.expanduser(os.getenv(ENV_F5_REF_TXT, "").strip())
    if ref_wav and ref_txt:
        for path, label in [(ref_wav, "AURICLE_F5_REF_WAV"), (ref_txt, "AURICLE_F5_REF_TXT")]:
            if Path(path).exists():
                _ok(label, f"({path})")
            else:
                _fail(label, f"file not found: {path}", issues)
    elif ref_wav or ref_txt:
        _warn("F5 ref config",
              "only one of AURICLE_F5_REF_WAV / AURICLE_F5_REF_TXT is set; "
              "both are required for cloning — will fall back to bundled voice")
    else:
        _info("F5 ref: not configured — will use bundled voice")


# ── section H: Kokoro-TTS dependencies ────────────────────────────────────────

def _check_kokoro_deps(issues: list[str]) -> None:
    _sec("Kokoro-TTS Dependencies")

    for pkg, pip_hint in [
        ("kokoro",    "kokoro"),
        ("soundfile", "soundfile"),
    ]:
        try:
            __import__(pkg)
            _ok(pkg)
        except ImportError:
            _fail(pkg, f"pip install {pip_hint}", issues)

    if shutil.which("espeak-ng"):
        _ok("espeak-ng (phonemizer backend)")
    else:
        _fail("espeak-ng",
              "not found on PATH — install with: apt-get install espeak-ng", issues)


# ── section I: audio devices ──────────────────────────────────────────────────

def _check_audio_devices(
    issues: list[str],
    audio_in: str,
    audio_out: str,
    mic_device: str,
    spk_device: str,
    binaries_ok: dict[str, bool],
) -> None:
    _sec("Audio Devices")

    if audio_in == "arecord":
        if not binaries_ok.get("arecord", False):
            _warn("Mic capture: skipped", "(arecord not found — see System Binaries above)")
        else:
            _check_mic_arecord(mic_device, issues)
    else:
        _check_mic_sounddevice(issues)

    if audio_out == "aplay":
        if not binaries_ok.get(APLAY_BIN, False):
            _warn("Speaker playback: skipped", "(aplay not found — see System Binaries above)")
        else:
            _check_speaker_aplay(spk_device, issues)
    else:
        _check_speaker_sounddevice(issues)


def _check_mic_arecord(device: str, issues: list[str]) -> None:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    try:
        result = subprocess.run(
            ["arecord", "-D", device, "-d", "1",
             "-f", "S16_LE", "-r", "16000", "-c", "1", "-q", tmp.name],
            capture_output=True,
            timeout=5,
        )
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace").strip()
            _fail("Mic capture (arecord)", stderr or "non-zero exit", issues)
            return
        try:
            import numpy as np
            with wave.open(tmp.name, "rb") as wf:
                raw = wf.readframes(wf.getnframes())
            samples = np.frombuffer(raw, dtype=np.int16)
            _report_peak(int(np.max(np.abs(samples))) if samples.size else 0, issues)
        except ImportError:
            _ok("Mic capture (arecord)", "(numpy unavailable — amplitude check skipped)")
        except Exception as e:
            _warn("Mic capture succeeded", f"amplitude check failed: {e}")
    except subprocess.TimeoutExpired:
        _fail("Mic capture (arecord)", "timed out (>5s)", issues)
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


def _check_speaker_aplay(device: str, issues: list[str]) -> None:
    result = subprocess.run(
        [APLAY_BIN, "-D", device, "-q", str(ASSET_NOTIFY)],
        capture_output=True,
        timeout=10,
    )
    if result.returncode == 0:
        _ok("Speaker playback (aplay)", "(notify chime played)")
    else:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        _fail("Speaker playback (aplay)", stderr or "non-zero exit", issues)


def _check_mic_sounddevice(issues: list[str]) -> None:
    try:
        import sounddevice as sd
        import numpy as np
    except ImportError:
        _warn("Mic capture (sounddevice): skipped",
              "(sounddevice or numpy not importable)")
        return

    device = _sd_device(ENV_SD_INPUT_DEVICE, DEFAULT_SD_INPUT_DEVICE)
    _info(f"sounddevice input: {repr(device) if device is not None else 'system default'}")

    try:
        recording = sd.rec(
            int(1.0 * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            device=device,
        )
        sd.wait()
        _report_peak(int(np.max(np.abs(recording))), issues)
    except Exception as e:
        _fail("Mic capture (sounddevice)", str(e), issues)


def _check_speaker_sounddevice(issues: list[str]) -> None:
    try:
        import sounddevice as sd
        import numpy as np
    except ImportError:
        _warn("Speaker playback (sounddevice): skipped",
              "(sounddevice or numpy not importable)")
        return

    device = _sd_device(ENV_SD_OUTPUT_DEVICE, DEFAULT_SD_OUTPUT_DEVICE)
    _info(f"sounddevice output: {repr(device) if device is not None else 'system default'}")

    try:
        with wave.open(str(ASSET_NOTIFY), "rb") as wf:
            raw = wf.readframes(wf.getnframes())
            samplerate = wf.getframerate()
        samples = np.frombuffer(raw, dtype=np.int16)
        sd.play(samples, samplerate=samplerate, device=device)
        sd.wait()
        _ok("Speaker playback (sounddevice)", "(notify chime played)")
    except Exception as e:
        _fail("Speaker playback (sounddevice)", str(e), issues)


# ── entry point ────────────────────────────────────────────────────────────────

def run_doctor() -> int:
    issues: list[str] = []

    print()
    print(_c("┌─────────────────────────────────────────────────────────┐", _C))
    print(_c("│           🩺 Auricle Engine Doctor                      │", _C))
    print(_c("└─────────────────────────────────────────────────────────┘", _C))

    _check_env()
    stt_backend, tts_backend, audio_in, audio_out, mic_device, spk_device = _check_config(issues)
    _check_python_deps(issues, stt_backend, tts_backend, audio_in, audio_out)
    binaries_ok = _check_binaries(issues, audio_in, audio_out)
    _check_files(issues, stt_backend)

    if stt_backend == "whisper":
        _check_whisper_deps(issues)

    if tts_backend == "f5-tts":
        _check_f5_deps(issues)

    if tts_backend == "kokoro-tts":
        _check_kokoro_deps(issues)

    _check_audio_devices(issues, audio_in, audio_out, mic_device, spk_device, binaries_ok)

    print()
    if issues:
        print(_c("─" * 60, _Y))
        print(_c(f"  Found {len(issues)} issue(s):", _Y, _B))
        print()
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        print()
        return 1

    print(_c("─" * 60, _G))
    print(_c("  All checks passed.", _G, _B))
    print()
    return 0


if __name__ == "__main__":
    sys.exit(run_doctor())
