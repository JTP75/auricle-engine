#!/usr/bin/env python3
"""
F5-TTS worker — runs in ~/f5-venv (Python 3.10, f5_tts, torch+CUDA).
Spawned by F5TTSProvider.load() in the hermes-auricle plugin. Do not run directly.

Protocol:
  Handshake (stdout, line):  READY\\n            → model loaded, ref resolved

  Request (stdin, binary):
    0x01 + uint32_be(len) + <len> UTF-8 text     → synthesize
    0x03  | EOF                                   → shutdown

  Response per synth (stdout, binary):
    0x01 + uint32_be(n) + <n> WAV bytes          → success (RIFF, 24 kHz mono s16le)
    0x00 + uint32_be(0)                          → empty / blank text, no audio
    0x02 + uint32_be(n) + <n> UTF-8 error text   → synthesis error
"""

import os
import sys

# Save the real IPC pipe (fd 1) before any imported library can write to it.
# Some f5_tts / torchaudio code paths print to sys.stdout; after this point
# those writes land on stderr (fd 2) instead of corrupting the binary protocol.
_IPC = os.fdopen(os.dup(1), "wb")
os.dup2(2, 1)
sys.stdout = sys.stderr

import argparse
import io
import struct
import wave

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from consts import (  # noqa: E402
    F5_BUNDLED_REF_RELPATH,
    F5_DEFAULT_REF_TEXT,
    F5_SAMPLE_RATE,
)


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _err(msg: str) -> None:
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


def _read_exact(fp, n: int) -> bytes:
    """Read exactly n bytes from a binary file-like, blocking until available."""
    buf = b""
    while len(buf) < n:
        chunk = fp.read(n - len(buf))
        if not chunk:
            return buf  # EOF — caller checks length
        buf += chunk
    return buf


def _out_line(text: str) -> None:
    """Write a text line to the IPC pipe (used for READY handshake)."""
    _IPC.write(text.encode() + b"\n")
    _IPC.flush()


def _out_frame(status: int, payload: bytes) -> None:
    """Write a binary response frame: 1-byte status + uint32_be(len) + payload."""
    _IPC.write(bytes([status]) + struct.pack(">I", len(payload)) + payload)
    _IPC.flush()


# ── Ref resolution ────────────────────────────────────────────────────────────

def _resolve_ref(ref_wav: str, ref_txt: str) -> tuple:
    """
    Return (ref_audio_path, ref_text) to use for synthesis.

    Both set → use them (error-exit if either file is missing).
    Exactly one set → warn + fall back to bundled sample.
    Neither set → bundled sample.
    """
    import f5_tts

    bundled_wav = os.path.join(list(f5_tts.__path__)[0], F5_BUNDLED_REF_RELPATH)

    have_wav = bool(ref_wav)
    have_txt = bool(ref_txt)

    if have_wav and have_txt:
        if not os.path.exists(ref_wav):
            _err(f"[f5_worker] ref wav not found: {ref_wav}")
            sys.exit(1)
        if not os.path.exists(ref_txt):
            _err(f"[f5_worker] ref txt not found: {ref_txt}")
            sys.exit(1)
        with open(ref_txt, "r", encoding="utf-8") as fh:
            text = fh.read().strip()
        _err(f"[f5_worker] using clone ref: {ref_wav}")
        return ref_wav, text

    if have_wav or have_txt:
        _err(
            "[f5_worker] WARNING: only one of --ref-wav / --ref-txt is set; "
            "both are required for cloning. Falling back to bundled voice."
        )

    _err("[f5_worker] using bundled reference voice")
    return bundled_wav, F5_DEFAULT_REF_TEXT


# ── Model loading ─────────────────────────────────────────────────────────────

def _load_model(model: str):
    import torch
    from f5_tts.api import F5TTS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _err(f"[f5_worker] loading {model!r} on {device}")
    return F5TTS(model=model, device=device)


# ── Synthesis ─────────────────────────────────────────────────────────────────

def _synth(tts, ref_audio: str, ref_text: str, text: str,
           steps: int, speed: float) -> bytes:
    """Synthesize text and return RIFF WAV bytes (24 kHz mono s16le)."""
    import numpy as np

    _err("[f5_worker] infer: start")
    wav, _sr, _ = tts.infer(
        ref_file=ref_audio,
        ref_text=ref_text,
        gen_text=text,
        nfe_step=steps,
        speed=speed,
        remove_silence=False,
    )
    _err("[f5_worker] infer: done")
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


# ── Main loop ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   default="F5TTS_v1_Base")
    parser.add_argument("--steps",   type=int,   default=5)
    parser.add_argument("--speed",   type=float, default=1.0)
    parser.add_argument("--ref-wav", default="", dest="ref_wav")
    parser.add_argument("--ref-txt", default="", dest="ref_txt")
    args = parser.parse_args()

    ref_audio, ref_text = _resolve_ref(args.ref_wav, args.ref_txt)
    tts = _load_model(args.model)
    _out_line("READY")

    stdin = sys.stdin.buffer
    while True:
        cmd_byte = stdin.read(1)
        if not cmd_byte:          # EOF — parent closed the pipe
            break
        cmd = cmd_byte[0]

        if cmd == 0x01:
            len_bytes = _read_exact(stdin, 4)
            if len(len_bytes) < 4:
                break
            n = struct.unpack(">I", len_bytes)[0]
            text_bytes = _read_exact(stdin, n)
            if len(text_bytes) < n:
                break
            text = text_bytes.decode("utf-8", errors="replace").strip()

            if not text:
                _out_frame(0x00, b"")
            else:
                try:
                    wav_bytes = _synth(tts, ref_audio, ref_text, text, args.steps, args.speed)
                    _err(f"[f5_worker] responding: {len(wav_bytes)} bytes")
                    _out_frame(0x01, wav_bytes)
                    _err("[f5_worker] response sent")
                except Exception as exc:
                    _err(f"[f5_worker] synthesis error: {exc}")
                    _out_frame(0x02, str(exc).encode("utf-8"))

        elif cmd == 0x03:
            break

        else:
            _err(f"[f5_worker] unknown command byte: {cmd:#04x}")
            _out_frame(0x00, b"")  # don't stall the parent


if __name__ == "__main__":
    main()
