#!/usr/bin/env python3
"""
Kokoro-TTS worker — runs in ~/kokoro-venv (Python 3.10, kokoro 0.9.x).
Spawned by KokoroTTSProvider.load() in auricle-engine. Do not run directly.

Protocol:
  Handshake (stdout, line):  READY\\n            → model loaded

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
# kokoro / huggingface / espeak-ng code paths print to sys.stdout; after this
# point those writes land on stderr (fd 2) instead of corrupting the binary protocol.
_IPC = os.fdopen(os.dup(1), "wb")
os.dup2(2, 1)
sys.stdout = sys.stderr

import argparse
import io
import struct
import wave

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from consts import KOKORO_SAMPLE_RATE  # noqa: E402


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


# ── Model loading ─────────────────────────────────────────────────────────────

def _load_pipeline(voice: str):
    from kokoro import KPipeline

    # Derive lang_code from the voice name prefix (af_/am_ → 'a', bf_/bm_ → 'b').
    lang = voice[0] if voice[:1] in ("a", "b") else "a"
    _err(f"[kokoro_worker] loading KPipeline (lang={lang!r}, voice={voice!r})")
    return KPipeline(lang_code=lang)


# ── Synthesis ─────────────────────────────────────────────────────────────────

def _synth(pipeline, voice: str, text: str) -> bytes:
    """Synthesize text and return RIFF WAV bytes (24 kHz mono s16le)."""
    import numpy as np

    _err("[kokoro_worker] infer: start")
    chunks = []
    for _gs, _ps, audio in pipeline(text, voice=voice, speed=1.0):
        chunks.append(audio)
    _err("[kokoro_worker] infer: done")

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


# ── Main loop ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--voice", default="af_heart")
    args = parser.parse_args()

    pipeline = _load_pipeline(args.voice)
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
                    wav_bytes = _synth(pipeline, args.voice, text)
                    if not wav_bytes:
                        _out_frame(0x00, b"")
                    else:
                        _err(f"[kokoro_worker] responding: {len(wav_bytes)} bytes")
                        _out_frame(0x01, wav_bytes)
                        _err("[kokoro_worker] response sent")
                except Exception as exc:
                    _err(f"[kokoro_worker] synthesis error: {exc}")
                    _out_frame(0x02, str(exc).encode("utf-8"))

        elif cmd == 0x03:
            break

        else:
            _err(f"[kokoro_worker] unknown command byte: {cmd:#04x}")
            _out_frame(0x00, b"")  # don't stall the parent


if __name__ == "__main__":
    main()
