#!/usr/bin/env python3
"""
Whisper STT worker — runs in a Python 3.10 venv with torch/transformers/webrtcvad.
Spawned by WhisperSTTProvider.load() in the hermes-auricle plugin. Do not run directly.

Protocol (stdin binary, stdout line-based UTF-8):
  IN  0x01 + AUDIO_CHUNK_BYTES bytes  → feed one OWW-cadence chunk
  IN  0x02                             → reset state (wakeword / utterance boundary)
  IN  0x03 or EOF                      → shutdown

  OUT READY\\n    → model loaded, ready for commands
  OUT OK\\n       → chunk accepted, no event
  OUT P\\n        → speech onset detected
  OUT F:<text>\\n → utterance complete, transcript is <text>

One response line is emitted per command. stderr is logging only.
"""

import argparse
import collections
import os
import sys

# Allow importing consts.py from the package directory (no heavy deps there).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from consts import (  # noqa: E402
    AUDIO_CHUNK_BYTES,
    SAMPLE_RATE,
    WHISPER_FRAME_BYTES,
    WHISPER_MIN_SPEECH_FRAMES,
    WHISPER_PADDING_MS,
    WHISPER_SILENCE_FRAMES,
    WHISPER_VAD_AGGRESSIVENESS,
    WHISPER_VAD_BLOCK_MS,
)


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _out(line: str) -> None:
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


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


# ── Model loading ─────────────────────────────────────────────────────────────

def _load_pipeline(model_id: str):
    import torch
    import webrtcvad
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline as hf_pipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if torch.cuda.is_available() else torch.float32

    _err(f"[whisper_worker] loading {model_id!r} on {device}")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    ).to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    pipe = hf_pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        torch_dtype=dtype,
        device=device,
    )
    vad = webrtcvad.Vad(WHISPER_VAD_AGGRESSIVENESS)
    is_multilingual = getattr(model.config, "is_multilingual", False)
    _err(f"[whisper_worker] ready (multilingual={is_multilingual})")
    return pipe, vad, is_multilingual


# ── VAD state ─────────────────────────────────────────────────────────────────

def _make_state() -> dict:
    padding_frames = WHISPER_PADDING_MS // WHISPER_VAD_BLOCK_MS
    return {
        "remainder":     b"",
        "voiced_frames": [],
        "ring_buffer":   collections.deque(maxlen=padding_frames),
        "triggered":     False,
        "silence_count": 0,
    }


def _reset_state(state: dict) -> None:
    state["remainder"]     = b""
    state["voiced_frames"] = []
    state["triggered"]     = False
    state["silence_count"] = 0
    state["ring_buffer"].clear()


# ── Feed logic (lifted verbatim from WhisperSTTProvider.feed) ─────────────────

def _feed(pcm_bytes: bytes, state: dict, pipe, vad, is_multilingual: bool) -> str:
    import numpy as np

    # Re-slice incoming 1280-byte OWW chunk into 960-byte VAD frames.
    data   = state["remainder"] + pcm_bytes
    offset = 0
    frames = []
    while offset + WHISPER_FRAME_BYTES <= len(data):
        frames.append(data[offset : offset + WHISPER_FRAME_BYTES])
        offset += WHISPER_FRAME_BYTES
    state["remainder"] = data[offset:]

    has_onset = False

    for pcm in frames:
        is_speech = vad.is_speech(pcm, SAMPLE_RATE)

        if not state["triggered"]:
            state["ring_buffer"].append((pcm, is_speech))
            num_voiced = sum(1 for _, s in state["ring_buffer"] if s)
            if num_voiced > 0.8 * state["ring_buffer"].maxlen:
                state["triggered"]     = True
                state["voiced_frames"] = [p for p, _ in state["ring_buffer"]]
                state["ring_buffer"].clear()
                state["silence_count"] = 0
                has_onset = True
        else:
            state["voiced_frames"].append(pcm)
            if is_speech:
                state["silence_count"] = 0
            else:
                state["silence_count"] += 1

            if state["silence_count"] >= WHISPER_SILENCE_FRAMES:
                state["triggered"]     = False
                state["silence_count"] = 0

                if len(state["voiced_frames"]) >= WHISPER_MIN_SPEECH_FRAMES:
                    audio_np  = np.frombuffer(b"".join(state["voiced_frames"]), dtype=np.int16)
                    audio_f32 = audio_np.astype(np.float32) / 32768.0
                    gen_kw    = {"language": "english"} if is_multilingual else {}
                    result    = pipe(
                        {"array": audio_f32, "sampling_rate": SAMPLE_RATE},
                        **({"generate_kwargs": gen_kw} if gen_kw else {}),
                    )
                    text = result["text"].strip()
                    if text:
                        state["voiced_frames"] = []
                        state["ring_buffer"].clear()
                        return f"F:{text}"

                state["voiced_frames"] = []
                state["ring_buffer"].clear()

    if has_onset:
        return "P"
    return "OK"


# ── Main loop ─────────────────────────────────────────────────────────────────

def _smoke(pipe, is_multilingual: bool) -> None:
    """Run a single inference pass on synthetic audio and report device + timing."""
    import time
    import numpy as np

    _err("[whisper_worker] smoke: generating 3s of test audio")
    rng   = np.random.default_rng(42)
    audio = rng.integers(-500, 500, size=SAMPLE_RATE * 3, dtype=np.int16).astype(np.float32) / 32768.0

    gen_kw = {"language": "english"} if is_multilingual else {}
    _err("[whisper_worker] smoke: running inference...")
    t0      = time.monotonic()
    result  = pipe(
        {"array": audio, "sampling_rate": SAMPLE_RATE},
        **({"generate_kwargs": gen_kw} if gen_kw else {}),
    )
    elapsed = time.monotonic() - t0

    _err(f"[whisper_worker] smoke OK: {elapsed:.3f}s  transcript={result['text']!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True, help="HuggingFace model ID")
    parser.add_argument("--smoke", action="store_true", help="Run inference smoke test and exit")
    args = parser.parse_args()

    pipe, vad, is_multilingual = _load_pipeline(args.model_id)

    if args.smoke:
        _smoke(pipe, is_multilingual)
        return

    state = _make_state()
    _out("READY")

    stdin = sys.stdin.buffer
    while True:
        cmd_byte = stdin.read(1)
        if not cmd_byte:           # EOF — parent closed the pipe
            break
        cmd = cmd_byte[0]

        if cmd == 0x01:
            pcm = _read_exact(stdin, AUDIO_CHUNK_BYTES)
            if len(pcm) < AUDIO_CHUNK_BYTES:
                break              # truncated read = EOF
            _out(_feed(pcm, state, pipe, vad, is_multilingual))

        elif cmd == 0x02:
            _reset_state(state)
            _out("OK")

        elif cmd == 0x03:
            break

        else:
            _err(f"[whisper_worker] unknown command byte: {cmd:#04x}")
            _out("OK")             # don't stall the parent


if __name__ == "__main__":
    main()
