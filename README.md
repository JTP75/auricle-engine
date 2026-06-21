# auricle-engine

A standalone, agent-agnostic voice engine. Owns wakeword detection, STT, TTS, audio I/O, and the FSM that coordinates them. Exposes a WebSocket server; any connector attaches to it to send response text and receive transcribed utterances.

---

## Architecture

**Ingress pipeline** — mic audio → OWW wakeword detection → STT → utterance event pushed to connected client

**Egress pipeline** — text deltas from client → segmentation → TTS synthesis → audio playback, with lookahead prefetch and barge-in preemption

**FSM** — 7-state machine (BOOTING / IDLE / AWAITING_UTTERANCE / UTTERANCE / DISPATCHED / SPEAKING / FATAL) with orthogonal `sleeping` and `muted` flags; gates what the ingress and egress pipelines do at each moment

**WebSocket server** — bidirectional channel for client↔engine communication; see [`docs/protocol.md`](docs/protocol.md)

---

## Requirements

### Python packages

```bash
# Core audio pipeline
pip install openwakeword numpy

# STT: vosk backend (default, CPU-only)
pip install vosk

# STT: whisper backend (GPU; requires a separate Python 3.10 venv)
python3.10 -m venv /path/to/whisper-venv
/path/to/whisper-venv/bin/pip install torch transformers accelerate webrtcvad

# TTS: edge-tts backend (default, cloud)
pip install edge-tts

# TTS: kokoro backend (local, CPU-friendly; requires espeak-ng on PATH)
sudo apt-get install espeak-ng
python3.10 -m venv ~/kokoro-venv
~/kokoro-venv/bin/pip install kokoro soundfile

# TTS: f5-tts backend (local, GPU)
python3.10 -m venv ~/f5-venv
~/f5-venv/bin/pip install f5-tts torch torchaudio

# WebSocket server
pip install websockets
```

### System packages (Debian/Ubuntu)

```bash
sudo apt install alsa-utils ffmpeg
```

### Models

Place in `models/` or override paths via config:

| File | What it is |
|------|-----------|
| `models/vosk-model/` | Vosk offline STT model directory |
| `models/wakeword.onnx` | openWakeWord custom wakeword model (ONNX) |
| `models/melspectrogram.onnx` | OWW melspec preprocessor |
| `models/embedding_model.onnx` | OWW embedding preprocessor |

---

## Installation

```bash
# TODO: installation instructions once packaging is defined
```

---

## Configuration

Configuration is read from `~/.auricle/config.yaml`. Environment variables take precedence over file values.

```yaml
# TODO: full config table once engine config is finalized
```

---

## Running

```bash
# TODO: entry point once defined
python -m auricle_engine
```

---

## WebSocket Protocol

See [`docs/protocol.md`](docs/protocol.md) for the full message schema, client ID contract, and connection lifecycle.

---

## Connectors

Connectors are separate repos that attach to the engine's WebSocket server:

| Connector | Agent backend |
|-----------|--------------|
| [`hermes-auricle`](https://github.com/nousresearch/hermes-auricle) | hermes-agent platform plugin |

---

## Diagnostics

```bash
python doctor.py
```

Checks Python dependencies, model files, system binaries, and exercises both audio devices (mic capture + speaker playback). Stop any running engine instance first — audio device tests need exclusive ALSA access.

---

## Project layout

```
auricle-engine/
  consts.py            all constants, env var names, defaults
  fsm.py               thread-safe FSM (7 states + orthogonal flags)
  providers.py         STTProvider / TTSProvider ABCs + implementations
  ingress.py           ingress pipeline loop (OWW + STT)
  egress.py            streaming TTS playback queue
  audio_buffer.py      ring buffer with TTS-active tracking for echo suppression
  audio_io.py          audio I/O abstraction (arecord/aplay + sounddevice backends)
  sleep.py             SleepDetector — spectral flux EMA for auto-sleep
  server.py            WebSocket server
  whisper_worker.py    STT subprocess (Python 3.10 + torch/transformers/webrtcvad)
  f5_worker.py         TTS subprocess (f5-tts venv)
  kokoro_worker.py     TTS subprocess (kokoro-tts venv)
  doctor.py            standalone diagnostic script
  assets/              auricle-wakeup / auricle-tosleep / auricle-notify / auricle-confused WAVs
  models/              model files (not committed)
  docs/
    protocol.md        WebSocket protocol specification
    uml/
      fsm.md           FSM state diagram (Mermaid)
      class.md         architecture class diagram (Mermaid)
```
