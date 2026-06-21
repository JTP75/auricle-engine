# auricle-engine

A standalone, agent-agnostic voice engine. Owns wakeword detection, STT, TTS, audio I/O, and the FSM that coordinates them. Exposes a WebSocket server on port 57310; any connector attaches to it to send response text and receive transcribed utterances.

- **Wakeword**: [openWakeWord](https://github.com/dscripka/openWakeWord) (ONNX, custom model)
- **STT**: [vosk](https://alphacephei.com/vosk/) (offline, CPU) or [distil-whisper](https://huggingface.co/distil-whisper/distil-large-v3) (GPU, subprocess)
- **TTS**: [edge-tts](https://github.com/rany2/edge-tts) (cloud, default), [F5-TTS](https://github.com/SWivid/F5-TTS) (local GPU, voice cloning), or [Kokoro-82M](https://github.com/hexgrad/kokoro) (local, fast)
- **Target hardware**: Raspberry Pi + Jabra Speak 510 USB

---

## Architecture

**Ingress pipeline** — mic audio (16kHz PCM, 40ms chunks) → OWW wakeword detection → STT → utterance event sent to connected client over WebSocket

**Egress pipeline** — `speak` message from client → segmentation by newline → TTS synthesis → audio playback with lookahead prefetch and barge-in preemption

**FSM** — 7 states (BOOTING → IDLE → AWAITING_UTTERANCE → UTTERANCE → DISPATCHED → SPEAKING → FATAL) with orthogonal `sleeping` and `muted` flags; gates which pipeline runs on each audio chunk

**WebSocket server** — bidirectional, single-client; see [`docs/protocol.md`](docs/protocol.md) for the full message schema

---

## Requirements

### Python packages (engine venv)

```bash
# Core audio pipeline (always required)
pip install openwakeword numpy websockets

# STT: vosk backend (default, CPU-only)
pip install vosk

# TTS: edge-tts backend (default, cloud)
pip install edge-tts

# STT: whisper backend — separate Python 3.10 venv
python3.10 -m venv /path/to/whisper-venv
/path/to/whisper-venv/bin/pip install torch transformers accelerate webrtcvad

# TTS: kokoro backend — separate Python 3.10 venv
sudo apt-get install espeak-ng
python3.10 -m venv ~/kokoro-venv
~/kokoro-venv/bin/pip install kokoro soundfile

# TTS: f5-tts backend — separate Python 3.10 venv with CUDA GPU
python3.10 -m venv ~/f5-venv
~/f5-venv/bin/pip install f5-tts torch torchaudio
```

### System packages (Debian/Ubuntu)

```bash
sudo apt install alsa-utils ffmpeg
```

### Models

Place in `models/` or point to them via env vars:

| File | What it is |
|------|-----------|
| `models/vosk-model/` | Vosk offline STT model directory |
| `models/wakeword.onnx` | openWakeWord custom wakeword model |
| `models/melspectrogram.onnx` | OWW melspec preprocessor |
| `models/embedding_model.onnx` | OWW embedding preprocessor |

---

## Installation

```bash
git clone <repo-url> auricle-engine
cd auricle-engine
python3 -m venv .venv
source .venv/bin/activate
pip install openwakeword numpy websockets vosk edge-tts
sudo apt install alsa-utils ffmpeg
# Place models in models/ (see above)
```

---

## Running

```bash
source .venv/bin/activate
python __main__.py
```

Listens on `ws://localhost:57310` by default. Set `AURICLE_ENGINE_PORT` to use a different port.

---

## Configuration

All settings are read from environment variables. There is no config file — set vars in a `.env` file, via the shell, or via a systemd unit.

### Audio

| Env var | Default | Description |
|---------|---------|-------------|
| `AURICLE_AUDIO_INPUT` | `arecord` | Input backend: `arecord` or `sounddevice` |
| `AURICLE_MIC_DEVICE` | `plughw:0,0` | ALSA device for `arecord` |
| `AURICLE_SD_INPUT_DEVICE` | *(system default)* | sounddevice input device (index or name) |
| `AURICLE_AUDIO_OUTPUT` | `aplay` | Output backend: `aplay` or `sounddevice` |
| `AURICLE_SPEAKER_DEVICE` | `plughw:0,0` | ALSA device for `aplay` |
| `AURICLE_SD_OUTPUT_DEVICE` | *(system default)* | sounddevice output device (index or name) |

### STT

| Env var | Default | Description |
|---------|---------|-------------|
| `AURICLE_STT_BACKEND` | `vosk` | STT backend: `vosk` or `whisper` |
| `AURICLE_VOSK_MODEL_PATH` | `models/vosk-model` | Path to vosk model directory (vosk only) |
| `AURICLE_WHISPER_PYTHON` | *(required)* | Python binary in the whisper venv (whisper only) |
| `AURICLE_WHISPER_MODEL_ID` | `distil-whisper/distil-large-v3` | HuggingFace model ID (whisper only) |

### TTS

| Env var | Default | Description |
|---------|---------|-------------|
| `AURICLE_TTS_BACKEND` | `edge-tts` | TTS backend: `edge-tts`, `f5-tts`, or `kokoro-tts` |
| `AURICLE_TTS_VOICE` | `en-GB-LibbyNeural` | edge-tts voice name (edge-tts only) |
| `AURICLE_F5_PYTHON` | *(required)* | Python binary in the f5-tts venv (f5-tts only) |
| `AURICLE_F5_MODEL` | `F5TTS_v1_Base` | F5-TTS model name (f5-tts only) |
| `AURICLE_F5_STEPS` | `5` | Flow-matching inference steps; lower = faster (f5-tts only) |
| `AURICLE_F5_SPEED` | `1.0` | Speech speed multiplier (f5-tts only) |
| `AURICLE_F5_REF_WAV` | *(optional)* | Reference WAV for voice cloning — 5–15s, 24kHz mono. Both `REF_WAV` and `REF_TXT` must be set together, or neither. |
| `AURICLE_F5_REF_TXT` | *(optional)* | Exact transcript of `F5_REF_WAV` (f5-tts only) |
| `AURICLE_KOKORO_PYTHON` | *(required)* | Python binary in the kokoro venv (kokoro-tts only) |
| `AURICLE_KOKORO_VOICE` | `af_heart` | Voice name. `af_*`/`am_*` = American English; `bf_*`/`bm_*` = British English (kokoro-tts only) |

### Wakeword (OWW)

| Env var | Default | Description |
|---------|---------|-------------|
| `AURICLE_OWW_WAKEWORD_MODEL_PATH` | `models/wakeword.onnx` | OWW custom wakeword model |
| `AURICLE_OWW_MELSPEC_MODEL_PATH` | `models/melspectrogram.onnx` | OWW melspec preprocessor |
| `AURICLE_OWW_EMBEDDING_MODEL_PATH` | `models/embedding_model.onnx` | OWW embedding preprocessor |

### Behavior

| Env var | Default | Description |
|---------|---------|-------------|
| `AURICLE_MUTE` | `false` | Disable wakeword detection on startup |
| `AURICLE_ACTIVE_LISTEN_DURATION` | `5` | Seconds of open listen after TTS ends (no wakeword needed) |
| `AURICLE_SLEEP_TIMEOUT` | `60` | Seconds of IDLE silence before auto-sleep |
| `AURICLE_SLEEP_WAKE_SENSITIVITY` | `3.0` | Flux multiplier above baseline to wake; lower = more sensitive |
| `AURICLE_SLEEP_FLUX_THRESHOLD` | `0.02` | Normalized flux EMA cutoff for "quiet" classification |
| `AURICLE_ENGINE_PORT` | `57310` | WebSocket server port |
| `AURICLE_ENGINE_HOST` | `localhost` | WebSocket server bind address. Set to `0.0.0.0` to accept connections from remote clients (e.g. connector on a different machine over Tailscale). |

---

## Voice commands

Matched against the full STT transcript (exact, case-insensitive). Wakeword must fire first.

| Say | Effect |
|-----|--------|
| "clear", "reset" | Speak confirmation, send `cmd: new` to connector |
| "stop" | Abort TTS, speak confirmation, send `cmd: stop` to connector |

---

## How it works

**Ingress:** An `arecord` subprocess (or sounddevice stream) feeds raw 16kHz PCM through a state-gated loop. In IDLE, OWW watches every 40ms chunk for the wakeword and `SleepDetector` tracks spectral flux for auto-sleep. In SPEAKING and DISPATCHED, OWW runs for barge-in. In AWAITING_UTTERANCE and UTTERANCE, the STT backend captures the utterance — gated by `AudioBuffer.tts_active` so speaker echo isn't fed into the model. When a final transcript arrives, it's sent to the connected client as `{t:"utterance"}`. Voice commands ("clear", "stop") are handled locally: the engine plays audio feedback and sends a `{t:"cmd"}` event to the client.

**Egress:** The client sends `{t:"speak", text:"..."}` with the full response. The engine segments by newline into sentences, enqueues them, and plays each via the active TTS backend. While sentence N plays, sentence N+1 is prefetched concurrently (lookahead). Barge-in (wakeword during SPEAKING) aborts playback immediately and re-enters AWAITING_UTTERANCE. TTS output is capped at 3000 characters.

**Auto-sleep:** After `AURICLE_SLEEP_TIMEOUT` seconds of acoustic inactivity in IDLE, the OWW model is gated off. Wake detection uses normalized spectral flux — stable background noise (fans, HVAC) stays below the threshold while any novel acoustic event (speech, knock) triggers an instant wake. The model stays loaded; sleep is a software flag.

**Subprocess workers:** Whisper STT, F5-TTS, and Kokoro-TTS each run in their own Python venv via a subprocess with a binary protocol over stdin/stdout. This isolates heavy ML deps from the engine's venv and allows each backend to use whatever Python version or CUDA wheel it needs.

---

## WebSocket Protocol

See [`docs/protocol.md`](docs/protocol.md).

---

## Connectors

| Connector | Agent backend |
|-----------|--------------|
| [hermes-auricle](../hermes-auricle/) | hermes-agent platform plugin |

---

## Diagnostics

```bash
python doctor.py
```

Checks Python dependencies, model files, system binaries, and exercises both audio devices. Stop any running engine instance first.

---

## Project layout

```
auricle-engine/
  __main__.py          entry point — boots providers, starts ingress thread, serves WebSocket
  server.py            WebSocket server — message routing, client_id assignment
  consts.py            all constants, env var names, defaults
  fsm.py               thread-safe FSM (7 states + sleeping/muted flags)
  ingress.py           audio capture loop (OWW + STT + sleep detection)
  egress.py            TTS playback queue with lookahead prefetch and barge-in
  providers.py         STTProvider / TTSProvider ABCs + implementations
  audio_buffer.py      ring buffer with TTS-active tracking for echo suppression
  audio_io.py          audio I/O abstraction (arecord/aplay + sounddevice)
  sleep.py             SleepDetector — normalized spectral flux EMA
  whisper_worker.py    Whisper STT subprocess (Python 3.10 venv, binary protocol)
  f5_worker.py         F5-TTS subprocess (f5-tts venv, binary protocol)
  kokoro_worker.py     Kokoro-TTS subprocess (kokoro venv, binary protocol)
  doctor.py            diagnostic script
  assets/              WAV chimes: wakeup / tosleep / notify / confused
  models/              ML model files (not committed — place locally)
  docs/
    protocol.md        WebSocket protocol specification
```
