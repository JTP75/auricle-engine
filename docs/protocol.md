# Auricle Engine WebSocket Protocol

The **connector** (hermes-auricle) runs the WebSocket server. The **engine**
connects to it as a client and reconnects automatically on disconnect.

---

## Client ID

Every message in both directions carries a `client_id` field. Only one engine
is supported today, but the field is required in the protocol now so that
multi-engine support can be added without a breaking change.

The connector assigns a `client_id` when the engine connects and sends it back
in the `ready` message. The engine must include this `client_id` in every
subsequent message it sends.

---

## Connection lifecycle

```
Engine                        Connector (hermes-auricle)
  |                               |
  |-------- WebSocket connect --->|
  |<--- {t:"ready", client_id} ---|   connector assigns client_id, signals ready
  |                               |
  |        ... session ...        |
  |                               |
  |---- WebSocket close --------->|   or connector closes on shutdown
```

If the engine disconnects and reconnects, the connector issues a new `client_id`.
The engine resets FSM to IDLE on disconnect.

---

## Message format

All messages are JSON objects with a `t` field identifying the message type.
Binary frames are not used.

---

## Client → Engine messages

### `speak` — full response text

```json
{"t": "speak", "client_id": "...", "text": "full response text here"}
```

Delivers the complete agent response. The engine segments by newline, synthesizes
TTS for each segment with lookahead prefetch, and plays them in order. The FSM
transitions SPEAKING → AWAITING_UTTERANCE when playback finishes.

> **Note:** The hermes adapter calls `send()` once per complete response, so
> streaming delta delivery is not needed yet. `speak` replaces the `delta`+`done`
> pattern described in earlier design documents.

### `abort` — cancel current output

```json
{"t": "abort", "client_id": "..."}
```

Instructs the engine to immediately stop TTS playback and drain the queue.
Sent when the agent issues `/stop` or the connector otherwise wants to
preempt the current response.

### `dispatched` — agent received utterance

```json
{"t": "dispatched", "client_id": "..."}
```

Sent after the connector has handed the utterance off to the agent and the
agent has begun processing. Transitions the FSM to DISPATCHED, which gates
the ingress loop out of UTTERANCE and into barge-in detection mode.

### `notify` — out-of-band proactive delivery

```json
{"t": "notify", "client_id": "...", "text": "proactive TTS text"}
```

Requests that the engine play the notify chime followed by the given text as TTS,
without going through the normal utterance/dispatch FSM flow. Used by the
connector's standalone sender (cron/proactive notifications). The engine replies
with `notify_done` when playback completes.

### `cmd` — session command acknowledgement

```json
{"t": "cmd", "client_id": "...", "name": "new"}
{"t": "cmd", "client_id": "...", "name": "stop"}
{"t": "cmd", "client_id": "...", "name": "clear"}
```

Notifies the engine that a session-level command was issued by the connector (e.g.
the connector dispatched `/new` to the agent). The engine logs this for
observability; no FSM action is taken since the engine already handled its side
when the voice command was detected.

---

## Engine → Client messages

All engine messages include `client_id`.

### `ready` — engine ready

```json
{"t": "ready", "client_id": "abc123"}
```

Sent immediately after a client connects. `client_id` is the identifier the
client must echo in all subsequent messages.

### `utterance` — user spoke

```json
{"t": "utterance", "client_id": "abc123", "text": "what is the weather today"}
```

The wakeword fired and STT produced a transcript. The connector should dispatch
this text to the agent as a new user message.

### `barge_in` — user interrupted TTS

```json
{"t": "barge_in", "client_id": "abc123"}
```

The wakeword fired while TTS was playing. The engine has already stopped audio
playback. The connector should cancel any in-flight agent response and prepare
to receive an `utterance` event immediately after.

### `cmd` — engine-initiated session command

```json
{"t": "cmd", "client_id": "abc123", "name": "new"}
```

The engine requests that the connector issue a session command to the agent.
Current cases:
- `"new"` — auto-sleep silence threshold crossed; connector should clear the
  session history (issue `/new` to the agent)

### `state` — FSM state change (informational)

```json
{"t": "state", "client_id": "abc123", "fsm": "IDLE", "sleeping": false, "muted": false}
```

Emitted on every FSM state transition. Connectors may use this for status
display or debugging but should not drive logic from it — use the semantic
event messages (`utterance`, `barge_in`, `cmd`) instead.

### `notify_done` — proactive delivery complete

```json
{"t": "notify_done", "client_id": "abc123"}
```

Sent after a `notify` request has finished playing (chime + TTS). The standalone
sender waits for this before closing the connection.

### `error` — engine error

```json
{"t": "error", "client_id": "abc123", "code": "fatal_audio", "message": "..."}
```

Sent before the engine closes the connection on a fatal error. The connector
should surface this to the user / operator and attempt reconnection.

---

## Notes

- The engine does not buffer messages while the client is disconnected. A
  reconnecting client starts fresh.
- Message ordering within a session is guaranteed (WebSocket is ordered).
- The engine sends `state` events for observability but the connector must not
  depend on them for correctness — all meaningful events have their own type.
