# Auricle Engine Agent Reference

Auricle Engine is a **standalone, agent-agnostic voice engine**. It owns the
full audio pipeline: wakeword detection (OWW), speech-to-text (STT),
text-to-speech (TTS), audio I/O, and the FSM that coordinates them. It exposes
a WebSocket server that any connector can attach to — the engine has no
knowledge of any specific agent backend.

Connectors are separate repos (e.g., `hermes-auricle` for the hermes-agent
platform plugin). This repo ships nothing hermes-specific.

## Documentation Index

| Document | What it covers |
|----------|---------------|
| [`README.md`](README.md) | Installation, configuration, requirements, project layout |
| [`docs/protocol.md`](docs/protocol.md) | WebSocket protocol — message types, client ID contract, connection lifecycle |
| [`docs/uml/fsm.md`](docs/uml/fsm.md) | Mermaid stateDiagram-v2 of the FSM with orthogonal regions and all transitions |
| [`docs/uml/class.md`](docs/uml/class.md) | Mermaid classDiagram of the engine architecture |
| [`doctor.py`](doctor.py) | Standalone diagnostic script — checks deps, model files, system binaries, and audio devices |

## Required behavior for agents

These rules apply to every task in this project unless explicitly overridden.
Bias: caution over speed on non-trivial work. Use judgment on trivial tasks.

### Rule 1 — Think Before Coding

State assumptions explicitly. If uncertain, ask rather than guess.
Present multiple interpretations when ambiguity exists.
Push back when a simpler approach exists.
Stop when confused. Name what's unclear.

### Rule 2 — Simplicity First

Minimum code that solves the problem. Nothing speculative.
No features beyond what was asked. No abstractions for single-use code.
Test: would a senior engineer say this is overcomplicated? If yes, simplify.

### Rule 3 — Surgical Changes

Touch only what you must. Clean up only your own mess.
Don't "improve" adjacent code, comments, or formatting.
Don't refactor what isn't broken. Match existing style.

### Rule 4 — Goal-Driven Execution

Define success criteria. Loop until verified.
Don't follow steps. Define success and iterate.
Strong success criteria let you loop independently.

### Rule 5 — Use the model only for judgment calls

Use me for: classification, drafting, summarization, extraction.
Do NOT use me for: routing, retries, deterministic transforms.
If code can answer, code answers.

### Rule 6 — IF YOU ARE CO-PILOT, IGNORE THIS RULE Token budgets are not advisory

Per-task: 4,000 tokens. Per-session: 30,000 tokens.
If approaching budget, summarize and start fresh.
Surface the breach. Do not silently overrun.

### Rule 7 — Surface conflicts, don't average them

If two patterns contradict, pick one (more recent / more tested).
Explain why. Flag the other for cleanup.
Don't blend conflicting patterns.

### Rule 8 — Read before you write

Before adding code, read exports, immediate callers, shared utilities.
"Looks orthogonal" is dangerous. If unsure why code is structured a way, ask.

### Rule 9 — Tests verify intent, not just behavior

Tests must encode WHY behavior matters, not just WHAT it does.
A test that can't fail when business logic changes is wrong.

### Rule 10 — Checkpoint after every significant step

Summarize what was done, what's verified, what's left.
Don't continue from a state you can't describe back.
If you lose track, stop and restate.

### Rule 11 — Match the codebase's conventions, even if you disagree

Conformance > taste inside the codebase.
If you genuinely think a convention is harmful, surface it. Don't fork silently.

### Rule 12 — Fail loud

"Completed" is wrong if anything was skipped silently.
"Tests pass" is wrong if any were skipped.
Default to surfacing uncertainty, not hiding it.

## Codebase-specific rules

### Rule 1 — Constants live in `consts.py`

Never hardcode numeric, string, or tuning values directly in module files.
All constants belong in `consts.py`. Import from there.
This applies to thresholds, timeouts, paths, defaults, and magic strings — anything that could need tuning without touching logic.

### Rule 2 — Do not auto-approve commits

Do not stage, commit, or push without being explicitly asked to do so.
When asked to commit, propose the message and wait for confirmation before running `git commit`.

### Rule 3 — Commit messages ≤ 100 characters

Subject line must be 100 characters or fewer.
Use the conventional format: `type: short description` (e.g. `fix: guard against empty transcript`).
No period at the end. No body unless the user asks for one.

### Rule 4 — Update docs when you update code

New feature additions must be documented with at least one line in `README.md`.
Patches and bug fixes should only be documented if they contradict what is in `README.md`.
The amount of documentation for an addition should reflect the scale of the code change.

### Rule 5 — Keep `doctor.py` in sync with the runtime

When adding any of the following, add the corresponding check to `doctor.py`:
- A new Python dependency (import check)
- A new model file or asset path (file-existence check)
- A new required system binary
- A new env var that controls audio device path, STT backend, or TTS backend

Constants used only by the doctor must use the `DOCTOR_` prefix and a `# ── Doctor (doctor.py only)` category comment in `consts.py`.

## Known Gotchas

### Gotcha 1 — This repo is agent-agnostic

Do not add logic that assumes hermes-agent (or any specific agent) on the other end.
Agent-specific behavior (message classification, session lifecycle commands, platform hints)
belongs in the connector repo.

If you find yourself importing anything from hermes or writing hermes-specific handling,
stop and move it to the connector.

### Gotcha 2 — client_id appears in every engine→client message

Every message the engine sends to a connected client must include the `client_id` field.
Only one client is supported today, but the field is required now so the protocol doesn't
need a breaking change when multi-client support is added.

See [`docs/protocol.md`](docs/protocol.md) for the full message schema.
