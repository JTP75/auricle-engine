import asyncio
import json
import logging
import os
import threading
import uuid
from typing import Optional

import websockets
import websockets.exceptions

from consts import (
    ASSET_NOTIFY,
    ENGINE_HOST,
    ENGINE_PORT,
    ENV_ENGINE_PORT,
    PROACTIVE_PRE_SPEECH_PAUSE,
    TTS_CLEARED,
    TTS_STOPPED,
    _CMD_CLEAR,
    _CMD_STOP,
)
from fsm import FSM, State

logger = logging.getLogger(__name__)


class AuricleServer:
    """
    WebSocket server for the auricle engine.

    Accepts a single client connection. Assigns a client_id on connect and
    includes it in every outbound message. Bridges inbound WebSocket messages
    to the audio pipeline (egress/FSM) and pushes engine events back to the
    connected client.
    """

    def __init__(self, fsm: FSM, egress, audio_output, stop_event: threading.Event) -> None:
        self._fsm          = fsm
        self._egress       = egress
        self._audio_output = audio_output
        self._stop_event   = stop_event

        self._ws:        Optional[object] = None
        self._client_id: Optional[str]   = None
        self._loop:      Optional[asyncio.AbstractEventLoop] = None

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    # ── Outbound ────────────────────────────────────────────────────────────

    async def send_event(self, msg: dict) -> None:
        """Send a JSON event to the connected client. No-op if no client."""
        ws = self._ws
        if ws is None:
            return
        try:
            await ws.send(json.dumps(msg))
        except Exception as exc:
            logger.debug("[auricle-engine] send_event failed: %s", exc)

    def send_event_threadsafe(self, msg: dict) -> None:
        """Schedule send_event from a non-asyncio thread (e.g. ingress)."""
        if self._loop is None:
            return
        asyncio.run_coroutine_threadsafe(self.send_event(msg), self._loop)

    # ── Dispatch fn (injected into ingress thread) ──────────────────────────

    async def _dispatch(self, text: str) -> None:
        """Coroutine called by the ingress thread via run_coroutine_threadsafe."""
        if text == _CMD_CLEAR:
            logger.info("[auricle-engine] voice command: clear")
            self._fsm.transition(State.IDLE)
            await self._egress.speak(TTS_CLEARED, priority=True)
            await self.send_event({"t": "cmd", "client_id": self._client_id, "name": "new"})
            return

        if text == _CMD_STOP:
            logger.info("[auricle-engine] voice command: stop")
            self._egress.abort()
            self._fsm.transition(State.IDLE)
            await self._egress.speak(TTS_STOPPED, priority=True)
            await self.send_event({"t": "cmd", "client_id": self._client_id, "name": "stop"})
            return

        logger.info("[auricle-engine] utterance: %r", text[:80])
        await self.send_event({"t": "utterance", "client_id": self._client_id, "text": text})

    def get_dispatch_fn(self):
        """Return the dispatch coroutine for injection into run_ingress_loop."""
        return self._dispatch

    # ── Inbound message handling ─────────────────────────────────────────────

    async def _handle_message(self, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("[auricle-engine] invalid JSON from client: %r", raw[:80])
            return

        t = msg.get("t")

        if t == "speak":
            text = msg.get("text", "")
            if not text:
                return
            logger.info("[auricle-engine] speak: %r", text[:80])
            self._egress.abort()
            self._egress.reset()
            self._egress.start_worker()
            self._fsm.transition(State.SPEAKING)
            await self._egress.process_delta(text, finalize=True)
            self._fsm.transition_if(State.SPEAKING, State.AWAITING_UTTERANCE)

        elif t == "abort":
            logger.info("[auricle-engine] abort received")
            self._egress.abort()
            self._fsm.transition_if(State.SPEAKING, State.IDLE)

        elif t == "dispatched":
            logger.info("[auricle-engine] dispatched — FSM → DISPATCHED")
            self._fsm.transition(State.DISPATCHED)

        elif t == "notify":
            text = msg.get("text", "")
            logger.info("[auricle-engine] notify: %r", text[:60])
            await self._audio_output.play_file(ASSET_NOTIFY)
            await asyncio.sleep(PROACTIVE_PRE_SPEECH_PAUSE)
            await self._egress.speak(text)
            await self.send_event({"t": "notify_done", "client_id": self._client_id})

        elif t == "cmd":
            logger.debug("[auricle-engine] cmd from connector: %s", msg.get("name"))

        else:
            logger.warning("[auricle-engine] unknown message type: %r", t)

    # ── Connection lifecycle ─────────────────────────────────────────────────

    async def _handle_connection(self, ws) -> None:
        if self._ws is not None:
            logger.warning("[auricle-engine] rejecting second client — only one supported")
            await ws.close(1008, "Only one client supported")
            return

        client_id = uuid.uuid4().hex[:8]
        self._ws        = ws
        self._client_id = client_id
        logger.info("[auricle-engine] client connected (id=%s)", client_id)

        await ws.send(json.dumps({"t": "ready", "client_id": client_id}))

        try:
            async for message in ws:
                await self._handle_message(message)
        except websockets.exceptions.ConnectionClosed:
            logger.info("[auricle-engine] client disconnected (id=%s)", client_id)
        finally:
            self._ws        = None
            self._client_id = None
            self._fsm.transition(State.IDLE)

    async def serve(self) -> None:
        """Start the WebSocket server. Blocks until cancelled."""
        host = ENGINE_HOST
        port = int(os.getenv(ENV_ENGINE_PORT, str(ENGINE_PORT)))
        logger.info("[auricle-engine] listening on ws://%s:%d", host, port)
        async with websockets.serve(self._handle_connection, host, port):
            await asyncio.Future()  # run forever
