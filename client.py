import asyncio
import json
import logging
import os
from typing import Optional

import websockets
import websockets.exceptions

from consts import (
    ASSET_NOTIFY,
    DEFAULT_CONNECTOR_URL,
    ENV_CONNECTOR_URL,
    PROACTIVE_PRE_SPEECH_PAUSE,
    RETRY_DELAY_SECONDS,
    TTS_CLEARED,
    TTS_STOPPED,
    _CMD_CLEAR,
    _CMD_STOP,
)
from fsm import FSM, State

logger = logging.getLogger(__name__)


class AuricleClient:
    """
    WebSocket client for the auricle engine.

    Connects to the connector's WebSocket server (hermes-auricle), receives
    speak/abort/notify commands, and pushes utterance/cmd events back.
    Reconnects automatically on disconnect.
    """

    def __init__(self, fsm: FSM, egress, audio_output, stop_event) -> None:
        self._fsm          = fsm
        self._egress       = egress
        self._audio_output = audio_output
        self._stop_event   = stop_event

        self._ws:        Optional[object] = None
        self._client_id: Optional[str]   = None
        self._loop:      Optional[asyncio.AbstractEventLoop] = None

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    # ── Outbound ─────────────────────────────────────────────────────────────

    async def send_event(self, msg: dict) -> None:
        """Send a JSON event to the connector. No-op if not connected."""
        ws = self._ws
        if ws is None:
            return
        try:
            await ws.send(json.dumps(msg))
        except Exception as exc:
            logger.debug("[auricle-engine] send_event failed: %s", exc)

    # ── Dispatch fn (injected into ingress thread) ───────────────────────────

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
        return self._dispatch

    # ── Inbound message handling ─────────────────────────────────────────────

    async def _handle_message(self, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("[auricle-engine] invalid JSON from connector: %r", raw[:80])
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

    async def run(self) -> None:
        """Connect to the connector and handle messages. Reconnects on disconnect."""
        url = os.getenv(ENV_CONNECTOR_URL, DEFAULT_CONNECTOR_URL)
        while not self._stop_event.is_set():
            try:
                logger.info("[auricle-engine] connecting to connector at %s", url)
                async with websockets.connect(url) as ws:
                    raw = await asyncio.wait_for(ws.recv(), timeout=10)
                    msg = json.loads(raw)
                    if msg.get("t") != "ready":
                        logger.error("[auricle-engine] unexpected handshake: %r", msg)
                    else:
                        self._ws        = ws
                        self._client_id = msg["client_id"]
                        logger.info("[auricle-engine] connected (client_id=%s)", self._client_id)
                        async for message in ws:
                            await self._handle_message(message)
            except websockets.exceptions.ConnectionClosed:
                logger.info("[auricle-engine] connector closed connection")
            except (OSError, ConnectionRefusedError, asyncio.TimeoutError) as exc:
                logger.warning("[auricle-engine] could not connect to connector: %s", exc)
            except Exception as exc:
                logger.warning("[auricle-engine] connection error: %s", exc)
            finally:
                self._ws        = None
                self._client_id = None
                self._fsm.transition(State.IDLE)

            if not self._stop_event.is_set():
                await asyncio.sleep(RETRY_DELAY_SECONDS)
                logger.info("[auricle-engine] retrying connector connection…")
