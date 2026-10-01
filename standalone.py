"""Standalone WebSocket server for out-of-process delivery.

Listens on its own port so that out-of-process senders (cron, `hermes send`)
can reach the engine's speaker without a running hermes gateway. Handles
`notify` requests (chime + TTS) and replies with `notify_done`.

This path is deliberately FSM-free, mirroring the connector connection's
`notify` handling: it speaks without touching utterance/session state.
"""
import asyncio
import json
import logging
import os
from typing import Optional

import websockets

from consts import (
    DEFAULT_STANDALONE_WS_HOST,
    DEFAULT_STANDALONE_WS_PORT,
    ENV_STANDALONE_WS_HOST,
    ENV_STANDALONE_WS_PORT,
)

logger = logging.getLogger(__name__)


# Single-client tracker: set while a connection is active.
_standalone_ws: Optional[object] = None


async def serve_standalone(egress, stop_event) -> None:
    """Run the standalone WS server until ``stop_event`` is set."""
    host = os.getenv(ENV_STANDALONE_WS_HOST, DEFAULT_STANDALONE_WS_HOST)
    port = int(os.getenv(ENV_STANDALONE_WS_PORT, str(DEFAULT_STANDALONE_WS_PORT)))

    async def handle_connection(ws) -> None:
        """One client at a time; second connections are rejected."""
        global _standalone_ws
        if _standalone_ws is not None:
            logger.warning("[auricle-engine] standalone: rejecting second connection")
            await ws.close(1008, "Only one client supported")
            return

        _standalone_ws = ws
        logger.info("[auricle-engine] standalone: client connected")
        try:
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                await _handle_standalone_message(ws, msg, egress)
        except websockets.exceptions.ConnectionClosed:
            logger.info("[auricle-engine] standalone: client disconnected")
        finally:
            _standalone_ws = None

    try:
        server = await websockets.serve(handle_connection, host, port)
    except OSError as exc:
        logger.error("[auricle-engine] standalone: could not bind ws://%s:%d: %s", host, port, exc)
        return

    logger.info("[auricle-engine] standalone listener on ws://%s:%d", host, port)
    await stop_event.wait()
    server.close()
    await server.wait_closed()
    logger.info("[auricle-engine] standalone listener closed")


async def _handle_standalone_message(ws, msg: dict, egress) -> None:
    t = msg.get("t")
    if t == "notify":
        text = msg.get("text", "")
        await egress.play_notify(text)
        try:
            await ws.send(json.dumps({"t": "notify_done"}))
        except websockets.exceptions.ConnectionClosed:
            pass
    else:
        logger.warning("[auricle-engine] standalone: unknown message type: %r", t)
