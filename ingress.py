import asyncio
import logging
import threading
import time
from typing import Callable, Coroutine

import numpy as np

from audio_buffer import AudioBuffer
from audio_io import AudioInput, AudioOutput
from consts import (
    ASSET_CONFUSED,
    ASSET_TOSLEEP,
    ASSET_WAKEUP,
    CLEAR_COMMANDS,
    MISINPUT_MAX_CONSECUTIVE,
    MISINPUT_PHRASES,
    STOP_COMMANDS,
    _CMD_CLEAR,
    _CMD_STOP,
)
from fsm import FSM, State
from sleep import SleepDetector, SleepSignal

logger = logging.getLogger(__name__)


def run_ingress_loop(
    *,
    audio_input: AudioInput,
    audio_output: AudioOutput,
    oww,
    wakeword_key: str,
    stt_provider,
    egress,
    audio_buffer: AudioBuffer,
    fsm: FSM,
    loop: asyncio.AbstractEventLoop,
    dispatch_fn: Callable[[str], Coroutine],
    stop_event: threading.Event,
    active_listen_duration: float,
    oww_threshold: float,
    sleep_detector: SleepDetector,
) -> None:
    """
    Synchronous ingress thread.

    Reads 1280-byte PCM chunks from the audio input, runs OWW for wakeword/barge-in
    detection, and feeds the STT backend for utterance capture. Dispatches final
    transcripts and internal commands to the event loop via asyncio.run_coroutine_threadsafe.
    """
    active_listen_deadline: float | None = None
    consecutive_misinputs: int = 0
    was_idle: bool = False

    while not stop_event.is_set():
        data = audio_input.read_chunk()
        if not data:
            logger.error("[auricle-engine] audio input closed unexpectedly — ingress exiting")
            break

        audio_buffer.append(data)
        state = fsm.get()

        # ── IDLE: wakeword detection + auto-sleep ──────────────────────────
        if state == State.IDLE:
            if not was_idle:
                sleep_detector.reset()
            was_idle = True

            if fsm.muted:
                continue

            sig = sleep_detector.feed(data)
            if sig is SleepSignal.SLEEP and not fsm.sleeping:
                fsm.sleeping = True
            elif sig is SleepSignal.WAKE and fsm.sleeping:
                fsm.sleeping = False

            if fsm.sleeping:
                continue

            audio = np.frombuffer(data, dtype=np.int16)
            prob  = oww.predict(audio).get(wakeword_key, 0.0)
            if prob >= oww_threshold:
                logger.info("[auricle-engine] wakeword detected (p=%.2f) → AWAITING_UTTERANCE", prob)
                oww.reset()
                stt_provider.reset()
                loop.call_soon_threadsafe(lambda: asyncio.ensure_future(audio_output.play_file(ASSET_WAKEUP)))
                fsm.transition(State.AWAITING_UTTERANCE)
                active_listen_deadline = None

        # ── SPEAKING: run OWW in parallel for barge-in ────────────────────
        elif state == State.SPEAKING:
            was_idle = False
            audio = np.frombuffer(data, dtype=np.int16)
            prob  = oww.predict(audio).get(wakeword_key, 0.0)
            if prob >= oww_threshold:
                logger.info("[auricle-engine] barge-in detected (p=%.2f)", prob)
                oww.reset()
                stt_provider.reset()
                loop.call_soon_threadsafe(egress.abort)
                loop.call_soon_threadsafe(lambda: asyncio.ensure_future(egress.play_file(ASSET_WAKEUP)))
                fsm.transition(State.AWAITING_UTTERANCE)
                active_listen_deadline = None

        # ── AWAITING_UTTERANCE / UTTERANCE: STT ───────────────────────────
        elif state in (State.AWAITING_UTTERANCE, State.UTTERANCE):
            was_idle = False
            if state == State.AWAITING_UTTERANCE and active_listen_deadline is None:
                active_listen_deadline = time.monotonic() + active_listen_duration
                logger.info("[auricle-engine] active-listen window armed (%.1fs)", active_listen_duration)

            if state == State.AWAITING_UTTERANCE and time.monotonic() >= active_listen_deadline:
                logger.info("[auricle-engine] active-listen expired → IDLE")
                oww.reset()
                stt_provider.reset()
                audio_output.play_file_sync(ASSET_TOSLEEP)
                fsm.transition(State.IDLE)
                active_listen_deadline = None
                continue

            oww.predict(np.frombuffer(data, dtype=np.int16))

            if audio_buffer.tts_active:
                final, partial = None, None
            else:
                final, partial = stt_provider.feed(data)

            if partial and state == State.AWAITING_UTTERANCE:
                logger.info("[auricle-engine] speech detected → UTTERANCE")
                active_listen_deadline = None
                fsm.transition(State.UTTERANCE)

            if final:
                logger.info("[auricle-engine] transcript: %r", final)
                active_listen_deadline = None
                oww.reset()
                stt_provider.reset()
                if _handle_transcript(final, fsm, loop, dispatch_fn):
                    consecutive_misinputs += 1
                    if consecutive_misinputs >= MISINPUT_MAX_CONSECUTIVE:
                        logger.info("[auricle-engine] misinput limit reached → IDLE")
                        audio_output.play_file_sync(ASSET_TOSLEEP)
                        fsm.transition(State.IDLE)
                        consecutive_misinputs = 0
                    else:
                        logger.info("[auricle-engine] misinput %d/%d → AWAITING_UTTERANCE",
                                    consecutive_misinputs, MISINPUT_MAX_CONSECUTIVE)
                        loop.call_soon_threadsafe(lambda: asyncio.ensure_future(audio_output.play_file(ASSET_CONFUSED)))
                        fsm.transition(State.AWAITING_UTTERANCE)
                else:
                    consecutive_misinputs = 0

        # ── DISPATCHED: agent is running, watch for wakeword to re-interrupt ─
        elif state in (State.DISPATCHED, State.BOOTING, State.FATAL):
            was_idle = False
            if state == State.DISPATCHED:
                audio = np.frombuffer(data, dtype=np.int16)
                prob  = oww.predict(audio).get(wakeword_key, 0.0)
                if prob >= oww_threshold:
                    logger.info("[auricle-engine] wakeword during dispatch (p=%.2f) → AWAITING_UTTERANCE", prob)
                    oww.reset()
                    stt_provider.reset()
                    loop.call_soon_threadsafe(egress.abort)
                    loop.call_soon_threadsafe(lambda: asyncio.ensure_future(egress.play_file(ASSET_WAKEUP)))
                    fsm.transition(State.AWAITING_UTTERANCE)
                    active_listen_deadline = None


def _handle_transcript(
    text: str,
    fsm: FSM,
    loop: asyncio.AbstractEventLoop,
    dispatch_fn: Callable[[str], Coroutine],
) -> bool:
    """Handle a finalized transcript. Returns True if it was a misinput."""
    lower = text.lower().strip()

    if lower in MISINPUT_PHRASES:
        logger.info("[auricle-engine] misinput detected: %r", text)
        return True

    if lower in CLEAR_COMMANDS:
        logger.info("[auricle-engine] command: clear")
        fsm.transition(State.IDLE)
        asyncio.run_coroutine_threadsafe(dispatch_fn(_CMD_CLEAR), loop)
        return False

    if lower in STOP_COMMANDS:
        logger.info("[auricle-engine] command: stop")
        fsm.transition(State.IDLE)
        asyncio.run_coroutine_threadsafe(dispatch_fn(_CMD_STOP), loop)
        return False

    fsm.transition(State.DISPATCHED)
    asyncio.run_coroutine_threadsafe(dispatch_fn(text), loop)
    return False
