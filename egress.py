import asyncio
import logging
import re
from pathlib import Path
from typing import Optional

from audio_buffer import AudioBuffer
from audio_io import AudioOutput, PlaybackHandle
from consts import TTS_MAX_CHARS

logger = logging.getLogger(__name__)

_TEXT_STREAM_DELIMITER = re.compile(r'\n+')
_NO_LOOKAHEAD = object()  # sentinel: no prefetched sentence waiting


def _segment(text: str) -> list[str]:
    return [s.strip() for s in _TEXT_STREAM_DELIMITER.split(text) if s.strip()]


class EgressController:
    """
    Manages streaming TTS playback for one agent response turn.

    Receives the full response text via process_delta(), segments it by
    newline into sentences, and plays them in order via a sequential
    asyncio queue with lookahead prefetch.
    """

    def __init__(
        self,
        tts_provider,
        barge_in_event: asyncio.Event,
        audio_buffer: AudioBuffer,
        audio_output: AudioOutput,
    ) -> None:
        self._tts          = tts_provider
        self._barge_in     = barge_in_event
        self._audio_buffer = audio_buffer
        self._audio_output = audio_output

        self._processed_len: int                    = 0
        self._text_buffer:   str                    = ""
        self._spoken_chars:  int                    = 0
        self._queue:         asyncio.Queue          = asyncio.Queue()
        self._active_handle: Optional[PlaybackHandle] = None
        self._worker_task:   Optional[asyncio.Task] = None
        self._generation:    int                    = 0

    def reset(self) -> None:
        self._processed_len = 0
        self._text_buffer   = ""
        self._spoken_chars  = 0
        self._queue         = asyncio.Queue()
        self._active_handle = None
        self._worker_task   = None
        self._barge_in.clear()
        self._audio_buffer.set_tts_active(False)

    def abort(self) -> None:
        """Forcefully abort the active egress playback task and clear the queue."""
        logger.info("[auricle-engine] aborting active egress playback")
        self._barge_in.set()
        self._audio_buffer.set_tts_active(False)
        self.kill_active()
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except (asyncio.QueueEmpty, ValueError):
                break

    def start_worker(self) -> None:
        self._generation += 1
        self._worker_task = asyncio.create_task(
            self._worker(self._queue, self._generation)
        )

    async def process_delta(self, cumulative_text: str, *, finalize: bool) -> None:
        if self._barge_in.is_set():
            logger.debug("[auricle-engine] process_delta ignored: barge-in event is set")
            return

        new_text = cumulative_text[self._processed_len:]
        self._processed_len = len(cumulative_text)
        self._text_buffer  += new_text

        sentences = _segment(self._text_buffer)

        if finalize:
            completed         = sentences
            self._text_buffer = ""
        elif len(sentences) > 1:
            completed         = sentences[:-1]
            self._text_buffer = sentences[-1]
        else:
            completed = []

        for sentence in completed:
            if self._spoken_chars >= TTS_MAX_CHARS:
                break
            self._spoken_chars += len(sentence)
            await self._queue.put(sentence)

        if finalize:
            await self._queue.put(None)  # sentinel — signals end of turn
            await self._queue.join()
            if self._worker_task:
                self._worker_task.cancel()

    def kill_active(self) -> None:
        """Kill the active playback. Safe to call from any thread."""
        handle = self._active_handle
        if handle is not None:
            handle.kill()

    async def _fetch_audio(self, sentence: str) -> bytes:
        """Collect all TTS audio bytes for a sentence into memory."""
        chunks: list[bytes] = []
        async for chunk in self._tts.stream_audio(sentence):
            if self._barge_in.is_set():
                return b""
            chunks.append(chunk)
        return b"".join(chunks)

    async def play_file(self, path: Path) -> None:
        """Play a WAV asset file directly (for notify/wakeup/tosleep/etc.)."""
        await self._audio_output.play_file(path)

    async def speak(self, text: str, *, priority: bool = False) -> None:
        """Synthesize and play a short phrase immediately, outside the worker queue.
        priority=True bypasses barge-in gating (for error/cleared system phrases)."""
        if priority:
            chunks: list[bytes] = []
            async for chunk in self._tts.stream_audio(text):
                chunks.append(chunk)
            audio = b"".join(chunks)
        else:
            audio = await self._fetch_audio(text)
        if audio:
            handle = await self._audio_output.play_bytes(audio)
            await handle.wait()

    async def _worker(self, queue: asyncio.Queue, my_gen: int) -> None:
        lookahead = _NO_LOOKAHEAD
        prefetch_bytes: Optional[bytes] = None

        def _stale() -> bool:
            return self._barge_in.is_set() or self._generation != my_gen

        while True:
            if _stale():
                if lookahead is not _NO_LOOKAHEAD:
                    queue.task_done()
                self._drain(queue)
                break

            if lookahead is not _NO_LOOKAHEAD:
                sentence, audio = lookahead, prefetch_bytes
                lookahead, prefetch_bytes = _NO_LOOKAHEAD, None
            else:
                sentence = await queue.get()
                audio = None

            if _stale():
                queue.task_done()
                self._drain(queue)
                break

            if sentence is None:
                queue.task_done()
                logger.info("[auricle-engine] TTS turn complete")
                break

            try:
                if not audio:
                    audio = await self._fetch_audio(sentence)
                handle = await self._audio_output.play_bytes(audio)

                if _stale():
                    handle.kill()
                    self._drain(queue)
                    break

                self._active_handle = handle
                self._audio_buffer.set_tts_active(True)

                play_task     = asyncio.create_task(handle.wait())
                get_next_task = asyncio.create_task(queue.get())

                done, _ = await asyncio.wait(
                    [play_task, get_next_task], return_when=asyncio.FIRST_COMPLETED
                )

                if get_next_task in done:
                    next_item = get_next_task.result()
                    lookahead = next_item
                    if next_item is not None and not _stale():
                        fetched, _ = await asyncio.gather(
                            self._fetch_audio(next_item),
                            play_task,
                            return_exceptions=True,
                        )
                        prefetch_bytes = fetched if isinstance(fetched, bytes) else None
                    else:
                        await play_task
                        prefetch_bytes = None
                else:
                    get_next_task.cancel()
                    await asyncio.gather(get_next_task, return_exceptions=True)

            except Exception as exc:
                logger.error("[auricle-engine] TTS playback error: %s", exc)
            finally:
                self._audio_buffer.set_tts_active(False)
                self._active_handle = None
                queue.task_done()

    def _drain(self, queue: Optional[asyncio.Queue] = None) -> None:
        q = queue if queue is not None else self._queue
        while not q.empty():
            try:
                q.get_nowait()
                q.task_done()
            except asyncio.QueueEmpty:
                break
