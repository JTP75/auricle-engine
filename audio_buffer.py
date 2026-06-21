import collections
import threading
import time
from typing import List


class AudioBuffer:
    """
    Unified ring buffer for all captured PCM audio.

    Every chunk is stored alongside the tts_active flag at capture time.
    Egress sets/clears tts_active around playback; ingress gates STT on it
    to prevent speaker echo from being fed into the STT backend.

    replay() returns only the chunks captured after the most recent TTS-active
    period, giving the STT backend echo-free look-back audio on state transitions.
    """

    def __init__(self, maxlen: int, tts_tail_seconds: float = 0.0) -> None:
        self._buf: collections.deque = collections.deque(maxlen=maxlen)
        self._tts_active = False
        self._tts_tail = tts_tail_seconds
        self._quiet_until: float = 0.0
        self._lock = threading.Lock()

    def append(self, chunk: bytes) -> None:
        with self._lock:
            self._buf.append((chunk, self._tts_active))

    def set_tts_active(self, active: bool) -> None:
        with self._lock:
            self._tts_active = active
            if not active:
                self._quiet_until = time.monotonic() + self._tts_tail

    @property
    def tts_active(self) -> bool:
        with self._lock:
            return self._tts_active or time.monotonic() < self._quiet_until

    def replay(self) -> List[bytes]:
        """Chunks captured after the most recent TTS-active period (echo-free)."""
        with self._lock:
            items = list(self._buf)
        last_tts = -1
        for i, (_, active) in enumerate(items):
            if active:
                last_tts = i
        return [chunk for chunk, _ in items[last_tts + 1:]]
