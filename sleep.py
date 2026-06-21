import logging
from enum import Enum, auto
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class SleepSignal(Enum):
    SLEEP = auto()   # flux EMA stayed low for full timeout → go to sleep
    WAKE  = auto()   # instantaneous flux spiked above baseline → wake up


class _Mode(Enum):
    COUNTING = auto()
    SLEEPING = auto()


class SleepDetector:
    """
    Detects extended acoustic silence using normalized spectral flux.

    Call feed() on each raw PCM chunk while the FSM is IDLE. Returns
    SleepSignal.SLEEP once the slow flux EMA has been below flux_threshold
    for timeout_chunks consecutive chunks. Returns SleepSignal.WAKE when
    a single chunk's instantaneous flux spikes above wake_multiplier times
    the baseline captured at sleep time.

    Uses normalized spectral flux (spectrum divided by total magnitude) so
    the metric is amplitude-independent — a fan at any volume registers near
    zero because its bin pattern doesn't reshape frame-to-frame.

    All numeric tuning values are passed via the constructor; no constants
    are defined in this module.
    """

    def __init__(
        self,
        timeout_seconds: float,
        sample_rate: int,
        chunk_bytes: int,
        flux_threshold: float,
        wake_multiplier: float,
        ema_alpha: float,
    ) -> None:
        bytes_per_sample = 2  # int16
        chunks_per_second = sample_rate * bytes_per_sample / chunk_bytes
        self._timeout_chunks  = int(timeout_seconds * chunks_per_second)
        self._timeout_seconds = timeout_seconds
        self._flux_threshold  = flux_threshold
        self._wake_multiplier = wake_multiplier
        self._ema_alpha       = ema_alpha

        self._prev_mag_norm:  Optional[np.ndarray] = None
        self._flux_ema:       float = 0.0
        self._low_flux_count: int   = 0
        self._sleep_baseline: float = 0.0
        self._mode = _Mode.COUNTING

    def reset(self) -> None:
        """Call when the FSM re-enters IDLE so the timer starts fresh."""
        self._prev_mag_norm  = None
        self._flux_ema       = 0.0
        self._low_flux_count = 0
        self._sleep_baseline = 0.0
        self._mode           = _Mode.COUNTING

    def feed(self, data: bytes) -> Optional[SleepSignal]:
        """Process one raw PCM chunk. Returns a signal or None."""
        audio    = np.frombuffer(data, dtype=np.int16).astype(np.float32)
        mag      = np.abs(np.fft.rfft(audio))
        total    = np.sum(mag)
        mag_norm = mag / (total + 1e-10)

        if self._prev_mag_norm is not None:
            flux = float(np.sum((mag_norm - self._prev_mag_norm) ** 2))
            self._flux_ema = self._ema_alpha * flux + (1.0 - self._ema_alpha) * self._flux_ema
        else:
            flux = 0.0
        self._prev_mag_norm = mag_norm

        if self._mode is _Mode.COUNTING:
            if self._flux_ema < self._flux_threshold:
                self._low_flux_count += 1
                if self._low_flux_count >= self._timeout_chunks:
                    self._sleep_baseline = max(self._flux_ema, 1e-10)
                    self._mode = _Mode.SLEEPING
                    logger.info(
                        "[auricle-engine] auto-sleep: %.0fs of silence → sleeping (ema=%.6f)",
                        self._timeout_seconds, self._flux_ema,
                    )
                    return SleepSignal.SLEEP
            else:
                self._low_flux_count = 0
            return None

        # SLEEPING: watch for instantaneous flux spike above the captured baseline
        wake_threshold = self._wake_multiplier * self._sleep_baseline
        if flux > wake_threshold:
            self._mode           = _Mode.COUNTING
            self._low_flux_count = 0
            logger.info(
                "[auricle-engine] auto-sleep: activity detected → waking (flux=%.6f, threshold=%.6f)",
                flux, wake_threshold,
            )
            return SleepSignal.WAKE
        return None
