"""
SleepDetector unit tests.

The detector uses normalized spectral flux (not amplitude) to distinguish
silence from speech. This matters because fans and HVAC produce constant
moderate amplitude — an amplitude-based detector would never sleep in a
real room, while this one settles as long as the spectrum doesn't change.

If SLEEP fires too early, the wakeword is missed while the user is speaking.
If SLEEP never fires, the OWW model runs forever at full CPU.
If WAKE fires on noise (not speech), the user gets a spurious wakeup.
"""

import numpy as np

from sleep import SleepDetector, SleepSignal

_CHUNK_BYTES = 1280
_SAMPLE_RATE = 16000
_N_SAMPLES = _CHUNK_BYTES // 2


def _detector(timeout_seconds=1.0, flux_threshold=0.02, wake_multiplier=3.0, ema_alpha=0.5):
    return SleepDetector(
        timeout_seconds=timeout_seconds,
        sample_rate=_SAMPLE_RATE,
        chunk_bytes=_CHUNK_BYTES,
        flux_threshold=flux_threshold,
        wake_multiplier=wake_multiplier,
        ema_alpha=ema_alpha,
    )


def _silent() -> bytes:
    """All-zero PCM: zero amplitude, zero spectral flux between frames."""
    return bytes(_CHUNK_BYTES)


def _sweep(i: int) -> bytes:
    """Sine at a different frequency each call — guarantees high spectral flux.

    Each successive chunk has energy at a different FFT bin, so the
    normalised magnitude spectrum shifts dramatically frame-to-frame.
    This reliably keeps flux_ema above the threshold.
    """
    freq = 200 + i * 100  # 200 Hz, 300 Hz, 400 Hz …
    t = np.linspace(0, _N_SAMPLES / _SAMPLE_RATE, _N_SAMPLES, endpoint=False)
    return (30000 * np.sin(2 * np.pi * freq * t)).astype(np.int16).tobytes()


def _spike() -> bytes:
    """Square wave at Nyquist — maximum spectral energy relative to silent baseline."""
    arr = np.array(
        [30000 if i % 2 == 0 else -30000 for i in range(_N_SAMPLES)],
        dtype=np.int16,
    )
    return arr.tobytes()


# ── basic behaviour ────────────────────────────────────────────────────────

def test_first_chunk_returns_none():
    # No prev_mag_norm on the first call — nothing to diff against.
    det = _detector()
    assert det.feed(_silent()) is None


def test_sleep_fires_after_sustained_silence():
    # chunks_per_second = 16000*2 / 1280 = 25; timeout_seconds=1.0 → 25 chunks
    det = _detector(timeout_seconds=1.0)
    fired = None
    for _ in range(60):  # well past the 25-chunk threshold
        sig = det.feed(_silent())
        if sig is SleepSignal.SLEEP:
            fired = sig
            break
    assert fired is SleepSignal.SLEEP


def test_no_sleep_with_spectrally_varied_audio():
    # Rapidly changing spectral content (like speech) must never trigger sleep,
    # even if the overall volume is constant.
    det = _detector(timeout_seconds=0.5)
    for i in range(60):
        result = det.feed(_sweep(i))
        assert result is not SleepSignal.SLEEP, f"SLEEP fired spuriously on chunk {i}"


def test_wake_fires_after_acoustic_spike_during_sleep():
    det = _detector(timeout_seconds=0.5, wake_multiplier=2.0)

    # Phase 1: go to sleep
    for _ in range(60):
        sig = det.feed(_silent())
        if sig is SleepSignal.SLEEP:
            break
    else:
        raise AssertionError("SLEEP never fired — check detector params")

    # Phase 2: spike should wake
    wake = None
    for _ in range(5):
        wake = det.feed(_spike())
        if wake is SleepSignal.WAKE:
            break
    assert wake is SleepSignal.WAKE


# ── reset ─────────────────────────────────────────────────────────────────

def test_reset_restarts_countdown():
    # reset() is called each time the FSM re-enters IDLE. Without it, partial
    # silence from a previous conversation would carry over and cause early sleep.
    det = _detector(timeout_seconds=1.0)  # 25 chunks to sleep

    for _ in range(15):  # 15 silent chunks (more than half way)
        det.feed(_silent())

    det.reset()

    # 15 more chunks after reset should not trigger sleep (only 15/25)
    results = [det.feed(_silent()) for _ in range(15)]
    assert SleepSignal.SLEEP not in results


def test_reset_while_sleeping_returns_to_counting():
    det = _detector(timeout_seconds=0.5)

    for _ in range(60):
        if det.feed(_silent()) is SleepSignal.SLEEP:
            break

    det.reset()

    # After reset, a spike should NOT produce WAKE (we're back in COUNTING mode)
    result = det.feed(_spike())
    assert result is not SleepSignal.WAKE


# ── post-wake behaviour ────────────────────────────────────────────────────

def test_after_wake_returns_none_on_next_silent_chunk():
    # After waking, detector is back in COUNTING mode — a single silent chunk
    # is nowhere near the timeout and should return None, not WAKE again.
    det = _detector(timeout_seconds=0.5, wake_multiplier=2.0)

    for _ in range(60):
        if det.feed(_silent()) is SleepSignal.SLEEP:
            break
    for _ in range(5):
        if det.feed(_spike()) is SleepSignal.WAKE:
            break

    result = det.feed(_silent())
    assert result is None
