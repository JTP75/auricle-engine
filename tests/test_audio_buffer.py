"""
AudioBuffer unit tests.

The buffer's primary job is echo suppression: STT must not process audio
that was played through the speaker. It does this by tagging each chunk with
tts_active at capture time and stripping TTS-era chunks from replay().

If replay() returns chunks from a TTS period, the speaker will feed back
into the STT model and produce phantom transcripts.
"""

import time

from audio_buffer import AudioBuffer

_CHUNK = b"\x01\x02" * 640  # 1280 bytes, non-zero


def _chunk(tag: int) -> bytes:
    """1280-byte chunk with a distinguishable first byte."""
    return bytes([tag]) + b"\x00" * 1279


# ── tts_active flag ────────────────────────────────────────────────────────

def test_tts_active_starts_false():
    assert AudioBuffer(maxlen=10).tts_active is False


def test_tts_active_set_true():
    buf = AudioBuffer(maxlen=10)
    buf.set_tts_active(True)
    assert buf.tts_active is True


def test_tts_active_cleared():
    buf = AudioBuffer(maxlen=10, tts_tail_seconds=0.0)
    buf.set_tts_active(True)
    buf.set_tts_active(False)
    assert buf.tts_active is False


def test_tts_tail_keeps_active_after_clear():
    # The tail window prevents STT from eating speaker echo immediately
    # after TTS ends — the speaker keeps vibrating for a few hundred ms.
    buf = AudioBuffer(maxlen=10, tts_tail_seconds=0.5)
    buf.set_tts_active(True)
    buf.set_tts_active(False)
    assert buf.tts_active is True  # tail still active


def test_tts_tail_expires():
    buf = AudioBuffer(maxlen=10, tts_tail_seconds=0.05)
    buf.set_tts_active(True)
    buf.set_tts_active(False)
    time.sleep(0.1)
    assert buf.tts_active is False


# ── replay echo-stripping ──────────────────────────────────────────────────

def test_replay_no_tts_returns_all():
    buf = AudioBuffer(maxlen=10, tts_tail_seconds=0.0)
    buf.append(_chunk(1))
    buf.append(_chunk(2))
    assert buf.replay() == [_chunk(1), _chunk(2)]


def test_replay_strips_tts_era_chunks():
    # Only the chunk captured AFTER the TTS period should come back.
    # Returning the TTS-era chunk would feed speaker audio into STT.
    buf = AudioBuffer(maxlen=10, tts_tail_seconds=0.0)
    buf.append(_chunk(1))           # pre-TTS
    buf.set_tts_active(True)
    buf.append(_chunk(2))           # during TTS — must be excluded
    buf.set_tts_active(False)
    buf.append(_chunk(3))           # post-TTS — must be included

    result = buf.replay()
    assert len(result) == 1
    assert result[0][0] == 3


def test_replay_strips_all_pre_last_tts_period():
    # Multiple TTS periods: only chunks after the LAST one are returned.
    buf = AudioBuffer(maxlen=20, tts_tail_seconds=0.0)
    buf.append(_chunk(1))
    buf.set_tts_active(True)
    buf.append(_chunk(2))
    buf.set_tts_active(False)
    buf.append(_chunk(3))           # between TTS periods
    buf.set_tts_active(True)
    buf.append(_chunk(4))
    buf.set_tts_active(False)
    buf.append(_chunk(5))           # only this should be returned

    result = buf.replay()
    assert [r[0] for r in result] == [5]


def test_replay_empty_buffer():
    assert AudioBuffer(maxlen=10).replay() == []


def test_replay_only_tts_chunks_returns_empty():
    buf = AudioBuffer(maxlen=10, tts_tail_seconds=0.0)
    buf.set_tts_active(True)
    buf.append(_chunk(1))
    buf.set_tts_active(False)
    assert buf.replay() == []


# ── ring buffer eviction ───────────────────────────────────────────────────

def test_maxlen_evicts_oldest():
    # When the buffer is full, the oldest chunk is dropped. If STT relies on
    # stale audio from before the ring wraps, it gets the wrong context.
    buf = AudioBuffer(maxlen=3, tts_tail_seconds=0.0)
    for i in range(5):
        buf.append(bytes([i]) * 1280)
    result = buf.replay()
    assert len(result) == 3
    assert [r[0] for r in result] == [2, 3, 4]
