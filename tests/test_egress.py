"""
EgressController tests — unit tests for segmentation, integration tests for
the full play pipeline with mocked TTS and audio output.

The egress is responsible for: splitting a response at newlines, synthesising
each sentence, playing them in order with lookahead prefetch, and stopping
cleanly when barge-in fires. A bug here means the wrong text gets spoken, or
TTS continues playing after the user interrupted.
"""

import asyncio

from audio_buffer import AudioBuffer
from egress import EgressController, _segment
from consts import TTS_MAX_CHARS


# ── _segment unit tests ────────────────────────────────────────────────────
# These are pure-function tests; no asyncio needed.

def test_segment_single_sentence():
    assert _segment("hello world") == ["hello world"]


def test_segment_splits_on_newline():
    assert _segment("first\nsecond") == ["first", "second"]


def test_segment_multiple_newlines_collapse():
    assert _segment("a\n\n\nb") == ["a", "b"]


def test_segment_strips_whitespace():
    assert _segment("  hello  \n  world  ") == ["hello", "world"]


def test_segment_empty_string():
    assert _segment("") == []


def test_segment_only_whitespace_and_newlines():
    assert _segment("  \n\n  ") == []


def test_segment_trailing_newline():
    assert _segment("hello\n") == ["hello"]


# ── Mocks ─────────────────────────────────────────────────────────────────

class _MockTTS:
    """Records which sentences were synthesised."""
    def __init__(self, audio: bytes = b"x" * 100):
        self._audio = audio
        self.calls: list[str] = []

    async def stream_audio(self, text: str):
        self.calls.append(text)
        if self._audio:
            yield self._audio


class _MockHandle:
    def __init__(self):
        self.killed = False

    async def wait(self):
        pass  # instant completion

    def kill(self):
        self.killed = True


class _MockOutput:
    """Records what was played."""
    def __init__(self):
        self.played: list[bytes] = []
        self._last_handle: _MockHandle | None = None

    async def play_bytes(self, audio_bytes: bytes) -> _MockHandle:
        self.played.append(audio_bytes)
        self._last_handle = _MockHandle()
        return self._last_handle

    async def play_file(self, path) -> None:
        pass

    def play_file_sync(self, path) -> None:
        pass


def _make_egress():
    tts = _MockTTS()
    out = _MockOutput()
    barge_in = asyncio.Event()
    buf = AudioBuffer(maxlen=50, tts_tail_seconds=0.0)
    egress = EgressController(tts, barge_in, buf, out)
    return egress, tts, out, barge_in


# ── EgressController integration tests ────────────────────────────────────

def test_single_sentence_plays():
    async def impl():
        egress, tts, out, _ = _make_egress()
        egress.reset()
        egress.start_worker()
        await egress.process_delta("hello world", finalize=True)
        assert tts.calls == ["hello world"]
        assert len(out.played) == 1

    asyncio.run(impl())


def test_two_sentences_both_play_in_order():
    # Segmentation and ordering matter: playing "world" before "hello" is wrong.
    async def impl():
        egress, tts, out, _ = _make_egress()
        egress.reset()
        egress.start_worker()
        await egress.process_delta("first\nsecond", finalize=True)
        assert tts.calls == ["first", "second"]
        assert len(out.played) == 2

    asyncio.run(impl())


def test_empty_text_produces_no_tts():
    async def impl():
        egress, tts, out, _ = _make_egress()
        egress.reset()
        egress.start_worker()
        await egress.process_delta("", finalize=True)
        assert tts.calls == []
        assert out.played == []

    asyncio.run(impl())


def test_whitespace_only_produces_no_tts():
    async def impl():
        egress, tts, out, _ = _make_egress()
        egress.reset()
        egress.start_worker()
        await egress.process_delta("  \n\n  ", finalize=True)
        assert tts.calls == []

    asyncio.run(impl())


def test_tts_max_chars_drops_sentences_past_cap():
    # Cumulative character count: once spoken_chars >= TTS_MAX_CHARS, further
    # sentences are silently dropped. This prevents runaway TTS on long responses.
    # s1 (1500 chars): 0 < 3000 → queued, spoken=1500
    # s2 (1500 chars): 1500 < 3000 → queued, spoken=3000
    # s3 (any):        3000 >= 3000 → DROPPED
    async def impl():
        egress, tts, out, _ = _make_egress()
        egress.reset()
        egress.start_worker()
        s1 = "a" * 1500
        s2 = "b" * 1500
        s3 = "c" * 100
        await egress.process_delta(f"{s1}\n{s2}\n{s3}", finalize=True)
        assert tts.calls == [s1, s2]

    asyncio.run(impl())


def test_barge_in_set_before_speak_skips_tts():
    # When a barge-in fires, the user is already speaking. Any queued TTS
    # must be dropped immediately — playing it would talk over the user.
    async def impl():
        egress, tts, out, barge_in = _make_egress()
        egress.reset()
        egress.start_worker()
        barge_in.set()
        await egress.process_delta("this should not play", finalize=True)
        assert tts.calls == []

    asyncio.run(impl())


def test_reset_clears_barge_in_for_next_turn():
    # After barge-in, the next agent response must play normally.
    # If reset() doesn't clear barge_in, every subsequent response is silently dropped.
    async def impl():
        egress, tts, out, barge_in = _make_egress()

        # First turn: barge-in fires
        egress.reset()
        egress.start_worker()
        barge_in.set()
        await egress.process_delta("interrupted", finalize=True)
        assert tts.calls == []

        # Second turn: reset clears the barge-in flag
        egress.abort()
        egress.reset()
        egress.start_worker()
        await egress.process_delta("after reset", finalize=True)
        assert "after reset" in tts.calls

    asyncio.run(impl())


def test_reset_clears_spoken_chars_counter():
    # spoken_chars must reset between turns, otherwise the second response
    # is silently capped by chars spoken in the first.
    async def impl():
        egress, tts, out, _ = _make_egress()

        # First turn: use up chars close to the cap
        egress.reset()
        egress.start_worker()
        await egress.process_delta("a" * (TTS_MAX_CHARS - 100), finalize=True)

        # Second turn: a short response should play in full
        egress.reset()
        egress.start_worker()
        await egress.process_delta("short response", finalize=True)
        assert "short response" in tts.calls

    asyncio.run(impl())


def test_abort_drains_queue():
    # abort() must leave the egress in a state where reset()+start_worker()
    # can start a fresh turn cleanly (no stuck queue.join()).
    async def impl():
        egress, tts, out, _ = _make_egress()
        egress.reset()
        egress.start_worker()
        egress.abort()
        egress.reset()
        egress.start_worker()
        await egress.process_delta("clean turn", finalize=True)
        assert "clean turn" in tts.calls

    asyncio.run(impl())


def test_tts_active_flag_set_during_playback():
    # AudioBuffer.tts_active must be True while audio is playing so the
    # ingress thread gates STT. If it's never set, speaker echo feeds STT.
    active_during = []

    class _TrackingOutput(_MockOutput):
        def __init__(self, buf):
            super().__init__()
            self._buf = buf

        async def play_bytes(self, audio_bytes):
            active_during.append(self._buf.tts_active)
            handle = _MockHandle()
            self.played.append(audio_bytes)
            return handle

    async def impl():
        tts = _MockTTS()
        buf = AudioBuffer(maxlen=50, tts_tail_seconds=0.0)
        out = _TrackingOutput(buf)
        barge_in = asyncio.Event()
        egress = EgressController(tts, barge_in, buf, out)
        egress.reset()
        egress.start_worker()
        await egress.process_delta("hello", finalize=True)
        assert any(active_during), "tts_active was never True during playback"

    asyncio.run(impl())
