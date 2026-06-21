"""
_handle_transcript unit + integration tests.

_handle_transcript() is the decision point after every STT result: is this
a misinput, a voice command, or a real utterance? Getting it wrong means either
dispatching garbage to the agent (misinput slips through) or silently dropping
real utterances (command false-positive).

These tests run _handle_transcript() inside a real asyncio loop so the
run_coroutine_threadsafe dispatch actually executes.
"""

import asyncio

from fsm import FSM, State
from ingress import _handle_transcript
from consts import _CMD_CLEAR, _CMD_STOP, MISINPUT_PHRASES, CLEAR_COMMANDS, STOP_COMMANDS


def _run(text: str, initial_state: State = State.IDLE):
    """
    Run _handle_transcript() and collect (result, final_state, dispatched_texts).

    Uses two asyncio.sleep(0) yields so run_coroutine_threadsafe callbacks
    are fully scheduled and executed before we inspect results.
    """
    dispatched: list[str] = []

    async def impl():
        loop = asyncio.get_running_loop()
        fsm = FSM()
        fsm.transition(initial_state)

        async def capture(t: str) -> None:
            dispatched.append(t)

        result = _handle_transcript(text, fsm, loop, capture)
        await asyncio.sleep(0)  # let call_soon_threadsafe schedule the task
        await asyncio.sleep(0)  # let the task coroutine run
        return result, fsm.get()

    result, state = asyncio.run(impl())
    return result, state, dispatched


# ── misinput detection ─────────────────────────────────────────────────────

def test_misinput_phrases_return_true():
    # Every phrase in the frozenset must be caught. Missing one means a bare
    # article like "the" dispatches as a real utterance to the agent.
    for phrase in MISINPUT_PHRASES:
        result, state, dispatched = _run(phrase)
        assert result is True, f"Expected misinput for {phrase!r}"
        assert dispatched == [], f"Misinput {phrase!r} was dispatched"


def test_misinput_is_case_insensitive():
    result, _, dispatched = _run("THE")
    assert result is True
    assert dispatched == []


def test_misinput_does_not_change_fsm_state():
    # The caller (ingress loop) handles FSM transitions for misinputs.
    # _handle_transcript must leave the state alone so the caller can decide
    # whether to play the confused chime or return to IDLE.
    _, state, _ = _run("the", initial_state=State.UTTERANCE)
    assert state == State.UTTERANCE


# ── voice commands ─────────────────────────────────────────────────────────

def test_clear_commands_dispatch_cmd_clear():
    for cmd in CLEAR_COMMANDS:
        result, state, dispatched = _run(cmd)
        assert result is False, f"Clear command {cmd!r} wrongly flagged as misinput"
        assert state == State.IDLE, f"FSM should be IDLE after clear, got {state}"
        assert dispatched == [_CMD_CLEAR], f"Wrong dispatch for {cmd!r}: {dispatched}"


def test_stop_commands_dispatch_cmd_stop():
    for cmd in STOP_COMMANDS:
        result, state, dispatched = _run(cmd)
        assert result is False
        assert state == State.IDLE
        assert dispatched == [_CMD_STOP], f"Wrong dispatch for {cmd!r}: {dispatched}"


def test_commands_are_case_insensitive():
    # "Clear." (with capital C) is in CLEAR_COMMANDS; the matching is done
    # after lower() so "CLEAR" should also match.
    result, state, dispatched = _run("CLEAR")
    assert result is False
    assert state == State.IDLE
    assert dispatched == [_CMD_CLEAR]


# ── normal utterances ──────────────────────────────────────────────────────

def test_normal_text_dispatches_as_utterance():
    result, state, dispatched = _run("what is the weather today")
    assert result is False
    assert state == State.DISPATCHED
    assert dispatched == ["what is the weather today"]


def test_normal_text_preserves_original_case():
    # Lower-casing is only used for matching; the original text must reach
    # the agent intact so the LLM sees proper capitalisation.
    _, _, dispatched = _run("Hey Hermes, what time is it?")
    assert dispatched == ["Hey Hermes, what time is it?"]


def test_normal_text_transitions_to_dispatched():
    _, state, _ = _run("turn on the lights")
    assert state == State.DISPATCHED


def test_multi_word_non_misinput_dispatches():
    # "in the" is a misinput but "in the morning" is a real utterance.
    result, state, dispatched = _run("in the morning")
    assert result is False
    assert state == State.DISPATCHED
    assert dispatched == ["in the morning"]


def test_misinput_substring_in_longer_phrase_passes():
    # "the" alone is a misinput; "the weather" is not.
    result, _, dispatched = _run("the weather")
    assert result is False
    assert dispatched == ["the weather"]
