"""
FSM unit tests.

The FSM is the arbiter of what every other component does at each moment.
A wrong transition silently corrupts the audio pipeline, so transitions must
be airtight — including under concurrent access from the ingress thread.
"""

import threading

from fsm import FSM, State


def test_initial_state():
    assert FSM().get() == State.BOOTING


def test_transition():
    fsm = FSM()
    fsm.transition(State.IDLE)
    assert fsm.get() == State.IDLE


def test_transition_full_happy_path():
    fsm = FSM()
    path = [
        State.IDLE,
        State.AWAITING_UTTERANCE,
        State.UTTERANCE,
        State.DISPATCHED,
        State.SPEAKING,
        State.IDLE,
    ]
    for state in path:
        fsm.transition(state)
        assert fsm.get() == state


def test_transition_to_fatal():
    fsm = FSM()
    fsm.transition(State.FATAL)
    assert fsm.get() == State.FATAL


def test_transition_if_succeeds_when_expected():
    fsm = FSM()
    fsm.transition(State.SPEAKING)
    result = fsm.transition_if(State.SPEAKING, State.AWAITING_UTTERANCE)
    assert result is True
    assert fsm.get() == State.AWAITING_UTTERANCE


def test_transition_if_no_op_when_not_expected():
    # Guards the egress→ingress hand-off: SPEAKING→AWAITING_UTTERANCE must not
    # fire if barge-in already moved the FSM to AWAITING_UTTERANCE.
    fsm = FSM()
    fsm.transition(State.AWAITING_UTTERANCE)
    result = fsm.transition_if(State.SPEAKING, State.IDLE)
    assert result is False
    assert fsm.get() == State.AWAITING_UTTERANCE  # unchanged


def test_muted_flag_independent_of_state():
    # muted is an orthogonal flag: it gates wakeword detection but never
    # changes the FSM state itself.
    fsm = FSM()
    assert fsm.muted is False
    fsm.muted = True
    fsm.transition(State.IDLE)
    assert fsm.get() == State.IDLE
    assert fsm.muted is True  # state transition must not clear the flag


def test_sleeping_flag_independent_of_state():
    fsm = FSM()
    assert fsm.sleeping is False
    fsm.sleeping = True
    fsm.transition(State.IDLE)
    assert fsm.sleeping is True


def test_is_idle_for_proactive_only_in_idle():
    # Proactive messages (cron notifications) should only play when the FSM
    # is IDLE — playing audio mid-utterance would corrupt the pipeline.
    fsm = FSM()
    fsm.transition(State.IDLE)
    assert fsm.is_idle_for_proactive() is True

    for state in [
        State.BOOTING,
        State.AWAITING_UTTERANCE,
        State.UTTERANCE,
        State.DISPATCHED,
        State.SPEAKING,
        State.FATAL,
    ]:
        fsm.transition(state)
        assert fsm.is_idle_for_proactive() is False, f"should be False in {state}"


def test_thread_safety_concurrent_transitions():
    # The ingress thread and the asyncio event loop both call transition().
    # This must never corrupt the internal state.
    fsm = FSM()
    fsm.transition(State.IDLE)
    errors = []

    def worker():
        for _ in range(2000):
            try:
                fsm.transition(State.IDLE)
                s = fsm.get()
                assert s in State.__members__.values()
                fsm.transition(State.SPEAKING)
                s = fsm.get()
                assert s in State.__members__.values()
            except Exception as exc:
                errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"Thread safety violations: {errors}"


def test_transition_if_thread_safety():
    # transition_if is used for the SPEAKING→AWAITING_UTTERANCE hand-off;
    # only one caller should win, never both.
    fsm = FSM()
    fsm.transition(State.SPEAKING)
    wins = []

    def try_transition():
        if fsm.transition_if(State.SPEAKING, State.AWAITING_UTTERANCE):
            wins.append(1)

    threads = [threading.Thread(target=try_transition) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(wins) == 1, "Exactly one thread should win the SPEAKING→AWAITING transition"
    assert fsm.get() == State.AWAITING_UTTERANCE
