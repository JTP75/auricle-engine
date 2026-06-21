import threading
from enum import Enum, auto


class State(Enum):
    BOOTING            = auto()
    IDLE               = auto()
    AWAITING_UTTERANCE = auto()
    UTTERANCE          = auto()
    DISPATCHED         = auto()
    SPEAKING           = auto()
    FATAL              = auto()


class FSM:
    """Thread-safe finite state machine for the auricle engine."""

    def __init__(self) -> None:
        self._state  = State.BOOTING
        self._lock   = threading.Lock()
        self.muted   = False  # orthogonal flag; not a state
        self.sleeping = False  # orthogonal flag; not a state

    def get(self) -> State:
        with self._lock:
            return self._state

    def transition(self, new_state: State) -> None:
        with self._lock:
            self._state = new_state

    def transition_if(self, expected: State, new_state: State) -> bool:
        """Transition only when current state matches expected. Returns True on success."""
        with self._lock:
            if self._state == expected:
                self._state = new_state
                return True
            return False

    def is_idle_for_proactive(self) -> bool:
        """True when the engine is idle (no active utterance/dispatch/TTS in progress)."""
        with self._lock:
            return self._state == State.IDLE
