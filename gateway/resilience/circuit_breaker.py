"""Per-provider circuit breaker: closed -> open (failure-rate over a rolling window) ->
half-open (single probe after a reset timeout) -> closed. Stops the gateway from hammering a
dead backend and lets it recover.

All methods are synchronous (no awaits), so they run atomically on the asyncio event loop —
no locking needed in a single-process app.
"""

import logging
import time
from collections import deque
from collections.abc import Callable
from enum import Enum

logger = logging.getLogger("gateway.breaker")


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    def __init__(
        self,
        name: str = "",
        *,
        failure_threshold: float = 0.5,
        window_size: int = 20,
        min_calls: int = 5,
        reset_timeout_s: float = 30.0,
        half_open_max: int = 1,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.min_calls = min_calls
        self.reset_timeout_s = reset_timeout_s
        self.half_open_max = half_open_max
        self.clock = clock

        self._state = CircuitState.CLOSED
        self._window: deque[bool] = deque(maxlen=window_size)  # True=success, False=failure
        self._opened_at = 0.0
        self._half_open_calls = 0

    @property
    def state(self) -> CircuitState:
        return self._state

    def allow(self) -> bool:
        """Whether a call may proceed now. Drives the OPEN -> HALF_OPEN time transition."""
        if self._state == CircuitState.OPEN:
            if self.clock() - self._opened_at >= self.reset_timeout_s:
                self._to_half_open()
            else:
                return False
        if self._state == CircuitState.HALF_OPEN:
            if self._half_open_calls < self.half_open_max:
                self._half_open_calls += 1
                return True
            return False
        return True  # CLOSED

    def record_success(self) -> None:
        if self._state == CircuitState.HALF_OPEN:
            self._to_closed()
        else:
            self._window.append(True)

    def record_failure(self) -> None:
        if self._state == CircuitState.HALF_OPEN:
            self._to_open()
            return
        self._window.append(False)
        if len(self._window) >= self.min_calls:
            failure_rate = self._window.count(False) / len(self._window)
            if failure_rate >= self.failure_threshold:
                self._to_open()

    def _to_open(self) -> None:
        self._transition(CircuitState.OPEN)
        self._opened_at = self.clock()
        self._half_open_calls = 0

    def _to_half_open(self) -> None:
        self._transition(CircuitState.HALF_OPEN)
        self._half_open_calls = 0

    def _to_closed(self) -> None:
        self._transition(CircuitState.CLOSED)
        self._window.clear()
        self._half_open_calls = 0

    def _transition(self, new_state: CircuitState) -> None:
        if new_state != self._state:
            logger.warning("circuit %r %s -> %s", self.name, self._state.value, new_state.value)
            self._state = new_state


class BreakerRegistry:
    """Lazily creates one CircuitBreaker per provider, all sharing the same config."""

    def __init__(self, **breaker_kwargs):
        self._kwargs = breaker_kwargs
        self._breakers: dict[str, CircuitBreaker] = {}

    def get(self, name: str) -> CircuitBreaker:
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(name=name, **self._kwargs)
        return self._breakers[name]

    def snapshot(self) -> dict[str, str]:
        return {name: b.state.value for name, b in self._breakers.items()}
