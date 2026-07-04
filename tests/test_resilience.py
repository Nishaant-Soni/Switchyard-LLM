import asyncio

import httpx
import pytest

from gateway.observability.cost import PriceBook
from gateway.providers.base import UpstreamError
from gateway.resilience.circuit_breaker import BreakerRegistry, CircuitBreaker, CircuitState
from gateway.resilience.retry import AllTargetsFailed, ResilientExecutor
from gateway.routing.policies import Target
from gateway.routing.signals import RoutingSignals
from gateway.schemas import ChatCompletionRequest, ChatCompletionResponse, Choice, Message, Usage


# --- fakes -------------------------------------------------------------------------------
class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def ok_response(model="m"):
    return ChatCompletionResponse(
        id="x",
        object="chat.completion",
        created=0,
        model=model,
        choices=[Choice(index=0, message=Message(role="assistant", content="hi"))],
        usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )


class FakeAdapter:
    """behavior is a response to return, or an Exception to raise, or a list to consume in order."""

    def __init__(self, name, behavior):
        self.name = name
        self.behavior = behavior
        self.calls = 0

    async def chat_completion(self, request):
        self.calls += 1
        b = self.behavior
        if isinstance(b, list):
            b = b[min(self.calls - 1, len(b) - 1)]
        if isinstance(b, Exception):
            raise b
        return b


class FakeRegistry:
    def __init__(self, adapters):
        self._adapters = adapters

    def get(self, name):
        return self._adapters.get(name)


async def _no_sleep(_):
    return None


def make_executor(**kwargs):
    return ResilientExecutor(
        BreakerRegistry(min_calls=2, window_size=10, failure_threshold=0.5),
        base_delay_s=0.0,
        sleep=_no_sleep,
        **kwargs,
    )


REQUEST = ChatCompletionRequest(model="fast", messages=[Message(role="user", content="hi")])


# --- circuit breaker ---------------------------------------------------------------------
def test_breaker_trips_on_failure_rate():
    b = CircuitBreaker("p", failure_threshold=0.5, min_calls=4, window_size=10)
    for _ in range(3):
        b.record_failure()
    assert b.state == CircuitState.CLOSED  # below min_calls
    b.record_failure()
    assert b.state == CircuitState.OPEN  # 4/4 failures >= 0.5
    assert b.allow() is False


def test_breaker_stays_closed_below_threshold():
    b = CircuitBreaker("p", failure_threshold=0.6, min_calls=4, window_size=10)
    b.record_success()
    b.record_success()
    b.record_failure()
    b.record_failure()
    assert b.state == CircuitState.CLOSED  # 2/4 = 0.5 < 0.6


def test_breaker_half_open_then_close_on_success():
    clock = FakeClock()
    b = CircuitBreaker("p", failure_threshold=0.5, min_calls=2, reset_timeout_s=30, clock=clock)
    b.record_failure()
    b.record_failure()
    assert b.state == CircuitState.OPEN
    assert b.allow() is False
    clock.advance(31)
    assert b.allow() is True  # probe permitted
    assert b.state == CircuitState.HALF_OPEN
    assert b.allow() is False  # only one probe at a time
    b.record_success()
    assert b.state == CircuitState.CLOSED


def test_breaker_half_open_reopens_on_failure():
    clock = FakeClock()
    b = CircuitBreaker("p", failure_threshold=0.5, min_calls=2, reset_timeout_s=30, clock=clock)
    b.record_failure()
    b.record_failure()
    clock.advance(31)
    assert b.allow() is True
    b.record_failure()  # probe fails
    assert b.state == CircuitState.OPEN


# --- executor fallback -------------------------------------------------------------------
def test_fallback_to_next_provider_on_5xx():
    reg = FakeRegistry(
        {
            "groq": FakeAdapter("groq", UpstreamError(500, {"error": "boom"})),
            "ollama": FakeAdapter("ollama", ok_response("ollama-m")),
        }
    )
    targets = [Target("groq", "a"), Target("ollama", "b")]
    resp, target = asyncio.run(make_executor().execute(targets, reg, REQUEST))
    assert target.provider == "ollama"
    assert resp.model == "ollama-m"


def test_fallback_on_429():
    reg = FakeRegistry(
        {
            "groq": FakeAdapter("groq", UpstreamError(429, {"error": "rate"})),
            "ollama": FakeAdapter("ollama", ok_response()),
        }
    )
    targets = [Target("groq", "a"), Target("ollama", "b")]
    _, target = asyncio.run(make_executor().execute(targets, reg, REQUEST))
    assert target.provider == "ollama"


def test_fallback_on_transport_error():
    reg = FakeRegistry(
        {
            "groq": FakeAdapter("groq", httpx.ConnectError("refused")),
            "ollama": FakeAdapter("ollama", ok_response()),
        }
    )
    targets = [Target("groq", "a"), Target("ollama", "b")]
    _, target = asyncio.run(make_executor().execute(targets, reg, REQUEST))
    assert target.provider == "ollama"


def test_non_retryable_4xx_surfaces_immediately():
    secondary = FakeAdapter("ollama", ok_response())
    reg = FakeRegistry(
        {
            "groq": FakeAdapter("groq", UpstreamError(400, {"error": "bad request"})),
            "ollama": secondary,
        }
    )
    targets = [Target("groq", "a"), Target("ollama", "b")]
    with pytest.raises(UpstreamError) as exc:
        asyncio.run(make_executor().execute(targets, reg, REQUEST))
    assert exc.value.status_code == 400
    assert secondary.calls == 0  # never fell through to the next provider


def test_all_targets_failed():
    reg = FakeRegistry(
        {
            "groq": FakeAdapter("groq", UpstreamError(500, {"error": "boom"})),
            "ollama": FakeAdapter("ollama", UpstreamError(503, {"error": "down"})),
        }
    )
    targets = [Target("groq", "a"), Target("ollama", "b")]
    with pytest.raises(AllTargetsFailed) as exc:
        asyncio.run(make_executor().execute(targets, reg, REQUEST))
    assert isinstance(exc.value.last_error, UpstreamError)


def test_open_circuit_is_skipped():
    executor = make_executor()
    # Trip groq's breaker before executing.
    groq_breaker = executor.breakers.get("groq")
    groq_breaker.record_failure()
    groq_breaker.record_failure()
    assert groq_breaker.state == CircuitState.OPEN

    groq = FakeAdapter("groq", ok_response("groq-m"))
    reg = FakeRegistry({"groq": groq, "ollama": FakeAdapter("ollama", ok_response("ollama-m"))})
    targets = [Target("groq", "a"), Target("ollama", "b")]
    _, target = asyncio.run(executor.execute(targets, reg, REQUEST))
    assert target.provider == "ollama"
    assert groq.calls == 0  # skipped without being called


def test_success_records_and_serves_first_healthy():
    groq = FakeAdapter("groq", ok_response("groq-m"))
    reg = FakeRegistry({"groq": groq, "ollama": FakeAdapter("ollama", ok_response())})
    targets = [Target("groq", "a"), Target("ollama", "b")]
    executor = make_executor()
    _, target = asyncio.run(executor.execute(targets, reg, REQUEST))
    assert target.provider == "groq"
    assert executor.breakers.get("groq").state == CircuitState.CLOSED


def test_executor_records_served_latency_to_signals():
    # The served provider's call latency is fed to the live-signal registry (latency-aware routing
    # reads it). A fake clock advanced inside the adapter call makes the measured latency exact.
    clock = FakeClock()
    signals = RoutingSignals(PriceBook({}))

    class SlowAdapter:
        def __init__(self, dt):
            self.dt = dt

        async def chat_completion(self, request):
            clock.advance(self.dt)  # simulate the call taking dt seconds
            return ok_response()

    reg = FakeRegistry({"groq": SlowAdapter(0.5)})
    executor = ResilientExecutor(
        BreakerRegistry(min_calls=2),
        base_delay_s=0.0,
        sleep=_no_sleep,
        clock=clock,
        signals=signals,
    )
    asyncio.run(executor.execute([Target("groq", "a")], reg, REQUEST))
    assert signals.latency_ms("groq") == 500.0  # 0.5 s -> 500 ms (first sample seeds the EWMA)
