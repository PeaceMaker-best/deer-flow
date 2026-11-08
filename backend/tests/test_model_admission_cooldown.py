"""Model graphs share provider cooldowns only within opted-in quota groups."""

import asyncio
import math
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from deerflow.agents.middlewares import llm_error_handling_middleware as errors
from deerflow.config.app_config import AppConfig, LlmCallConfig
from deerflow.config.model_config import RequestAdmissionConfig
from deerflow.config.sandbox_config import SandboxConfig
from deerflow.models import request_admission as admission


class ScriptedModel(BaseChatModel):
    fails: bool = True
    status: int = 429
    hint: bool = True

    @property
    def _llm_type(self) -> str:
        return "offline-admission-probe"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs: Any):
        if self.fails:
            request = httpx.Request("POST", "https://offline.invalid/model")
            response = httpx.Response(self.status, headers={"retry-after-ms": "10000"} if self.hint else {}, request=request)
            raise httpx.HTTPStatusError("rate limit" if self.status == 429 else "unavailable", request=request, response=response)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="OK"))])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs: Any):
        return self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


@pytest.fixture
def clock(monkeypatch) -> Iterator[list[float]]:
    current = [0.0]
    monkeypatch.setattr(admission, "monotonic", lambda: current[0])
    monkeypatch.setattr(admission, "_registry", {})
    monkeypatch.setattr(errors, "_PROCESS_LIMITER", None)
    monkeypatch.setattr(errors, "_CAP_RESOLVED", False)
    yield current


@pytest.mark.asyncio
@pytest.mark.parametrize("same_group", [False, True], ids=["independent", "shared"])
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("status,hint,should_pause", [(429, True, True), (429, False, False), (503, True, False)], ids=["rate-limited", "no-hint", "not-rate-limit"])
async def test_provider_cooldown_is_shared_by_the_quota_group(clock, asynchronous, status, hint, should_pause, same_group):
    policy = RequestAdmissionConfig(requests_per_minute=6000, group="audit-shared")
    owner = admission.get_request_admission("producer", policy)
    peer_policy = policy if same_group else policy.model_copy(update={"group": "independent"})
    peer = admission.get_request_admission("peer", peer_policy)
    assert (owner is peer) is same_group
    model = ScriptedModel(rate_limiter=owner, status=status, hint=hint)
    config = AppConfig(sandbox=SandboxConfig(use="test"), llm_call=LlmCallConfig(retry_max_attempts=1))
    graph = create_agent(model=model, middleware=[errors.LLMErrorHandlingMiddleware(app_config=config)])
    payload = {"messages": [HumanMessage(content="Return a short answer")]}
    if asynchronous:
        await graph.ainvoke(payload)
    else:
        graph.invoke(payload)
    clock[0] = 1.0  # Normal RPM spacing has passed; the provider's 10s hint has not.
    dispatched = peer.acquire(blocking=False)
    assert dispatched is not (should_pause and same_group)


def test_cooldown_extends_the_schedule_without_spending_a_permit(clock):
    limiter = admission.RequestAdmission(RequestAdmissionConfig(requests_per_minute=60))
    limiter.defer_for(10)
    clock[0] = 1
    limiter.defer_for(2)  # A later, shorter hint cannot undo the earlier one.
    assert not limiter.acquire(blocking=False)
    clock[0] = 10
    assert limiter.acquire(blocking=False)
    assert not limiter.acquire(blocking=False)
    clock[0] = 11
    assert limiter.acquire(blocking=False)


def test_long_cooldown_does_not_poll_at_the_high_rpm_interval(clock):
    limiter = admission.RequestAdmission(RequestAdmissionConfig(requests_per_minute=60000))
    limiter.defer_for(60)
    assert limiter._delay(300) == pytest.approx(0.05)
    clock[0] = 59.999
    assert limiter._delay(300) == pytest.approx(0.001)


@pytest.mark.parametrize("seconds", [math.inf, math.nan, -1])
def test_invalid_cooldown_does_not_poison_future_admission(clock, seconds):
    limiter = admission.RequestAdmission(RequestAdmissionConfig(requests_per_minute=60))
    with pytest.raises(ValueError, match="finite and non-negative"):
        limiter.defer_for(seconds)
    assert limiter.acquire(blocking=False)


@pytest.mark.asyncio
async def test_cooldown_preserves_fifo_and_cancelled_waiter_cleanup(clock):
    limiter = admission.RequestAdmission(RequestAdmissionConfig(requests_per_minute=60))
    limiter.defer_for(10)
    first, cancelled, last = [asyncio.create_task(limiter.aacquire()) for _ in range(3)]
    try:
        await asyncio.sleep(0)
        assert len(limiter._waiters) == 3
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        assert len(limiter._waiters) == 2
        clock[0] = 10
        assert await asyncio.wait_for(first, 1)
        assert not last.done()
        clock[0] = 11
        assert await asyncio.wait_for(last, 1)
        assert not limiter._waiters
    finally:
        for task in (first, cancelled, last):
            task.cancel()
        await asyncio.gather(first, cancelled, last, return_exceptions=True)


@pytest.mark.asyncio
async def test_wait_deadline_still_applies_during_provider_cooldown(clock):
    limiter = admission.RequestAdmission(RequestAdmissionConfig(requests_per_minute=60, max_wait_seconds=1))
    limiter.defer_for(10)
    waiter = asyncio.create_task(limiter.aacquire())
    try:
        await asyncio.sleep(0)
        clock[0] = 2
        with pytest.raises(admission.AdmissionError, match="timed out"):
            await asyncio.wait_for(waiter, 1)
        assert not limiter._waiters
        clock[0] = 10
        assert limiter.acquire(blocking=False)
    finally:
        waiter.cancel()
        await asyncio.gather(waiter, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_waiter", [False, True], ids=["resume", "cancel"])
async def test_peer_agent_waits_without_dispatching_and_can_be_cancelled(clock, cancel_waiter):
    policy = RequestAdmissionConfig(requests_per_minute=6000, group="agent-graphs")
    limiter = admission.get_request_admission("producer", policy)
    config = AppConfig(sandbox=SandboxConfig(use="test"), llm_call=LlmCallConfig(retry_max_attempts=1))
    producer = create_agent(model=ScriptedModel(rate_limiter=limiter), middleware=[errors.LLMErrorHandlingMiddleware(app_config=config)])
    peer = create_agent(model=ScriptedModel(fails=False, rate_limiter=admission.get_request_admission("peer", policy)), middleware=[errors.LLMErrorHandlingMiddleware(app_config=config)])
    payload = {"messages": [HumanMessage(content="Return a short answer")]}
    await producer.ainvoke(payload)
    clock[0] = 1
    task = asyncio.create_task(peer.ainvoke(payload))
    try:
        for _ in range(100):
            if limiter._waiters or task.done():
                break
            await asyncio.sleep(0.005)
        assert len(limiter._waiters) == 1
        assert not task.done()
        if cancel_waiter:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert not limiter._waiters
        clock[0] = 10
        result = await asyncio.wait_for(peer.ainvoke(payload) if cancel_waiter else task, 1)
        assert result["messages"][-1].content == "OK"
        assert not limiter._waiters
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
