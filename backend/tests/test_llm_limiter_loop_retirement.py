"""A discarded cross-loop wakeup must not reserve provider capacity forever."""

import asyncio
import importlib.util
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from deerflow.agents.middlewares.llm_error_handling_middleware import _ProcessWideLimiter


def _park_on_another_loop(limiter):
    loop = asyncio.new_event_loop()
    # Deliberately abandon a suspended task to model an unclean loop shutdown.
    # Record its expected destruction diagnostic rather than logging it globally.
    errors = []
    loop.set_exception_handler(lambda _loop, context: errors.append(context))
    task = loop.create_task(limiter.acquire_async())
    loop.run_until_complete(asyncio.sleep(0))
    assert not task.done()
    return loop, task, errors


def _dispose(loop, task):
    if not loop.is_closed():
        loop.close()
    task.get_coro().close()


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [1, 2])
@pytest.mark.parametrize("close_before_release", [False, True])
async def test_new_async_call_recovers_a_grant_discarded_by_loop_close(limit, close_before_release):
    limiter = _ProcessWideLimiter(limit)
    for _ in range(limit):
        limiter.acquire_sync()
    loop, task, errors = await asyncio.to_thread(_park_on_another_loop, limiter)
    try:
        if close_before_release:
            loop.close()
            limiter.release()
        else:
            limiter.release()  # call_soon_threadsafe accepts the wakeup.
            loop.close()  # The accepted callback never gets a chance to run.
        await asyncio.wait_for(limiter.acquire_async(), timeout=1)
        assert limiter.in_flight == limit
        limiter.release()
        assert limiter.in_flight == limit - 1
        assert not errors
    finally:
        _dispose(loop, task)
        for _ in range(limit - 1):
            limiter.release()


@pytest.mark.asyncio
async def test_closed_loop_does_not_reclaim_an_already_acquired_permit():
    limiter = _ProcessWideLimiter(1)

    def acquire_then_close():
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(limiter.acquire_async())
        finally:
            loop.close()

    await asyncio.to_thread(acquire_then_close)
    # Once acquire returns, only the caller's release may return its permit.
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(limiter.acquire_async(), timeout=0.2)
    assert limiter.in_flight == 1
    limiter.release()
    await asyncio.wait_for(limiter.acquire_async(), timeout=1)
    limiter.release()
    assert limiter.in_flight == 0


@pytest.mark.asyncio
async def test_reclaiming_several_grants_and_late_cleanup_preserves_the_cap():
    limiter = _ProcessWideLimiter(3)
    for _ in range(3):
        limiter.acquire_sync()
    parked = [await asyncio.to_thread(_park_on_another_loop, limiter) for _ in range(3)]
    try:
        for _ in parked:
            limiter.release()
        for loop, _, _ in parked:
            loop.close()
        for _ in range(3):
            await asyncio.wait_for(limiter.acquire_async(), timeout=1)
        # Finalizing old waiters after their grants were reused must not free
        # the replacement callers' permits a second time.
        for loop, task, _ in parked:
            _dispose(loop, task)
        assert limiter.in_flight == 3
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(limiter.acquire_async(), timeout=0.2)
        for _ in range(3):
            limiter.release()
        assert limiter.in_flight == 0
    finally:
        for loop, task, _ in parked:
            _dispose(loop, task)


@pytest.mark.asyncio
async def test_isolated_subagent_loop_shutdown_does_not_strand_the_lead_loop(monkeypatch):
    # conftest installs an executor stub to break an import cycle. Load the
    # production shutdown separately, as the direct-result graph tests do.
    path = Path(__file__).parents[1] / "packages/harness/deerflow/subagents/executor.py"
    spec = importlib.util.spec_from_file_location("_llm_limiter_shutdown_executor", path)
    executor = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, executor)
    spec.loader.exec_module(executor)

    limiter = _ProcessWideLimiter(1)
    limiter.acquire_sync()
    loop, task, _ = await asyncio.to_thread(_park_on_another_loop, limiter)
    monkeypatch.setattr(executor, "_isolated_subagent_loop", loop)
    monkeypatch.setattr(executor, "_isolated_subagent_loop_thread", None)
    monkeypatch.setattr(executor, "_isolated_subagent_loop_started", None)
    monkeypatch.setattr(executor, "_isolated_subagent_loop_shutdown_pending", False)
    try:
        limiter.release()
        executor._shutdown_isolated_subagent_loop()
        assert loop.is_closed()
        await asyncio.wait_for(limiter.acquire_async(), timeout=1)
        limiter.release()
        assert limiter.in_flight == 0
    finally:
        _dispose(loop, task)


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [1, 2])
async def test_already_queued_async_call_recovers_without_another_release(limit):
    limiter = _ProcessWideLimiter(limit)
    for _ in range(limit):
        limiter.acquire_sync()
    loop, task, _ = await asyncio.to_thread(_park_on_another_loop, limiter)
    follower = asyncio.create_task(limiter.acquire_async())
    await asyncio.sleep(0)
    try:
        assert not follower.done()
        limiter.release()
        loop.close()
        await asyncio.wait_for(follower, timeout=1)
        assert limiter.in_flight == limit
        limiter.release()
        assert limiter.in_flight == limit - 1
    finally:
        follower.cancel()
        await asyncio.gather(follower, return_exceptions=True)
        _dispose(loop, task)
        for _ in range(limit - 1):
            limiter.release()


@pytest.mark.parametrize("limit", [1, 2])
def test_queued_sync_call_recovers_without_another_release(limit, monkeypatch):
    limiter = _ProcessWideLimiter(limit)
    for _ in range(limit):
        limiter.acquire_sync()
    loop, task, _ = _park_on_another_loop(limiter)
    queued = threading.Event()
    entered = threading.Event()
    original_wait = limiter._cond.wait

    def wait_until_notified(timeout=None):
        # Signal while holding the condition lock. release() cannot transfer
        # the permit until the follower is genuinely waiting on capacity.
        queued.set()
        return original_wait(timeout)

    monkeypatch.setattr(limiter._cond, "wait", wait_until_notified)

    def follower():
        limiter.acquire_sync()
        entered.set()
        limiter.release()

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(follower)
        try:
            assert queued.wait(1)
            limiter.release()
            loop.close()
            assert entered.wait(1), "closed-loop reservation stranded a healthy synchronous caller"
            future.result(timeout=1)
            assert limiter.in_flight == limit - 1
        finally:
            # Also join the worker when testing the broken baseline.
            if not entered.is_set():
                limiter.release()
            future.result(timeout=2)
            _dispose(loop, task)
            for _ in range(limit - 1):
                limiter.release()


@pytest.mark.asyncio
async def test_a_paused_but_open_loop_keeps_its_reservation():
    limiter = _ProcessWideLimiter(1)
    limiter.acquire_sync()
    loop, task, _ = await asyncio.to_thread(_park_on_another_loop, limiter)
    limiter.release()
    try:
        # A stopped loop can legally be restarted. It still owns its grant.
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(limiter.acquire_async(), timeout=0.2)
        await asyncio.to_thread(loop.run_until_complete, task)
        assert limiter.in_flight == 1
        limiter.release()
        await asyncio.wait_for(limiter.acquire_async(), timeout=1)
        limiter.release()
        assert limiter.in_flight == 0
    finally:
        _dispose(loop, task)
