"""A durable retry must not overlap its still-unwinding native execution."""

from __future__ import annotations

import asyncio
import atexit
import importlib.util
import sys
import threading
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from deerflow.config.subagent_batches_config import SubagentBatchesConfig
from deerflow.config.subagent_runtime_config import SubagentRuntimeConfig
from deerflow.subagents import batch_service as service_module
from deerflow.subagents.batch_service import SubagentBatchService


class _Status(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

    @property
    def is_terminal(self):
        return self in {_Status.COMPLETED, _Status.FAILED}


def _item():
    return {
        "id": "item-retry",
        "item_key": "stable-record-key",
        "prompt": "Process one record idempotently.",
        "batch": {
            "id": "batch-retry",
            "thread_id": "thread-retry",
            "user_id": "user-retry",
            "run_id": "run-retry",
            "execution_spec": {"subagent_config": {"name": "general-purpose", "description": "test", "system_prompt": "Work carefully."}, "parent_model": "test-model"},
        },
    }


async def _until(predicate, timeout=3):
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.005)


@pytest_asyncio.fixture
async def faulted_worker(monkeypatch):
    result = SimpleNamespace(status=_Status.RUNNING, result=None, error=None, stop_reason=None, token_usage_records=None, completed_at=None, execution_done_event=threading.Event())
    repository = SimpleNamespace(
        mark_item_running=AsyncMock(side_effect=RuntimeError("mark-running unavailable")),
        renew_item_lease=AsyncMock(return_value={"valid": True, "cancel_requested": False}),
        finalize_item=AsyncMock(return_value=True),
        requeue_item_after_admission_failure=AsyncMock(return_value=True),
    )
    cancelled, cleaned, tasks = [], [], []

    class Executor:
        def __init__(self, **_kwargs):
            pass

        def execute_async(self, _prompt, task_id=None):
            return "execution-retry"

    monkeypatch.setattr(service_module, "SubagentStatus", _Status)
    monkeypatch.setattr(service_module, "SubagentExecutor", Executor)
    monkeypatch.setattr(service_module, "get_background_task_result", lambda _: result)
    monkeypatch.setattr(service_module, "request_cancel_background_task", cancelled.append)
    monkeypatch.setattr(service_module, "cleanup_background_task", cleaned.append)
    monkeypatch.setattr(service_module, "resolve_subagent_model_name", lambda *_args, **_kwargs: "test-model")
    monkeypatch.setattr("deerflow.tools.get_available_tools", lambda **_: [])
    # Subsecond lease/grace is test-only; avoid a ten-second minimum-config
    # timeout in every cancellation/failure regression.
    config = SubagentBatchesConfig.model_construct(lease_seconds=0.6, poll_interval_seconds=0.01)
    service = SubagentBatchService(repository=repository, config=config, runtime_config=SubagentRuntimeConfig(max_running=1), app_config=SimpleNamespace())

    def start():
        task = asyncio.create_task(service._execute_item(_item()))
        tasks.append(task)
        return task

    yield SimpleNamespace(service=service, repository=repository, result=result, cancelled=cancelled, cleaned=cleaned, start=start)
    result.execution_done_event.set()
    for task in tasks:
        if not task.done():
            task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_mark_failure_cancels_then_waits_for_execution_exit_before_retry(faulted_worker):
    worker = faulted_worker
    task = worker.start()
    await _until(lambda: worker.cancelled or task.done())
    assert worker.cancelled == ["execution-retry"]
    worker.repository.finalize_item.assert_not_awaited()
    assert not task.done()
    assert worker.cleaned == []

    # Business terminality can precede stream/extension/capacity teardown.
    worker.result.status = _Status.FAILED
    await asyncio.sleep(0.06)
    worker.repository.finalize_item.assert_not_awaited()
    assert not task.done()
    worker.result.execution_done_event.set()
    await asyncio.wait_for(task, timeout=1)

    worker.repository.finalize_item.assert_awaited_once()
    assert worker.repository.finalize_item.await_args.kwargs["succeeded"] is False
    assert worker.repository.finalize_item.await_args.kwargs["error"] == "mark-running unavailable"
    assert worker.cleaned == ["execution-retry"]


@pytest.mark.asyncio
async def test_missing_background_result_does_not_release_item_for_retry(faulted_worker, monkeypatch):
    worker = faulted_worker
    monkeypatch.setattr(service_module, "get_background_task_result", lambda _: None)
    await asyncio.wait_for(worker.start(), timeout=1)
    worker.repository.finalize_item.assert_not_awaited()
    worker.repository.requeue_item_after_admission_failure.assert_not_awaited()
    assert worker.cancelled == ["execution-retry"]


@pytest.mark.asyncio
@pytest.mark.parametrize("renewal", [{"valid": False, "cancel_requested": True}, OSError("lease store unavailable")])
async def test_unsafe_lease_during_drain_never_finalizes_even_after_execution_exits(faulted_worker, renewal):
    worker = faulted_worker
    if isinstance(renewal, Exception):
        worker.repository.renew_item_lease.side_effect = renewal
    else:
        worker.repository.renew_item_lease.return_value = renewal
    task = worker.start()
    await _until(lambda: worker.repository.renew_item_lease.await_count or task.done())
    worker.repository.renew_item_lease.assert_awaited()
    assert not task.done(), "loss of lease must not abandon the cleanup grace"
    worker.repository.finalize_item.assert_not_awaited()
    worker.result.execution_done_event.set()
    await asyncio.wait_for(task, timeout=1)
    worker.repository.finalize_item.assert_not_awaited()
    assert worker.cancelled == ["execution-retry"]


@pytest.mark.asyncio
async def test_drain_renews_lease_and_remembers_a_later_loss(faulted_worker):
    worker = faulted_worker
    worker.service._config = worker.service._config.model_copy(update={"lease_seconds": 1.4})
    worker.repository.renew_item_lease.side_effect = [{"valid": True, "cancel_requested": False}, {"valid": False, "cancel_requested": True}]
    task = worker.start()
    await _until(lambda: worker.repository.renew_item_lease.await_count >= 2 or task.done())
    assert worker.repository.renew_item_lease.await_count >= 2
    assert not task.done()
    worker.result.execution_done_event.set()
    await asyncio.wait_for(task, timeout=1)
    worker.repository.finalize_item.assert_not_awaited()
    assert worker.cancelled == ["execution-retry"]


@pytest.mark.asyncio
async def test_monitor_renewal_error_waits_for_exit_and_preserves_original_error(faulted_worker):
    worker = faulted_worker
    worker.repository.mark_item_running.side_effect = None
    worker.repository.mark_item_running.return_value = True
    worker.repository.renew_item_lease.side_effect = [RuntimeError("original renewal failure"), {"valid": True, "cancel_requested": False}]
    task = worker.start()
    await _until(lambda: worker.cancelled or task.done())
    assert worker.cancelled == ["execution-retry"]
    worker.repository.finalize_item.assert_not_awaited()
    worker.result.status = _Status.FAILED
    worker.result.error = "child cancelled while unwinding"
    worker.result.execution_done_event.set()
    await asyncio.wait_for(task, timeout=1)
    worker.repository.finalize_item.assert_awaited_once()
    assert worker.repository.finalize_item.await_args.kwargs["error"] == "original renewal failure"


@pytest.mark.asyncio
async def test_drain_grace_is_bounded_without_making_retry_safe(faulted_worker):
    worker = faulted_worker
    await asyncio.wait_for(worker.start(), timeout=1.5)
    assert not worker.result.execution_done_event.is_set()
    worker.repository.finalize_item.assert_not_awaited()
    assert worker.cancelled == ["execution-retry"]
    assert worker.cleaned == ["execution-retry"]


@pytest.mark.asyncio
async def test_host_cancellation_during_drain_propagates_without_retry(faulted_worker):
    worker = faulted_worker
    task = worker.start()
    await _until(lambda: worker.cancelled or task.done())
    assert worker.cancelled == ["execution-retry"]
    task.cancel("shutdown")
    task.cancel("shutdown-again")
    with pytest.raises(asyncio.CancelledError):
        await task
    worker.repository.finalize_item.assert_not_awaited()
    assert not worker.result.execution_done_event.is_set()
    assert worker.cancelled == ["execution-retry"]
    # #5221's registry cleanup defers actual removal until execution exit.
    assert worker.cleaned == ["execution-retry"]


@pytest.mark.asyncio
async def test_pre_dispatch_failure_uses_existing_retry_path(faulted_worker, monkeypatch):
    worker = faulted_worker

    def broken_catalog(**_kwargs):
        raise ValueError("cannot construct tools")

    monkeypatch.setattr("deerflow.tools.get_available_tools", broken_catalog)
    await worker.start()
    worker.repository.finalize_item.assert_awaited_once()
    assert worker.repository.finalize_item.await_args.kwargs["error"] == "cannot construct tools"
    assert worker.cancelled == [] and worker.cleaned == []


@pytest.mark.asyncio
@pytest.mark.parametrize("admission_failure", [False, True])
async def test_terminal_persistence_failure_is_not_rewritten_as_execution_failure(faulted_worker, admission_failure):
    worker = faulted_worker
    worker.result.status = _Status.FAILED if admission_failure else _Status.COMPLETED
    worker.result.result = None if admission_failure else "completed report"
    worker.result.admission_failure = admission_failure
    worker.result.execution_done_event.set()
    persistence = worker.repository.requeue_item_after_admission_failure if admission_failure else worker.repository.finalize_item
    persistence.side_effect = OSError("terminal commit unavailable")
    with pytest.raises(OSError, match="terminal commit unavailable"):
        await worker.start()
    persistence.assert_awaited_once()
    if admission_failure:
        worker.repository.finalize_item.assert_not_awaited()
    else:
        assert persistence.await_args.kwargs["succeeded"] is True
    assert worker.cancelled == []


@pytest.mark.asyncio
async def test_cancelled_terminal_write_defers_real_registry_cleanup_until_exit(faulted_worker, monkeypatch):
    worker = faulted_worker
    name = "deerflow.subagents._batch_retry_cleanup_test_executor"
    spec = importlib.util.spec_from_file_location(name, Path(service_module.__file__).with_name("executor.py"))
    assert spec is not None and spec.loader is not None
    executor_module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, executor_module)
    spec.loader.exec_module(executor_module)
    result = executor_module.SubagentResult(task_id="execution-retry", trace_id="retry-cleanup", status=executor_module.SubagentStatus.COMPLETED, result="report")
    result._background_execution = True
    with executor_module._background_tasks_lock:
        executor_module._background_tasks[result.task_id] = result
    monkeypatch.setattr(service_module, "SubagentStatus", executor_module.SubagentStatus)
    monkeypatch.setattr(service_module, "get_background_task_result", executor_module.get_background_task_result)
    monkeypatch.setattr(service_module, "cleanup_background_task", executor_module.cleanup_background_task)
    worker.repository.finalize_item.side_effect = asyncio.CancelledError("shutdown during outcome write")
    try:
        with pytest.raises(asyncio.CancelledError):
            await worker.start()
        worker.repository.finalize_item.assert_awaited_once()
        assert worker.repository.finalize_item.await_args.kwargs["succeeded"] is True
        assert worker.cancelled == []
        assert not result.execution_done_event.is_set()
        assert executor_module.get_background_task_result(result.task_id) is result
        assert result.task_id in executor_module._background_cleanup_requested
        assert worker.service._execution_ids == {}

        # The executor, not the cancelled supervisor, owns eventual removal.
        executor_module._mark_background_execution_done(result)
        assert executor_module.get_background_task_result(result.task_id) is None
        assert result.task_id not in executor_module._background_cleanup_requested
    finally:
        executor_module._mark_background_execution_done(result)
        executor_module.cleanup_background_task(result.task_id)
        atexit.unregister(executor_module._shutdown_isolated_subagent_loop)


@pytest.mark.asyncio
async def test_sqlite_retry_is_unclaimable_until_real_isolated_execution_finishes(tmp_path, monkeypatch):
    """Real SQLite, execution registry, cross-loop cancellation, and exit fence.

    The child body is a deterministic stand-in for an Agent stream with slow
    teardown. Scheduling, registry ownership and durable retry are not mocked.
    """
    from deerflow.config.app_config import AppConfig
    from deerflow.config.database_config import DatabaseConfig
    from deerflow.persistence.engine import close_engine, get_session_factory, init_engine_from_config
    from deerflow.persistence.subagent_batches import SubagentBatchRepository

    # Isolate the real executor from conftest's cycle-breaking mock without
    # replacing shared package attributes or its process-wide test registry.
    name = "deerflow.subagents._batch_retry_test_executor"
    spec = importlib.util.spec_from_file_location(name, Path(service_module.__file__).with_name("executor.py"))
    assert spec is not None and spec.loader is not None
    executor_module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, executor_module)
    spec.loader.exec_module(executor_module)
    entered, teardown_started, release = threading.Event(), threading.Event(), threading.Event()
    holders = []
    parent_loop = asyncio.get_running_loop()
    child_loops = []

    async def slow_child(_self, _prompt, result_holder=None):
        holders.append(result_holder)
        child_loops.append(asyncio.get_running_loop())
        result_holder.status = executor_module.SubagentStatus.RUNNING
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            pass
        finally:
            teardown_started.set()
            await asyncio.to_thread(release.wait)
        result_holder.try_set_terminal(executor_module.SubagentStatus.CANCELLED, error="child cancelled")
        return result_holder

    monkeypatch.setattr(executor_module.SubagentExecutor, "_aexecute", slow_child)
    for attr in ("SubagentExecutor", "SubagentStatus", "get_background_task_result", "request_cancel_background_task", "cleanup_background_task"):
        monkeypatch.setattr(service_module, attr, getattr(executor_module, attr))
    monkeypatch.setattr("deerflow.tools.get_available_tools", lambda **_: [])
    task = None
    try:
        await init_engine_from_config(DatabaseConfig(backend="sqlite", sqlite_dir=str(tmp_path)))
        repository = SubagentBatchRepository(get_session_factory())
        batch = _item()["batch"]
        await repository.create_batch(
            batch_id=batch["id"],
            user_id=batch["user_id"],
            thread_id=batch["thread_id"],
            run_id=batch["run_id"],
            tool_call_id="batch-call",
            submission_key="retry-test",
            title="Retry supervision",
            subagent_type="general-purpose",
            items=[{"key": "stable-record-key", "prompt": "Process one record."}],
            max_live_items=1,
            max_running_items=1,
            max_attempts=2,
            execution_spec=batch["execution_spec"],
        )
        service = SubagentBatchService(
            repository=repository,
            config=SubagentBatchesConfig(lease_seconds=10, poll_interval_seconds=0.1),
            runtime_config=SubagentRuntimeConfig(max_running=1),
            app_config=AppConfig.model_validate({"sandbox": {"use": "test"}}),
        )
        item = (await repository.claim_items(now=datetime.now(UTC), lease_owner=service._lease_owner, lease_seconds=10, limit=1))[0]
        marked = asyncio.Event()

        async def broken_mark(*_args, **_kwargs):
            assert entered.is_set()
            marked.set()
            raise OSError("mark-running write failed")

        monkeypatch.setattr(repository, "mark_item_running", broken_mark)
        task = asyncio.create_task(service._execute_item(item))
        await asyncio.wait_for(marked.wait(), timeout=3)
        rows = await repository.list_items(batch["id"], user_id=batch["user_id"])
        assert rows[0]["status"] == "leased", "retry became visible while the original execution still owns its side effects"
        assert await repository.claim_items(now=datetime.now(UTC), lease_owner="another-worker", lease_seconds=10, limit=1) == []
        await _until(teardown_started.is_set)
        assert child_loops == [child_loops[0]] and child_loops[0] is not parent_loop
        assert not holders[0].execution_done_event.is_set()
        assert not task.done()

        release.set()
        await asyncio.wait_for(task, timeout=3)
        assert holders[0].execution_done_event.is_set()
        rows = await repository.list_items(batch["id"], user_id=batch["user_id"])
        assert rows[0]["status"] == "queued"
        assert rows[0]["error"] == "mark-running write failed"
        retry = (await repository.claim_items(now=datetime.now(UTC), lease_owner="another-worker", lease_seconds=10, limit=1))[0]
        assert retry["id"] == item["id"] and retry["item_key"] == "stable-record-key"
        assert retry["attempt"] == 2
    finally:
        release.set()
        for holder in holders:
            executor_module.request_cancel_background_task(holder.task_id)
        if task is not None:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        for holder in holders:
            await asyncio.to_thread(holder.execution_done_event.wait, 3)
            executor_module.cleanup_background_task(holder.task_id)
        await asyncio.to_thread(executor_module._shutdown_isolated_subagent_loop)
        atexit.unregister(executor_module._shutdown_isolated_subagent_loop)
        await close_engine()
