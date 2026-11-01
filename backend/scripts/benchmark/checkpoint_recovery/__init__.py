"""Offline process-restart observations for the public DeerFlow SDK factory."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sqlite3
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[3]
SCENARIOS = ("uninterrupted", "crash_after_checkpoint", "crash_before_tool_checkpoint")
OPERATIONS = ("alpha", "beta")
PROMPT = "Record alpha and beta, once each, then report their labels."


def _worker_environment(root: Path) -> dict[str, str]:
    # An allowlist avoids inheriting model credentials, user configuration,
    # Python startup hooks or a caller's package search path.
    env = {name: os.environ[name] for name in ("PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP") if name in os.environ}
    env.update(
        HOME=str(root / "home"),
        DEER_FLOW_HOME=str(root / "home"),
        PYTHONPATH=str(BACKEND),
        PYTHONNOUSERSITE="1",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONUTF8="1",
    )
    return env


def _wait_ready(child: subprocess.Popen, ready: Path, deadline: float) -> None:
    while not ready.exists():
        if child.poll() is not None:
            raise RuntimeError("worker exited before the fault boundary")
        if time.monotonic() >= deadline:
            raise TimeoutError("worker did not reach the fault boundary")
        time.sleep(0.01)
    if child.poll() is not None:
        raise RuntimeError("worker exited before it could be killed")


def _phase(root: Path, phase: str, boundary: str, timeout_seconds: float) -> tuple[int, int, dict]:
    command = [sys.executable, "-m", "scripts.benchmark.checkpoint_recovery._worker", str(root), phase, boundary]
    deadline = time.monotonic() + timeout_seconds
    with (root / f"{phase}.log").open("w", encoding="utf-8") as log:
        with subprocess.Popen(command, cwd=BACKEND, env=_worker_environment(root), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT) as child:
            try:
                if phase == "initial" and boundary != "none":
                    ready = root / "ready.json"
                    _wait_ready(child, ready, deadline)
                    if json.loads(ready.read_text(encoding="utf-8")) != {"boundary": boundary}:
                        raise ValueError("unexpected fault boundary")
                    child.kill()
                    child.wait(timeout=5)
                    if child.returncode == 0:
                        raise RuntimeError("fault worker did not terminate abnormally")
                    return child.pid, child.returncode, {}
                child.wait(timeout=max(0.001, deadline - time.monotonic()))
                if child.returncode != 0:
                    raise RuntimeError(f"{phase} worker exited with code {child.returncode}")
                result = json.loads((root / f"{phase}.json").read_text(encoding="utf-8"))
                return child.pid, child.returncode, result
            finally:
                if child.poll() is None:
                    child.kill()
                child.wait(timeout=5)


def _run_case(scenario: str, root: Path, *, timeout_seconds: float) -> dict:
    """Observe one fresh database; no result is inferred from a model's claim."""
    if scenario not in SCENARIOS:
        raise ValueError("unknown recovery scenario")
    boundary = {"uninterrupted": "none", "crash_after_checkpoint": "after", "crash_before_tool_checkpoint": "before"}[scenario]
    result = {"scenario": scenario, "expected_observation_match": False}
    phase = "initial"
    try:
        root.mkdir(parents=True, exist_ok=True)
        pid, code, output = _phase(root, "initial", boundary, timeout_seconds)
        result.update(first_pid=pid, first_process_exit=code)
        if boundary != "none":
            phase = "resume"
            started = time.perf_counter()
            pid, _, output = _phase(root, "resume", boundary, timeout_seconds)
            result.update(resume_pid=pid, resume_process_wall_ms=round((time.perf_counter() - started) * 1000, 3))
        effects = (root / "effects.jsonl").read_text(encoding="utf-8").splitlines()
        effects = [json.loads(line) for line in effects]
        counts = Counter(effects)
        duplicates = sum(max(0, count - 1) for count in counts.values())
        expected = ["alpha", "alpha", "beta"] if boundary == "before" else list(OPERATIONS)
        prior = output["checkpoint_tool_results"]
        expected_prior = ["alpha"] if boundary == "after" else []
        result.update(
            effects=effects,
            tool_executions=len(effects),
            duplicate_effects=duplicates,
            final_answer=output["final_answer"],
            resume_checkpoint_tool_results=prior,
            resume_execution_ms=output["execution_ms"] if boundary != "none" else None,
            checkpoint_file_bytes=sum(path.stat().st_size for path in root.glob("checkpoints.sqlite*")),
            expected_observation_match=effects == expected and prior == expected_prior and output["final_answer"] == "alpha|beta",
        )
    except (OSError, RuntimeError, ValueError, TimeoutError, subprocess.TimeoutExpired) as error:
        # Child logs are private temporary diagnostics, not public provider or
        # environment dumps. Report a stable error class, never a traceback.
        result.update(error=type(error).__name__, failed_phase=phase)
    return result


def _provenance() -> dict:
    env = {key: os.environ[key] for key in ("PATH", "SYSTEMROOT", "WINDIR") if key in os.environ}
    env.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")
    repo = BACKEND.parent
    try:
        revision = subprocess.run(["git", "-c", f"safe.directory={repo}", "rev-parse", "HEAD"], cwd=repo, env=env, check=True, capture_output=True, text=True, timeout=5).stdout.strip()
        dirty = bool(subprocess.run(["git", "-c", f"safe.directory={repo}", "status", "--porcelain"], cwd=repo, env=env, check=True, capture_output=True, text=True, timeout=5).stdout)
    except (OSError, subprocess.SubprocessError):
        revision, dirty = None, None
    source = "".join(name + "\0" + Path(__file__).with_name(name).read_text(encoding="utf-8") + "\0" for name in ("__init__.py", "__main__.py", "_worker.py"))
    return {
        "git_revision": revision,
        "git_dirty": dirty,
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "python": platform.python_version(),
        "platform": platform.system(),
        "sqlite": sqlite3.sqlite_version,
        "packages": {name: importlib.metadata.version(name) for name in ("langchain", "langgraph", "langgraph-checkpoint-sqlite")},
    }


def run_suite(*, repeat: int = 1, timeout_seconds: float = 30) -> dict:
    """Run scripted controls; timings are observations, never pass thresholds."""
    if isinstance(repeat, bool) or not isinstance(repeat, int) or not 1 <= repeat <= 10:
        raise ValueError("repeat must be an integer from 1 to 10")
    if isinstance(timeout_seconds, bool) or not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 300:
        raise ValueError("timeout_seconds must be finite, positive and at most 300")
    fixture = {"synthetic": True, "version": 1, "operations": OPERATIONS, "prompt": PROMPT, "model": "state-driven-script-v1", "checkpoint_mode": "full", "durability": "sync", "scenarios": SCENARIOS}
    cases = []
    with tempfile.TemporaryDirectory(prefix="deerflow-recovery-") as temporary:
        for iteration in range(repeat):
            for scenario in SCENARIOS:
                case = _run_case(scenario, Path(temporary) / f"{iteration}-{scenario}", timeout_seconds=timeout_seconds)
                cases.append({"iteration": iteration, **case})
    return {
        "schema_version": 1,
        "fixture": {**fixture, "sha256": hashlib.sha256(json.dumps(fixture, sort_keys=True).encode("utf-8")).hexdigest()},
        "config": {"repeat": repeat, "timeout_seconds": timeout_seconds, "clock": "perf_counter; informational wall time"},
        "provenance": _provenance(),
        "cases": cases,
        "summary": {"cases": len(cases), "matching_observations": sum(case["expected_observation_match"] for case in cases)},
    }
