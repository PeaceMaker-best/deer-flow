"""Exercise recovery through fresh processes, the SDK factory and real SQLite."""

import json
import subprocess

import pytest

from scripts.benchmark.checkpoint_recovery import run_suite
from scripts.benchmark.checkpoint_recovery.__main__ import main


def test_recovery_matrix_measures_the_committed_and_uncommitted_windows(tmp_path, capsys):
    output = tmp_path / "recovery.json"
    assert main(["--output", str(output)]) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert json.loads(capsys.readouterr().out) == report
    assert report["schema_version"] == 1
    assert report["fixture"]["synthetic"] is True
    assert len(report["fixture"]["sha256"]) == 64
    assert report["summary"]["matching_observations"] == 3
    cases = {case["scenario"]: case for case in report["cases"]}
    control = cases["uninterrupted"]
    safe = cases["crash_after_checkpoint"]
    unsafe = cases["crash_before_tool_checkpoint"]
    assert control["effects"] == safe["effects"] == ["alpha", "beta"]
    assert unsafe["effects"] == ["alpha", "alpha", "beta"]
    assert safe["duplicate_effects"] == 0
    assert unsafe["duplicate_effects"] == 1
    assert safe["resume_checkpoint_tool_results"] == ["alpha"]
    assert unsafe["resume_checkpoint_tool_results"] == []
    for case in (safe, unsafe):
        assert case["first_process_exit"] != 0
        assert case["first_pid"] != case["resume_pid"]
        assert case["resume_execution_ms"] >= 0
        assert case["checkpoint_file_bytes"] > 0
    assert all(case["final_answer"] == "alpha|beta" for case in cases.values())


@pytest.mark.parametrize("repeat,timeout", [(0, 30), (11, 30), (1, 0), (1, float("nan")), (1, float("inf")), (1, True), (1, 301)])
def test_invalid_controls_are_rejected_before_spawning(monkeypatch, repeat, timeout):
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid controls must not spawn a worker")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    with pytest.raises(ValueError):
        run_suite(repeat=repeat, timeout_seconds=timeout)


def test_timeout_reaps_the_actual_child(monkeypatch, tmp_path):
    import scripts.benchmark.checkpoint_recovery as benchmark

    children = []
    original = subprocess.Popen

    def capture(*args, **kwargs):
        child = original(*args, **kwargs)
        children.append(child)
        return child

    def expired(*args, **kwargs):
        raise TimeoutError("controlled ready deadline")

    monkeypatch.setattr(subprocess, "Popen", capture)
    monkeypatch.setattr(benchmark, "_wait_ready", expired)
    result = benchmark._run_case("crash_after_checkpoint", tmp_path, timeout_seconds=30)
    assert result["expected_observation_match"] is False
    assert result["error"] == "TimeoutError"
    assert result["failed_phase"] == "initial"
    assert len(children) == 1
    assert children[0].poll() is not None


def test_worker_environment_excludes_credentials_and_config_selectors(monkeypatch, tmp_path):
    import scripts.benchmark.checkpoint_recovery as benchmark

    for name in ("GITHUB_TOKEN", "OPENAI_API_KEY", "DEER_FLOW_CONFIG_PATH", "PYTHONHOME"):
        monkeypatch.setenv(name, "must-not-reach-worker")
    env = benchmark._worker_environment(tmp_path)
    assert all(name not in env for name in ("GITHUB_TOKEN", "OPENAI_API_KEY", "DEER_FLOW_CONFIG_PATH", "PYTHONHOME"))
    assert env["DEER_FLOW_HOME"] == str(tmp_path / "home")
    assert env["PYTHONNOUSERSITE"] == "1"


def test_cli_exports_the_same_report_and_refuses_to_overwrite(monkeypatch, tmp_path, capsys):
    import scripts.benchmark.checkpoint_recovery.__main__ as cli

    report = {"summary": {"matching_observations": 0, "cases": 1}}
    monkeypatch.setattr(cli, "run_suite", lambda **kwargs: report)
    output = tmp_path / "report.json"
    assert main(["--output", str(output)]) == 1
    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert json.loads(capsys.readouterr().out) == report
    with pytest.raises(SystemExit):
        main(["--output", str(output)])
    assert json.loads(output.read_text(encoding="utf-8")) == report
