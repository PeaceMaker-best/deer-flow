"""Synthetic trajectory goldens and compatibility with the real Journal/store."""

import json
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from scripts.benchmark.run_tool_evidence import evaluate, main, read_events


def event(kind, content, **metadata):
    return {"thread_id": "t", "run_id": "r", "event_type": kind, "category": "trace" if kind in {"run.start", "run.terminal"} else "message", "content": content, "metadata": metadata}


def call(call_id="c", name="web_fetch", caller="lead_agent"):
    return event("llm.ai.response", AIMessage(content="", tool_calls=[{"id": call_id, "name": name, "args": {"private": "DO_NOT_EXPORT"}}]).model_dump(), caller=caller)


def result(call_id="c", status="success"):
    return event("llm.tool.result", ToolMessage(content="DO_NOT_EXPORT", tool_call_id=call_id, status=status).model_dump())


def terminal(status="success"):
    return event("run.terminal", {"status": status}, protocol_version=1, recovered=False)


def numbered(*events):
    return [dict(row, seq=index) for index, row in enumerate((event("run.start", {}), *events), 1)]


def check(*events, required=("web_fetch",)):
    return evaluate(numbered(*events), required)


def test_success_is_an_observed_tool_result_not_an_answer_claim():
    report = check(call(), result(), terminal())
    assert report["verdict"] == "pass"
    assert report["tools"]["web_fetch"] == {"calls": 1, "success": 1, "error": 0, "missing": 0}
    assert "DO_NOT_EXPORT" not in json.dumps(report)
    assert check(event("llm.ai.response", {"type": "ai", "content": "I fetched the page"}, caller="lead_agent"), terminal())["verdict"] == "fail"


@pytest.mark.parametrize("tail,verdict,code", [([terminal()], "unverifiable", "missing_result"), ([result(status="error"), terminal()], "fail", "required_tool_not_successful")])
def test_missing_and_explicit_error_are_distinct(tail, verdict, code):
    report = check(call(), *tail)
    assert report["verdict"] == verdict
    assert code in {issue["code"] for issue in report["issues"]}


def test_recovery_can_satisfy_a_required_tool_without_erasing_failed_attempts():
    report = check(call(), result(status="error"), call("retry"), result("retry"), terminal())
    assert report["verdict"] == "pass"
    assert report["tools"]["web_fetch"]["error"] == 1


@pytest.mark.parametrize("status", ["error", "interrupted", "timeout"])
def test_graph_end_does_not_override_final_run_outcome(status):
    report = check(call(), result(), event("run.end", {}, status="success"), terminal(status))
    assert report["verdict"] == "fail"
    assert report["run_status"] == status


@pytest.mark.parametrize("rows", [[], [call(), result()], [call(), result(), terminal(), terminal()], [call(), result(), terminal(), call("late")]])
def test_incomplete_or_conflicting_terminal_evidence_never_passes(rows):
    assert check(*rows)["verdict"] == "unverifiable"


def test_reused_ids_are_ambiguous_even_when_different_callers_share_a_name():
    assert check(call(), call(caller="subagent:researcher"), result(), terminal())["verdict"] == "unverifiable"
    assert check(call(), result(), call(), result(), terminal())["verdict"] == "unverifiable"


def test_subagent_calls_cannot_satisfy_a_lead_agent_requirement():
    assert check(call(caller="subagent:researcher"), result(), terminal())["verdict"] == "fail"


def test_each_required_tool_needs_its_own_result_and_hash_is_reproducible():
    rows = numbered(call(), result(), terminal())
    report = evaluate(rows, ("web_search", "web_fetch", "web_search"))
    assert report["verdict"] == "fail"
    assert report["unsatisfied_tools"] == ["web_search"]
    assert report["events_sha256"] == evaluate([dict(reversed(list(row.items()))) for row in rows], ("web_fetch",))["events_sha256"]


@pytest.mark.parametrize(
    "mutate",
    [lambda rows: rows[1].update(run_id="other"), lambda rows: rows[1].update(seq=1), lambda rows: rows[1].update(seq=True), lambda rows: rows[1].update(metadata={}), lambda rows: rows[-1]["metadata"].update(protocol_version=True)],
)
def test_malformed_scope_order_or_protocol_fails_closed(mutate):
    rows = numbered(call(), result(), terminal())
    mutate(rows)
    assert evaluate(rows, ("web_fetch",))["verdict"] == "unverifiable"


def test_orphan_result_cannot_be_used_as_success_evidence():
    assert check(result(), terminal())["verdict"] == "unverifiable"


@pytest.mark.parametrize("field,value", [("event_type", []), ("content", []), ("content", {"type": "tool", "tool_call_id": "c", "status": []})])
def test_malformed_records_return_a_verdict_instead_of_crashing(field, value):
    rows = numbered(call(), result(), terminal())
    rows[2][field] = value
    assert evaluate(rows, ("web_fetch",))["verdict"] == "unverifiable"


def test_result_order_start_boundary_and_thread_scoped_sequence_gaps():
    assert check(result(), call(), terminal())["verdict"] == "unverifiable"
    rows = numbered(call(), result(), terminal())
    assert evaluate(rows[1:], ("web_fetch",))["verdict"] == "unverifiable"
    for row in rows:
        row["seq"] *= 10  # Other runs in the same thread can consume sequence numbers.
    assert evaluate(rows, ("web_fetch",))["verdict"] == "pass"


def test_duplicate_result_and_issue_output_bound():
    assert check(call(), result(), result(), terminal())["verdict"] == "unverifiable"
    report = check(*(result(str(index)) for index in range(40)), terminal())
    assert len(report["issues"]) == 20
    assert report["omitted_issues"] == 21


def test_tool_summary_bound_is_explicitly_unverifiable(monkeypatch):
    import scripts.benchmark.run_tool_evidence as module

    monkeypatch.setattr(module, "MAX_TOOLS", 1)
    report = check(call(), result(), call("other", name="web_search"), result("other"), terminal())
    assert report["verdict"] == "unverifiable"
    assert len(report["tools"]) == 1


@pytest.mark.parametrize("raw", [b'{"seq":1,"seq":2}', b'{"seq":NaN}', b"[]"])
def test_reader_rejects_ambiguous_json(tmp_path, raw):
    path = tmp_path / "events.jsonl"
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="line 1"):
        read_events(path)


def test_reader_uses_physical_lines_and_rejects_corrupt_tails(tmp_path):
    path = tmp_path / "events.jsonl"
    rows = numbered(call(), result(), terminal())
    rows[2]["content"]["content"] = "valid\u2028Unicode\u2029text"
    path.write_bytes(b"".join(json.dumps(row, ensure_ascii=False).encode() + b"\n" for row in rows))
    assert read_events(path) == rows
    with path.open("ab") as stream:
        stream.write(b'{"private":"DO_NOT_EXPORT')
    with pytest.raises(ValueError, match="line 5") as error:
        read_events(path)
    assert "DO_NOT_EXPORT" not in str(error.value)


def test_reader_bounds_bytes_and_records(tmp_path, monkeypatch):
    import scripts.benchmark.run_tool_evidence as module

    path = tmp_path / "events.jsonl"
    path.write_text("{}\n{}\n", encoding="utf-8")
    monkeypatch.setattr(module, "MAX_EVENTS", 1)
    with pytest.raises(ValueError, match="event limit"):
        read_events(path)
    monkeypatch.setattr(module, "MAX_BYTES", 2)
    with pytest.raises(ValueError, match="byte limit"):
        read_events(path)


def test_cli_json_and_exit_codes(tmp_path, capsys):
    path = tmp_path / "events.jsonl"
    for rows, code in [(numbered(call(), result(), terminal()), 0), (numbered(terminal()), 1), (numbered(call(), terminal()), 2)]:
        path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
        assert main([str(path), "--require-tool", "web_fetch"]) == code
        assert json.loads(capsys.readouterr().out)["schema_version"] == 1
    path.write_bytes(b"\xff")
    assert main([str(path), "--require-tool", "web_fetch"]) == 2
    assert json.loads(capsys.readouterr().out)["verdict"] == "unverifiable"


@pytest.mark.asyncio
async def test_real_journal_and_jsonl_store_produce_accepted_evidence(tmp_path):
    from deerflow.runtime.events.store.jsonl import JsonlRunEventStore
    from deerflow.runtime.events.terminal import confirm_terminal_event
    from deerflow.runtime.journal import RunJournal

    store = JsonlRunEventStore(tmp_path)
    journal = RunJournal("r", "t", store)
    root_id, llm_id = uuid4(), uuid4()
    journal.on_chain_start({"name": "LangGraph"}, {}, run_id=root_id)
    journal.on_chat_model_start({}, [[]], run_id=llm_id, tags=["lead_agent"])
    message = AIMessage(content="", tool_calls=[{"id": "c", "name": "web_fetch", "args": {}}])
    journal.on_llm_end(LLMResult(generations=[[ChatGeneration(message=message)]]), run_id=llm_id, tags=["lead_agent"])
    journal.on_tool_end(ToolMessage(content="page", tool_call_id="c"), run_id=uuid4())
    await journal.close()
    await confirm_terminal_event(store, thread_id="t", run_id="r", status="success", user_id=None, recovered=False, authorized=lambda: True)
    rows = read_events(tmp_path / "threads/t/runs/r.jsonl")
    assert evaluate(rows, ("web_fetch",))["verdict"] == "pass"
