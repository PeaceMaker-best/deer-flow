"""Offline assertions about recorded lead-agent tool results, not task correctness."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from deerflow.runtime.events.catalog import LLM_AI_RESPONSE_EVENT, LLM_TOOL_RESULT_EVENT, RUN_START_EVENT, RUN_TERMINAL_EVENT
from deerflow.runtime.events.terminal import TERMINAL_PROTOCOL_VERSION, TERMINAL_STATUSES

MAX_BYTES = 32 * 1024 * 1024
MAX_EVENTS = 100_000
MAX_ISSUES = 20
MAX_TOOLS = 256


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError("Non-finite JSON number")


def read_events(path: Path) -> list[dict]:
    """Bound reads and reject corrupt records rather than silently scoring a subset."""
    events, total = [], 0
    with path.open("rb") as stream:
        while line := stream.readline(MAX_BYTES + 1):
            total += len(line)
            if total > MAX_BYTES:
                raise ValueError("Trace exceeds byte limit")
            if not line.strip():
                raise ValueError("Blank trace record")
            if len(events) >= MAX_EVENTS:
                raise ValueError("Trace exceeds event limit")
            try:
                row = json.loads(line.decode("utf-8"), object_pairs_hook=_unique_keys, parse_constant=_reject_constant)
                if not isinstance(row, dict):
                    raise ValueError("Expected event object")
            except (ValueError, UnicodeError, RecursionError):
                # Neither provider payloads nor parser snippets belong in a report.
                raise ValueError(f"Invalid JSON event at line {len(events) + 1}") from None
            events.append(row)
    return events


def evaluate(events: list[dict], required_tools: tuple[str, ...]) -> dict:
    """Join canonical AI tool calls and ToolMessages within one persisted Run.

    Tool results have no caller scope in today's Journal. Repeated provider IDs
    are therefore ambiguous even if their callers differ or results are ordered.
    Never guess which execution produced a result.
    """
    issues, tools, calls, results = [], {}, {}, {}
    issue_count, uncertain, run_status = 0, False, None

    def issue(code, seq=None, *, unknown=True):
        nonlocal issue_count, uncertain
        issue_count += 1
        uncertain |= unknown
        if len(issues) < MAX_ISSUES:
            issues.append({"code": code, "seq": seq})

    identity, previous, terminals = None, 0, []
    for row in events:
        seq = row.get("seq")
        scope = (row.get("thread_id"), row.get("run_id"))
        if not all(isinstance(value, str) and value for value in scope) or (identity is not None and scope != identity):
            issue("mixed_or_missing_run_identity")
        identity = identity or scope
        if type(seq) is not int or seq <= previous:
            issue("invalid_sequence")
            continue
        previous = seq
        kind, content, metadata = row.get("event_type"), row.get("content"), row.get("metadata")
        if not isinstance(kind, str) or not kind:
            issue("invalid_event_type", seq)
            continue
        if kind == RUN_TERMINAL_EVENT.event_type:
            terminals.append(row)
        if kind not in {LLM_AI_RESPONSE_EVENT.event_type, LLM_TOOL_RESULT_EVENT.event_type}:
            continue
        if row.get("category") != "message" or not isinstance(content, dict):
            issue("invalid_message", seq)
            continue
        if kind == LLM_AI_RESPONSE_EVENT.event_type:
            tool_calls = content.get("tool_calls", [])
            if content.get("type") != "ai" or not isinstance(tool_calls, list) or content.get("invalid_tool_calls"):
                issue("invalid_ai_message", seq)
                continue
            caller = metadata.get("caller") if isinstance(metadata, dict) else None
            for call in tool_calls:
                call_id, name = (call.get("id"), call.get("name")) if isinstance(call, dict) else (None, None)
                if not all(isinstance(value, str) and 0 < len(value) <= 256 for value in (call_id, name, caller)):
                    issue("invalid_tool_call", seq)
                    continue
                if call_id in calls:
                    issue("ambiguous_tool_call_id", seq)
                calls.setdefault(call_id, []).append((name, caller, seq))
        else:
            call_id = content.get("tool_call_id")
            if content.get("type") != "tool" or not isinstance(call_id, str) or not call_id or not isinstance(content.get("status"), str) or content["status"] not in {"success", "error"}:
                issue("invalid_tool_result", seq)
                continue
            results.setdefault(call_id, []).append((content["status"], seq))

    if not events or events[0].get("event_type") != RUN_START_EVENT.event_type or events[0].get("category") != RUN_START_EVENT.category:
        issue("missing_run_start")
    if len(terminals) != 1 or terminals[0] is not events[-1]:
        issue("missing_or_nonfinal_terminal")
    else:
        terminal = terminals[0]
        content, metadata = terminal.get("content"), terminal.get("metadata")
        if (
            terminal.get("category") != RUN_TERMINAL_EVENT.category
            or not isinstance(content, dict)
            or set(content) != {"status"}
            or not isinstance(content["status"], str)
            or content["status"] not in TERMINAL_STATUSES
            or not isinstance(metadata, dict)
            or type(metadata.get("protocol_version")) is not int
            or metadata["protocol_version"] != TERMINAL_PROTOCOL_VERSION
            or type(metadata.get("recovered")) is not bool
        ):
            issue("invalid_terminal")
        else:
            run_status = content["status"]

    for call_id, entries in calls.items():
        outcomes = results.get(call_id, [])
        for name, caller, seq in entries:
            if caller != "lead_agent":
                continue
            if name not in tools and len(tools) >= MAX_TOOLS:
                issue("tool_limit_exceeded", seq)
                continue
            counts = tools.setdefault(name, dict.fromkeys(("calls", "success", "error", "missing"), 0))
            counts["calls"] += 1
            if len(entries) == 1 and len(outcomes) == 1 and outcomes[0][1] > seq:
                counts[outcomes[0][0]] += 1
            else:
                counts["missing"] += 1
                issue("missing_result" if not outcomes else "ambiguous_result", seq)
    for call_id, outcomes in results.items():
        if call_id not in calls or len(outcomes) != 1 or outcomes[0][1] <= calls[call_id][0][2]:
            issue("unmatched_or_duplicate_result", outcomes[0][1])
    required = sorted(set(required_tools))
    missing = [name for name in required if not tools.get(name, {}).get("success", 0)]
    if missing:
        issue("required_tool_not_successful", unknown=False)
    if run_status is not None and run_status != "success":
        issue("run_not_successful", unknown=False)
    return {
        "schema_version": 1,
        "assertion": "recorded-lead-tool-success",
        "events_sha256": hashlib.sha256(json.dumps(events, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest(),
        "verdict": "unverifiable" if uncertain else "fail" if missing or run_status != "success" else "pass",
        "run_status": run_status,
        "event_count": len(events),
        "required_tools": required,
        "unsatisfied_tools": missing,
        "tools": dict(sorted(tools.items())),
        "issues": issues,
        "omitted_issues": max(0, issue_count - MAX_ISSUES),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="One run's complete RunEventStore JSONL export")
    parser.add_argument("--require-tool", action="append", required=True, help="Require a recorded successful lead-agent result; repeat for multiple tools")
    args = parser.parse_args(argv)
    if any(not name.strip() or len(name) > 256 for name in args.require_tool):
        parser.error("Tool names must contain 1–256 characters")
    try:
        report = evaluate(read_events(args.trace), tuple(args.require_tool))
    except (OSError, ValueError, RecursionError):
        report = {"schema_version": 1, "verdict": "unverifiable", "issues": [{"code": "unreadable_or_invalid_trace"}]}
    print(json.dumps(report, indent=2))
    return {"pass": 0, "fail": 1, "unverifiable": 2}[report["verdict"]]


if __name__ == "__main__":
    raise SystemExit(main())
