# Offline tool-result evidence checks

From `backend/`, inspect one completed run's local JSONL event file:

```bash
uv run python -m scripts.benchmark.run_tool_evidence \
  .deer-flow/threads/THREAD/runs/RUN.jsonl \
  --require-tool web_search --require-tool web_fetch
```

Use the file from the configured `JsonlRunEventStore` base directory, or export
**all** pages of one run's `RunEventStore.list_events()` records as physical
UTF-8 JSON lines. The input is local and read-only. No models, credentials,
servers, datasets, downloads or shell commands are invoked by the checker.

The deterministic assertion is deliberately narrow: the final recorded Run
status is `success`, and every named tool has at least one lead-agent call with
a matching `ToolMessage(status="success")`. A plausible final answer, a tool
request without a result, a child-agent call, or `run.end` cannot satisfy it.
Failed attempts remain counted when a later successful retry satisfies it.

Exit codes and JSON `verdict` values:

| Code | Verdict | Meaning |
| --- | --- | --- |
| 0 | `pass` | The recorded evidence satisfies the requested tool assertions. |
| 1 | `fail` | Well-formed bounded evidence lacks a required recorded success, or the final Run failed, timed out or was interrupted. |
| 2 | `unverifiable` | Missing, malformed or ambiguous evidence prevents a reliable assertion. |

Require a `run.start` boundary, one final protocol-v1 `run.terminal`, one Run
identity, and strictly increasing thread sequence numbers (gaps are valid).
Missing results, orphan/duplicate results, repeated provider call IDs, corrupt
JSON and incomplete terminal evidence fail closed. Results currently carry no
caller identity, so reused IDs are ambiguous even across different callers or
sequential calls. The checker rejects those traces rather than guessing.

Reports contain event counts, satisfied/unsatisfied requirements, a canonical
event-content SHA-256, at most 256 tool summaries, and up to 20 failure codes with
sequence positions. They omit message bodies, tool arguments, results and parser
error snippets. Input is bounded to 32 MiB
and 100,000 events; exceeding the tool summary budget is also unverifiable.
Keep raw traces private. Use the hash and sequence positions
to locate the same evidence locally when comparing model/runtime variants.

This is an evidence assertion, **not an overall Agent score**. Recorded success
does not prove task correctness, artifact delivery, external side effects, or
semantic success of tools that return error text with a success status (#3164).
The tool neither verifies an Evidence Seal nor authenticates an imported file;
absence of evidence is not proof that an external action never happened. Historical
traces lacking the required terminal protocol remain unverifiable. It does not
launch runs, implement replay, judge answer claims, or change runtime behavior.

Offline synthetic goldens and real Journal/JSONL serialization compatibility:

```bash
uv run pytest tests/test_run_tool_evidence.py -q
```
