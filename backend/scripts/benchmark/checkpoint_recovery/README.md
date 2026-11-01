# Offline SDK checkpoint recovery observations

This matrix evaluates one reliability dimension raised in
[#3351](https://github.com/bytedance/deer-flow/issues/3351): whether a checkpointed
Agent continues correctly in a fresh process, and whether its external effects
are repeated. It uses the public `create_deerflow_agent` factory, its native
middleware chain with `RuntimeFeatures(sandbox=False)`, the real ToolNode and
`SqliteSaver`. Only the chat model and the two harmless fixture operations are
scripted. No provider, model credentials, sandbox, Gateway or external dataset
is needed.

From `backend/`, in the locked environment:

```bash
uv run --locked python -m scripts.benchmark.checkpoint_recovery --repeat 3 --output /tmp/recovery-report.json
uv run --locked pytest tests/test_bench_checkpoint_recovery.py -q
```

The output path must be new. Without `--output`, the JSON is printed to stdout.
`--timeout-seconds` is positive and at most 300, bounding each disposable worker; `--repeat` accepts 1–10.
Workers use a small environment allowlist, isolated home directories and unique
temporary databases. Every started child is killed/reaped on failure before
temporary storage is removed. Reports contain synthetic observations and source
provenance; private worker logs and databases are temporary.

## Three controls

The fixture asks for two file-backed operations, `alpha` then `beta`. Each real
tool execution appends a JSON line to a ledger and flushes it with `fsync`.
The ledger is independent of the model's final answer and checkpoint messages.
It deliberately has no idempotency key or deduplication.

| Scenario | Fault boundary | Expected ledger |
| --- | --- | --- |
| `uninterrupted` | No process fault | `alpha, beta` |
| `crash_after_checkpoint` | Native interrupt after the tools node; verify the saved tool result and pending work, then kill the process | `alpha, beta` |
| `crash_before_tool_checkpoint` | First tool has flushed its effect, but has not returned its result; kill the process inside that tool | `alpha, alpha, beta` |

Each faulted case starts a **new Python process**, rebuilds the same SDK Agent,
opens the existing SQLite database and invokes the graph with `None`. No test
restores messages by hand. The scripted model selects its next operation from
the actual ToolMessages it receives; losing the checkpoint therefore cannot
silently pass the ledger oracle.

The third control demonstrates the boundary between state recovery and
exactly-once external actions: a tool can produce an effect before LangGraph
can save its returned ToolMessage. Resumption can replay that action. Real
side-effecting tools need their own idempotency/reconciliation contract.

## Read the report

- `effects`, `tool_executions` and `duplicate_effects` come from the real ledger.
- `resume_checkpoint_tool_results` comes from the new Agent's checkpoint read,
  before resumption: `['alpha']` after the committed boundary, `[]` before it.
- `first_process_exit`, `first_pid` and `resume_pid` expose the process fault
  and fresh-worker boundary.
- `final_answer` is collected from the graph's resulting AIMessage.
- `resume_execution_ms` measures resumed graph execution;
  `resume_process_wall_ms` also includes process startup and imports. Timings
  are informational, with no pass/fail performance threshold.
- `checkpoint_file_bytes` counts the disposable SQLite files after execution.
- `expected_observation_match` checks the scenario's **declared observation**.
  The expected duplicate in the unsafe window matches that control; it is not
  a claim that the application achieved exactly-once execution.

The report pins fixture/model-rule version and SHA-256, control parameters,
source hash, Git revision/dirty status, Python, OS, SQLite and package versions.
If Git metadata is unavailable, revision/dirty are `null`; the source hash and
package versions remain available. Exit status is nonzero if a case fails to
produce its expected observation or a worker fails/times out.

## Scope

This is a synthetic reliability probe of the **full-checkpoint SDK factory**,
using explicit synchronous checkpoint durability for reproducible boundaries.
The factory rejects persisted delta mode, so this matrix does not bypass that
guard or compare delta behavior. It does not measure business-task success,
real-model reasoning, token costs, abrupt disk/power loss, Gateway takeover,
multi-replica coordination or protection against arbitrary mid-transaction
failures. Killing after a committed checkpoint is an intentionally controlled
boundary. The unsafe-window control exposes one concrete uncommitted-effect
case, rather than asserting a general recovery guarantee.
