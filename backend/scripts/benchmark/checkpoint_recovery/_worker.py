"""One disposable process in the checkpoint-recovery benchmark."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite import SqliteSaver

from deerflow.agents.factory import create_deerflow_agent
from deerflow.agents.features import RuntimeFeatures
from deerflow.runtime.checkpoint_mode import ensure_checkpoint_mode_compatible, inject_checkpoint_mode
from scripts.benchmark.checkpoint_recovery import OPERATIONS, PROMPT


class ScriptedModel(BaseChatModel):
    @property
    def _llm_type(self):
        return "checkpoint-recovery-script-v1"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        completed = [message.content for message in messages if isinstance(message, ToolMessage) and message.name == "record_effect"]
        if len(completed) < len(OPERATIONS):
            operation = OPERATIONS[len(completed)]
            answer = AIMessage(content="", tool_calls=[{"name": "record_effect", "args": {"operation": operation}, "id": f"call-{operation}"}])
        else:
            answer = AIMessage(content="|".join(completed))
        return ChatResult(generations=[ChatGeneration(message=answer)])


def _park(root: Path, boundary: str):
    temporary = root / "ready.tmp"
    temporary.write_text(json.dumps({"boundary": boundary}), encoding="utf-8")
    temporary.replace(root / "ready.json")
    threading.Event().wait()


def run_worker(root: Path, phase: str, boundary: str) -> None:
    @tool
    def record_effect(operation: str) -> str:
        """Append one operation to the synthetic durable side-effect ledger."""
        if operation not in OPERATIONS:
            raise ValueError("unknown synthetic operation")
        with (root / "effects.jsonl").open("a", encoding="utf-8", newline="\n") as ledger:
            ledger.write(json.dumps(operation) + "\n")
            ledger.flush()
            os.fsync(ledger.fileno())
        if phase == "initial" and boundary == "before" and operation == "alpha":
            _park(root, "before")
        return operation

    config = {"configurable": {"thread_id": "synthetic-recovery"}, "recursion_limit": 32}
    inject_checkpoint_mode(config, "full")
    with SqliteSaver.from_conn_string(str(root / "checkpoints.sqlite")) as saver:
        ensure_checkpoint_mode_compatible(saver, config, "full")
        agent = create_deerflow_agent(ScriptedModel(), [record_effect], features=RuntimeFeatures(sandbox=False), checkpointer=saver)
        before = agent.get_state(config)
        prior = [message.content for message in before.values.get("messages", []) if isinstance(message, ToolMessage) and message.name == "record_effect"]
        context = {"thread_id": "synthetic-recovery", "run_id": f"{phase}-{os.getpid()}", "user_id": "synthetic-benchmark"}
        started = time.perf_counter()
        state = agent.invoke(
            {"messages": [HumanMessage(content=PROMPT)]} if phase == "initial" else None,
            config,
            context=context,
            durability="sync",
            interrupt_after=["tools"] if phase == "initial" and boundary == "after" else None,
        )
        elapsed = (time.perf_counter() - started) * 1000
        if phase == "initial" and boundary == "after":
            snapshot = agent.get_state(config)
            if not snapshot.next or not any(isinstance(message, ToolMessage) and message.content == "alpha" for message in snapshot.values.get("messages", [])):
                raise RuntimeError("the real graph did not checkpoint the completed tool")
            _park(root, "after")
        answer = next((message.content for message in reversed(state["messages"]) if isinstance(message, AIMessage)), "")
        (root / f"{phase}.json").write_text(json.dumps({"final_answer": answer, "checkpoint_tool_results": prior, "execution_ms": round(elapsed, 3)}), encoding="utf-8")


if __name__ == "__main__":
    run_worker(Path(sys.argv[1]), sys.argv[2], sys.argv[3])
