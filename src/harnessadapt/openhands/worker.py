"""Standalone entrypoint executed with the official SDK inside a task container.

Only transport, telemetry and benchmark patch extraction are local code.
Agent, Conversation, prompts and all tools come from the official packages.
"""
from __future__ import annotations

import gzip
from importlib.metadata import version
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile


def git_patch(workspace: str, base_commit: str, initial_untracked=()) -> str:
    """Diff against the initial commit, including new files, preserving the real index."""
    with tempfile.TemporaryDirectory(prefix="openhands-patch-") as temporary:
        env = dict(os.environ, GIT_INDEX_FILE=temporary + "/index", GIT_CONFIG_NOSYSTEM="1")

        def git(*args):
            return subprocess.run(["git", "-c", "core.hooksPath=/dev/null", *args],
                                  cwd=workspace, env=env, check=True, capture_output=True,
                                  timeout=30).stdout

        git("read-tree", base_commit)
        # R2E adds untracked fixtures (e.g. r2e_tests) during container setup.
        # They belong to the benchmark, not to the agent's submission.
        exclusions = [":(top,exclude,literal)" + name for name in initial_untracked]
        git("add", "-A", "--", ".", *exclusions)
        patch = git("diff", "--cached", "--binary", "--no-ext-diff", "--no-textconv", base_commit, "--")
        if len(patch) > 8 * 1024 * 1024:
            raise ValueError("Submission exceeds 8 MiB")
        return patch.decode("utf-8")


def execute(request: dict, budget: int) -> dict:
    expected = request["version"]
    packages = {name: version(name) for name in ("openhands-sdk", "openhands-tools")}
    if any(value != expected for value in packages.values()):
        raise RuntimeError(f"Official package version mismatch: {packages}; expected {expected}")

    from openhands.sdk import LLM, Agent, Conversation, Tool
    from openhands.sdk.event import LLMConvertibleEvent
    from openhands.sdk.conversation.state import ConversationExecutionStatus
    from openhands.tools.file_editor import FileEditorTool
    from openhands.tools.task_tracker import TaskTrackerTool
    from openhands.tools.terminal import TerminalTool

    result = {"exit_status": "Error", "submission": "", "messages": [], "events": [],
              "n_calls": 0, "diagnostics": {}, "provenance": {
                  "backend": "openhands_sdk", "packages": packages,
                  "upstream": "https://github.com/OpenHands/software-agent-sdk/tree/v" + expected}}
    diagnostics = result["diagnostics"]
    error_codes = []

    def on_event(event):
        result["events"].append(event.model_dump(mode="json"))
        if isinstance(event, LLMConvertibleEvent):
            message = event.to_llm_message().model_dump(mode="json", exclude_none=True)
            # Display/analysis transcript only; training tokens come from Slime.
            content = message.get("content", [])
            if isinstance(content, list):
                message["content"] = "\n".join(str(part.get("text", "")) for part in content)
            result["messages"].append(message)
        if getattr(event, "code", None):
            error_codes.append(event.code)

    def on_completion(_filename: str, log_data: str):
        # Public telemetry hook. Discard request kwargs, which may hold secrets.
        data = json.loads(log_data)
        result["n_calls"] += 1
        response = data.get("response", {})
        usage = response.get("usage") or {}
        if "prompt_tokens" in usage:
            diagnostics["last_prompt_tokens"] = usage["prompt_tokens"]
            diagnostics["last_remaining_tokens"] = max(0, request["context_tokens"] - usage["prompt_tokens"])
        if any(choice.get("finish_reason") == "length" for choice in response.get("choices", [])):
            diagnostics["generation_length_exit"] = True
            conversation.pause()
        error_type = str((data.get("error") or {}).get("type", ""))
        if "Timeout" in error_type:
            diagnostics["model_request_timeout"] = True
        if "ContextWindow" in error_type:
            diagnostics["context_budget_exit"] = True

    base_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=request["workspace"],
                                 check=True, capture_output=True, text=True, timeout=30).stdout.strip()
    initial_untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=request["workspace"], check=True, capture_output=True, timeout=30,
    ).stdout.decode("utf-8").split("\0")
    initial_untracked = [name for name in initial_untracked if name]
    llm = LLM(
        model=request["model"], base_url=request["base_url"], api_key=request["api_key"],
        temperature=request["temperature"], top_p=request["top_p"],
        max_input_tokens=request["context_tokens"], max_output_tokens=request["max_output_tokens"],
        timeout=min(request["request_timeout"], budget), num_retries=0,
        api_mode="chat", stream=False, native_tool_calling=True, caching_prompt=False,
        reasoning_effort=None, log_completions=True,
    )
    # Upstream builds the system prompt and supplies ThinkTool/FinishTool.
    agent = Agent(llm=llm, condenser=None, tools=[
        Tool(name=TerminalTool.name, params={"terminal_type": "subprocess"}),
        Tool(name=FileEditorTool.name), Tool(name=TaskTrackerTool.name),
    ])
    conversation = Conversation(agent=agent, workspace=request["workspace"],
                                max_iteration_per_run=request["turns"],
                                callbacks=[on_event], visualizer=None)

    def timeout(_signum, _frame):
        raise TimeoutError("OpenHands agent wall time exceeded")

    previous_handler = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(budget)
    try:
        conversation.send_message(request["task"])
        # send_message lazily initializes the agent and its LLM registry. Attach
        # after that initialization so registry metric reset cannot drop us.
        conversation.agent.llm.telemetry.set_log_completions_callback(on_completion)
        conversation.run()
        result["sdk_status"] = conversation.state.execution_status.value
        if conversation.state.execution_status == ConversationExecutionStatus.FINISHED:
            result["submission"] = git_patch(request["workspace"], base_commit, initial_untracked)
            result["exit_status"] = "Submitted" if result["submission"].strip() else "EmptySubmission"
        elif conversation.state.execution_status == ConversationExecutionStatus.STUCK:
            result["exit_status"] = "Stuck"
        elif "MaxIterationsReached" in error_codes:
            result["exit_status"] = "LimitsExceeded"
        else:
            result["error"] = ", ".join(error_codes) or result["sdk_status"]
    except Exception as exc:
        # Conversation.run wraps provider errors; inspect their causes too.
        chain = []
        current = exc
        while current is not None and id(current) not in {id(item) for item in chain}:
            chain.append(current)
            current = current.__cause__ or current.__context__
        names = " ".join(type(item).__name__ for item in chain)
        if "Timeout" in names or diagnostics.get("model_request_timeout"):
            result["exit_status"] = "TimeExceeded"
        if "ContextWindow" in names:
            diagnostics["context_budget_exit"] = True
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
        try:
            conversation.close()
        except Exception as exc:
            result.update(exit_status="Error", error=f"SDK cleanup failed: {type(exc).__name__}: {exc}")
    if diagnostics.get("generation_length_exit"):
        result["exit_status"] = "GenerationLengthExceeded"
    elif diagnostics.get("context_budget_exit"):
        result["exit_status"] = "ContextBudgetExceeded"
    if result["exit_status"] != "Submitted":
        result["submission"] = ""
    diagnostics["last_history_assistant_turns"] = sum(m.get("role") == "assistant" for m in result["messages"])
    diagnostics["last_history_reasoning_turns"] = sum(bool(m.get("reasoning_content")) for m in result["messages"])
    return result


def main() -> None:
    source, target = Path(sys.argv[1]), Path(sys.argv[2])
    request = json.loads(source.read_text())
    source.unlink()
    os.environ["OH_PERSISTENCE_DIR"] = str(target.parent / "openhands")
    try:
        result = execute(request, int(sys.argv[3]))
    except Exception as exc:
        result = {"exit_status": "Error", "error": f"{type(exc).__name__}: {exc}"}
    if request["api_key"] and result.get("error"):
        result["error"] = result["error"].replace(request["api_key"], "[redacted]")
    # Do not replace strings inside the patch: placeholder keys such as 'local'
    # are common source identifiers. SDK models redact their own secret fields.
    serialized = json.dumps(result, ensure_ascii=False)
    with gzip.open(target, "wt", encoding="utf-8") as stream:
        stream.write(serialized)


if __name__ == "__main__":
    main()
