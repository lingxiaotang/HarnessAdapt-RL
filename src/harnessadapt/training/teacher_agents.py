"""Non-interactive teacher agents. CLI executables are supplied by the operator.

Codex uses the Responses protocol, so HARNESSADAPT_HARNESS_TEACHER_URL must
support it. No CLI installation or download is performed by this module.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile


def agent_executable(backend: str) -> str:
    if backend not in {"codex", "opencode"}:
        raise ValueError(f"Unsupported teacher agent backend: {backend!r}")
    variable = f"HARNESSADAPT_{backend.upper()}_BIN"
    executable = shutil.which(os.environ.get(variable, backend))
    if not executable:
        raise RuntimeError(
            f"Teacher requires an externally installed {backend} CLI; "
            f"set {variable} to its executable path. No HTTP fallback is available."
        )
    return str(Path(executable).resolve())


def _run(command: list[str], prompt: str, *, cwd: Path, env: dict[str, str],
         run_dir: Path, timeout: int) -> list[dict]:
    """Retain agent events and diagnostics, including on failure or timeout."""
    events_path = run_dir / "events.jsonl"
    stderr_path = run_dir / "stderr.log"
    with events_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(
            command, cwd=cwd, env=env, stdin=subprocess.PIPE, stdout=stdout,
            stderr=stderr, text=True, encoding="utf-8", start_new_session=True,
        )
        try:
            process.communicate(prompt, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            # Also stop shell commands spawned by the agent.
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
            raise TimeoutError(f"Teacher agent exceeded {timeout}s; see {stderr_path}") from exc
    if process.returncode:
        raise RuntimeError(f"Teacher agent exited with {process.returncode}; see {stderr_path}")
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if any(event.get("type") in {"error", "turn.failed"} for event in events):
        raise RuntimeError(f"Teacher agent reported failure; see {events_path}")
    return events


def call_teacher_agent(prompt: str, *, backend: str, model: str, base_url: str,
                       api_key: str, run_dir: Path, workspace: Path | None = None,
                       timeout: int = 600) -> str:
    """Run one fresh agent session; return its final answer, not tool output.

    OpenCode is restricted to inline skill analysis. Codex can inspect, edit,
    and test the supplied candidate workspace through its normal agent loop.
    """
    executable = agent_executable(backend)
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "prompt.md").write_text(prompt, encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="harnessadapt-teacher-") as temporary:
        temporary = Path(temporary)
        cwd = workspace.resolve() if workspace is not None else temporary / "workspace"
        cwd.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        if backend == "codex":
            answer_path = run_dir / "last_message.md"
            answer_path.unlink(missing_ok=True)
            command = [
                executable, "--ask-for-approval", "never", "exec",
                "--model", model, "--sandbox", "workspace-write" if workspace else "read-only",
                "--skip-git-repo-check", "--json", "--output-last-message", str(answer_path),
            ]
            if base_url:
                command += [
                    "-c", 'model_provider="harnessadapt"',
                    "-c", 'model_providers.harnessadapt.name="HarnessAdapt teacher"',
                    "-c", f"model_providers.harnessadapt.base_url={json.dumps(base_url.rstrip('/'))}",
                    "-c", 'model_providers.harnessadapt.wire_api="responses"',
                    "-c", 'model_providers.harnessadapt.env_key="HARNESSADAPT_AGENT_API_KEY"',
                ]
                env["HARNESSADAPT_AGENT_API_KEY"] = api_key
            elif api_key:
                env["CODEX_API_KEY"] = api_key
            command.append("-")
            _run(command, prompt, cwd=cwd, env=env, run_dir=run_dir, timeout=timeout)
            answer = answer_path.read_text(encoding="utf-8") if answer_path.is_file() else ""
        else:
            if workspace is not None:
                raise ValueError("OpenCode skill teacher must not receive a harness workspace")
            # Use a dedicated primary agent with every tool denied, including MCP tools.
            env["XDG_CONFIG_HOME"] = str(temporary / "config")
            env["OPENCODE_CONFIG_DIR"] = str(temporary / "config" / "opencode")
            env.pop("OPENCODE_CONFIG", None)
            env["OPENCODE_DISABLE_AUTOUPDATE"] = "true"
            env["HARNESSADAPT_AGENT_API_KEY"] = api_key
            env["OPENCODE_CONFIG_CONTENT"] = json.dumps({
                "$schema": "https://opencode.ai/config.json",
                "autoupdate": False,
                "provider": {"harnessadapt": {
                    "npm": "@ai-sdk/openai-compatible", "name": "HarnessAdapt teacher",
                    "options": {"baseURL": base_url, "apiKey": "{env:HARNESSADAPT_AGENT_API_KEY}"},
                    "models": {model: {"name": model}},
                }},
                "agent": {"teacher": {
                    "description": "Analyze only the supplied historical trajectories",
                    "mode": "primary", "permission": {"*": "deny"}, "temperature": 0.2,
                }},
            })
            command = [executable, "run", "--format", "json", "--agent", "teacher",
                       "--model", f"harnessadapt/{model}"]
            events = _run(command, prompt, cwd=cwd, env=env, run_dir=run_dir, timeout=timeout)
            answer = "\n".join(event.get("part", {}).get("text", "") for event in events if event.get("type") == "text")
        if not answer.strip():
            raise RuntimeError(f"Teacher agent returned no final answer; see {run_dir}")
        return answer.strip()
