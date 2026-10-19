"""Run a policy with H1, H2, or the official OpenHands SDK on a task image.

Inference pairs the trained policy with H2 (or H1) without task-specific training
skills or teacher calls. Accompanying general harness instructions, tools, and
runtime hooks come from the selected harness artifact.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import time
import types
from pathlib import Path

from . import policy, runtime
from .model_utils import HEAD_TAIL_OBSERVATION_TEMPLATE, _normalize_tool_arguments_for_template

ROOT = Path(__file__).resolve().parents[2]


class DockerRuntime:
    """Small bridge to the command-policy environment retained from the runner."""

    def __init__(self, container):
        self.container = container

    def run(self, code, *, timeout=45, workdir="/testbed"):
        command = f"timeout {int(timeout)} {code}"
        result = self.container.exec_run(["/bin/sh", "-lc", command], workdir=workdir)
        output = result.output.decode("utf-8", errors="replace")
        status = "0" if result.exit_code == 0 else f"Error: Exit code {result.exit_code}"
        return output, status


class ContextModel:
    """Cap generation against the packaged tokenizer and chat template."""

    def __init__(self, model, tokenizer, context_length, request_timeout, deadline):
        self._model = model
        self._tokenizer = tokenizer
        self.model_context_length = context_length
        self.safety_margin = 256
        self.request_timeout = request_timeout
        self.deadline = deadline

    def __getattr__(self, name):
        return getattr(self._model, name)

    def remaining_context(self, messages, tools):
        prepared = self._model._prepare_messages_for_api(messages)
        prepared = _normalize_tool_arguments_for_template(prepared)
        ids = self._tokenizer.apply_chat_template(
            prepared, tools=tools, tokenize=True, add_generation_prompt=True,
        )
        return max(0, self.model_context_length - self.safety_margin - len(ids))

    def query(self, messages, **kwargs):
        from minisweagent.exceptions import LimitsExceeded, TimeExceeded

        if time.monotonic() >= self.deadline:
            raise TimeExceeded({"role": "exit", "content": "TimeExceeded",
                                "extra": {"exit_status": "TimeExceeded", "submission": ""}})
        remaining = self.remaining_context(messages, self._evolved_harness.schemas())
        if remaining < 256:
            raise LimitsExceeded({"role": "exit", "content": "ContextBudgetExceeded",
                                  "extra": {"exit_status": "ContextBudgetExceeded", "submission": ""}})
        kwargs.update(max_tokens=remaining, timeout=max(1, min(self.request_timeout,
                       self.deadline - time.monotonic())))
        message = self._model.query(messages, **kwargs)
        response = message.get("extra", {}).get("response", {})
        choices = response.get("choices", []) if isinstance(response, dict) else []
        if choices and choices[0].get("finish_reason") == "length":
            raise LimitsExceeded({"role": "exit", "content": "GenerationLengthExceeded",
                                  "extra": {"exit_status": "GenerationLengthExceeded", "submission": ""}})
        return message


def make_base(args, tokenizer):
    from minisweagent.agents.default import DefaultAgent
    from minisweagent.models.litellm_model import LitellmModel

    base = types.ModuleType("publication_inference_base")

    def case_model(*unused, **kwargs):
        inner = LitellmModel(
            model_name="openai/" + args.model.removeprefix("openai/"),
            model_kwargs={"api_base": args.base_url.rstrip("/"),
                          "api_key": os.environ.get("HARNESSADAPT_API_KEY", "local"),
                          "temperature": args.temperature, "top_p": 1.0,
                          "num_retries": 0},
            cost_tracking="ignore_errors",
            observation_template=HEAD_TAIL_OBSERVATION_TEMPLATE,
        )
        return ContextModel(inner, tokenizer, args.context_length, args.request_timeout,
                            time.monotonic() + args.wall_time)

    base._case_model = case_model
    base._agent_class = lambda: DefaultAgent
    base._run_case = lambda **kwargs: None  # Training wrapper is unused here.
    base._install_format_error_history_recovery = lambda model: model
    base._normalize_tool_arguments_for_template = _normalize_tool_arguments_for_template
    return base


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harness", type=str.lower, choices=["h1", "h2", "openhands"], default="h2")
    parser.add_argument("--benchmark", type=str.lower, choices=["verified", "multilingual", "swepro"], default="verified",
                        help="Benchmark environment convention (swepro links /app -> /testbed and disables network)")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models/update45")
    parser.add_argument("--base-url", default="http://127.0.0.1:31980/v1")
    parser.add_argument("--model", default="harnessadapt-update45")
    parser.add_argument("--task-image", required=True,
                        help="Prepared image with the task's starting repository at /testbed")
    parser.add_argument("--issue-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--context-length", type=int, default=131072)
    parser.add_argument("--turns", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--step-timeout", type=int, default=45)
    parser.add_argument("--request-timeout", type=int, default=600)
    parser.add_argument("--wall-time", type=int, default=5400)
    parser.add_argument("--container-memory", default="8g")
    parser.add_argument("--openhands-runtime", type=Path,
                        help="Exported official SDK runtime; defaults to HARNESSADAPT_OPENHANDS_RUNTIME")
    parser.add_argument("--max-output-tokens", type=int, default=8192,
                        help="Official SDK per-request output cap (inference only)")
    args = parser.parse_args(argv)
    for name in ("context_length", "turns", "step_timeout", "request_timeout", "wall_time", "max_output_tokens"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not args.temperature >= 0:
        parser.error("--temperature must be nonnegative")
    issue = args.issue_file.read_text(encoding="utf-8").strip()
    if not issue:
        parser.error("--issue-file must contain the task description")
    official = args.harness == "openhands"
    container_options = {}
    if official:
        from .openhands import provenance
        from .openhands.bridge import ModelConnection, OfficialAgent, docker_kwargs, runtime_path

        if args.openhands_runtime:
            os.environ["HARNESSADAPT_OPENHANDS_RUNTIME"] = str(args.openhands_runtime)
        container_options = docker_kwargs()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    # No existing outputs are overwritten. Do not serialize API credentials.
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(task_skills=False, teachers=False, grader=False)
    if official:
        config.update(provenance=provenance(),
                      runtime_manifest=json.loads((runtime_path() / "manifest.json").read_text()))
    (args.output_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")

    import docker
    if not official:
        from transformers import Qwen2TokenizerFast

        tokenizer = Qwen2TokenizerFast.from_pretrained(str(args.checkpoint), local_files_only=True)
        base = make_base(args, tokenizer)
        runtime._install(base, ROOT / "harnesses" / args.harness)
    client = docker.from_env()
    container = None
    result = {"submitted": False, "exit_status": "Error", "graded": False}
    agent = None
    try:
        # Use a local task image; no benchmark data or images are bundled.
        if args.benchmark == "swepro" and not official:
            container_options.setdefault("network_mode", "none")
        container = client.containers.run(
            args.task_image, entrypoint=["/bin/sh", "-c"], command="exec sleep infinity",
            detach=True, working_dir="/" if args.benchmark == "swepro" else "/testbed",
            mem_limit=args.container_memory,
            pids_limit=512, labels={"harnessadapt.publication_task": "true"},
            **container_options,
        )
        if args.benchmark == "swepro":
            setup = container.exec_run(
                [
                    "/bin/sh",
                    "-lc",
                    "set -eu; "
                    "if [ -d /app ] && git -C /app rev-parse --is-inside-work-tree >/dev/null 2>&1; then "
                    "  if [ ! -e /testbed ]; then ln -s /app /testbed; fi; "
                    "  [ \"$(readlink -f /testbed)\" = /app ] || exit 42; "
                    "elif [ -d /testbed ] && git -C /testbed rev-parse --is-inside-work-tree >/dev/null 2>&1; then :; "
                    "else exit 43; fi",
                ],
                workdir="/",
            )
            if setup.exit_code != 0:
                raise RuntimeError(f"SWE-bench Pro /app -> /testbed setup failed (exit {setup.exit_code})")
        environment = policy._make_environment_cls()(
            DockerRuntime(container), timeout=args.step_timeout,
            hard_step_timeout=args.step_timeout + 30,
        )
        if official:
            model = ModelConnection(
                base_url=args.base_url.rstrip("/"),
                api_key=os.environ.get("HARNESSADAPT_API_KEY", "local"),
                model="openai/" + args.model.removeprefix("openai/"),
                temperature=args.temperature, context_tokens=args.context_length,
                max_output_tokens=args.max_output_tokens, request_timeout=args.request_timeout,
                deadline=time.monotonic() + args.wall_time,
            )
            agent_class = OfficialAgent
        else:
            model = base._case_model(tokenizer=tokenizer)
            agent_class = base._agent_class()
        agent = agent_class(
            model, environment, system_template=policy.SYSTEM_TEMPLATE,
            step_limit=args.turns, cost_limit=0.0,
            wall_time_limit_seconds=args.wall_time, max_consecutive_format_errors=3,
            output_path=None,
        )
        exit_info = agent.run(issue) or {}
        patch = str(exit_info.get("submission") or "")
        submitted = exit_info.get("exit_status") == "Submitted" and bool(patch.strip())
        result.update(submitted=submitted, exit_status=exit_info.get("exit_status", "Unknown"),
                      turns=agent.n_calls)
        if official:
            result.update(provenance=agent.backend_provenance, sdk_status=exit_info.get("sdk_status"))
        else:
            result.update(agent._evolved_harness.diagnostics())
        if submitted:
            (args.output_dir / "submission.patch").write_text(patch, encoding="utf-8")
    except Exception as exc:
        # Only aggregate status is saved; no conversation or raw trajectory.
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if container is not None:
            try:
                container.remove(force=True)
            except Exception as exc:
                result["cleanup_error"] = f"{type(exc).__name__}: {exc}"
        client.close()
        (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["submitted"] and not result.get("cleanup_error") else 1


if __name__ == "__main__":
    sys.exit(main())
