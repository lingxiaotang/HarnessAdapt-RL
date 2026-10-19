"""Container transport and benchmark protocol for the official OpenHands agent.

The training process does not import the SDK: its Python 3.12 dependencies live
in an isolated runtime. The worker uses upstream Agent/Conversation and tools.
Slime's OpenAI adapter remains authoritative for training token IDs, log
probabilities and masks; no transcript is retokenized here for training.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass, field
import gzip
import json
import math
import os
from pathlib import Path
import secrets
import shlex
import time

from . import BACKEND, RUNTIME_TARGET, VERSION, provenance


def enabled() -> bool:
    backend = os.environ.get("HARNESSADAPT_AGENT_BACKEND", "miniswe")
    if backend not in ("miniswe", BACKEND):
        raise ValueError(f"Unsupported agent backend: {backend}")
    return backend == BACKEND


def runtime_path() -> Path:
    value = os.environ.get("HARNESSADAPT_OPENHANDS_RUNTIME", "")
    if not value:
        raise ValueError("Set HARNESSADAPT_OPENHANDS_RUNTIME to the exported official SDK runtime")
    path = Path(value).expanduser().resolve(strict=True)
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("packages") != provenance()["packages"] or manifest.get("backend") != BACKEND:
        raise ValueError(f"Expected official OpenHands SDK and tools {VERSION}; rebuild the runtime")
    # The interpreter symlink resolves only at the fixed container mount point.
    if not os.path.lexists(path / "venv/bin/python"):
        raise FileNotFoundError(f"Incomplete OpenHands runtime: {path}")
    return path


def docker_kwargs() -> dict:
    return {"network_mode": "host", "volumes": {
        str(runtime_path()): {"bind": RUNTIME_TARGET, "mode": "ro"},
    }}


def docker_run_args() -> list[str]:
    return ["--network", "host", "--volume", f"{runtime_path()}:{RUNTIME_TARGET}:ro"]


@dataclass
class ModelConnection:
    """Connection settings plus diagnostics consumed by the existing grader."""

    base_url: str
    api_key: str = field(repr=False)
    model: str = "openai/slime-actor"
    temperature: float = 1.0
    top_p: float = 1.0
    context_tokens: int = 65536
    max_output_tokens: int = 8192
    request_timeout: int = 1200
    deadline: float = math.inf
    last_prompt_tokens: int | None = None
    last_remaining_tokens: int | None = None
    last_history_assistant_turns: int = 0
    last_history_reasoning_turns: int = 0
    context_budget_exit: bool = False
    generation_length_exit: bool = False
    model_request_timeout: bool = False


class OfficialAgent:
    """Runner facade only; agent decisions and tools execute in upstream SDK."""

    def __init__(self, model, environment, *, step_limit=60,
                 wall_time_limit_seconds=1400, **_mini_swe_options):
        self.model = model
        self.environment = environment
        self.step_limit = step_limit
        self.wall_time = wall_time_limit_seconds
        self.messages = []
        self.events = []
        self.n_calls = 0
        self.backend_provenance = provenance()

    def _exec(self, command: str, *, timeout=30) -> str:
        result = self.environment.execute({"command": command}, cwd="/testbed", timeout=timeout)
        if result.get("returncode") != 0 or result.get("exception_info"):
            # Never echo the command: uploads contain the session credential.
            if result.get("returncode") in (124, 137):
                raise TimeoutError("Official OpenHands worker exceeded its deadline")
            raise RuntimeError("OpenHands container transport failed (exit "
                               + str(result.get("returncode")) + ")")
        return str(result.get("output", ""))

    def _upload(self, target: str, data: bytes) -> None:
        encoded = base64.b64encode(data).decode("ascii")
        temporary = shlex.quote(target + ".b64")
        self._exec(f"umask 077; : > {temporary}")
        for offset in range(0, len(encoded), 16000):
            self._exec("printf %s " + shlex.quote(encoded[offset:offset + 16000]) + " >> " + temporary)
        self._exec(f"umask 077; base64 -d {temporary} > {shlex.quote(target)} && rm {temporary}")

    def run(self, task: str) -> dict:
        directory = "/tmp/harnessadapt-oh-" + secrets.token_hex(12)
        self._exec("umask 077; mkdir " + shlex.quote(directory))
        deadline = min(self.model.deadline, time.monotonic() + self.wall_time)
        try:
            request = {
                "version": VERSION, "task": task, "workspace": "/testbed",
                "base_url": self.model.base_url, "api_key": self.model.api_key,
                "model": self.model.model, "temperature": self.model.temperature,
                "top_p": self.model.top_p, "context_tokens": self.model.context_tokens,
                "max_output_tokens": self.model.max_output_tokens,
                "request_timeout": self.model.request_timeout, "turns": self.step_limit,
            }
            self._upload(directory + "/worker.py", Path(__file__).with_name("worker.py").read_bytes())
            self._upload(directory + "/request.json", json.dumps(request).encode())
            remaining = math.floor(deadline - time.monotonic())
            if remaining <= 0:
                raise TimeoutError("OpenHands deadline expired during worker setup")
            command = shlex.join([
                RUNTIME_TARGET + "/venv/bin/python", "-I", "-B", directory + "/worker.py",
                directory + "/request.json", directory + "/result.json.gz", str(remaining),
            ])
            self._exec(command + " > " + shlex.quote(directory + "/worker.log") + " 2>&1",
                       timeout=remaining + 10)
            # Chunked reads work with environments that cap command output.
            source = shlex.quote(directory + "/result.json.gz")
            size = int(self._exec(f"wc -c < {source}").strip())
            if not 0 < size <= 64 * 1024 * 1024:
                raise RuntimeError("Invalid OpenHands result size")
            chunks = []
            for offset in range(0, size, 8192):
                output = self._exec(f"dd if={source} bs=8192 skip={offset // 8192} count=1 2>/dev/null | base64")
                chunks.append(base64.b64decode(output))
            result = json.loads(gzip.decompress(b"".join(chunks)))
            return self._accept_result(result)
        finally:
            try:
                self._exec("rm -rf -- " + shlex.quote(directory))
            except Exception:
                pass  # The case runner also removes the task container.

    def _accept_result(self, result: dict) -> dict:
        if result.get("provenance") != provenance():
            raise RuntimeError("Worker did not report the pinned official OpenHands packages: "
                               + str(result.get("error", "missing provenance")))
        self.messages = result.get("messages", [])
        self.events = result.get("events", [])
        self.n_calls = int(result.get("n_calls", 0))
        for name in ("last_prompt_tokens", "last_remaining_tokens", "last_history_assistant_turns",
                     "last_history_reasoning_turns", "context_budget_exit", "generation_length_exit",
                     "model_request_timeout"):
            if name in result.get("diagnostics", {}):
                setattr(self.model, name, result["diagnostics"][name])
        status = result.get("exit_status", "Error")
        patch = str(result.get("submission", ""))
        if status == "Submitted" and (
            result.get("sdk_status") != "finished" or not patch.strip()
            or self.model.generation_length_exit or self.model.context_budget_exit
            or self.model.model_request_timeout
        ):
            raise RuntimeError("Invalid official OpenHands submission")
        if status == "Error":
            raise RuntimeError("Official OpenHands failed: " + str(result.get("error", "unknown SDK error")))
        return {"exit_status": status, "submission": patch if status == "Submitted" else "",
                "backend": BACKEND, "sdk_status": result.get("sdk_status"),
                "error": result.get("error", "")}
