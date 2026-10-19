"""Run the OpenAI/Anthropic adapter in a dedicated process.

The adapter's asyncio loop normally shares the RolloutManager process with the
synchronous Mini-SWE agent threads (184 of them at production concurrency).
CPython serialises bytecode on a single GIL, so the loop cannot reliably be
re-scheduled when an executor job finishes or an upstream response arrives.
Measured on a 943K-turn run, that cost ~1.0 s per turn waiting to resume after
render, another ~1.0 s after decode, and left a p50 event-loop lag of 1.28 s.
Giving the adapter its own interpreter removes it from that contention.

Only the control plane needs a bridge: the agent-facing data plane is already
HTTP. ``open_session`` / ``finish_session`` / ``drop_session`` are served under
``/_control`` on the same port and speak ``Sample.to_dict()``/``from_dict()``,
so the parent-side proxy is signature-compatible with ``BaseAdapter``.
"""

from __future__ import annotations

import argparse
import asyncio
import atexit
import ctypes
import json
import logging
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import aiohttp
from aiohttp import web

from slime.utils.types import Sample

logger = logging.getLogger(__name__)

CONTROL_PREFIX = "/_control"
_PR_SET_PDEATHSIG = 1


# --------------------------------------------------------------- child side --


def add_control_routes(app: web.Application, adapter: Any) -> None:
    """Expose the adapter's session lifecycle over HTTP."""

    async def healthz(_request: web.Request) -> web.Response:
        return web.json_response({"ok": True, "pid": os.getpid()})

    async def open_session(request: web.Request) -> web.Response:
        body = await request.json()
        adapter.open_session(
            body["sid"],
            sampling_defaults=body.get("sampling_defaults") or {},
            max_context_tokens=int(body.get("max_context_tokens") or 0),
            min_generation_tokens=int(body.get("min_generation_tokens") or 0),
        )
        return web.json_response({"ok": True})

    async def finish_session(request: web.Request) -> web.Response:
        body = await request.json()
        samples = await adapter.finish_session(
            body["sid"],
            base_sample=Sample.from_dict(body["base_sample"]),
            reward=body.get("reward", 0.0),
            extra_metadata=body.get("extra_metadata") or {},
            wait_timeout=float(body.get("wait_timeout", 5.0)),
        )
        return web.json_response({"samples": [s.to_dict() for s in samples]})

    async def drop_session(request: web.Request) -> web.Response:
        body = await request.json()
        await adapter.drop_session(
            body["sid"], wait_timeout=float(body.get("wait_timeout", 5.0))
        )
        return web.json_response({"ok": True})

    app.router.add_get(f"{CONTROL_PREFIX}/healthz", healthz)
    app.router.add_post(f"{CONTROL_PREFIX}/open_session", open_session)
    app.router.add_post(f"{CONTROL_PREFIX}/finish_session", finish_session)
    app.router.add_post(f"{CONTROL_PREFIX}/drop_session", drop_session)


def _build_adapter(args: argparse.Namespace):
    from transformers import AutoTokenizer

    from slime.agent.adapters.anthropic import AnthropicAdapter
    from slime.agent.adapters.openai import OpenAIAdapter

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path, trust_remote_code=True
    )
    cls = AnthropicAdapter if args.protocol == "anthropic" else OpenAIAdapter
    adapter = cls(
        tokenizer=tokenizer,
        sglang_url=args.sglang_url,
        tool_parser=args.tool_parser or None,
        reasoning_parser=args.reasoning_parser or None,
        max_turns_per_sid=args.max_turns_per_sid or None,
        fork_threshold_tokens=args.fork_threshold_tokens or None,
    )
    from .recovery_trajectory import RecoveryTrajectoryManager
    adapter.manager = RecoveryTrajectoryManager(fork_threshold_tokens=args.fork_threshold_tokens or None)
    return adapter


def child_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--sglang-url", required=True)
    parser.add_argument("--protocol", default="openai", choices=("openai", "anthropic"))
    parser.add_argument("--tool-parser", default="")
    parser.add_argument("--reasoning-parser", default="")
    parser.add_argument("--max-turns-per-sid", type=int, default=0)
    parser.add_argument("--fork-threshold-tokens", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--port-file", required=True)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, stream=sys.stdout, format="[adapter] %(message)s"
    )
    adapter = _build_adapter(args)
    add_control_routes(adapter.app, adapter)

    async def serve() -> None:
        from slime.agent.aiohttp_threaded import FilteredAccessLogger

        runner = web.AppRunner(
            adapter.app,
            handler_cancellation=True,
            access_log_class=FilteredAccessLogger,
        )
        await runner.setup()
        site = web.TCPSite(runner, args.host, args.port)
        await site.start()
        port = args.port
        for sock in site._server.sockets:  # type: ignore[attr-defined]
            port = sock.getsockname()[1]
            break
        # Publish atomically so the parent never reads a half-written file.
        path = Path(args.port_file)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"port": port, "pid": os.getpid()}))
        tmp.replace(path)
        logger.info("listening on %s:%d pid=%d", args.host, port, os.getpid())
        while True:
            await asyncio.sleep(3600)

    asyncio.run(serve())


# -------------------------------------------------------------- parent side --


def _set_pdeathsig() -> None:
    """Ask the kernel to SIGTERM this child if the parent dies.

    Ray can hard-kill the RolloutManager actor, in which case atexit never
    runs; without this the adapter would survive and keep its port bound.
    """
    try:
        ctypes.CDLL("libc.so.6").prctl(_PR_SET_PDEATHSIG, signal.SIGTERM)
    except Exception:  # pragma: no cover - non-glibc platforms
        pass


class AdapterProcessError(RuntimeError):
    pass


class AdapterProcess:
    """Parent-side handle and ``BaseAdapter``-compatible control proxy."""

    def __init__(
        self,
        *,
        tokenizer_path: str,
        sglang_url: str,
        protocol: str = "openai",
        tool_parser: str | None = None,
        reasoning_parser: str | None = None,
        max_turns_per_sid: int | None = None,
        fork_threshold_tokens: int | None = None,
        host: str = "127.0.0.1",
        port: int = 0,
        start_timeout_sec: float = 600.0,
    ) -> None:
        self.host = host
        self._proc: subprocess.Popen | None = None
        self._port_file = Path(
            tempfile.mkstemp(prefix="slime-adapter-", suffix=".json")[1]
        )
        self._port_file.unlink(missing_ok=True)
        self._session: aiohttp.ClientSession | None = None

        cmd = [
            sys.executable,
            "-u",
            "-m",
            "harnessadapt.training.runtime.recovery_adapter_process",
            "--tokenizer-path", tokenizer_path,
            "--sglang-url", sglang_url,
            "--protocol", protocol,
            "--tool-parser", tool_parser or "",
            "--reasoning-parser", reasoning_parser or "",
            "--max-turns-per-sid", str(max_turns_per_sid or 0),
            "--fork-threshold-tokens", str(fork_threshold_tokens or 0),
            "--host", host,
            "--port", str(port),
            "--port-file", str(self._port_file),
        ]
        # Propagate the parent's import path: the child resolves ``slime`` the
        # same way regardless of how the parent was launched (Ray runtime env,
        # pytest rootdir, plain shell).
        env = dict(os.environ)
        existing = env.get("PYTHONPATH", "")
        entries = [p for p in sys.path if p] + ([existing] if existing else [])
        env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(entries))
        self._proc = subprocess.Popen(cmd, preexec_fn=_set_pdeathsig, env=env)
        atexit.register(self.stop)
        self.port = self._await_ready(start_timeout_sec)
        self.url = f"http://{host}:{self.port}"
        logger.info(
            "[adapter_process] pid=%d url=%s", self._proc.pid, self.url
        )

    # -- lifecycle ---------------------------------------------------------

    def _await_ready(self, timeout: float) -> int:
        import requests

        deadline = time.monotonic() + timeout
        port = None
        while time.monotonic() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                raise AdapterProcessError(
                    f"adapter process exited during startup "
                    f"(code {self._proc.returncode})"
                )
            if port is None and self._port_file.exists():
                try:
                    port = int(json.loads(self._port_file.read_text())["port"])
                except (ValueError, KeyError, json.JSONDecodeError):
                    port = None
            if port is not None:
                try:
                    r = requests.get(
                        f"http://{self.host}:{port}{CONTROL_PREFIX}/healthz",
                        timeout=2,
                    )
                    if r.status_code == 200:
                        return port
                except requests.RequestException:
                    pass
            time.sleep(0.5)
        raise AdapterProcessError("adapter process did not become ready in time")

    def check_alive(self) -> None:
        """Raise if the adapter died; callers must not silently lose sessions."""
        if self._proc is None or self._proc.poll() is not None:
            code = self._proc.returncode if self._proc else None
            raise AdapterProcessError(f"adapter process is not running (code {code})")

    def stop(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        atexit.unregister(self.stop)
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        self._port_file.unlink(missing_ok=True)

    # -- control plane -----------------------------------------------------

    async def _http(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=None, sock_read=900),
                trust_env=False,
            )
        return self._session

    async def _post(self, route: str, payload: dict) -> dict:
        self.check_alive()
        session = await self._http()
        async with session.post(f"{self.url}{CONTROL_PREFIX}/{route}", json=payload) as r:
            text = await r.text()
            if r.status >= 400:
                raise AdapterProcessError(f"{route} failed {r.status}: {text[:400]}")
            return json.loads(text)

    def open_session(
        self,
        sid: str,
        *,
        sampling_defaults: dict | None = None,
        max_context_tokens: int = 0,
        min_generation_tokens: int = 0,
    ) -> None:
        # Kept synchronous to match BaseAdapter. It runs once per case (not per
        # turn) and the POST is loopback-local, so the sub-millisecond block on
        # the rollout loop is not worth an API change.
        import requests

        self.check_alive()
        r = requests.post(
            f"{self.url}{CONTROL_PREFIX}/open_session",
            json={
                "sid": sid,
                "sampling_defaults": sampling_defaults or {},
                "max_context_tokens": int(max_context_tokens or 0),
                "min_generation_tokens": int(min_generation_tokens or 0),
            },
            timeout=60,
        )
        if r.status_code >= 400:
            raise AdapterProcessError(
                f"open_session failed {r.status_code}: {r.text[:400]}"
            )

    async def finish_session(
        self,
        sid: str,
        *,
        base_sample: Sample,
        reward: float = 0.0,
        extra_metadata: dict | None = None,
        wait_timeout: float = 5.0,
    ) -> list[Sample]:
        data = await self._post(
            "finish_session",
            {
                "sid": sid,
                "base_sample": base_sample.to_dict(),
                "reward": reward,
                "extra_metadata": extra_metadata or {},
                "wait_timeout": wait_timeout,
            },
        )
        return [Sample.from_dict(d) for d in data["samples"]]

    async def drop_session(self, sid: str, *, wait_timeout: float = 5.0) -> None:
        await self._post("drop_session", {"sid": sid, "wait_timeout": wait_timeout})

    async def aclose(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None


if __name__ == "__main__":
    child_main()

