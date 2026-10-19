#!/usr/bin/env python3
"""Continuously record Slime training, GPU, memory, storage, and Docker health."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FATAL_RE = re.compile(
    r"(CUDA out of memory|OutOfMemoryError|RayTaskError|RuntimeError:|"
    r"Traceback \(most recent call last\)|"
    r"\bKilled(?:\s+process\b|\s+by\b|\s*$)|Evicted|NCCL.*error)",
    re.IGNORECASE,
)
STEP_PATTERNS = (
    re.compile(r"'(?:train|rollout)/step':\s*([0-9]+)"),
    re.compile(r"(?:train|rollout)/step[=: ]+([0-9]+)"),
    re.compile(r"rollout\s+([0-9]+)\s+custom metrics"),
    re.compile(r"step\s+([0-9]+):"),
)
PROGRESS_MARKER_RE = re.compile(
    r"(custom metrics:|perf [0-9]+:|train/step|rollout/step|"
    r"sav(?:e|ing|ed).{0,40}checkpoint|update weights? finished)",
    re.IGNORECASE,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _tail(path: Path, max_bytes: int = 524_288) -> tuple[str, os.stat_result | None]:
    try:
        stat = path.stat()
        with path.open("rb") as handle:
            handle.seek(max(0, stat.st_size - max_bytes))
            text = handle.read().decode("utf-8", errors="replace")
        return text.replace("\r", "\n"), stat
    except OSError:
        return "", None


def _run(command: list[str], timeout: float = 5.0) -> dict[str, Any]:
    try:
        result = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "error": str(exc)}
    return {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def _filesystem(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path)}
    try:
        stat = os.statvfs(path)
    except OSError as exc:
        result["error"] = str(exc)
        return result
    block_size = stat.f_frsize or stat.f_bsize
    total = stat.f_blocks * block_size
    free = stat.f_bfree * block_size
    available = stat.f_bavail * block_size
    used = total - free
    denominator = used + available
    result.update(
        {
            "total_bytes": total,
            "used_bytes": used,
            "available_bytes": available,
            "available_percent": (
                100.0 * available / denominator if denominator else 0.0
            ),
            "inode_total": stat.f_files,
            "inode_available": stat.f_favail,
            "inode_available_percent": (
                100.0 * stat.f_favail / stat.f_files if stat.f_files else 0.0
            ),
        }
    )
    return result


def _memory() -> dict[str, Any]:
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, raw = line.split(":", 1)
            fields = raw.split()
            if fields:
                values[key] = int(fields[0]) * 1024
    except (OSError, ValueError):
        return {"error": "cannot parse /proc/meminfo"}
    total = values.get("MemTotal")
    available = values.get("MemAvailable", values.get("MemFree"))
    return {
        "total_bytes": total,
        "available_bytes": available,
        "used_bytes": (
            max(0, total - available)
            if total is not None and available is not None
            else None
        ),
        "cache_bytes": values.get("Cached", 0)
        + values.get("SReclaimable", 0),
        "swap_total_bytes": values.get("SwapTotal"),
        "swap_free_bytes": values.get("SwapFree"),
    }


def _cgroup_memory() -> dict[str, Any]:
    """Read cgroup-v2 memory accounting when exposed by the container runtime."""
    root = Path("/sys/fs/cgroup")

    def read_number(name: str) -> int | None:
        try:
            raw = (root / name).read_text().strip()
        except OSError:
            return None
        if raw == "max":
            return None
        try:
            return int(raw)
        except ValueError:
            return None

    events: dict[str, int] = {}
    try:
        for line in (root / "memory.events").read_text().splitlines():
            key, value = line.split(maxsplit=1)
            events[key] = int(value)
    except (OSError, ValueError):
        pass
    return {
        "current_bytes": read_number("memory.current"),
        "limit_bytes": read_number("memory.max"),
        "peak_bytes": read_number("memory.peak"),
        "swap_current_bytes": read_number("memory.swap.current"),
        "swap_limit_bytes": read_number("memory.swap.max"),
        "events": events,
    }


def _gpus() -> dict[str, Any]:
    if shutil.which("nvidia-smi") is None:
        return {"error": "nvidia-smi not found"}
    result = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        timeout=5,
    )
    if not result["ok"]:
        return {"error": result.get("stderr") or result.get("error")}
    rows = []
    try:
        for line in result["stdout"].splitlines():
            index, used, total, utilization = (
                field.strip() for field in line.split(",")
            )
            rows.append(
                {
                    "index": int(index),
                    "used_mib": int(used),
                    "total_mib": int(total),
                    "utilization_percent": int(utilization),
                }
            )
    except (TypeError, ValueError):
        return {"error": "invalid nvidia-smi output"}
    return {
        "rows": rows,
        "active_count": sum(
            row["used_mib"] >= 1024 or row["utilization_percent"] > 0
            for row in rows
        ),
    }


def _processes() -> dict[str, Any]:
    coordinators: list[int] = []
    ray = sglang = docker_related = 0
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():
            continue
        try:
            command = (
                (directory / "cmdline")
                .read_bytes()
                .replace(b"\0", b" ")
                .decode("utf-8", errors="replace")
            )
        except OSError:
            continue
        lowered = command.lower()
        if "python3 -u train.py" in command or "python -u train.py" in command:
            coordinators.append(int(directory.name))
        if any(
            marker in lowered
            for marker in ("raylet", "gcs_server", "default_worker.py")
        ):
            ray += 1
        if "sglang" in lowered or "scheduler" in lowered and "gpu" in lowered:
            sglang += 1
        if "r2egym" in lowered or "mini-swe" in lowered:
            docker_related += 1
    return {
        "coordinator_pids": sorted(coordinators),
        "ray_processes": ray,
        "sglang_processes": sglang,
        "r2e_processes": docker_related,
    }


def _docker_container_limits(container: Any) -> dict[str, Any]:
    """Return the cgroup limits from a Docker SDK container inspect result."""

    attrs = getattr(container, "attrs", {}) or {}
    host_config = attrs.get("HostConfig") or {}
    labels = (attrs.get("Config") or {}).get("Labels") or {}

    def integer(name: str) -> int | None:
        value = host_config.get(name)
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    return {
        "managed_r2e": str(labels.get("self_evolve.r2e") or "") == "1",
        "memory_bytes": integer("Memory"),
        "memory_swap_bytes": integer("MemorySwap"),
        "pids_limit": integer("PidsLimit"),
        # False is the safe Docker default: a cgroup-local OOM may kill the
        # offending container process instead of escalating to a host OOM.
        "oom_kill_disabled": bool(host_config.get("OomKillDisable", False)),
    }


def _docker() -> dict[str, Any]:
    client = None
    try:
        import docker

        client = docker.from_env(timeout=10)
        containers = client.containers.list()
        info = client.info()
        disk: dict[str, Any] = {}
        disk_error = None
        try:
            raw_disk = client.df()
            disk_containers = raw_disk.get("Containers") or []
            disk_volumes = raw_disk.get("Volumes") or []
            disk = {
                "layers_size_bytes": raw_disk.get("LayersSize"),
                "image_count": len(raw_disk.get("Images") or []),
                "container_count": len(disk_containers),
                "container_size_rw_bytes": sum(
                    int(item.get("SizeRw") or 0) for item in disk_containers
                ),
                "volume_count": len(disk_volumes),
                "volume_size_bytes": sum(
                    int((item.get("UsageData") or {}).get("Size") or 0)
                    for item in disk_volumes
                    if int((item.get("UsageData") or {}).get("RefCount") or 0) > 0
                ),
            }
        except Exception as exc:
            disk_error = f"{type(exc).__name__}: {exc}"
        r2e_containers = [
            container
            for container in containers
            if not str(container.name).startswith("slime-r2e-")
        ]
        r2e_limits = [
            _docker_container_limits(container) for container in r2e_containers
        ]
        rows = []
        for container in containers[:100]:
            row = {
                "id": container.short_id,
                "image": (
                    container.image.tags[0]
                    if container.image.tags
                    else container.image.short_id
                ),
                "name": container.name,
                "status": container.status,
                "resource_limits": _docker_container_limits(container),
            }
            rows.append(row)
        result = {
            "running_count": len(containers),
            "r2e_count": len(r2e_containers),
            "r2e_resource_limits": {
                "managed_count": sum(
                    bool(limits["managed_r2e"]) for limits in r2e_limits
                ),
                "memory_limited_count": sum(
                    int(limits["memory_bytes"] or 0) > 0
                    for limits in r2e_limits
                ),
                "memory_swap_equal_count": sum(
                    int(limits["memory_bytes"] or 0) > 0
                    and limits["memory_swap_bytes"] == limits["memory_bytes"]
                    for limits in r2e_limits
                ),
                "pids_limited_count": sum(
                    int(limits["pids_limit"] or 0) > 0
                    for limits in r2e_limits
                ),
                "oom_kill_disabled_count": sum(
                    bool(limits["oom_kill_disabled"]) for limits in r2e_limits
                ),
                "memory_values_bytes": sorted(
                    {
                        int(limits["memory_bytes"])
                        for limits in r2e_limits
                        if int(limits["memory_bytes"] or 0) > 0
                    }
                ),
                "memory_swap_values_bytes": sorted(
                    {
                        int(limits["memory_swap_bytes"])
                        for limits in r2e_limits
                        if int(limits["memory_swap_bytes"] or 0) > 0
                    }
                ),
                "pids_values": sorted(
                    {
                        int(limits["pids_limit"])
                        for limits in r2e_limits
                        if int(limits["pids_limit"] or 0) > 0
                    }
                ),
            },
            "container_list_truncated": len(containers) > len(rows),
            "driver": info.get("Driver"),
            "docker_root_dir": info.get("DockerRootDir"),
            "disk": disk,
            "containers": rows,
        }
        if disk_error:
            result["disk_error"] = disk_error
        return result
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass


def _newest_event(tensorboard_dir: Path) -> dict[str, Any] | None:
    candidates = []
    try:
        for path in tensorboard_dir.rglob("events.out.tfevents.*"):
            try:
                candidates.append((path.stat().st_mtime, path))
            except OSError:
                continue
    except OSError:
        return None
    if not candidates:
        return None
    mtime, path = max(candidates)
    try:
        stat = path.stat()
    except OSError:
        return None
    return {"path": str(path), "mtime": mtime, "size": stat.st_size}


def _progress(log_file: Path) -> dict[str, Any]:
    text, stat = _tail(log_file)
    step_matches = [
        (match.start(), int(match.group(1)))
        for pattern in STEP_PATTERNS
        for match in pattern.finditer(text)
    ]
    failure_lines = [
        line.strip()[-500:]
        for line in text.splitlines()
        if FATAL_RE.search(line)
    ]
    progress_lines = [
        line.strip()[-1000:]
        for line in text.splitlines()
        if PROGRESS_MARKER_RE.search(line)
    ]
    return {
        "log_path": str(log_file),
        "log_mtime": stat.st_mtime if stat else None,
        "log_size": stat.st_size if stat else None,
        "step": max(step_matches)[1] if step_matches else None,
        "last_progress_marker": progress_lines[-1] if progress_lines else None,
        "last_failure": failure_lines[-1] if failure_lines else None,
    }


class HealthMonitor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.stop = False
        self.last_semantic_signature: tuple[Any, ...] | None = None
        self.last_log_signature: tuple[Any, ...] | None = None
        self.last_progress_at = time.time()
        self.last_log_activity_at = time.time()
        self.last_docker: dict[str, Any] | None = None
        self.last_seen_step: int | None = None
        self.last_seen_progress_marker: str | None = None
        args.output.parent.mkdir(parents=True, exist_ok=True)
        self.handle = args.output.open("a", encoding="utf-8", buffering=1)

    def request_stop(self, *_args: Any) -> None:
        self.stop = True

    def sample(self, reason: str) -> dict[str, Any]:
        progress = _progress(self.args.log_file)
        event = _newest_event(self.args.tensorboard_dir)
        if progress.get("step") is not None:
            self.last_seen_step = int(progress["step"])
        else:
            progress["step"] = self.last_seen_step
        if progress.get("last_progress_marker") is not None:
            self.last_seen_progress_marker = str(
                progress["last_progress_marker"]
            )
        else:
            progress["last_progress_marker"] = self.last_seen_progress_marker
        semantic_signature = (
            progress.get("step"),
            progress.get("last_progress_marker"),
            event.get("path") if event else None,
            event.get("size") if event else None,
        )
        # Do not treat arbitrary log growth as training progress: a wedged
        # server can print the same warning forever. Conversely, do not reset
        # the timer when an old marker falls out of the bounded log tail.
        if any(value is not None for value in semantic_signature) and (
            semantic_signature != self.last_semantic_signature
        ):
            self.last_semantic_signature = semantic_signature
            self.last_progress_at = time.time()
        log_signature = (
            progress.get("log_mtime"),
            progress.get("log_size"),
        )
        if log_signature != self.last_log_signature:
            self.last_log_signature = log_signature
            self.last_log_activity_at = time.time()
        processes = _processes()
        stale_seconds = max(0.0, time.time() - self.last_progress_at)
        log_stale_seconds = max(0.0, time.time() - self.last_log_activity_at)
        running = bool(processes["coordinator_pids"]) or processes["ray_processes"] > 0
        stalled = running and stale_seconds >= self.args.stall_seconds
        record = {
            "schema_version": 1,
            "timestamp_utc": _utc_now(),
            "reason": reason,
            "status": (
                "shutdown"
                if reason == "shutdown"
                else "stalled"
                if stalled
                else "failed"
                if progress.get("last_failure") and not running
                else "running"
                if running
                else "waiting"
            ),
            "stalled": stalled,
            "stale_seconds": stale_seconds,
            "semantic_progress_stale_seconds": stale_seconds,
            "log_activity_stale_seconds": log_stale_seconds,
            "processes": processes,
            "progress": progress,
            "tensorboard_event": event,
            "gpus": _gpus(),
            "memory": _memory(),
            "cgroup_memory": _cgroup_memory(),
            "filesystems": {
                "workspace": _filesystem(self.args.workspace),
                "run_root": _filesystem(self.args.run_root),
            },
        }
        if reason in {"startup", "detail", "shutdown"}:
            self.last_docker = _docker()
        if self.last_docker is not None:
            # Carry the most recent expensive Docker snapshot into cheap
            # interval records so the live watcher does not alternate between
            # known and unknown container counts.
            record["docker"] = self.last_docker
        return record

    def write(self, reason: str) -> None:
        record = self.sample(reason)
        self.handle.write(
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        )
        self.handle.flush()
        os.fsync(self.handle.fileno())
        if record["stalled"]:
            print(
                f"Slime health monitor: no step/checkpoint/TensorBoard progress for "
                f"{record['stale_seconds']:.0f}s",
                flush=True,
            )

    def run(self) -> int:
        self.write("startup")
        last_detail = time.monotonic()
        try:
            while not self.stop:
                deadline = time.monotonic() + self.args.interval
                while not self.stop and time.monotonic() < deadline:
                    time.sleep(min(0.5, deadline - time.monotonic()))
                if self.stop:
                    break
                now = time.monotonic()
                reason = (
                    "detail"
                    if now - last_detail >= self.args.detail_interval
                    else "interval"
                )
                if reason == "detail":
                    last_detail = now
                self.write(reason)
            self.write("shutdown")
            return 0
        finally:
            self.handle.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--log-file", type=Path, required=True)
    parser.add_argument("--tensorboard-dir", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--detail-interval", type=float, default=600.0)
    parser.add_argument("--stall-seconds", type=float, default=3600.0)
    args = parser.parse_args()
    if min(args.interval, args.detail_interval, args.stall_seconds) <= 0:
        parser.error("all intervals must be positive")
    return args


def main() -> int:
    args = parse_args()
    monitor = HealthMonitor(args)
    signal.signal(signal.SIGTERM, monitor.request_stop)
    signal.signal(signal.SIGINT, monitor.request_stop)
    return monitor.run()


if __name__ == "__main__":
    raise SystemExit(main())

