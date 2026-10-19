"""Background training supervisor that monitors host memory and disk usage."""
from __future__ import annotations

import argparse
import datetime
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from .config import WORKSPACE, inside, read_json, write_json


def memory_percent(text: str) -> float:
    values = {
        line.split(":", 1)[0]: int(line.split()[1])
        for line in text.splitlines()
        if ":" in line
    }
    total, available = values["MemTotal"], values["MemAvailable"]
    return 100.0 * (total - available) / total


def disk_percent(stat) -> float:
    used = stat.f_blocks - stat.f_bfree
    denominator = used + stat.f_bavail
    return 100.0 * used / denominator


def docker(*args: str, timeout: int = 30) -> str:
    return subprocess.run(
        ["docker", *args], text=True, capture_output=True, check=True, timeout=timeout
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=80.0)
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--stop-after", type=int, help="Defaults to total_steps (46)")
    parser.add_argument("--continuation", action="store_true")
    args = parser.parse_args()

    root = inside(args.run_dir)
    config = read_json(root / "config.json")
    stop_after = args.stop_after or config["total_steps"]
    lock = (root / "resource_guard.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    with (root / "training_background.log").open("a", buffering=1) as log:
        module = (
            "harnessadapt.training.continuation"
            if args.continuation
            else "harnessadapt.training"
        )
        cmd = [sys.executable, "-u", "-m", module]
        if not args.continuation:
            cmd.append("train")
        cmd.extend(["--run-dir", str(root), "--stop-after", str(stop_after)])
        child = subprocess.Popen(
            cmd,
            cwd=WORKSPACE,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        while child.poll() is None:
            mem = memory_percent(Path("/proc/meminfo").read_text(encoding="utf-8"))
            dsk = disk_percent(os.statvfs(root))
            status = {
                "time": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "launcher_pid": child.pid,
                "memory_percent": round(mem, 2),
                "disk_percent": round(dsk, 2),
            }
            write_json(root / "resource_guard_status.json", status)
            if mem >= args.threshold or dsk >= args.threshold:
                os.killpg(child.pid, signal.SIGTERM)
                return 2
            time.sleep(args.interval)
        return child.returncode or 0


if __name__ == "__main__":
    raise SystemExit(main())
