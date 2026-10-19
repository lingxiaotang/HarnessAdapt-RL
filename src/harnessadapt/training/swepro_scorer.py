"""Official SWE-bench Pro V2 clean-container `test.sh` replay scorer (642 tasks).

Ported from `trajectory_meta_skill_teacher/eval_base_v7_swepro_v2_20261005/score_v2.py`
and `eval_phase1_60turn_step20_v7_swepro_v2_20261005/score_v2.py`.
Applies each submitted patch in a pristine task container with `/tests` mounted
read-only, runs `/tests/test.sh`, and reads `/logs/verifier/reward.txt`.
Unsubmitted or empty patches receive 0 under `strict_explicit_submission_zero`;
timed-out inference or grading attempts receive 0 under `case_timeout_zero`.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import subprocess
import tomllib
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from threading import BoundedSemaphore
from typing import Any

from .config import WORKSPACE, inside, read_json, write_json
from .evaluation_paths import evaluation_directory

DEFAULT_SWEPRO_TASKS = Path(
    os.environ.get(
        "HARNESSADAPT_SWEPRO_TASKS",
        str(WORKSPACE / "swebench_pro_v2_upstream_20260923" / "v2" / "tasks"),
    )
)
DOCKER_LIFECYCLE_SLOTS = BoundedSemaphore(8)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _docker_prefix() -> tuple[str, ...]:
    try:
        if os.readlink("/proc/self/ns/mnt") != os.readlink("/proc/1/ns/mnt"):
            return ("nsenter", "-t", "1", "-m", "--", "docker")
    except OSError:
        pass
    return ("docker",)


def docker(*args: str, timeout: int = 90, stdout=None) -> subprocess.CompletedProcess:
    """Bound concurrent Docker container lifecycle calls while allowing parallel test execution."""
    slots = DOCKER_LIFECYCLE_SLOTS if args and args[0] in {"run", "cp", "rm"} else nullcontext()
    with slots:
        return subprocess.run(
            (*_docker_prefix(), *args),
            text=True,
            stdout=stdout if stdout is not None else subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )


def grade_swepro_task(
    task_id: str,
    patch: str,
    result_dir: Path,
    *,
    tasks_root: Path = DEFAULT_SWEPRO_TASKS,
    label: str = "swepro",
) -> dict[str, Any]:
    """Replay official SWE-bench Pro V2 `/tests/test.sh` in a fresh container."""
    tasks_root = Path(tasks_root).resolve()
    task = tasks_root / task_id
    if not task.is_dir() or task.parent != tasks_root:
        raise ValueError(f"Unknown SWE-bench Pro task: {task_id}")
    config = tomllib.loads((task / "task.toml").read_text(encoding="utf-8"))
    image = config["environment"]["docker_image"]
    if image.rsplit(":", 1)[1] != task_id:
        raise ValueError(f"Task/image mismatch for {task_id}: {image}")

    result_dir.mkdir(parents=True, exist_ok=True)
    uid = hashlib.sha256(f"{label}:{task_id}".encode("utf-8")).hexdigest()[:14]
    attempt_id = uuid.uuid4().hex
    container = f"swepro-v2-grade-{uid}-{attempt_id[:12]}"
    canonical_dir = result_dir
    attempt_dir = canonical_dir / "attempts" / attempt_id
    attempt_dir.mkdir(parents=True)
    (attempt_dir / "tmp").mkdir()
    (attempt_dir / "verifier").mkdir()

    record: dict[str, Any] = {
        "task_id": task_id,
        "image": image,
        "time": now(),
        "patch_bytes": len(patch.encode("utf-8")),
        "patch_sha256": hashlib.sha256(patch.encode("utf-8")).hexdigest(),
        "reward_known": False,
        "reward": None,
        "container": container,
        "attempt_id": attempt_id,
        "log_dir": str(attempt_dir),
    }
    patch_file = attempt_dir / "submitted.patch"
    patch_file.write_text(patch, encoding="utf-8")
    test_log = attempt_dir / "test.log"
    try:
        start = docker(
            "run",
            "--rm",
            "-d",
            "--name",
            container,
            "--label",
            f"self_evolve.swepro_v2={canonical_dir.parent}",
            "--network",
            "bridge",
            "--memory",
            "4g",
            "--cpus",
            "1",
            "-v",
            f"{(task / 'tests').resolve()}:/tests:ro",
            "-v",
            f"{attempt_dir.resolve()}:/logs:rw",
            "-w",
            "/",
            image,
            "sleep",
            "3300",
            timeout=300,
        )
        if start.returncode != 0:
            record["error"] = "container start: " + str(start.stdout)[-1000:]
            return record
        record["container_started"] = True
        if patch.strip():
            copied = docker("cp", str(patch_file), f"{container}:/tmp/model.patch", timeout=120)
            if copied.returncode != 0:
                record["error"] = "docker cp: " + str(copied.stdout)[-1000:]
                return record
            apply = docker(
                "exec",
                "-w",
                "/",
                container,
                "bash",
                "-c",
                "cd /app 2>/dev/null || cd /testbed; "
                "git apply --verbose /tmp/model.patch "
                "|| git apply --3way /tmp/model.patch "
                "|| patch --fuzz=3 -p1 -i /tmp/model.patch",
                timeout=300,
            )
            record["apply_returncode"] = apply.returncode
            (attempt_dir / "apply.log").write_text(str(apply.stdout), encoding="utf-8")
        else:
            record["apply_returncode"] = None
        with test_log.open("w", encoding="utf-8") as output:
            test = docker(
                "exec",
                "-e",
                "TMPDIR=/logs/tmp",
                "-w",
                "/",
                container,
                "bash",
                "/tests/test.sh",
                timeout=3120,
                stdout=output,
            )
        record["test_returncode"] = test.returncode
        reward_file = attempt_dir / "verifier" / "reward.txt"
        if reward_file.is_file() and reward_file.read_text(encoding="utf-8").strip() in ("0", "1"):
            record["reward"] = int(reward_file.read_text(encoding="utf-8").strip())
            record["reward_known"] = True
        else:
            record["error"] = "Official verifier did not produce a valid reward.txt"
        return record
    except subprocess.TimeoutExpired as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        record.update(
            reward=0,
            reward_known=True,
            timed_out=True,
            protocol="case_timeout_zero",
            timeout_stage="official_grading",
            timeout_seconds=exc.timeout,
            retryable=False,
        )
        return record
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        return record
    finally:
        try:
            cleanup = docker("rm", "-f", container, timeout=60)
            if cleanup.returncode:
                record["cleanup_error"] = str(cleanup.stdout)[-1000:]
        except Exception as exc:
            record["cleanup_error"] = f"{type(exc).__name__}: {exc}"
        payload = json.dumps(record, ensure_ascii=False, indent=2) + "\n"
        (attempt_dir / "result.json").write_text(payload, encoding="utf-8")
        output_path = canonical_dir / "result.json"
        temporary = canonical_dir / "result.json.tmp"
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(output_path)


def _trajectory_dir(run_dir: Path, step: int, repetition: int = 1) -> Path:
    return evaluation_directory(run_dir, "swepro", step, repetition) / "trajectories"


def score_swepro_run(
    run_dir: Path,
    *,
    step: int = 45,
    repetition: int = 1,
    tasks_root: Path = DEFAULT_SWEPRO_TASKS,
    workers: int = 128,
    label: str | None = None,
) -> dict[str, Any]:
    """Score all 642 SWE-bench Pro V2 trajectories in `run_dir` and write `official_v2_summary.json`."""
    run_dir = inside(run_dir)
    tasks_root = Path(tasks_root).resolve()
    label = label or run_dir.name
    evaluation_root = evaluation_directory(run_dir, "swepro", step, repetition)
    trajectory_dir = _trajectory_dir(run_dir, step, repetition)
    results = evaluation_root / "official_v2_results"
    results.mkdir(parents=True, exist_ok=True)

    files = sorted(trajectory_dir.rglob("*.json.gz"))
    if len(files) != 642:
        raise ValueError(f"Expected 642 trajectories in {trajectory_dir}, found {len(files)}")

    records: dict[str, dict[str, Any]] = {}
    for path in files:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            row = json.load(stream)
        key = str(row["instance_id"])
        if key in records:
            raise ValueError(f"Duplicate trajectory for task: {key}")
        submitted = bool(row.get("official_submitted"))
        patch = str(row.get("explicit_submission_patch") or "") if submitted else ""
        timed_out = bool((row.get("outcome") or {}).get("timed_out"))
        records[key] = {"patch": patch, "submitted": submitted, "timed_out": timed_out}

    if tasks_root.is_dir():
        expected = {x.name for x in tasks_root.iterdir() if x.is_dir()}
        if set(records) != expected:
            raise ValueError("Trajectory task set differs from official SWE-bench Pro V2 task directory")

    todo: list[tuple[str, str]] = []
    for task_id, info in sorted(records.items()):
        patch = info["patch"]
        submitted = info["submitted"]
        timed_out = info["timed_out"]
        patch_sha = hashlib.sha256(patch.encode("utf-8")).hexdigest()
        result_file = results / task_id / "result.json"
        if result_file.is_file():
            old = read_json(result_file)
            if old.get("reward_known") and old.get("patch_sha256") == patch_sha:
                continue
        if timed_out or not submitted or not patch.strip():
            directory = results / task_id
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "submitted.patch").write_text(patch, encoding="utf-8")
            write_json(
                result_file,
                {
                    "task_id": task_id,
                    "time": now(),
                    "reward_known": True,
                    "reward": 0,
                    "submitted": submitted,
                    "timed_out": timed_out,
                    "patch_bytes": len(patch.encode("utf-8")),
                    "patch_sha256": patch_sha,
                    "protocol": "case_timeout_zero" if timed_out else "strict_explicit_submission_zero",
                },
            )
            continue
        todo.append((task_id, patch))

    print(
        json.dumps(
            {
                "variant": label,
                "step": step,
                "repetition": repetition,
                "benchmark": "swepro",
                "total": len(records),
                "to_grade": len(todo),
                "submitted": sum(v["submitted"] for v in records.values()),
            }
        ),
        flush=True,
    )

    if todo:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    grade_swepro_task,
                    task_id,
                    patch,
                    results / task_id,
                    tasks_root=tasks_root,
                    label=label,
                ): task_id
                for task_id, patch in todo
            }
            for index, future in enumerate(as_completed(futures), 1):
                try:
                    outcome = future.result()
                except Exception as exc:
                    print(json.dumps({"task": futures[future], "error": str(exc)}), flush=True)
                    continue
                if index % 10 == 0 or not outcome.get("reward_known"):
                    print(
                        json.dumps(
                            {
                                "graded": index,
                                "of": len(todo),
                                "task": outcome["task_id"],
                                "reward": outcome.get("reward"),
                                "error": outcome.get("error", ""),
                            }
                        ),
                        flush=True,
                    )

    graded: dict[str, dict[str, Any]] = {}
    for task_id in records:
        path = results / task_id / "result.json"
        if path.is_file():
            graded[task_id] = read_json(path)

    known = sum(bool(x.get("reward_known")) for x in graded.values())
    passed = sum(x.get("reward") == 1 for x in graded.values())
    submitted_count = sum(v["submitted"] for v in records.values())
    timeout_zero = sum(x.get("protocol") == "case_timeout_zero" for x in graded.values())
    summary = {
        "variant": label,
        "step": step,
        "repetition": repetition,
        "benchmark": "swepro",
        "total": 642,
        "graded": len(graded),
        "known": known,
        "passed": passed,
        "pass_at_1": passed / 642.0,
        "submitted": submitted_count,
        "timeout_zero": timeout_zero,
        "unknown": sorted(k for k, v in graded.items() if not v.get("reward_known")),
        "time": now(),
        "protocol": "official_v2_test_sh_clean_replay",
        "complete": known == 642,
    }
    write_json(evaluation_root / "official_v2_summary.json", summary)
    write_json(evaluation_root / "summary.json", summary)
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--step", type=int, default=45)
    parser.add_argument("--repetition", type=int, default=1)
    parser.add_argument("--tasks-root", type=Path, default=DEFAULT_SWEPRO_TASKS)
    parser.add_argument("--smoke-task")
    parser.add_argument("--workers", type=int, default=128)
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 128:
        parser.error("--workers must be between 1 and 128")
    if args.smoke_task:
        task = Path(args.tasks_root).resolve() / args.smoke_task
        if not task.is_dir():
            parser.error(f"Unknown smoke task: {args.smoke_task}")
        smoke_root = (args.run_dir or Path(".")).resolve() / "scoring_preflight" / task.name
        for label, patch in (("empty", ""), ("gold", (task / "solution" / "gold_patch.diff").read_text(encoding="utf-8"))):
            result = grade_swepro_task(
                task.name,
                patch,
                smoke_root / label,
                tasks_root=args.tasks_root,
                label="preflight_" + label,
            )
            print(json.dumps({"case": label, **result}), flush=True)
            if not result["reward_known"] or result["reward"] != (1 if label == "gold" else 0):
                return 1
        return 0
    if not args.run_dir:
        parser.error("Specify --run-dir or --smoke-task")
    result = score_swepro_run(
        args.run_dir,
        step=args.step,
        repetition=args.repetition,
        tasks_root=args.tasks_root,
        workers=args.workers,
    )
    print(json.dumps(result), flush=True)
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
