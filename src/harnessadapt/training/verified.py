"""Prepare and validate SWE-bench Verified (500), SWE-bench Multilingual (300), and SWE-bench Pro V2 (642) inputs.

Ported from:
- `trajectory_meta_skill_teacher/training/verified.py`
- `trajectory_meta_skill_teacher/eval_step15_v5_multilingual_128k_20260928/evaluate.py`
- `trajectory_meta_skill_teacher/eval_swepro_v2_three_way_20260923/prepare.py` & `evaluate.py`
"""
from __future__ import annotations

import argparse
import json
import os
import tomllib
from pathlib import Path
from typing import Any

from .config import WORKSPACE, inside, read_json, sha256, write_json
from .multilingual_scorer import verify_multilingual_parsers

from .evaluation_paths import BENCHMARKS, input_directory

MULTILINGUAL_DATASET = "SWE-bench/SWE-bench_Multilingual"
MULTILINGUAL_REVISION = "846e647b9f33c0b51b739d005d13d85493c9af09"
MULTILINGUAL_TASKS = 300
DEFAULT_MULTILINGUAL_SOURCE = Path(
    os.environ.get(
        "HARNESSADAPT_MULTILINGUAL_SOURCE",
        str(WORKSPACE / "swe_multilingual_image_download_20260922" / "dataset.jsonl"),
    )
)

SWEPRO_DATASET = "ScaleAI/SWE-bench_Pro"
SWEPRO_DATASET_REVISION = "2d52cb3df914a3fcf80c7f66738b3a88ae37fc50"
SWEPRO_UPSTREAM_REVISION = "66f92766bba642462d4bbe5479e83f91f9211862"
SWEPRO_TASKS = 642
DEFAULT_SWEPRO_TASKS_DIR = Path(
    os.environ.get(
        "HARNESSADAPT_SWEPRO_TASKS",
        str(WORKSPACE / "swebench_pro_v2_upstream_20260923" / "v2" / "tasks"),
    )
)


def _check_no_training_overlap(root: Path, rows: list[dict[str, Any]]) -> None:
    if not (root / "train.jsonl").is_file():
        return
    training = [
        json.loads(line)
        for line in (root / "train.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    train_ids = {r["label"] for r in training}
    train_issues = {
        r["metadata"]["instance"]["problem_statement"].strip()
        for r in training
        if (r.get("metadata") or {}).get("instance", {}).get("problem_statement")
    }
    for row in rows:
        instance = row["metadata"]["instance"]
        if row["label"] in train_ids or instance.get("problem_statement", "").strip() in train_issues:
            raise ValueError(f"Overlap detected between training pool and evaluation task {row['label']}")


def prepare_verified(root: Path) -> None:
    """Prepare and validate SWE-bench Verified (500 tasks)."""
    target = input_directory(root, "verified") / "verified.jsonl"
    manifest_path = input_directory(root, "verified") / "verified_manifest.json"
    if target.exists() and manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest.get("dataset") == "princeton-nlp/SWE-bench_Verified":
            validate_verified(root)
            return
    try:
        from datasets import load_dataset
        from swebench.harness.test_spec.test_spec import make_test_spec
    except ImportError as exc:
        raise RuntimeError(
            "Preparing Verified data requires the dependencies in requirements-data.txt"
        ) from exc
    target.parent.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(
        "princeton-nlp/SWE-bench_Verified", split="test", cache_dir=str(input_directory(root, "verified") / "dataset_cache")
    )
    if len(dataset) != 500:
        raise ValueError("SWE-bench Verified must contain 500 tasks")
    rows = []
    for raw in dataset:
        instance = dict(raw)
        spec = make_test_spec(instance)
        instance["eval_script"] = spec.eval_script
        instance.pop("patch", None)
        instance.pop("hints_text", None)
        instance["image_name"] = (
            "docker.io/swebench/sweb.eval.x86_64."
            + instance["instance_id"].replace("__", "_1776_")
            + ":latest"
        ).lower()
        rows.append(
            {
                "prompt": [{"role": "user", "content": instance["problem_statement"]}],
                "label": instance["instance_id"],
                "metadata": {
                    "data_source": "swe-bench_verified",
                    "instance": instance,
                    "curriculum": {"medium_hint": "", "stage2_use_hint": False},
                },
            }
        )
    target.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    write_json(
        manifest_path,
        {
            "dataset": "princeton-nlp/SWE-bench_Verified",
            "benchmark": "verified",
            "split": "test",
            "sha256": sha256(target),
            "tasks": 500,
            "temperature": 1.0,
            "max_iterations": 150,
            "context_tokens": 131072,
            "n_samples": 1,
            "teacher_access": False,
        },
    )
    validate_verified(root)


def validate_verified(root: Path) -> list[dict[str, Any]]:
    manifest = read_json(input_directory(root, "verified") / "verified_manifest.json")
    path = input_directory(root, "verified") / "verified.jsonl"
    if sha256(path) != manifest["sha256"]:
        raise ValueError("Verified input changed")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    labels = [row["label"] for row in rows]
    if manifest.get("tasks") != 500 or len(rows) != 500 or len(set(labels)) != 500:
        raise ValueError("Verified input must contain exactly 500 unique tasks")
    for row in rows:
        instance = row["metadata"]["instance"]
        if row["metadata"]["curriculum"].get("medium_hint") or instance.get("patch") or instance.get("hints_text"):
            raise ValueError("Privileged solution or skill found in Verified input")
        if not instance.get("eval_script"):
            raise ValueError("Missing official evaluation script in Verified input")
    _check_no_training_overlap(root, rows)
    return rows


def prepare_multilingual(root: Path, *, source: Path | None = None) -> None:
    """Prepare and validate SWE-bench Multilingual (300 tasks, revision 846e647b9f33c0b51b739d005d13d85493c9af09)."""
    target = input_directory(root, "multilingual") / "verified.jsonl"
    manifest_path = input_directory(root, "multilingual") / "verified_manifest.json"
    if target.exists() and manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest.get("dataset") == MULTILINGUAL_DATASET:
            validate_multilingual(root)
            return

    source_path = Path(source or DEFAULT_MULTILINGUAL_SOURCE)
    raw_rows: list[dict[str, Any]]
    if source_path.is_file():
        raw_rows = [
            json.loads(line)
            for line in source_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError(
                "Preparing Multilingual data requires requirements-data.txt or a local dataset.jsonl"
            ) from exc
        ds = load_dataset(
            MULTILINGUAL_DATASET,
            split="test",
            revision=MULTILINGUAL_REVISION,
            cache_dir=str(input_directory(root, "multilingual") / "dataset_cache"),
        )
        raw_rows = [dict(r) for r in ds]

    if len(raw_rows) != MULTILINGUAL_TASKS:
        raise ValueError(f"SWE-bench Multilingual must contain {MULTILINGUAL_TASKS} tasks, found {len(raw_rows)}")

    target.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for raw in raw_rows:
        instance = dict(raw)
        instance.pop("patch", None)
        instance.pop("hints_text", None)
        instance["image_name"] = instance["image"]
        rows.append(
            {
                "prompt": [{"role": "user", "content": instance["problem_statement"]}],
                "label": instance["instance_id"],
                "metadata": {
                    "data_source": "swe-bench_verified",
                    "benchmark": "swe-bench_multilingual",
                    "instance": instance,
                    "curriculum": {"medium_hint": "", "stage2_use_hint": False},
                },
            }
        )
    target.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    write_json(
        manifest_path,
        {
            "dataset": MULTILINGUAL_DATASET,
            "benchmark": "multilingual",
            "split": "test",
            "tasks": MULTILINGUAL_TASKS,
            "sha256": sha256(target),
            "source_revision": MULTILINGUAL_REVISION,
            "solver": "original_mini_swe",
            "teacher_access": False,
            "compatibility_note": "data_source uses legacy verified dispatch only; official Multilingual eval_script/log_parser used",
        },
    )
    validate_multilingual(root)


def validate_multilingual(root: Path) -> list[dict[str, Any]]:
    manifest = read_json(input_directory(root, "multilingual") / "verified_manifest.json")
    path = input_directory(root, "multilingual") / "verified.jsonl"
    if sha256(path) != manifest["sha256"] or manifest.get("tasks") != MULTILINGUAL_TASKS:
        raise ValueError("Multilingual input hash or task count mismatch")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != MULTILINGUAL_TASKS or len({r["label"] for r in rows}) != MULTILINGUAL_TASKS:
        raise ValueError("Multilingual input must contain 300 unique tasks")
    for row in rows:
        instance = row["metadata"]["instance"]
        if "patch" in instance or "hints_text" in instance:
            raise ValueError("Privileged solution or hints leaked to Multilingual input")
        if row["prompt"] != [{"role": "user", "content": instance["problem_statement"]}]:
            raise ValueError("Multilingual prompt differs from official problem_statement")
        if row["metadata"]["curriculum"].get("medium_hint") or row["metadata"]["curriculum"].get("stage2_use_hint"):
            raise ValueError("Unexpected skill/hint in Multilingual evaluation input")
        if instance.get("image_name") != instance.get("image"):
            raise ValueError(f"Multilingual image_name mismatch for {row['label']}")
        if not (
            instance.get("eval_script")
            and instance.get("log_parser")
            and instance.get("FAIL_TO_PASS")
            and instance.get("eval_type") == "pass_and_fail"
        ):
            raise ValueError(f"Incomplete Multilingual grading metadata for {row['label']}")
    verify_multilingual_parsers(rows)
    _check_no_training_overlap(root, rows)
    return rows


def prepare_swepro(root: Path, *, tasks_root: Path | None = None) -> None:
    """Prepare and validate SWE-bench Pro V2 (642 tasks, official V2 clean replay protocol)."""
    target = input_directory(root, "swepro") / "verified.jsonl"
    manifest_path = input_directory(root, "swepro") / "verified_manifest.json"
    if target.exists() and manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest.get("dataset") == SWEPRO_DATASET:
            validate_swepro(root)
            return

    tasks_dir = Path(tasks_root or DEFAULT_SWEPRO_TASKS_DIR)
    if not tasks_dir.is_dir():
        raise FileNotFoundError(
            f"SWE-bench Pro V2 tasks directory not found: {tasks_dir}. "
            "Set HARNESSADAPT_SWEPRO_TASKS or pass --swepro-tasks."
        )
    task_dirs = sorted(p for p in tasks_dir.iterdir() if p.is_dir())
    if len(task_dirs) != SWEPRO_TASKS:
        raise ValueError(f"Expected {SWEPRO_TASKS} official SWE-bench Pro V2 task directories, found {len(task_dirs)}")

    target.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for task in task_dirs:
        task_id = task.name
        instruction = (task / "instruction.md").read_text(encoding="utf-8").strip()
        task_config = tomllib.loads((task / "task.toml").read_text(encoding="utf-8"))
        image = task_config["environment"]["docker_image"]
        if image.rsplit(":", 1)[1] != task_id:
            raise ValueError(f"SWE-bench Pro image mismatch for {task_id}: {image}")
        if not instruction or not (task / "tests" / "test.sh").is_file():
            raise ValueError(f"Incomplete SWE-bench Pro task directory: {task_id}")
        instance = {
            "instance_id": task_id,
            "problem_statement": instruction,
            "image_name": image,
            "image": image,
            "eval_script": "# official V2 clean replay happens after inference",
        }
        rows.append(
            {
                "prompt": [{"role": "user", "content": instruction}],
                "label": task_id,
                "metadata": {
                    "data_source": "swe-bench_verified",
                    "instance": instance,
                    "curriculum": {"medium_hint": "", "stage2_use_hint": False},
                },
            }
        )
    target.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    write_json(
        manifest_path,
        {
            "dataset": SWEPRO_DATASET,
            "benchmark": "swepro",
            "config": "default",
            "split": "test",
            "tasks": SWEPRO_TASKS,
            "sha256": sha256(target),
            "dataset_revision": SWEPRO_DATASET_REVISION,
            "upstream_revision": SWEPRO_UPSTREAM_REVISION,
            "explicit_submission_required": True,
            "score_protocol": "offline_clean_v2_test_sh_replay",
        },
    )
    validate_swepro(root)


def validate_swepro(root: Path) -> list[dict[str, Any]]:
    manifest = read_json(input_directory(root, "swepro") / "verified_manifest.json")
    path = input_directory(root, "swepro") / "verified.jsonl"
    if sha256(path) != manifest["sha256"] or manifest.get("tasks") != SWEPRO_TASKS:
        raise ValueError("Frozen SWE-bench Pro V2 input hash or task count mismatch")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != SWEPRO_TASKS or len({r["label"] for r in rows}) != SWEPRO_TASKS:
        raise ValueError("Incomplete SWE-bench Pro V2 input (expected 642 unique tasks)")
    for row in rows:
        instance = row["metadata"]["instance"]
        if row["label"] != instance.get("instance_id"):
            raise ValueError(f"Mismatched SWE-bench Pro task ID: {row['label']}")
        if row["prompt"] != [{"role": "user", "content": instance["problem_statement"]}]:
            raise ValueError("SWE-bench Pro prompt differs from official instruction")
        if any(key in instance for key in ("patch", "test_patch", "hints_text", "fail_to_pass", "pass_to_pass")):
            raise ValueError("Privileged grader data leaked to SWE-bench Pro solver metadata")
        if row["metadata"]["curriculum"].get("medium_hint"):
            raise ValueError("Unexpected skill/hint in SWE-bench Pro V2 evaluation")
    _check_no_training_overlap(root, rows)
    return rows


def prepare_benchmark(
    root: Path,
    benchmark: str = "verified",
    *,
    multilingual_source: Path | None = None,
    swepro_tasks: Path | None = None,
) -> None:
    benchmark = benchmark.strip().lower()
    if benchmark == "verified":
        prepare_verified(root)
    elif benchmark == "multilingual":
        prepare_multilingual(root, source=multilingual_source)
    elif benchmark == "swepro":
        prepare_swepro(root, tasks_root=swepro_tasks)
    else:
        raise ValueError(f"Unknown benchmark {benchmark!r}; choose from {BENCHMARKS}")


def validate_benchmark(root: Path, benchmark: str = "verified") -> list[dict[str, Any]]:
    benchmark = benchmark.strip().lower()
    if benchmark == "verified":
        return validate_verified(root)
    if benchmark == "multilingual":
        return validate_multilingual(root)
    if benchmark == "swepro":
        return validate_swepro(root)
    raise ValueError(f"Unknown benchmark {benchmark!r}; choose from {BENCHMARKS}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--benchmark", choices=BENCHMARKS, default="verified")
    parser.add_argument("--multilingual-source", type=Path)
    parser.add_argument("--swepro-tasks", type=Path)
    args = parser.parse_args()
    prepare_benchmark(
        inside(args.run_dir),
        args.benchmark,
        multilingual_source=args.multilingual_source,
        swepro_tasks=args.swepro_tasks,
    )
