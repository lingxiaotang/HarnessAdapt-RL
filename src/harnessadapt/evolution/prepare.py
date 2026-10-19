"""Initialize a tool and hook evolution directory (H0 -> H1 in Epoch 1, H1 -> H2 in Epoch 2)."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .common import BASE, IMAGE, PACKAGE_ROOT, TARGETS, files, sha, write
from harnessadapt.policy import INSTANCE_TEMPLATE, SYSTEM_TEMPLATE


def prepare(root: Path | str, *, source_pool: Path | None = None, epoch1_checkpoint: Path | None = None) -> dict:
    root = Path(root).resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"Use a fresh experiment directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(PACKAGE_ROOT / "data/sampling_plan.json", root / "sampling_plan.json")

    pool_path = Path(source_pool or (PACKAGE_ROOT / "data/all_cached_train.jsonl"))
    if pool_path.is_file():
        with pool_path.open(encoding="utf-8") as stream, (root / "pool.jsonl").open("w", encoding="utf-8") as output:
            for line in stream:
                row = json.loads(line)
                row["metadata"].pop("task_specific_skill", None)
                row["metadata"].pop("task_skill_provenance", None)
                row["metadata"]["curriculum"] = {"medium_hint": "", "stage2_use_hint": False}
                output.write(json.dumps(row, ensure_ascii=False) + "\n")

    for version, path in TARGETS.items():
        shutil.copytree(path, root / "targets" / version, ignore=shutil.ignore_patterns("__pycache__"))

    write(
        root / "native_templates.json",
        {"SYSTEM_TEMPLATE": SYSTEM_TEMPLATE, "INSTANCE_TEMPLATE": INSTANCE_TEMPLATE},
    )
    config = {
        "schema_version": 1,
        "run_dir": str(root),
        "model": str(BASE),
        "image": IMAGE,
        "epoch1_checkpoint": str(epoch1_checkpoint or (PACKAGE_ROOT / "models/update45")),
        "seed": 20260720,
        "tasks": 46,
        "rollouts_per_task": 4,
        "turns": 60,
        "context_tokens": 65536,
        "temperature": 1.0,
        "workers": 64,
        "setup_workers": 8,
        "setup_timeout": 300,
        "request_timeout": 1200,
        "wall_time": 1400,
        "case_timeout": 2420,
        "grade_timeout": 90,
        "container_memory_gib": 8,
        "server_port": 31980,
        "server_tp": 4,
        "server_max_requests": 96,
        "watchdog_threshold": 80,
        "watchdog_interval": 10,
        "target_manifest": {v: files(root / "targets" / v) for v in TARGETS},
        "sampling_plan_sha256": sha(root / "sampling_plan.json"),
    }
    write(root / "config.json", config)
    write(
        root / "status.json",
        {
            "stage": "prepared",
            "branch": "h1",
            "incumbent": None,
            "accepted": [],
            "rejected": [],
            "rounds": [],
        },
    )
    (root / "requests").mkdir()
    return config


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--source-pool", type=Path)
    p.add_argument("--epoch1-checkpoint", type=Path)
    args = p.parse_args()
    print(json.dumps(prepare(args.run_dir, source_pool=args.source_pool, epoch1_checkpoint=args.epoch1_checkpoint), indent=2))
