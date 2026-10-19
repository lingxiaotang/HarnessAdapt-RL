"""Materialize and validate the 1,058-task R2E-Gym-Lite training pool (Appendix A.1 & Table 3)."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from .config import DEFAULT_INPUT, DEFAULT_POOL_METADATA, sha256, write_json


def read_pool(path: Path, *, require_manifest: bool = False) -> list[dict]:
    manifest_path = path.with_suffix(".manifest.json")
    if require_manifest and not manifest_path.is_file():
        raise ValueError(f"Missing task-pool completion manifest: {manifest_path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            sha256(path) != manifest["input_sha256"]
            or len(rows) != manifest["task_count"]
            or len({r["label"] for r in rows}) != manifest["task_count"]
        ):
            raise ValueError(f"Task pool differs from completion manifest: {path}")
    return rows


def publish_pool(output: Path, records: list[dict], manifest: dict) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.with_suffix(".manifest.json").exists():
        raise FileExistsError(output)
    fd, name = tempfile.mkstemp(prefix=output.name + ".", suffix=".tmp", dir=output.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            for row in records:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        manifest = dict(manifest, input_sha256=sha256(temporary), task_count=len(records))
        os.link(temporary, output)
        write_json(output.with_suffix(".manifest.json"), manifest)
    finally:
        temporary.unlink(missing_ok=True)


def build_cached_pool(output: Path = DEFAULT_INPUT, *, cache: Path | None = None) -> Path:
    """Build the published task pool, downloading the dataset if no cache exists."""
    pool_meta = json.loads(DEFAULT_POOL_METADATA.read_text(encoding="utf-8"))
    target_ids = set(pool_meta["task_ids"])
    assisted_ids = set(pool_meta["epoch2_assisted_task_ids"])
    cache = Path(cache or Path.home() / ".cache/huggingface/datasets/R2E-Gym___r2_e-gym-lite")
    shards = sorted(cache.rglob("*-train-*.arrow"))
    revision = os.environ.get("HARNESSADAPT_TRAIN_DATASET_REVISION")

    def source_rows():
        if shards:
            try:
                import pyarrow.ipc as ipc
            except ImportError as exc:
                raise RuntimeError("Install requirements-data.txt to prepare training data") from exc
            for path in shards:
                with path.open("rb") as stream:
                    yield from ipc.open_stream(stream).read_all().to_pylist()
        else:
            try:
                from datasets import load_dataset
            except ImportError as exc:
                raise RuntimeError(
                    "No cached training data found. Install requirements-data.txt "
                    "to download R2E-Gym/R2E-Gym-Lite, or provide --source."
                ) from exc
            yield from load_dataset("R2E-Gym/R2E-Gym-Lite", split="train", revision=revision)

    records_by_id = {}
    total = 0
    for raw in source_rows():
        total += 1
        label = f"{raw['repo_name'].replace('/', '_')}__{raw['commit_hash'][:12]}"
        if label not in target_ids:
            continue
        record = {
            "prompt": [{"role": "user", "content": raw["problem_statement"].strip()}],
            "label": label,
            "metadata": {
                "data_source": "R2E-Gym/R2E-Gym-Lite",
                "instance": dict(raw, instance_id=label),
                "curriculum": {
                    "instance_id": label,
                    "medium_hint": "",
                    "stage2_use_hint": label in assisted_ids,
                },
            },
        }
        if label in records_by_id and records_by_id[label] != record:
            raise ValueError(f"Conflicting dataset revisions for task {label}")
        records_by_id[label] = record
    if len(records_by_id) != 1058:
        raise ValueError(f"Expected 1,058 pool tasks, found {len(records_by_id)}")
    ordered_records = [records_by_id[tid] for tid in pool_meta["task_ids"]]
    publish_pool(
        output,
        ordered_records,
        {
            "dataset": "R2E-Gym/R2E-Gym-Lite",
            "split": "train",
            "dataset_size": total,
            "selection": "1,058-task training pool matching Table 3 of the paper",
            "task_count": len(ordered_records),
            "source_shards": [str(p) for p in shards],
            "requested_revision": revision if not shards else None,
        },
    )
    return output
