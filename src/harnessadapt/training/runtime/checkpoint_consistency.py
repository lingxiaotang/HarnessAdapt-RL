#!/usr/bin/env python3
"""Validate that a Slime model checkpoint has matching curriculum progress."""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path


ITERATION_DIR_RE = re.compile(r"iter_(\d+)")
STATE_FILE_RE = re.compile(r"global_dataset_state_dict_(\d+)[.]pt")
LATEST_MARKER = "latest_checkpointed_iteration.txt"


def _read_latest(checkpoint_dir: Path) -> int | None:
    marker = checkpoint_dir / LATEST_MARKER
    try:
        value = marker.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(f"invalid checkpoint marker {marker}: {value!r}") from exc


def _model_iterations(checkpoint_dir: Path) -> dict[int, Path]:
    result: dict[int, Path] = {}
    if not checkpoint_dir.exists():
        return result
    for path in checkpoint_dir.iterdir():
        match = ITERATION_DIR_RE.fullmatch(path.name)
        if match is None or path.is_symlink() or not path.is_dir():
            continue
        # Megatron's distributed checkpoint is not considered usable until
        # both its metadata and common state have landed.
        if not (path / ".metadata").is_file() or not (path / "common.pt").is_file():
            continue
        if not any(path.glob("*.distcp")):
            continue
        result[int(match.group(1))] = path
    return result


def _all_iteration_dirs(checkpoint_dir: Path) -> dict[int, Path]:
    result: dict[int, Path] = {}
    if not checkpoint_dir.exists():
        return result
    for path in checkpoint_dir.iterdir():
        match = ITERATION_DIR_RE.fullmatch(path.name)
        if match is not None and not path.is_symlink() and path.is_dir():
            result[int(match.group(1))] = path
    return result


def _data_iterations(checkpoint_dir: Path) -> dict[int, Path]:
    result: dict[int, Path] = {}
    rollout_dir = checkpoint_dir / "rollout"
    if not rollout_dir.is_dir():
        return result
    for path in rollout_dir.iterdir():
        match = STATE_FILE_RE.fullmatch(path.name)
        if (
            match is None
            or path.is_symlink()
            or not path.is_file()
            or path.stat().st_size <= 0
        ):
            continue
        result[int(match.group(1))] = path
    return result


def _all_data_state_files(checkpoint_dir: Path) -> dict[int, Path]:
    result: dict[int, Path] = {}
    rollout_dir = checkpoint_dir / "rollout"
    if not rollout_dir.is_dir():
        return result
    for path in rollout_dir.iterdir():
        match = STATE_FILE_RE.fullmatch(path.name)
        if match is not None and not path.is_symlink() and path.is_file():
            result[int(match.group(1))] = path
    return result


def _atomic_write_marker(checkpoint_dir: Path, iteration: int) -> None:
    marker = checkpoint_dir / LATEST_MARKER
    temporary = marker.with_suffix(marker.suffix + ".tmp")
    temporary.write_text(f"{iteration}\n", encoding="utf-8")
    os.replace(temporary, marker)


def _quarantine_path(path: Path, quarantine: Path, stamp: str) -> Path:
    quarantine.mkdir(parents=True, exist_ok=True)
    target = quarantine / f"{path.name}.{stamp}"
    suffix = 0
    while target.exists():
        suffix += 1
        target = quarantine / f"{path.name}.{stamp}.{suffix}"
    os.replace(path, target)
    return target


def _unpublished_artifacts(
    checkpoint_dir: Path, selected: int | None
) -> list[tuple[str, Path]]:
    """Return checkpoint artifacts newer than the authoritative marker."""

    result: list[tuple[str, Path]] = []
    for iteration, path in sorted(_all_iteration_dirs(checkpoint_dir).items()):
        if selected is None or iteration > selected:
            result.append(("model", path))
    for iteration, path in sorted(_all_data_state_files(checkpoint_dir).items()):
        if selected is None or iteration > selected:
            result.append(("rollout", path))
    return result


def _quarantine_artifacts(
    checkpoint_dir: Path, artifacts: list[tuple[str, Path]]
) -> list[str]:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    quarantine = checkpoint_dir / ".incomplete"
    quarantined: list[str] = []
    for kind, path in artifacts:
        if path.exists():
            quarantined.append(
                str(_quarantine_path(path, quarantine / kind, stamp))
            )
    return quarantined


def ensure_consistent_checkpoint(
    checkpoint_dir: Path, *, repair: bool
) -> dict[str, object]:
    """Return the selected iteration and optionally roll back an incomplete head."""

    checkpoint_dir = checkpoint_dir.resolve()
    latest = _read_latest(checkpoint_dir)
    if latest is None:
        artifacts = _unpublished_artifacts(checkpoint_dir, None)
        if not artifacts:
            return {"status": "fresh", "latest": None, "selected": None}
        if not repair:
            raise RuntimeError(
                f"checkpoint marker is absent but {len(artifacts)} unpublished "
                f"artifact(s) exist in {checkpoint_dir}"
            )
        quarantined = _quarantine_artifacts(checkpoint_dir, artifacts)
        return {
            "status": "repaired_fresh",
            "latest": None,
            "selected": None,
            "quarantined": quarantined,
        }

    models = _model_iterations(checkpoint_dir)
    data = _data_iterations(checkpoint_dir)
    complete = sorted(set(models).intersection(data))
    if latest in complete:
        # Async checkpoint writers can die after creating the next model
        # directory or data-state file but before publishing the marker.  The
        # current marker is still resumable, yet replaying the next iteration
        # on top of those stale shards is unsafe.
        artifacts = _unpublished_artifacts(checkpoint_dir, latest)
        if not artifacts:
            return {"status": "complete", "latest": latest, "selected": latest}
        if not repair:
            raise RuntimeError(
                f"latest checkpoint {latest} is complete but "
                f"{len(artifacts)} unpublished newer artifact(s) exist"
            )
        quarantined = _quarantine_artifacts(checkpoint_dir, artifacts)
        return {
            "status": "repaired",
            "latest": latest,
            "selected": latest,
            "quarantined": quarantined,
        }

    candidates = [iteration for iteration in complete if iteration < latest]
    if not candidates:
        raise RuntimeError(
            f"latest checkpoint {latest} is missing either model or rollout "
            f"state and no earlier complete checkpoint exists in {checkpoint_dir}"
        )
    selected = candidates[-1]
    if not repair:
        raise RuntimeError(
            f"latest checkpoint {latest} is incomplete; complete fallback "
            f"{selected} exists but repair is disabled"
        )

    # A replay from ``selected`` will eventually save these iteration numbers
    # again. Move stale artifacts out of Megatron's namespace so the new save
    # cannot merge with partial distributed shards.
    quarantined = _quarantine_artifacts(
        checkpoint_dir, _unpublished_artifacts(checkpoint_dir, selected)
    )
    _atomic_write_marker(checkpoint_dir, selected)
    return {
        "status": "repaired",
        "latest": latest,
        "selected": selected,
        "quarantined": quarantined,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--repair", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = ensure_consistent_checkpoint(args.checkpoint_dir, repair=args.repair)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

