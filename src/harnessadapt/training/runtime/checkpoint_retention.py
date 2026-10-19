#!/usr/bin/env python3
"""Prune completed Megatron checkpoints while preserving resumability."""

from __future__ import annotations

import argparse
import re
import shutil
import signal
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path


ITERATION_DIR_RE = re.compile(r"iter_(\d+)")
QUARANTINED_ITERATION_RE = re.compile(r"iter_(\d+)[.].+")
QUARANTINED_STATE_RE = re.compile(r"global_dataset_state_dict_(\d+)[.]pt[.].+")
LATEST_MARKER = "latest_checkpointed_iteration.txt"


def _log(message: str) -> None:
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"{timestamp} checkpoint-retention: {message}", flush=True)


def _read_latest(checkpoint_dir: Path) -> int | None:
    marker = checkpoint_dir / LATEST_MARKER
    try:
        value = marker.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    try:
        return int(value)
    except ValueError:
        _log(f"ignoring invalid marker {marker}: {value!r}")
        return None


def prune_completed(
    checkpoint_dir: Path,
    keep_last: int,
    protect_iterations: set[int] | frozenset[int] = frozenset(),
) -> list[Path]:
    """Delete old completed ``iter_*`` directories and return deleted paths.

    Megatron updates ``latest_checkpointed_iteration.txt`` after a checkpoint
    is complete. Directories newer than that marker may still be in flight and
    are never considered for deletion.
    """

    if keep_last < 1:
        raise ValueError("keep_last must be at least 1")

    checkpoint_dir = checkpoint_dir.resolve()
    latest = _read_latest(checkpoint_dir)
    if latest is None:
        return []

    completed: list[tuple[int, Path]] = []
    try:
        children = list(checkpoint_dir.iterdir())
    except FileNotFoundError:
        return []

    for path in children:
        match = ITERATION_DIR_RE.fullmatch(path.name)
        if match is None or path.is_symlink() or not path.is_dir():
            continue
        iteration = int(match.group(1))
        if iteration <= latest:
            completed.append((iteration, path))

    completed.sort(key=lambda item: item[0])
    deleted: list[Path] = []
    keep_iterations = {
        iteration for iteration, _path in completed[-keep_last:]
    } | set(protect_iterations)
    for iteration, path in completed:
        if iteration in keep_iterations:
            continue
        # Revalidate the exact target immediately before the destructive step.
        if (
            path.parent.resolve() != checkpoint_dir
            or path.is_symlink()
            or ITERATION_DIR_RE.fullmatch(path.name) is None
            or iteration > latest
        ):
            continue
        shutil.rmtree(path)
        deleted.append(path)
        _log(f"deleted completed checkpoint {path}")

    # Startup recovery quarantines an incomplete head instead of deleting it.
    # Reclaim that copy only after replay has produced a complete replacement
    # at the same iteration.
    quarantine = checkpoint_dir / ".incomplete"
    current_states = checkpoint_dir / "rollout"
    for kind, pattern in (
        ("model", QUARANTINED_ITERATION_RE),
        ("rollout", QUARANTINED_STATE_RE),
    ):
        directory = quarantine / kind
        if not directory.is_dir():
            continue
        for path in list(directory.iterdir()):
            match = pattern.fullmatch(path.name)
            if match is None or path.is_symlink():
                continue
            iteration = int(match.group(1))
            replacement_model = checkpoint_dir / f"iter_{iteration:07d}"
            replacement_state = (
                current_states / f"global_dataset_state_dict_{iteration}.pt"
            )
            latest_model = checkpoint_dir / f"iter_{latest:07d}"
            latest_state = (
                current_states / f"global_dataset_state_dict_{latest}.pt"
            )
            replacement_complete = (
                replacement_model.is_dir() and replacement_state.is_file()
            )
            superseded_by_later = (
                latest > iteration
                and latest_model.is_dir()
                and latest_state.is_file()
            )
            if (
                iteration > latest
                or not (replacement_complete or superseded_by_later)
            ):
                continue
            if path.is_dir():
                shutil.rmtree(path)
            elif path.is_file():
                path.unlink()
            else:
                continue
            deleted.append(path)
            _log(f"deleted superseded quarantined checkpoint artifact {path}")
    return deleted


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--keep-last", type=int, required=True)
    parser.add_argument(
        "--protect-iterations",
        type=int,
        nargs="*",
        default=[],
        help="Completed checkpoint iterations to retain in addition to --keep-last.",
    )
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.keep_last < 1:
        parser.error("--keep-last must be at least 1")
    if args.interval <= 0:
        parser.error("--interval must be positive")
    if any(iteration < 0 for iteration in args.protect_iterations):
        parser.error("--protect-iterations must be non-negative")
    args.protect_iterations = set(args.protect_iterations)
    return args


def main() -> int:
    args = _parse_args()
    if args.once:
        prune_completed(
            args.checkpoint_dir,
            args.keep_last,
            args.protect_iterations,
        )
        return 0

    stop = threading.Event()

    def request_stop(_signum: int, _frame: object) -> None:
        stop.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    _log(
        f"watching {args.checkpoint_dir.resolve()}; "
        f"keeping the latest {args.keep_last} completed checkpoint(s), "
        f"protected iterations={sorted(args.protect_iterations)}"
    )
    while not stop.is_set():
        try:
            prune_completed(
                args.checkpoint_dir,
                args.keep_last,
                args.protect_iterations,
            )
        except Exception as exc:  # Keep training alive; surface errors in our log.
            _log(f"prune failed: {type(exc).__name__}: {exc}")
        stop.wait(args.interval)
    return 0


if __name__ == "__main__":
    sys.exit(main())

