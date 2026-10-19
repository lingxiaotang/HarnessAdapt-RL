"""Periodic 5-update checkpoint saving and SWE-bench Verified evaluation barriers (Appendix B.4)."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from .config import inside, read_json, write_json
from .evaluation_paths import evaluation_directory, input_directory


def evaluation_steps(total_steps: int = 46, *, interval: int = 5, completed_step: int = 0, extra_steps=()):
    """Return periodic checkpoint/evaluation barriers (updates 5, 10, ..., 45)."""
    if interval <= 0 or total_steps < 0 or completed_step < 0:
        raise ValueError("Invalid checkpoint/evaluation schedule")
    barriers = list(range(interval, total_steps + 1, interval))
    return [
        step for step in sorted(set(barriers) | set(extra_steps))
        if completed_step < step <= total_steps
    ]


def require_optimizer_complete(root: Path, step: int) -> None:
    record = read_json(root / "steps" / f"{step:04d}" / "optimizer_complete.json")
    if record.get("step") != step or not all(
        record.get(key)
        for key in (
            "optimizer_call_completed",
            "inference_weights_synchronized",
            "nonzero_parameter_update_observed",
        )
    ):
        raise RuntimeError(f"Optimizer update {step} has not completed successfully")


def check_checkpoint(directory: Path, step: int, *, require_latest: bool = True) -> Path:
    iteration = step - 1
    pointer = directory / "latest_checkpointed_iteration.txt"
    latest = int(pointer.read_text(encoding="utf-8").strip())
    if require_latest and latest != iteration:
        raise ValueError(f"Checkpoint pointer does not match completed step {step}: {pointer}")
    if not require_latest and latest < iteration:
        raise ValueError(f"Checkpoint step {step} is newer than the latest checkpoint: {pointer}")
    checkpoint = directory / f"iter_{iteration:07d}"
    required = [
        checkpoint / "common.pt",
        checkpoint / "metadata.json",
        directory / "rollout" / f"global_dataset_state_dict_{iteration}.pt",
    ]
    shards = list(checkpoint.glob("*.distcp"))
    if not shards or any(not p.is_file() or p.stat().st_size == 0 for p in [*required, *shards]):
        raise ValueError(f"Incomplete checkpoint/data-source state: {checkpoint}")
    return checkpoint


def evaluation_summary(root: Path, step: int, *, benchmark: str = "verified", repetition: int = 1) -> dict:
    directory = evaluation_directory(root, benchmark, step, repetition)
    records = []
    for path in sorted((directory / "trajectories").rglob("*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            records.append(json.load(stream))
    expected = {
        json.loads(line)["label"]
        for line in (input_directory(root, benchmark) / "verified.jsonl").read_text(encoding="utf-8").splitlines()
    }
    ids = [r.get("instance_id") for r in records]
    complete = len(ids) == len(expected) and len(set(ids)) == len(expected) and set(ids) == expected
    passed = sum(r.get("official_reward") == 1 for r in records)
    result = {
        "step": step,
        "benchmark": benchmark,
        "repetition": repetition,
        "complete": complete,
        "records": len(records),
        "passed": passed,
        "pass_at_1": passed / len(expected) if expected else 0.0,
        "submitted": sum(bool(r.get("official_submitted")) for r in records),
    }
    write_json(directory / "summary.json", result)
    if not complete:
        raise RuntimeError(f"Incomplete {benchmark} evaluation at step {step}: {result}")
    return result


def run(root: Path, stop_after: int) -> None:
    from .__main__ import launch, preflight
    from .verified import prepare_verified

    config = preflight(root)
    if not 1 <= stop_after <= config["total_steps"]:
        raise ValueError("Requested stop must be within the training schedule")
    prepare_verified(root)
    previous = config.get("resume_from_step", 0)
    if not 0 <= previous <= stop_after:
        raise ValueError("Resume step must not exceed the requested stop")
    barriers = evaluation_steps(
        stop_after,
        interval=config.get("save_interval", 5),
        completed_step=previous,
        extra_steps=config.get("extra_evaluation_steps", ()),
    )
    for step in barriers:
        step_directory = evaluation_directory(root, "verified", step).parent
        completed = [p for p in step_directory.glob("run_*/summary.json") if read_json(p).get("complete")]
        if completed:
            require_optimizer_complete(root, step)
            previous = step
            continue
        if not (root / "steps" / f"{step:04d}" / "optimizer_complete.json").exists():
            write_json(
                root / "continuation_status.json",
                {"stage": "training", "from_step": previous, "target_step": step},
            )
            launch(root, config, stop_after=step)
        require_optimizer_complete(root, step)
        check_checkpoint(root / "checkpoints", step)
        write_json(root / "continuation_status.json", {"stage": "evaluation", "step": step})
        # Preserve failed attempts on resume and allocate a fresh repetition.
        repetitions = [int(p.name.removeprefix("run_")) for p in step_directory.glob("run_*")
                       if p.is_dir() and p.name.removeprefix("run_").isdigit()]
        repetition = max(repetitions, default=0) + 1
        launch(root, config, stop_after=step, evaluation=True, repetition=repetition)
        evaluation_summary(root, step, repetition=repetition)
        previous = step
    # Periodic evaluation stops at update 45, but the optimizer schedule ends at
    # update 46. Run the tail without adding an off-schedule evaluation.
    if previous < stop_after:
        marker = root / "steps" / f"{stop_after:04d}" / "optimizer_complete.json"
        if not marker.exists():
            write_json(
                root / "continuation_status.json",
                {"stage": "training", "from_step": previous, "target_step": stop_after},
            )
            launch(root, config, stop_after=stop_after)
    require_optimizer_complete(root, stop_after)
    write_json(root / "continuation_status.json", {"stage": "complete", "completed_step": stop_after})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--stop-after", type=int, default=46)
    args = parser.parse_args()
    run(inside(args.run_dir), args.stop_after)
