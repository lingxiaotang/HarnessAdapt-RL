"""Static rollout runner for skill-free baselines (GRPO, OpenHands-GRPO, Frozen-H1)."""
from __future__ import annotations

import os
from pathlib import Path

from .config import read_json, write_json
from .rollout import phase
from .recipes import validate_recipe


def generate_rollout(args, rollout_id: int, data_source, evaluation: bool = False):
    if evaluation:
        raise RuntimeError("Evaluation uses the standalone evaluation entrypoint")
    run = Path(os.environ["R2E_COEVOLVE_ROOT"])
    config = read_json(run / "config.json")
    return run_static(args, rollout_id, data_source, run, config)


def run_static(args, rollout_id: int, data_source, run: Path, config: dict):
    validate_recipe(config)
    step = rollout_id + 1
    root = run / "steps" / f"{step:04d}"
    root.mkdir(parents=True, exist_ok=True)
    harness = None
    if config.get("fixed_harness_source"):
        from .fixed_harness import validate_release

        harness = run / "initial_harness"
        validate_release(run, harness)
    release = {
        "step": step,
        "harness": str(harness) if harness else "H0",
        "skills": None,
        "tools": read_json(harness / "release.json")["tools"] if harness else [],
        "hooks": read_json(harness / "release.json")["hooks"] if harness else [],
        "teacher_invoked": False,
        "static": True,
        "backend": config.get("backend", "miniswe"),
    }
    if config.get("backend") == "openhands_sdk":
        from harnessadapt.openhands import provenance

        release.update(harness="OpenHands", upstream=provenance())
    write_json(root / "release.json", release)
    write_json(
        root / "acceptance.json",
        {"status": "official_openhands_baseline" if config.get("backend") == "openhands_sdk"
         else ("fixed_harness_no_skills" if harness else "grpo_h0_baseline"), "accepted": False},
    )
    write_json(root / "progress.json", {"stage": "training_rollout", "step": step})
    output, stats = phase(
        args, rollout_id, data_source, config, root / "training", harness, {}, "training", training=True
    )
    write_json(root / "progress.json", {"stage": "awaiting_optimizer", "step": step, "sampling": stats})
    return output
