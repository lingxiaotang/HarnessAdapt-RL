"""Discover and freeze H_pre before the first policy optimizer update."""
from __future__ import annotations

from pathlib import Path

from .config import acceptance, read_json, write_json
from .fixed_harness import hashes


def frozen_harness(run: Path) -> Path:
    manifest = read_json(run / "pre_evolution" / "release.json")
    harness = Path(manifest["harness"])
    if hashes(harness) != manifest["harness_hashes"]:
        raise ValueError("Frozen pre-evolved harness changed")
    return harness


def prepare_harness(args, run: Path, config: dict, rollout_id: int) -> Path:
    """Use fresh preparation/comparison data while leaving training cursors alone.

    Called inside the first rollout request: Slime has synchronized the base
    policy but has not yet called its optimizer. All discovery rounds therefore
    use the same base weights. Subsequent updates only verify the frozen bundle.
    """
    root = run / "pre_evolution"
    if (root / "release.json").exists():
        return frozen_harness(run)
    if rollout_id != 0 or any(run.glob("steps/*/optimizer_complete.json")):
        raise RuntimeError("Cannot discover H_pre after policy optimization has started")

    from slime.rollout.data_source import RolloutDataSource
    from .rollout import phase
    from .sampling import planned_ids
    from .teacher import propose_harness_change, update_skills

    rounds = int(config.get("pre_evolution_rounds", config["evolution_rounds_per_epoch"]))
    if not 1 <= rounds <= min(5, config["steps_per_epoch"]):
        raise ValueError("Pre-evolution requires between one and five base-policy rounds")
    current = run / "initial_harness"
    history = []
    for index in range(rounds):
        round_root = root / "rounds" / f"{index + 1:04d}"
        result_path = round_root / "result.json"
        if result_path.exists():
            result = read_json(result_path)
            if result["parent_hashes"] != hashes(current):
                raise ValueError("Pre-evolution parent changed while resuming")
            current = Path(result["harness"])
            if hashes(current) != result["harness_hashes"]:
                raise ValueError("Pre-evolution candidate changed while resuming")
            history.append(result["decision"])
            continue

        ids = planned_ids(config, index)
        parent_hashes = hashes(current)
        preparation = round_root / "preparation"
        # A completed phase is immutable on resume; incomplete directories are
        # handled by phase(), which discards their partial sampling artifacts.
        if not (preparation / "summary.json").exists():
            phase(
                args, index, RolloutDataSource(args), config,
                preparation, current, {}, "pre_evolution_preparation",
            )
        skills_result = update_skills(
            round_root / "teacher", preparation / "markdown", current, {},
            dict(config, teacher_expected_task_ids=ids),
        )
        skills = read_json(Path(skills_result["skills"]))
        candidate = propose_harness_change(
            round_root / "proposal", preparation / "markdown", current,
            history, config, epoch=0, round_index=index + 1,
        )
        decision = {"accepted": False, "status": "no_change"}
        if candidate is not None:
            stats = {}
            for name, harness in (("current", current), ("candidate", candidate)):
                output = round_root / "comparison" / name
                summary = output / "summary.json"
                if summary.exists():
                    stats[name] = read_json(summary)
                else:
                    _, stats[name] = phase(
                        args, index, RolloutDataSource(args), config,
                        output, harness, skills, f"pre_evolution_{name}",
                    )
            decision = acceptance(stats["current"], stats["candidate"])
            if decision["accepted"]:
                current = candidate
        write_json(result_path, {
            "parent_hashes": parent_hashes,
            "harness": str(current),
            "harness_hashes": hashes(current),
            "decision": decision,
        })
        history.append(decision)

    write_json(root / "release.json", {
        "harness": str(current),
        "harness_hashes": hashes(current),
        "base_model": config["model"],
        "rounds": rounds,
        "optimizer_updates": 0,
        "decisions": history,
    })
    return frozen_harness(run)
