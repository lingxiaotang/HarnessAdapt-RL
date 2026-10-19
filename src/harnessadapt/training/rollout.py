"""On-policy RL rollout orchestration: preparation, skill generation, tool/hook evolution, and GRPO training.

Implements Section 3.1–3.3 and Appendix B.1–B.3:
- Epoch 1 (updates 1–23): assists all 1,058 tasks; rounds 1–5 evolve H0 -> H1 when enabled.
- Epoch 2 (updates 24–46): assists the fixed 512-task subset with regenerated skills;
  rounds 1–5 (updates 24–28) evolve H1 -> H2 when enabled.
- Candidate harnesses are evaluated on 184 fresh comparison rollouts against the current harness
  and accepted iff neither pass_rate nor trainable_rate decreases and at least one strictly increases.
- Preparation and comparison rollouts are never reused for policy optimization; fresh training
  rollouts are collected under the retained harness and task-specific skills.
"""
from __future__ import annotations

import copy
import gzip
import json
import os
import shutil
import time
from pathlib import Path

from .config import acceptance, read_json, sha256, write_json
from .diagnostics import runtime_errors
from .sampling import PlannedBatch, assisted_ids, planned_ids


def flatten(samples):
    result = []
    for sample in samples:
        result.extend(flatten(sample) if isinstance(sample, (list, tuple)) else [sample])
    return result


def gradient_signal(row: dict, varied: bool) -> tuple[bool, int]:
    """Determine whether a rollout is eligible for policy updates (Appendix B.2).

    Truncated rollouts and rollouts in equal-reward (all-pass or all-fail) groups
    are excluded from trainable_rate.
    """
    signal = row.get("training_signal") or {}
    if signal.get("after_reward_processing"):
        eligible = bool(signal.get("enters_policy_gradient")) and varied
        return eligible, int(signal.get("trainable_token_count") or 0) if eligible else 0
    eligible = bool(signal.get("eligible_before_group_mask")) and varied
    return eligible, int(signal.get("trainable_token_count_before_group_mask") or 0) if eligible else 0


def collection(root: Path, config: dict) -> dict:
    grouped: dict[str, list[dict]] = {}
    for path in sorted((root / "trajectories/train").glob("*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            record = json.load(stream)
        grouped.setdefault(record["instance_id"], []).append(record)
    rows = [row for group in grouped.values() for row in group]
    complete = len(grouped) == config["tasks"] and all(
        len(g) == config["rollouts_per_task"] for g in grouped.values()
    )
    if config.get("task_ids") is not None:
        complete = complete and set(grouped) == set(config["task_ids"])
    if not complete:
        raise ValueError(f"Incomplete phase {root}: {len(rows)} trajectories / {len(grouped)} tasks")
    gradient = passed = submitted = errors = tokens = 0
    hook_errors = tool_exceptions = 0
    for group in grouped.values():
        # Exclude group if any rollout has an unknown reward (Appendix B.3)
        unknown_reward = any(
            not (row.get("outcome") or {}).get("reward_known", True) for row in group
        )
        # Exclude truncated rollouts before checking whether remaining attempts have mixed outcomes (Appendix B.2)
        non_truncated = [
            row
            for row in group
            if not (row.get("outcome") or {}).get("context_budget_exit")
            and not (row.get("outcome") or {}).get("generation_length_exit")
            and not (row.get("outcome") or {}).get("limits_exceeded")
        ]
        rewards = [float(row.get("official_reward") or 0) for row in non_truncated]
        varied = (not unknown_reward) and (
            (max(rewards) - min(rewards) > 1e-12) if len(rewards) >= 2 else False
        )
        for row in group:
            row_hook_errors, row_tool_exceptions = runtime_errors(row)
            hook_errors += row_hook_errors
            tool_exceptions += row_tool_exceptions
            is_truncated = bool(
                (row.get("outcome") or {}).get("context_budget_exit")
                or (row.get("outcome") or {}).get("generation_length_exit")
                or (row.get("outcome") or {}).get("limits_exceeded")
            )
            eligible, token_count = gradient_signal(row, varied and not is_truncated)
            gradient += eligible
            tokens += token_count
            reward_known = bool((row.get("outcome") or {}).get("reward_known", True))
            passed += reward_known and float(row.get("official_reward") or 0) == 1
            submitted += bool(row.get("official_submitted"))
            errors += bool((row.get("outcome") or {}).get("error")) or not reward_known
    n = len(rows)
    return {
        "complete": complete,
        "tasks": len(grouped),
        "task_ids": sorted(grouped),
        "trajectories": n,
        "passed": passed,
        "gradient": gradient,
        "pass_rate": passed / n if n else 0.0,
        "trainable_rate": gradient / n if n else 0.0,
        "submitted": submitted,
        "errors": errors,
        "hook_errors": hook_errors,
        "tool_exceptions": tool_exceptions,
        "effective_tokens": tokens,
    }


def export_markdown(root: Path, config: dict) -> None:
    from .runtime.trajectory_markdown import write_index, write_trajectory_markdown

    out_dir = root / "markdown"
    manifest_path = out_dir / "_manifest.json"
    if manifest_path.exists():
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    entries = []
    for path in sorted((root / "trajectories/train").glob("*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            record = json.load(stream)
        md_path = write_trajectory_markdown(record, out_dir)
        records.append(record)
        messages = record.get("messages") or []
        issue = ""
        for msg in messages:
            if msg.get("role") == "user":
                issue = str(msg.get("content") or "")
                break
        entries.append(
            {
                "source": path.name,
                "source_sha256": sha256(path),
                "markdown": md_path.name,
                "instance_id": record.get("instance_id", ""),
                "session_id": record.get("session_id", ""),
                "official_reward": record.get("official_reward", 0),
                "official_submitted": bool(record.get("official_submitted", False)),
                "exit_status": str((record.get("outcome") or {}).get("exit_status") or ""),
                "issue": issue,
            }
        )
    write_index(records, out_dir)
    write_json(manifest_path, {"records": entries, "trajectory_count": len(entries)})


def phase(
    args,
    rollout_id: int,
    data_source,
    config: dict,
    root: Path,
    harness: Path | None,
    skills: dict[str, str],
    name: str,
    *,
    training: bool = False,
    task_ids: list[str] | None = None,
):
    from slime.rollout.sglang_rollout import generate_rollout as stock
    from .phase_runtime import configure
    from .runtime.reward_processing import post_process_rewards

    ids = planned_ids(config, rollout_id)
    if task_ids is not None:
        ids = list(task_ids)
        args = copy.copy(args)
        args.rollout_batch_size = len(ids)
        args.over_sampling_batch_size = len(ids)
    phase_config = dict(config, tasks=len(ids), task_ids=ids)
    assistance = [x for x in ids if x in set(assisted_ids(config, rollout_id))]
    harness_ids = ids if harness is not None else assistance

    root.mkdir(parents=True, exist_ok=True)
    # Retrying a failed phase must not mix old trajectories with fresh samples.
    # Training always resamples; non-training callers may reuse completed phases.
    for directory in ("trajectories", "markdown"):
        path = root / directory
        if path.exists():
            shutil.rmtree(path)
    (root / "summary.json").unlink(missing_ok=True)
    write_json(
        root / "batch.json",
        {
            "phase": name,
            "step": rollout_id + 1,
            "task_ids": ids,
            "assisted_task_ids": assistance,
            "harness": "OpenHands" if config.get("backend") == "openhands_sdk" else str(harness or "H0"),
            "backend": config.get("backend", "miniswe"),
        },
    )
    started = time.time()
    configure(
        root=root,
        harness=harness,
        skills=skills,
        phase=name,
        step=rollout_id + 1,
        assisted_task_ids=harness_ids,
        native_skills=bool(skills and harness is None),
        backend=config.get("backend", "miniswe"),
    )
    source = PlannedBatch(data_source, ids, rollout_id, config, training=training)
    output = stock(args, rollout_id, source, evaluation=False)
    flat = flatten(output.samples)
    post_process_rewards(args, flat)
    stats = collection(root, phase_config)
    stats["elapsed_seconds"] = time.time() - started
    if not training:
        export_markdown(root, phase_config)
    write_json(root / "summary.json", stats)
    return output, stats


def generate_rollout(args, rollout_id: int, data_source, evaluation: bool = False):
    """Execute one complete update of HarnessAdapt-RL or an ablated configuration."""
    if evaluation:
        raise RuntimeError("Evaluation uses the standalone evaluation entrypoint")
    run = Path(os.environ["R2E_COEVOLVE_ROOT"])
    config = read_json(run / "config.json")
    from .recipes import validate_recipe

    validate_recipe(config)
    pre_evolved = None
    if config.get("pre_evolve_on_base") and not config.get("fixed_harness_source"):
        from .pre_evolution import prepare_harness

        pre_evolved = prepare_harness(args, run, config, rollout_id)

    if not config.get("skill_variant") and not config.get("evolve_epoch1") and not config.get("evolve_epoch2"):
        from .static_rollout import run_static

        return run_static(args, rollout_id, data_source, run, config)

    from slime.rollout.data_source import RolloutDataSource
    from .teacher import propose_harness_change, update_skills

    step = rollout_id + 1
    steps_per_epoch = int(config.get("steps_per_epoch", 23))
    epoch = 1 if step <= steps_per_epoch else 2
    round_in_epoch = step if epoch == 1 else (step - steps_per_epoch)
    max_rounds = int(config.get("evolution_rounds_per_epoch", 5))
    evolve_this_step = (
        (epoch == 1 and bool(config.get("evolve_epoch1")) and round_in_epoch <= max_rounds)
        or (epoch == 2 and bool(config.get("evolve_epoch2")) and round_in_epoch <= max_rounds)
    )

    root = run / "steps" / f"{step:04d}"
    root.mkdir(parents=True, exist_ok=True)

    if step == 1:
        current_harness = pre_evolved or (
            run / "initial_harness"
            if (config.get("fixed_harness_source") or config.get("evolve_epoch1"))
            else None
        )
        old_skills: dict[str, str] = {}
    else:
        prior = read_json(run / "steps" / f"{step - 1:04d}" / "release.json")
        current_harness = Path(prior["harness"]) if prior.get("harness") not in (None, "H0") else None
        if pre_evolved is not None and current_harness != pre_evolved:
            raise ValueError("Pre-Evolved training must retain its frozen H_pre")
        # Skills for the second epoch are regenerated rather than reused from the first (Appendix A.2)
        if step == steps_per_epoch + 1:
            old_skills = {}
        else:
            old_skills = read_json(Path(prior["skills"])) if prior.get("skills") else {}

    ids = planned_ids(config, rollout_id)
    assistance = assisted_ids(config, rollout_id)

    # 1. Preparation rollouts (4 per assisted task: 4,232 in Epoch 1; 2,048 in Epoch 2; Table 6)
    skills: dict[str, str] = {}
    prep_dir = root / "preparation"
    if assistance:
        write_json(root / "progress.json", {"stage": "preparation", "step": step, "assisted_tasks": len(assistance)})
        phase(
            args,
            rollout_id,
            RolloutDataSource(args),
            config,
            prep_dir,
            current_harness,
            {},
            "preparation",
            task_ids=assistance,
        )
        # 2. Skill generation from current-policy preparation trajectories (Section 3.1)
        write_json(root / "progress.json", {"stage": "skills", "step": step})
        skills_result = update_skills(
            root / "teacher",
            prep_dir / "markdown",
            current_harness or (run / "initial_harness"),
            old_skills,
            dict(config, teacher_expected_task_ids=assistance),
        )
        skills = read_json(Path(skills_result["skills"]))
        if skills_result.get("meta_updated"):
            current_harness = Path(skills_result["harness"])

    # 3. Tool & hook evolution and Pareto selection (rounds 1–5 of each epoch; Section 3.2 & 3.3)
    decision_record = {"accepted": False, "status": "harness_fixed"}
    if evolve_this_step and current_harness is not None:
        write_json(root / "progress.json", {"stage": "harness_proposal", "step": step, "epoch": epoch, "round": round_in_epoch})
        previous_proposals = [
            read_json(p) for p in sorted(run.glob("steps/*/acceptance.json"))
        ]
        candidate = propose_harness_change(
            root / "evolution",
            prep_dir / "markdown",
            current_harness,
            previous_proposals,
            config,
            epoch=epoch,
            round_index=round_in_epoch,
        )
        if candidate is not None:
            # Run 184 fresh comparison rollouts per side (46 tasks x 4 rollouts; Appendix B.3)
            stats = {}
            for arm_name, arm_harness in (("current", current_harness), ("candidate", candidate)):
                target = root / "comparison" / arm_name
                write_json(root / "progress.json", {"stage": "harness_comparison", "arm": arm_name, "step": step})
                _, stats[arm_name] = phase(
                    args,
                    rollout_id,
                    RolloutDataSource(args),
                    config,
                    target,
                    arm_harness,
                    {t: skills.get(t, "") for t in assistance},
                    f"comparison_{arm_name}",
                )
            decision_record = acceptance(stats["current"], stats["candidate"])
            if decision_record["accepted"]:
                current_harness = candidate
        else:
            decision_record = {"accepted": False, "status": "no_change"}

    write_json(root / "acceptance.json", decision_record)
    release = {
        "step": step,
        "epoch": epoch,
        "round_in_epoch": round_in_epoch,
        "harness": str(current_harness) if current_harness else "H0",
        "skills": str(root / "teacher/task_skills.json") if assistance else None,
        "assisted_task_ids": assistance,
        "harness_accepted": decision_record["accepted"],
    }
    write_json(root / "release.json", release)

    # 4. Fresh training rollouts under the selected harness and skills for GRPO update (Section 3.3)
    write_json(root / "progress.json", {"stage": "training_rollout", "step": step})
    output, train_stats = phase(
        args,
        rollout_id,
        data_source,
        config,
        root / "training",
        current_harness,
        {t: skills.get(t, "") for t in assistance},
        "training",
        training=True,
    )
    write_json(root / "progress.json", {"stage": "awaiting_optimizer", "step": step, "sampling": train_stats})
    return output
