"""Aggregate Mini-SWE/R2E rollout health into Slime's normal metric stream."""

from __future__ import annotations

import json
import logging
import math
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Iterable

from slime.utils.types import Sample

logger = logging.getLogger(__name__)

def _stage_eval_context(rollout_id: int, args: Any) -> dict[str, Any] | None:
    """Describe the configured curriculum snapshot for this evaluation."""

    update = rollout_id + 1
    plan = os.environ.get("R2E_STAGE_EVAL_PLAN", "")
    label = ""
    stage_index = -1
    for index, item in enumerate(plan.split(",")):
        raw_update, separator, raw_label = item.strip().partition(":")
        if not separator:
            continue
        try:
            planned_update = int(raw_update)
        except ValueError:
            continue
        if planned_update == update:
            label = _metric_component(raw_label)
            stage_index = index
            break
    if not label or stage_index < 0:
        return None

    from .curriculum import NUM_TRAIN_CASES, NUM_STAGES

    prompts_per_update = int(getattr(args, "rollout_batch_size", 0) or 0)
    prompts_seen = max(0, update * prompts_per_update)
    stage_prompt_counts = []
    for stage in range(NUM_STAGES):
        stage_start = stage * NUM_TRAIN_CASES
        stage_prompt_counts.append(
            min(NUM_TRAIN_CASES, max(0, prompts_seen - stage_start))
        )
    return {
        "label": label,
        "stage_index": stage_index,
        "training_hint_mode": os.environ.get(
            "R2E_HINT_MODE", "skills"
        ),
        "update": update,
        "rollout_id": rollout_id,
        "prompts_per_update": prompts_per_update,
        "training_prompts_seen": prompts_seen,
        "stage_prompt_counts": {
            "epoch_1_full_assisted": stage_prompt_counts[0] if len(stage_prompt_counts) > 0 else 0,
            "epoch_2_subset_assisted": stage_prompt_counts[1] if len(stage_prompt_counts) > 1 else 0,
        },
        "target_stage_complete": (
            stage_index < len(stage_prompt_counts)
            and stage_prompt_counts[stage_index] == NUM_TRAIN_CASES
        ),
        "later_stage_prompt_exposure": sum(
            stage_prompt_counts[stage_index + 1 :]
        ),
    }


def _write_stage_eval_summary(
    context: dict[str, Any],
    metrics: dict[str, float],
    *,
    sample_count: int,
) -> str:
    """Atomically save one human-readable summary per curriculum stage."""

    raw_dir = os.environ.get("R2E_STAGE_EVAL_DIR", "").strip()
    if not raw_dir:
        return ""
    target_dir = Path(raw_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / (
        f"{context['stage_index'] + 1}_{context['label']}"
        f"_update_{context['update']:03d}.json"
    )
    temporary = target.with_suffix(target.suffix + f".{os.getpid()}.tmp")
    payload = {
        "format_version": 1,
        "timestamp_utc": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "evaluation_protocol": "swe-bench_verified_fresh_container_pass_at_1",
        "evaluation_hint_used": False,
        "sample_count": sample_count,
        **context,
        "metrics": metrics,
    }
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, target)
    return str(target)


def _flatten_samples(values: Iterable[Any]) -> list[Sample]:
    flattened: list[Sample] = []
    for value in values:
        if isinstance(value, Sample):
            flattened.append(value)
        elif isinstance(value, (list, tuple)):
            flattened.extend(_flatten_samples(value))
    return flattened


def _metadata(sample: Sample) -> dict[str, Any]:
    return sample.metadata if isinstance(sample.metadata, dict) else {}


def _reward(sample: Sample) -> float:
    value = sample.reward
    if isinstance(value, dict):
        for key in ("reward", "score", "value"):
            if key in value:
                value = value[key]
                break
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _flag(sample: Sample, key: str) -> bool:
    return bool(_metadata(sample).get(key))


def _number(sample: Sample, key: str, default: float = 0.0) -> float:
    try:
        return float(_metadata(sample).get(key, default) or 0.0)
    except (TypeError, ValueError):
        return float(default)


def _trainable(sample: Sample) -> bool:
    return not sample.remove_sample and bool(
        sample.loss_mask and any(sample.loss_mask)
    )


def _rate(samples: list[Sample], predicate) -> float:
    return (
        sum(bool(predicate(sample)) for sample in samples) / len(samples)
        if samples
        else 0.0
    )


def _grouped(samples: list[Sample]) -> dict[str, list[Sample]]:
    groups: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        key = (
            f"group:{sample.group_index}"
            if sample.group_index is not None
            else f"sample:{sample.index}"
        )
        groups[key].append(sample)
    return groups


def _metric_component(value: Any) -> str:
    component = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower())
    return component.strip("_")[:80] or "unknown"


def _group_advantages(args: Any, samples: list[Sample]) -> dict[int, float]:
    """Mirror Slime's current GRPO reward normalization for diagnostics."""
    result: dict[int, float] = {}
    for group in _grouped(samples).values():
        rewards = [_reward(sample) for sample in group]
        if getattr(args, "rewards_normalization", True):
            mean = fmean(rewards)
            centered = [reward - mean for reward in rewards]
            if (
                getattr(args, "advantage_estimator", "grpo")
                in {"grpo", "gspo", "cispo"}
                and getattr(args, "grpo_std_normalization", True)
            ):
                # torch.std defaults to the unbiased estimator. A singleton
                # group is not expected here but remains finite for monitoring.
                if len(centered) > 1:
                    variance = sum(value * value for value in centered) / (
                        len(centered) - 1
                    )
                    std = math.sqrt(variance)
                else:
                    std = 0.0
                centered = [value / (std + 1e-6) for value in centered]
            advantages = centered
        else:
            advantages = rewards
        for sample, advantage in zip(group, advantages, strict=True):
            result[id(sample)] = float(advantage)
    return result


def aggregate_r2e_metrics(
    args: Any,
    values: Iterable[Any],
    *,
    is_training: bool = True,
) -> dict[str, float]:
    samples = _flatten_samples(values)
    if not samples:
        return {}

    physical_samples = samples
    if any((_metadata(sample)).get('recovery_segments') for sample in samples):
        from .recovery_segments import metric_views
        samples = metric_views(samples)

    rewards = [_reward(sample) for sample in samples]
    lengths = [
        int(
            _number(
                sample,
                "trajectory_response_tokens",
                int(sample.response_length or 0),
            )
        )
        for sample in samples
    ]
    hinted = [sample for sample in samples if _flag(sample, "hint_used")]
    unhinted = [sample for sample in samples if not _flag(sample, "hint_used")]
    groups = _grouped(samples)

    metrics: dict[str, float] = {
        "generate/batch_num_seq": float(len(samples)),
        "generate/min_num_tokens": float(min(lengths)),
        "generate/max_num_tokens": float(max(lengths)),
        "generate/avg_num_tokens": fmean(lengths),
        "generate/std_num_tokens": pstdev(lengths) if len(lengths) > 1 else 0.0,
        "generate/avg_tokens_non_zero_rewards": (
            fmean(
                length
                for length, reward in zip(lengths, rewards, strict=True)
                if reward > 0
            )
            if any(reward > 0 for reward in rewards)
            else 0.0
        ),
        "generate/avg_tokens_zero_rewards": (
            fmean(
                length
                for length, reward in zip(lengths, rewards, strict=True)
                if reward == 0
            )
            if any(reward == 0 for reward in rewards)
            else 0.0
        ),
        "reward/avg_raw_reward": fmean(rewards),
        "reward/mean_positive_reward": fmean(max(reward, 0.0) for reward in rewards),
        # Compatibility alias for the old SkyRL dashboard. Slime also emits
        # rollout/response_len/mean from the effective loss mask.
        "policy/response_length": fmean(lengths),
        "rollout/error_rate": _rate(
            samples,
            lambda sample: (
                bool(_metadata(sample).get("rollout_error"))
                and not _flag(sample, "external_stop")
                and not _flag(sample, "timed_out")
                and not _flag(sample, "repeated_format_error")
            )
            or not bool(_metadata(sample).get("reward_known", True)),
        ),
        "rollout/truncated_rate": _rate(
            samples,
            lambda sample: sample.status == Sample.Status.TRUNCATED
            or _flag(sample, "generation_length_exit"),
        ),
        "rollout/case_timeout_rate": _rate(
            samples, lambda sample: _flag(sample, "timed_out")
        ),
        "rollout/placeholder_rate": _rate(
            samples,
            lambda sample: sample.remove_sample
            or bool(_metadata(sample).get("masked_reason")),
        ),
        "rollout/external_stop_masked_rate": _rate(
            samples,
            lambda sample: _flag(sample, "external_stop") and not _trainable(sample),
        ),
        "rollout/trainable_rate": _rate(samples, _trainable),
        "rollout/reward_preserved_placeholder_rate": _rate(
            samples,
            lambda sample: not _trainable(sample)
            and bool(_metadata(sample).get("reward_known")),
        ),
        "rollout/submitted_rate": _rate(
            samples, lambda sample: _flag(sample, "submitted")
        ),
        "rollout/not_submitted_masked_rate": _rate(
            samples,
            lambda sample: not _flag(sample, "submitted")
            and not _trainable(sample),
        ),
        "rollout/patch_available_rate": _rate(
            samples, lambda sample: _flag(sample, "patch_available")
        ),
        "rollout/reward_unknown_rate": _rate(
            samples,
            lambda sample: not bool(
                _metadata(sample).get("reward_known", False)
            ),
        ),
        "rollout/adapter_sample_count_invalid_rate": _rate(
            samples,
            lambda sample: int(_number(sample, "adapter_sample_count", 0)) != 1,
        ),
    }

    # Throughput counts physical sequences; rewards/eligibility count rollouts.
    metrics['generate/training_segment_count'] = float(len(physical_samples))
    metrics['rollout/adapter_sample_count_invalid_rate'] = _rate(
        samples, lambda s: int(_number(s, 'adapter_sample_count', 0)) < 1
        or (int(_number(s, 'adapter_sample_count', 0)) != 1
            and not _metadata(s).get('recovery_segments')))
    full_forward_tokens = sum(len(sample.tokens or []) for sample in physical_samples)
    trainable_samples = [sample for sample in physical_samples if _trainable(sample)]
    compacted_forward_tokens = sum(
        len(sample.tokens or []) for sample in trainable_samples
    )
    metrics.update(
        {
            "rollout/forward_sample_count": float(len(trainable_samples)),
            "rollout/dropped_zero_loss_sample_count": float(
                len(physical_samples) - len(trainable_samples)
            ),
            "rollout/forward_sample_rate": len(trainable_samples) / len(physical_samples),
            "perf/forward_input_token_count": float(compacted_forward_tokens),
            "perf/dropped_forward_input_token_count": float(
                full_forward_tokens - compacted_forward_tokens
            ),
            "perf/forward_input_token_fraction": (
                compacted_forward_tokens / full_forward_tokens
                if full_forward_tokens
                else 0.0
            ),
        }
    )

    timeout_samples = [
        sample for sample in samples if _flag(sample, "timed_out")
    ]
    metrics["rollout/timeout_with_patch_rate"] = (
        sum(_flag(sample, "patch_available") for sample in timeout_samples)
        / len(timeout_samples)
        if timeout_samples
        else 0.0
    )
    metrics["rollout/timeout_patch_capture_error_rate"] = (
        sum(
            bool(_metadata(sample).get("unsubmitted_patch_error"))
            for sample in timeout_samples
        )
        / len(timeout_samples)
        if timeout_samples
        else 0.0
    )
    unsubmitted_samples = [
        sample for sample in samples if not _flag(sample, "submitted")
    ]
    unsubmitted_with_patch = [
        sample
        for sample in unsubmitted_samples
        if _flag(sample, "unsubmitted_worktree_patch_available")
    ]
    unsubmitted_graded = [
        sample
        for sample in unsubmitted_with_patch
        if _flag(sample, "unsubmitted_patch_reward_known")
    ]
    counterfactual_passes = sum(
        _number(sample, "unsubmitted_patch_reward") > 0
        for sample in unsubmitted_graded
    )
    metrics.update(
        {
            "rollout/unsubmitted_worktree_patch_rate": _rate(
                samples,
                lambda sample: _flag(
                    sample, "unsubmitted_worktree_patch_available"
                ),
            ),
            "rollout/unsubmitted_worktree_patch_given_unsubmitted_rate": (
                len(unsubmitted_with_patch) / len(unsubmitted_samples)
                if unsubmitted_samples
                else 0.0
            ),
            "rollout/unsubmitted_patch_grade_attempt_rate": _rate(
                samples,
                lambda sample: _flag(
                    sample, "unsubmitted_patch_grade_attempted"
                ),
            ),
            "rollout/unsubmitted_patch_grade_known_rate": (
                len(unsubmitted_graded) / len(unsubmitted_with_patch)
                if unsubmitted_with_patch
                else 0.0
            ),
            "rollout/unsubmitted_patch_verified_pass_rate": (
                counterfactual_passes / len(unsubmitted_graded)
                if unsubmitted_graded
                else 0.0
            ),
            "rollout/unsubmitted_patch_apply_failed_rate": _rate(
                samples,
                lambda sample: _flag(
                    sample, "unsubmitted_patch_apply_failed"
                ),
            ),
            "rollout/unsubmitted_patch_grade_timeout_rate": _rate(
                samples,
                lambda sample: _flag(
                    sample, "unsubmitted_patch_grade_timed_out"
                ),
            ),
            # Diagnostic upper bound only: this is what pass@1 would be if a
            # cleanly applicable worktree diff counted as submission.
            "eval/reward_if_worktree_patch_accepted": (
                (
                    sum(rewards)
                    + counterfactual_passes
                )
                / len(samples)
            ),
            "eval/unsubmitted_worktree_patch_pass_count": float(
                counterfactual_passes
            ),
        }
    )

    diagnostic_keys = (
        "token_ids_missing",
        "logprobs_missing",
        "text_diverge",
        "token_retokenized",
        "tool_parser_normalized",
        "format_error_recovered",
        "format_error_history_preserved",
        "format_error_recovery",
        "format_error_arguments_normalized",
        "sampled_without_eos",
        "zero_variance_group_masked",
        "adapter_fork",
        "weight_version_missing",
        "weight_version_mixed",
        "context_budget_exit",
        "generation_length_exit",
        "model_request_timeout",
        "agent_time_exceeded",
        "limits_exceeded",
        "repeated_format_error",
        "external_stop",
        "verified_fresh_container",
        "verified_patch_apply_failed",
        "unsubmitted_worktree_patch_available",
        "unsubmitted_patch_grade_attempted",
        "unsubmitted_patch_reward_known",
        "unsubmitted_patch_apply_failed",
        "unsubmitted_patch_grade_timed_out",
    )
    for key in diagnostic_keys:
        metrics[f"rollout/{key}_rate"] = _rate(
            samples, lambda sample, metric_key=key: _flag(sample, metric_key)
        )
    observed_weight_versions = [
        int(_number(sample, "weight_version_min", -1))
        for sample in samples
        if int(_number(sample, "weight_version_min", -1)) >= 0
    ] + [
        int(_number(sample, "weight_version_max", -1))
        for sample in samples
        if int(_number(sample, "weight_version_max", -1)) >= 0
    ]
    metrics["rollout/weight_version_min"] = float(
        min(observed_weight_versions) if observed_weight_versions else -1
    )
    metrics["rollout/weight_version_max"] = float(
        max(observed_weight_versions) if observed_weight_versions else -1
    )
    masked_reason_counts: dict[str, int] = defaultdict(int)
    exit_status_counts: dict[str, int] = defaultdict(int)
    for sample in samples:
        metadata = _metadata(sample)
        if reason := metadata.get("masked_reason"):
            masked_reason_counts[_metric_component(reason)] += 1
        if status := metadata.get("exit_status"):
            exit_status_counts[_metric_component(status)] += 1
    for reason, count in masked_reason_counts.items():
        metrics[f"rollout/masked_reason/{reason}_rate"] = count / len(samples)
    for status, count in exit_status_counts.items():
        metrics[f"rollout/exit_status/{status}_rate"] = count / len(samples)
        status_samples = [
            sample
            for sample in samples
            if _metric_component(_metadata(sample).get("exit_status")) == status
        ]
        metrics[f"rollout/exit_status/{status}_reward_mean"] = fmean(
            _reward(sample) for sample in status_samples
        )
        metrics[f"rollout/exit_status/{status}_agent_calls_mean"] = fmean(
            _number(sample, "agent_calls") for sample in status_samples
        )
        metrics[f"rollout/exit_status/{status}_tokens_mean"] = fmean(
            _number(sample, "trajectory_response_tokens") for sample in status_samples
        )

    metrics.update(
        {
            "rollout/agent_calls_mean": fmean(
                _number(sample, "agent_calls") for sample in samples
            ),
            "rollout/case_elapsed_seconds_mean": fmean(
                _number(sample, "case_elapsed_seconds") for sample in samples
            ),
            "rollout/case_elapsed_seconds_max": max(
                _number(sample, "case_elapsed_seconds") for sample in samples
            ),
            "rollout/container_start_wait_seconds_mean": fmean(
                _number(sample, "container_start_wait_seconds")
                for sample in samples
            ),
            "rollout/container_start_wait_seconds_max": max(
                _number(sample, "container_start_wait_seconds")
                for sample in samples
            ),
            "rollout/format_error_recovered_turns_mean": fmean(
                _number(sample, "format_error_recovered_turns")
                for sample in samples
            ),
            "rollout/format_error_recovery_turns_mean": fmean(
                _number(sample, "format_error_recovery_turns")
                for sample in samples
            ),
            "rollout/sampled_without_eos_turns_mean": fmean(
                _number(sample, "sampled_without_eos_turns")
                for sample in samples
            ),
            "rollout/bad_json_response_rate": _rate(
                samples, lambda sample: _number(sample, "bad_json_turns") > 0
            ),
            "rollout/no_tool_response_rate": _rate(
                samples, lambda sample: _number(sample, "no_tool_turns") > 0
            ),
            "rollout/bad_json_tokens_mean": fmean(
                _number(sample, "bad_json_tokens") for sample in samples
            ),
            "rollout/no_tool_tokens_mean": fmean(
                _number(sample, "no_tool_tokens") for sample in samples
            ),
        }
    )
    assistant_turns = sum(_number(sample, "assistant_turns") for sample in samples)
    history_assistant_turns = sum(
        _number(sample, "history_assistant_turns") for sample in samples
    )
    history_reasoning_turns = sum(
        _number(sample, "history_reasoning_turns") for sample in samples
    )
    format_error_turns = sum(
        _number(sample, "bad_json_turns")
        + _number(sample, "no_tool_turns")
        for sample in samples
    )
    transition_pairs = sum(
        _number(sample, "format_transition_pairs") for sample in samples
    )
    previous_error_turns = sum(
        _number(sample, "previous_error_turns") for sample in samples
    )
    error_after_error_turns = sum(
        _number(sample, "error_after_error_turns") for sample in samples
    )
    previous_ok_turns = sum(
        _number(sample, "previous_ok_turns") for sample in samples
    )
    error_after_ok_turns = sum(
        _number(sample, "error_after_ok_turns") for sample in samples
    )
    recovery_turns = sum(
        _number(sample, "format_error_recovery_turns") for sample in samples
    )
    streak_count = sum(
        _number(sample, "format_error_streak_count") for sample in samples
    )
    trajectory_max_streaks = [
        _number(sample, "format_error_max_streak") for sample in samples
    ]
    terminal_streaks = [
        _number(sample, "format_error_terminal_streak") for sample in samples
    ]
    error_after_error_rate = (
        error_after_error_turns / previous_error_turns
        if previous_error_turns
        else 0.0
    )
    error_after_ok_rate = (
        error_after_ok_turns / previous_ok_turns if previous_ok_turns else 0.0
    )
    metrics.update(
        {
            "format/assistant_turn_count": assistant_turns,
            "format/history_assistant_turn_count": history_assistant_turns,
            "format/history_reasoning_turn_count": history_reasoning_turns,
            "format/history_reasoning_retention_rate": (
                history_reasoning_turns / history_assistant_turns
                if history_assistant_turns
                else 1.0
            ),
            "format/history_reasoning_missing_turns_mean": fmean(
                _number(sample, "history_reasoning_missing_turns")
                for sample in samples
            ),
            "format/turn_error_rate": (
                format_error_turns / assistant_turns if assistant_turns else 0.0
            ),
            "format/transition_pair_count": transition_pairs,
            "format/previous_error_support": previous_error_turns,
            "format/previous_ok_support": previous_ok_turns,
            "format/p_error_given_previous_error": error_after_error_rate,
            "format/recovery_rate_after_error": (
                recovery_turns / previous_error_turns
                if previous_error_turns
                else 0.0
            ),
            "format/recovery_turn_count": recovery_turns,
            "format/p_error_given_previous_ok": error_after_ok_rate,
            "format/error_stickiness_lift": (
                error_after_error_rate - error_after_ok_rate
            ),
            "format/error_streak_count": streak_count,
            "format/error_streak_length_mean": (
                format_error_turns / streak_count if streak_count else 0.0
            ),
            "format/trajectory_max_error_streak_mean": fmean(
                trajectory_max_streaks
            ),
            "format/trajectory_max_error_streak_max": max(
                trajectory_max_streaks
            ),
            "format/terminal_error_streak_mean": fmean(terminal_streaks),
            "format/terminal_error_streak_max": max(terminal_streaks),
        }
    )

    trainable_groups = 0
    nonzero_variance_groups = 0
    effective_groups = 0
    passed_groups = 0
    for group in groups.values():
        has_trainable = any(_trainable(sample) for sample in group)
        group_rewards = [_reward(sample) for sample in group]
        has_variance = min(group_rewards) != max(group_rewards)
        trainable_groups += int(has_trainable)
        nonzero_variance_groups += int(has_variance)
        effective_groups += int(has_trainable and has_variance)
        passed_groups += int(any(reward > 0 for reward in group_rewards))
    group_count = len(groups)
    metrics.update(
        {
            "rollout/trainable_group_rate": trainable_groups / group_count,
            "rollout/nonzero_variance_group_rate": (
                nonzero_variance_groups / group_count
            ),
            "rollout/effective_gradient_group_rate": (
                effective_groups / group_count
            ),
            f"reward/avg_pass_at_{int(getattr(args, 'n_samples_per_prompt', 1))}": (
                passed_groups / group_count
            ),
        }
    )

    stages = [_number(sample, "curriculum_stage", -1) for sample in samples]
    metrics.update(
        {
            "curriculum/stage": fmean(stages),
            "curriculum/stage_min": min(stages),
            "curriculum/stage_max": max(stages),
            "curriculum/hint_fraction": len(hinted) / len(samples),
        }
    )
    if hinted:
        metrics["reward/hinted"] = fmean(_reward(sample) for sample in hinted)
    if unhinted:
        metrics["reward/unhinted"] = fmean(
            _reward(sample) for sample in unhinted
        )

    legal_tokens = sum(_number(sample, "legal_tool_tokens") for sample in samples)
    non_target_tokens = sum(
        _number(sample, "non_target_language_tokens") for sample in samples
    )
    metrics["degeneration/non_target_language_token_rate"] = (
        non_target_tokens / legal_tokens if legal_tokens else 0.0
    )
    sampled_tokens = sum(
        _number(sample, "sampled_token_count") for sample in samples
    )
    sampled_unique_tokens = sum(
        _number(sample, "sampled_unique_token_count") for sample in samples
    )
    sampled_16grams = sum(
        _number(sample, "sampled_16gram_count") for sample in samples
    )
    repeated_16grams = sum(
        _number(sample, "repeated_16gram_count") for sample in samples
    )
    metrics.update(
        {
            "degeneration/sampled_token_unique_fraction": (
                sampled_unique_tokens / sampled_tokens if sampled_tokens else 0.0
            ),
            "degeneration/repeated_16gram_rate": (
                repeated_16grams / sampled_16grams if sampled_16grams else 0.0
            ),
            "degeneration/high_repetition_trajectory_rate": _rate(
                samples,
                lambda sample: _number(
                    sample, "sampled_16gram_repetition_rate"
                )
                >= 0.10,
            ),
        }
    )

    bad_json_tokens = sum(_number(sample, "bad_json_tokens") for sample in samples)
    no_tool_tokens = sum(_number(sample, "no_tool_tokens") for sample in samples)
    invalid_tokens = bad_json_tokens + no_tool_tokens
    response_tokens = sum(lengths)
    trainable_tokens = sum(
        sum(int(value) for value in (sample.loss_mask or []))
        if _trainable(sample)
        else 0
        for sample in samples
    )
    error_tokens_in_loss = sum(
        (
            _number(sample, "bad_json_tokens")
            + _number(sample, "no_tool_tokens")
        )
        if _trainable(sample)
        else 0
        for sample in samples
    )
    metrics.update(
        {
            "format_reward/bad_json_token_count": bad_json_tokens,
            "format_reward/no_tool_token_count": no_tool_tokens,
            "format_reward/invalid_token_count": invalid_tokens,
            "format_reward/error_tokens_in_loss": error_tokens_in_loss,
            "format_reward/invalid_response_token_fraction": (
                invalid_tokens / response_tokens if response_tokens else 0.0
            ),
            "format_reward/trainable_token_count": float(trainable_tokens),
        }
    )

    advantages = _group_advantages(args, samples)
    actual_token_counts = {
        id(sample): (
            sum(int(value) for value in (sample.loss_mask or []))
            if _trainable(sample)
            else 0
        )
        for sample in samples
    }
    raw_token_counts = {
        id(sample): int(_number(sample, "raw_trainable_tokens"))
        for sample in samples
    }
    actual_token_total = sum(actual_token_counts.values())
    actual_net = sum(
        advantages.get(id(sample), 0.0) * actual_token_counts[id(sample)]
        for sample in samples
    )
    counterfactual_net = sum(
        advantages.get(id(sample), 0.0) * raw_token_counts[id(sample)]
        for sample in samples
    )
    masked_counterfactual_net = sum(
        advantages.get(id(sample), 0.0) * raw_token_counts[id(sample)]
        for sample in samples
        if not _trainable(sample)
    )
    external_counterfactual_net = sum(
        advantages.get(id(sample), 0.0) * raw_token_counts[id(sample)]
        for sample in samples
        if _flag(sample, "external_stop")
    )
    metrics.update(
        {
            # This reproduces the old SkyRL early-warning scalar before any
            # optional global advantage whitening.
            "loss/avg_raw_advantages": (
                actual_net / actual_token_total if actual_token_total else 0.0
            ),
            "gradient/actual_net_token_advantage_millions": actual_net / 1e6,
            "gradient/counterfactual_net_token_advantage_millions": (
                counterfactual_net / 1e6
            ),
            "gradient/masked_counterfactual_net_token_advantage_millions": (
                masked_counterfactual_net / 1e6
            ),
            "gradient/external_stop_counterfactual_net_token_advantage_millions": (
                external_counterfactual_net / 1e6
            ),
        }
    )
    bad_adv_sum = no_tool_adv_sum = 0.0
    bad_adv_count = no_tool_adv_count = 0.0
    positive_bad_json_tokens = positive_no_tool_tokens = 0.0
    for sample in samples:
        advantage = advantages.get(id(sample), 0.0)
        bad_count = _number(sample, "bad_json_tokens")
        no_tool_count = _number(sample, "no_tool_tokens")
        bad_adv_sum += advantage * bad_count
        no_tool_adv_sum += advantage * no_tool_count
        bad_adv_count += bad_count
        no_tool_adv_count += no_tool_count
        if advantage > 0:
            positive_bad_json_tokens += bad_count
            positive_no_tool_tokens += no_tool_count
    metrics.update(
        {
            "format/bad_json_positive_advantage_rate": (
                positive_bad_json_tokens / bad_adv_count
                if bad_adv_count
                else 0.0
            ),
            "format/no_tool_positive_advantage_rate": (
                positive_no_tool_tokens / no_tool_adv_count
                if no_tool_adv_count
                else 0.0
            ),
            "format/error_token_positive_advantage_rate": (
                (positive_bad_json_tokens + positive_no_tool_tokens)
                / (bad_adv_count + no_tool_adv_count)
                if bad_adv_count + no_tool_adv_count
                else 0.0
            ),
            "format/mean_task_advantage_on_bad_json": (
                bad_adv_sum / bad_adv_count if bad_adv_count else 0.0
            ),
            "format/mean_task_advantage_on_no_tool": (
                no_tool_adv_sum / no_tool_adv_count
                if no_tool_adv_count
                else 0.0
            ),
        }
    )

    for status in {
        _metric_component(_metadata(sample).get("exit_status"))
        for sample in samples
    }:
        status_samples = [
            sample
            for sample in samples
            if _metric_component(_metadata(sample).get("exit_status")) == status
        ]
        status_actual = sum(
            advantages.get(id(sample), 0.0) * actual_token_counts[id(sample)]
            for sample in status_samples
        )
        status_counterfactual = sum(
            advantages.get(id(sample), 0.0) * raw_token_counts[id(sample)]
            for sample in status_samples
        )
        metrics[
            f"gradient/exit_status/{status}_net_token_advantage_millions"
        ] = (status_actual / 1e6)
        metrics[
            f"gradient/exit_status/{status}_counterfactual_net_token_advantage_millions"
        ] = (status_counterfactual / 1e6)

    if not is_training:
        metrics["eval/no_tool_rate"] = metrics[
            "rollout/no_tool_response_rate"
        ]
        metrics["eval/repeated_format_error_rate"] = metrics[
            "rollout/repeated_format_error_rate"
        ]
        metrics["eval/verified_fresh_container_rate"] = metrics[
            "rollout/verified_fresh_container_rate"
        ]
        metrics["eval/verified_patch_apply_failure_rate"] = metrics[
            "rollout/verified_patch_apply_failed_rate"
        ]
    return metrics


def log_rollout_data(
    rollout_id: int,
    args: Any,
    samples: list[Sample],
    rollout_extra_metrics: dict[str, Any] | None,
    rollout_time: float,
) -> bool:
    """Slime ``--custom-rollout-log-function-path`` hook."""
    # This hook runs before Slime converts samples into train data. Apply the
    # project zero-variance mask here as well as in reward post-processing so
    # monitoring and the eventual token-mean denominator observe the same
    # trainable set.
    from .reward_processing import mask_zero_variance_groups

    mask_zero_variance_groups(args, samples)
    metrics = aggregate_r2e_metrics(args, samples, is_training=True)
    metrics["timing/generate"] = float(rollout_time)
    if rollout_extra_metrics is not None:
        rollout_extra_metrics.update(metrics)
    else:
        from slime.utils import logging_utils
        from slime.utils.metric_utils import compute_rollout_step

        metrics["rollout/step"] = compute_rollout_step(args, rollout_id)
        logging_utils.log(args, metrics, step_key="rollout/step")
    logger.info("[slime-r2e] rollout %d custom metrics: %s", rollout_id, metrics)
    return False


def log_eval_rollout_data(
    rollout_id: int,
    args: Any,
    data: dict[str, dict[str, Any]],
    extra_metrics: dict[str, Any] | None,
) -> bool:
    """Slime ``--custom-eval-rollout-log-function-path`` hook."""
    samples = [
        sample
        for item in data.values()
        for sample in _flatten_samples(item.get("samples") or [])
    ]
    raw = aggregate_r2e_metrics(args, samples, is_training=False)
    metrics = {
        key if key.startswith("eval/") else f"eval/{key}": value
        for key, value in raw.items()
    }
    from slime.utils import logging_utils
    from slime.utils.metric_utils import compute_rollout_step

    metrics["eval/step"] = compute_rollout_step(args, rollout_id)
    stage_context = _stage_eval_context(rollout_id, args)
    if stage_context is not None:
        stage_root = f"eval_stage/{stage_context['label']}"
        stage_metric_sources = {
            "verified_pass_at_1": "eval/reward/avg_raw_reward",
            "relaxed_worktree_patch_pass_at_1": (
                "eval/reward_if_worktree_patch_accepted"
            ),
            "submitted_rate": "eval/rollout/submitted_rate",
            "case_timeout_rate": "eval/rollout/case_timeout_rate",
            "model_request_timeout_rate": (
                "eval/rollout/model_request_timeout_rate"
            ),
            "fresh_container_rate": (
                "eval/verified_fresh_container_rate"
            ),
            "patch_apply_failure_rate": (
                "eval/verified_patch_apply_failure_rate"
            ),
        }
        for name, source in stage_metric_sources.items():
            if source in metrics:
                metrics[f"{stage_root}/{name}"] = metrics[source]
        metrics[f"{stage_root}/training_prompts_seen"] = float(
            stage_context["training_prompts_seen"]
        )
        metrics[f"{stage_root}/later_stage_prompt_exposure"] = float(
            stage_context["later_stage_prompt_exposure"]
        )
        metrics[f"{stage_root}/target_stage_complete"] = float(
            stage_context["target_stage_complete"]
        )

    logging_utils.log(args, metrics, step_key="eval/step")
    summary_path = ""
    if stage_context is not None:
        try:
            summary_path = _write_stage_eval_summary(
                stage_context,
                metrics,
                sample_count=len(samples),
            )
        except Exception as exc:
            logger.warning(
                "[slime-r2e] failed to save stage evaluation summary: %s",
                exc,
            )
    logger.info(
        "[slime-r2e] eval %d stage=%s summary=%s custom metrics: %s",
        rollout_id,
        stage_context["label"] if stage_context else "periodic",
        summary_path or "none",
        metrics,
    )
    return False
