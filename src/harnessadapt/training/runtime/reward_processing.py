"""GRPO group-mean reward centering, unknown-reward group exclusion, and zero-variance group masking (Appendix B.1–B.3)."""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import fmean
from typing import Any


def _flat_samples(samples: list[Any]) -> list[Any]:
    if any(isinstance(sample, (list, tuple)) for sample in samples):
        raise TypeError(
            "R2E reward post-processing expects Slime's flattened training samples"
        )
    return samples


def _groups(samples: list[Any]) -> list[list[int]]:
    grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        group_index = getattr(sample, "group_index", None)
        key = (
            ("group", int(group_index))
            if group_index is not None
            else ("sample", index)
        )
        grouped[key].append(index)
    return list(grouped.values())


def _is_truncated(sample: Any) -> bool:
    md = sample.metadata if isinstance(getattr(sample, "metadata", None), dict) else {}
    return bool(
        md.get("context_budget_exit")
        or md.get("generation_length_exit")
        or md.get("limits_exceeded")
    )


def mask_zero_variance_groups(args: Any, samples: list[Any]) -> list[float]:
    """Exclude groups with unknown rewards (Appendix B.3) or without mixed non-truncated outcomes (Appendix B.2)."""

    flat = _flat_samples(samples)
    raw_rewards = [float(sample.get_reward_value(args)) for sample in flat]
    for indices in _groups(flat):
        unknown_reward = any(
            not (
                flat[index].metadata
                if isinstance(getattr(flat[index], "metadata", None), dict)
                else {}
            ).get("reward_known", True)
            for index in indices
        )
        non_truncated_rewards = [
            raw_rewards[index]
            for index in indices
            if not _is_truncated(flat[index])
        ]
        mixed_outcomes = (
            (max(non_truncated_rewards) - min(non_truncated_rewards) > 1e-12)
            if len(non_truncated_rewards) >= 2
            else False
        )
        if not unknown_reward and mixed_outcomes:
            continue
        for index in indices:
            sample = flat[index]
            sample.loss_mask = [0] * int(sample.response_length or 0)
            metadata = (
                dict(sample.metadata)
                if isinstance(getattr(sample, "metadata", None), dict)
                else {}
            )
            if unknown_reward:
                metadata["unknown_reward_group_masked"] = True
            else:
                metadata["zero_variance_group_masked"] = True
            sample.metadata = metadata
    return raw_rewards


def post_process_rewards(args: Any, samples: list[Any]) -> tuple[list[float], list[float]]:
    """Compute GRPO group-mean centered advantages without std normalization (Table 5)."""

    flat = _flat_samples(samples)
    if any((sample.metadata or {}).get("recovery_segments") for sample in flat):
        from .recovery_segments import rollout_groups

        rollouts = rollout_groups(flat)
        representatives = [flat[indices[0]] for indices in rollouts]
        for indices in rollouts:
            rewards = [float(flat[i].get_reward_value(args)) for i in indices]
            if any(value != rewards[0] for value in rewards):
                raise ValueError("Recovery segments have inconsistent rewards")
        import copy

        proxies = [copy.copy(sample) for sample in representatives]
        for proxy in proxies:
            proxy.metadata = {**(proxy.metadata or {}), "recovery_segments": False}
        raw_rollouts, processed_rollouts = post_process_rewards(args, proxies)
        raw, processed = [0.0] * len(flat), [0.0] * len(flat)
        for indices, proxy, reward, advantage in zip(
            rollouts, proxies, raw_rollouts, processed_rollouts, strict=True
        ):
            for i in indices:
                raw[i], processed[i] = reward, advantage
                if (proxy.metadata or {}).get("zero_variance_group_masked") or (
                    proxy.metadata or {}
                ).get("unknown_reward_group_masked"):
                    flat[i].loss_mask = [0] * int(flat[i].response_length or 0)
                    flat[i].metadata = {
                        **(flat[i].metadata or {}),
                        "zero_variance_group_masked": bool(
                            (proxy.metadata or {}).get("zero_variance_group_masked")
                        ),
                        "unknown_reward_group_masked": bool(
                            (proxy.metadata or {}).get("unknown_reward_group_masked")
                        ),
                    }
        return raw, processed
    raw_rewards = mask_zero_variance_groups(args, flat)
    processed = list(raw_rewards)

    normalize = bool(getattr(args, "rewards_normalization", True))
    estimator = str(getattr(args, "advantage_estimator", "grpo"))
    normalize_std = bool(getattr(args, "grpo_std_normalization", False))

    for indices in _groups(flat):
        group_rewards = [raw_rewards[index] for index in indices]
        if not normalize:
            continue
        mean = fmean(group_rewards)
        centered = [reward - mean for reward in group_rewards]
        if estimator in {"grpo", "gspo", "cispo"} and normalize_std:
            if len(centered) > 1:
                variance = sum(value * value for value in centered) / (
                    len(centered) - 1
                )
                std = math.sqrt(variance)
            else:
                std = 0.0
            centered = [value / (std + 1e-6) for value in centered]
        for index, value in zip(indices, centered, strict=True):
            processed[index] = float(value)

    return raw_rewards, processed
