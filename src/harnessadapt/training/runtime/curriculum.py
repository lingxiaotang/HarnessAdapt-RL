"""Two-epoch skill-assistance scheduling helpers (Appendix A.1 & B.1).

Epoch 1 (updates 1–23, 1,058 task visits) assists all 1,058 tasks.
Epoch 2 (updates 24–46, 1,058 task visits) assists the fixed 512-task subset.
"""
from __future__ import annotations

from typing import Any, Mapping

NUM_TRAIN_CASES = 1058
STAGE2_HINT_CASES = 512
TRAIN_PROMPTS_PER_STEP = 46
TRAJECTORIES_PER_PROMPT = 4
NUM_STAGES = 2
TOTAL_TRAIN_PROMPTS = NUM_TRAIN_CASES * NUM_STAGES
NUM_TRAIN_UPDATES = TOTAL_TRAIN_PROMPTS // TRAIN_PROMPTS_PER_STEP
GLOBAL_BATCH_SIZE = TRAIN_PROMPTS_PER_STEP * TRAJECTORIES_PER_PROMPT
EVAL_AFTER_UPDATES = tuple(range(5, NUM_TRAIN_UPDATES + 1, 5))

if TOTAL_TRAIN_PROMPTS % TRAIN_PROMPTS_PER_STEP:
    raise RuntimeError("Total curriculum prompts must be divisible by prompts per step")


def curriculum_stage(group_index: int, *, num_cases: int = NUM_TRAIN_CASES) -> int:
    if num_cases <= 0:
        raise ValueError("num_cases must be positive")
    if group_index < 0:
        raise ValueError("group_index must be non-negative")
    return min(group_index // num_cases, NUM_STAGES - 1)


def hint_for_sample(
    group_index: int,
    curriculum: Mapping[str, Any],
    *,
    evaluation: bool = False,
    num_cases: int = NUM_TRAIN_CASES,
    hint_mode: str = "skills",
) -> tuple[int, bool, str]:
    """Return `(stage, use_hint, hint_text)` for one Slime sample."""
    if hint_mode == "skills":
        text = "" if evaluation else str(curriculum.get("medium_hint") or "").strip()
        return -1, bool(text), text
    if hint_mode not in {"curriculum", "no_hint"}:
        raise ValueError(f"hint_mode must be 'skills', 'curriculum', or 'no_hint', got {hint_mode!r}")
    if evaluation:
        return -1, False, ""

    stage = curriculum_stage(group_index, num_cases=num_cases)
    if hint_mode == "no_hint":
        use_hint = False
    elif stage == 0:
        use_hint = True
    else:
        use_hint = bool(curriculum.get("stage2_use_hint"))

    hint = str(curriculum.get("medium_hint") or "").strip() if use_hint else ""
    return stage, bool(hint), hint
