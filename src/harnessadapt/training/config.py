"""Training hyperparameters, schedule defaults, and Pareto harness selection rule (Table 5 & Appendix B)."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
PROJECT = PACKAGE_ROOT
WORKSPACE = Path(os.environ.get("HARNESSADAPT_WORKSPACE", str(PACKAGE_ROOT.parent))).resolve()
DEFAULT_INPUT = Path(
    os.environ.get("HARNESSADAPT_TRAIN_POOL", str(PACKAGE_ROOT / "data" / "all_cached_train.jsonl"))
)
DEFAULT_POOL_METADATA = PACKAGE_ROOT / "data" / "training_pool_1058.json"
DEFAULT_PROMPTS_DIR = PACKAGE_ROOT / "prompts"

DEFAULTS = {
    "schema_version": 2,
    # Schedule (Table 5 & Appendix B.1)
    "policy": "Qwen3.5-9B",
    "pool_tasks": 1058,
    "tasks": 46,
    "epochs": 2,
    "steps_per_epoch": 23,
    "total_steps": 46,
    "full_assisted_steps": 23,
    "assisted_steps": 46,
    "epoch2_assisted_tasks": 512,
    "rollouts_per_task": 4,
    "probe_rollouts_per_task": 4,
    "teacher_trajectories_per_task": 2,
    "evolution_rounds_per_epoch": 5,
    # Optimization (Table 5)
    "optimizer": "adam",
    "adam_beta1": 0.9,
    "adam_beta2": 0.98,
    "adam_eps": 1e-8,
    "weight_decay": 0.1,
    "learning_rate": 1e-6,
    "lr_decay_style": "constant",
    "advantage_estimator": "grpo",
    "advantage_std_normalization": False,
    "ppo_clip_low": 0.2,
    "ppo_clip_high": 0.28,
    "kl_coef": 0.0,
    "entropy_coef": 0.0,
    "grad_clip": 1.0,
    # Rollout budgets (Appendix A.3 & Table 5)
    "turns": 60,
    "context_tokens": 65536,
    "temperature": 1.0,
    # Evaluation budgets (Appendix A.3 & B.4)
    "verified_tasks": 500,
    "verified_turns": 150,
    "verified_context_tokens": 131072,
    "verified_temperature": 1.0,
    "verified_samples": 1,
    "evaluation_repetitions": 3,
    "save_interval": 5,
    "evaluated_checkpoint_step": 45,
    # Teachers (Section 4.1 & Appendix B.1)
    "skill_teacher_backend": "opencode",
    "skill_teacher_model": "glm-5.3-flash",
    "harness_teacher_backend": "codex",
    "harness_teacher_model": "gpt-5.6-sol",
    "teacher_concurrency": 32,
    "teacher_starts_per_minute": 90,
    "format_retries": 3,
    "meta_skill_limit": 10,
    "meta_word_limit": 1000,
    "mechanism_skill_word_limit": 120,
    # System & Concurrency (Table 5)
    "precision": "bf16",
    "gpus": 8,
    "tp_size": 2,
    "cp_size": 2,
    "pp_size": 1,
    "case_concurrency": 128,
    "setup_concurrency": 8,
    "setup_timeout": 300,
    "model_request_timeout": 1200,
    "agent_wall_time": 1400,
    "case_timeout": 2420,
    "grade_timeout": 90,
    "verified_agent_wall_time": 5400,
    "verified_grade_timeout": 1200,
    "verified_case_timeout": 8100,
    "verified_model_request_timeout": 600,
    # Seeds
    "seed": 20260720,
    "half_assignment_seed": 20260721,
    # Paths / environment overrides
    "model": os.environ.get("HARNESSADAPT_BASE_MODEL", str(WORKSPACE / "models" / "Qwen3.5-9B")),
    "reference": os.environ.get("HARNESSADAPT_REF_MODEL", str(WORKSPACE / "models" / "Qwen3.5-9B_torch_dist")),
    "image": os.environ.get("HARNESSADAPT_TRAIN_IMAGE", "harnessadapt-training:20261019"),
}


def inside(path: Path) -> Path:
    return path.resolve()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def relative_change_pct(current_rate: float, candidate_rate: float) -> float | None:
    """Relative percentage change defined in Equation (2) of Appendix C.8."""
    if current_rate <= 0:
        return None
    return 100.0 * (candidate_rate - current_rate) / current_rate


def acceptance(current: dict, candidate: dict) -> dict:
    """Training-aware Pareto harness selection rule (Section 3.3 & Appendix B.2).

    Accepts a candidate harness iff neither the pass rate nor the trainable rate
    decreases and at least one strictly increases over the 184-rollout comparison.
    """
    n = int(current.get("trajectories") or 0)
    valid = (
        n > 0
        and n == int(candidate.get("trajectories") or 0)
        and bool(current.get("complete", False))
        and bool(candidate.get("complete", False))
        and bool(current.get("task_ids"))
        and current.get("task_ids") == candidate.get("task_ids")
        and not current.get("errors", 0)
        and not candidate.get("errors", 0)
        and not current.get("hook_errors", 0)
        and not current.get("tool_exceptions", 0)
        and not candidate.get("hook_errors", 0)
        and not candidate.get("tool_exceptions", 0)
    )
    pass_current = current["passed"] / n if n else 0.0
    pass_candidate = candidate["passed"] / n if n else 0.0
    trainable_current = current["gradient"] / n if n else 0.0
    trainable_candidate = candidate["gradient"] / n if n else 0.0

    pass_delta = candidate["passed"] - current["passed"]
    trainable_delta = candidate["gradient"] - current["gradient"]

    accepted = bool(
        valid
        and pass_delta >= 0
        and trainable_delta >= 0
        and (pass_delta > 0 or trainable_delta > 0)
    )
    return {
        "accepted": accepted,
        "status": "accepted" if accepted else ("rejected" if valid else "inconclusive"),
        "rule": "pass_rate(candidate) >= pass_rate(current) and trainable_rate(candidate) >= trainable_rate(current) with at least one strict increase",
        "n": n,
        "pass_rate_current": pass_current,
        "pass_rate_candidate": pass_candidate,
        "trainable_rate_current": trainable_current,
        "trainable_rate_candidate": trainable_candidate,
        "pass_delta_count": pass_delta,
        "trainable_delta_count": trainable_delta,
        "pass_delta_pp": 100.0 * (pass_candidate - pass_current),
        "trainable_delta_pp": 100.0 * (trainable_candidate - trainable_current),
        "pass_rel_change_pct": relative_change_pct(pass_current, pass_candidate),
        "trainable_rel_change_pct": relative_change_pct(trainable_current, trainable_candidate),
        "current": current,
        "candidate": candidate,
    }
