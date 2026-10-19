"""Shared definitions and Pareto selection rule for tool and hook evolution (Section 3.2 & 3.3)."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
PROJECT = PACKAGE_ROOT
WORKSPACE = Path(os.environ.get("HARNESSADAPT_WORKSPACE", str(PACKAGE_ROOT.parent))).resolve()
BASE = Path(os.environ.get("HARNESSADAPT_BASE_MODEL", str(WORKSPACE / "models" / "Qwen3.5-9B")))
TARGETS = {
    "h1": PACKAGE_ROOT / "harnesses" / "h1",
    "h2": PACKAGE_ROOT / "harnesses" / "h2",
}
IMAGE = os.environ.get(
    "HARNESSADAPT_SERVER_IMAGE",
    "slimerl/slime@sha256:a97ec147e37bef050337a9b229036eda00b4aa9c4d02b31a0109dc850f8ca342",
)


def read(path: Path | str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path: Path | str, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def sha(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def files(path: Path | str) -> dict[str, str]:
    return {
        str(p.relative_to(path)): sha(p)
        for p in sorted(Path(path).rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts
    }


def runtime_fingerprint() -> dict[str, str]:
    paths = [
        PACKAGE_ROOT / "src/harnessadapt/policy.py",
        PACKAGE_ROOT / "src/harnessadapt/runtime.py",
        PACKAGE_ROOT / "src/harnessadapt/training/evolved_runtime.py",
        PACKAGE_ROOT / "src/harnessadapt/training/runtime/generate.py",
        PACKAGE_ROOT / "src/harnessadapt/training/runtime/reward_processing.py",
        PACKAGE_ROOT / "src/harnessadapt/training/runtime/fast_model.py",
        Path(__file__).parent / "evaluate.py",
    ]
    return {str(p.relative_to(PACKAGE_ROOT)): sha(p) for p in sorted(paths) if p.is_file()}


def decision(parent: dict, candidate: dict) -> dict:
    """Pareto acceptance rule on pass_rate and trainable_rate (Section 3.3 & Appendix B.2)."""
    valid = (
        parent.get("complete", False)
        and candidate.get("complete", False)
        and parent.get("task_ids") == candidate.get("task_ids")
        and parent.get("trajectories") == candidate.get("trajectories") > 0
        and not parent.get("infrastructure_errors")
        and not candidate.get("infrastructure_errors")
        and not parent.get("hook_errors", 0)
        and not parent.get("tool_exceptions", 0)
    )
    dg = candidate["gradient"] - parent["gradient"]
    dp = candidate["passed"] - parent["passed"]
    broken = bool(candidate.get("hook_errors", 0) or candidate.get("tool_exceptions", 0))
    accepted = valid and not broken and dg >= 0 and dp >= 0 and (dg > 0 or dp > 0)
    n = parent["trajectories"]
    return {
        "accepted": bool(accepted),
        "status": "accepted" if accepted else ("inconclusive" if not valid else "rejected"),
        "trainable_delta": dg,
        "pass_delta": dp,
        "trainable_delta_pp": 100.0 * dg / n if n else None,
        "pass_delta_pp": 100.0 * dp / n if n else None,
        "pass_rel_change_pct": (100.0 * dp / parent["passed"]) if parent.get("passed") else None,
        "trainable_rel_change_pct": (100.0 * dg / parent["gradient"]) if parent.get("gradient") else None,
        "rule": "neither pass_rate nor trainable_rate decreases, and at least one strictly increases",
        "candidate_runtime_errors": broken,
        "parent": parent,
        "candidate": candidate,
    }
