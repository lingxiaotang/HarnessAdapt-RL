"""Evaluation runtime adapters for SWE-bench Verified, SWE-bench Multilingual, and SWE-bench Pro V2.

Ported from:
- `trajectory_meta_skill_teacher/eval_*_multilingual_*/runtime.py`
- `trajectory_meta_skill_teacher/eval_swepro_v2_three_way_20260923/runtime.py`
- `trajectory_meta_skill_teacher/eval_base_v7_swepro_v2_20261005/runtime.py`

Supports both native (`H0`) and evolved (`H1` / `H2`) harnesses without
task-specific training skills or teacher calls.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .multilingual_scorer import score_multilingual_subprocess
from .runtime import generate as native


def _portable_python_volume() -> list[str]:
    path = os.environ.get("R2E_PORTABLE_PYTHON_PATH", "").strip()
    if path and Path(path).is_dir():
        return ["-v", f"{Path(path).resolve()}:/opt/v5python:ro"]
    return []


def multilingual_environment(instance: dict[str, Any]):
    """Create a Multilingual evaluation container preserving language toolchains in image ENV."""
    from minisweagent.environments import get_environment

    return get_environment(
        {
            "environment_class": "docker",
            "image": native._verified_image_name(instance),
            "cwd": "/testbed",
            "timeout": native.CONFIG.step_timeout,
            "pull_timeout": native.CONFIG.setup_timeout,
            "executable": "docker",
            "run_args": native._verified_docker_run_args() + _portable_python_volume(),
            "interpreter": ["bash", "--noprofile", "--norc", "-c"],
            "env": {
                "PAGER": "cat",
                "MANPAGER": "cat",
                "LESS": "-R",
                "PIP_PROGRESS_BAR": "off",
                "TQDM_DISABLE": "1",
            },
        }
    )


def pro_environment(instance: dict[str, Any]):
    """Create a SWE-bench Pro V2 evaluation container with `/app -> /testbed` symlink and `--network none`."""
    from minisweagent.environments import get_environment

    environment = get_environment(
        {
            "environment_class": "docker",
            "image": native._verified_image_name(instance),
            "cwd": "/",
            "timeout": native.CONFIG.step_timeout,
            "pull_timeout": native.CONFIG.setup_timeout,
            "executable": "docker",
            "run_args": native._verified_docker_run_args() + _portable_python_volume() + ["--network", "none"],
            "interpreter": ["bash", "--noprofile", "--norc", "-c"],
            "env": {
                "PAGER": "cat",
                "MANPAGER": "cat",
                "LESS": "-R",
                "PIP_PROGRESS_BAR": "off",
                "TQDM_DISABLE": "1",
            },
        }
    )
    # SWE-bench Pro V2 images place the repository at /app, whereas Mini-SWE prompts
    # and H1/H2 tools operate under /testbed. Link /testbed -> /app before turn 1.
    setup = environment.execute(
        {
            "command": """
set -eu
if [ -d /app ] && git -C /app rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  if [ ! -e /testbed ]; then ln -s /app /testbed; fi
  [ "$(readlink -f /testbed)" = /app ] || exit 42
elif [ -d /testbed ] && git -C /testbed rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  :
else
  exit 43
fi
git -C /testbed rev-parse --is-inside-work-tree
"""
        },
        cwd="/",
        timeout=30,
    )
    if setup.get("returncode") != 0:
        native._close_verified_environment(environment)
        raise RuntimeError("Pro V2 repository setup failed: " + str(setup.get("output"))[-500:])
    environment.config.cwd = "/testbed"
    return environment


def defer_pro_grade(instance: dict[str, Any], patch: str, *, case_deadline: float, label_prefix: str):
    """Defer SWE-bench Pro V2 grading to offline clean-container replay (`swepro_scorer.py`)."""
    return native._VerifiedPatchGrade(
        attempted=False,
        reward=0.0,
        reward_known=True,
        resolution_status="PENDING_OFFICIAL_V2_REPLAY",
    )


async def generate(args, base_sample, sampling_params, evaluation: bool = False):
    """Unified evaluation entrypoint for Verified, Multilingual, and SWE-bench Pro V2."""
    if not evaluation:
        raise RuntimeError("benchmark_runtime.generate is evaluation-only")
    benchmark = os.environ.get("R2E_BENCHMARK_KIND", "verified").strip().lower()
    if benchmark == "multilingual":
        native._new_verified_environment = multilingual_environment
        native._score_swebench_verified_output = score_multilingual_subprocess
    elif benchmark == "swepro":
        native._new_verified_environment = pro_environment
        native._grade_verified_patch_fresh = defer_pro_grade
    elif benchmark != "verified":
        raise ValueError(f"Unsupported evaluation benchmark: {benchmark!r}")

    if os.environ.get("R2E_EVOLVED_HARNESS"):
        from . import evolved_runtime

        return await evolved_runtime.generate(args, base_sample, sampling_params, evaluation=True)
    return await native.generate(args, base_sample, sampling_params, evaluation=True)
