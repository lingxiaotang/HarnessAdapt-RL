"""Per-batch runtime boundary: task-specific skill injection and reversible harness loading."""
from __future__ import annotations

import copy
import os
from contextvars import ContextVar
from pathlib import Path

_ORIGINALS = None
_BATCH = None
_CASE_ASSISTED = ContextVar("harnessadapt_case_assisted", default=False)


def _install_mixed_dispatch(base, evolved, harness, assisted_task_ids, native_add_turn_reminder=None):
    originals = dict(_ORIGINALS)
    loaded = evolved._install(base, harness)
    enhanced = {key: getattr(base, key) for key in originals}

    def dispatch(name):
        def routed(*args, **kwargs):
            assisted = _CASE_ASSISTED.get()
            functions = enhanced if assisted else originals
            if name == "_agent_class" and not assisted and native_add_turn_reminder is not None:
                kwargs = dict(kwargs, add_turn_reminder=native_add_turn_reminder)
            return functions[name](*args, **kwargs)

        return routed

    for name in ("_case_model", "_agent_class", "_write_training_trajectory_record"):
        setattr(base, name, dispatch(name))

    def run_case(**kwargs):
        instance = kwargs.get("instance") or {}
        declared = str(instance.get("instance_id") or "")
        session_task = evolved._instance_id_from_session(str(kwargs.get("session_id") or ""))
        task_id = declared or session_task
        if not declared and task_id:
            kwargs = dict(kwargs, instance={**instance, "instance_id": task_id})
        assisted = task_id in assisted_task_ids
        token = _CASE_ASSISTED.set(assisted)
        try:
            if not assisted:
                kwargs = dict(kwargs, hint="", opsd_note_override="", opsd_priority_block_override="")
            return (enhanced if assisted else originals)["_run_case"](**kwargs)
        finally:
            _CASE_ASSISTED.reset(token)

    base._run_case = run_case
    evolved._ACTIVE_HARNESS = loaded
    evolved._BASE_GENERATE_MODULE = base


def configure(
    *,
    root: Path,
    harness: Path | None,
    skills: dict,
    phase: str,
    step: int,
    assisted_task_ids=None,
    native_skills: bool = False,
    backend: str = "miniswe",
):
    global _ORIGINALS, _BATCH
    if backend not in ("miniswe", "openhands_sdk"):
        raise ValueError(f"Unsupported agent backend: {backend}")
    if backend == "openhands_sdk" and (harness or skills or native_skills):
        raise ValueError("Official OpenHands cannot load a local harness or teacher skills")
    os.environ["HARNESSADAPT_AGENT_BACKEND"] = backend
    selected = frozenset(assisted_task_ids) if assisted_task_ids is not None else None
    from .runtime import generate as base
    from harnessadapt.training import evolved_runtime as evolved

    if _ORIGINALS is None:
        _ORIGINALS = {
            k: getattr(base, k)
            for k in ("_case_model", "_agent_class", "_write_training_trajectory_record", "_run_case")
        }
    for key, value in _ORIGINALS.items():
        setattr(base, key, value)
    evolved._ACTIVE_HARNESS = None
    evolved._BASE_GENERATE_MODULE = None

    trajectory_dir = root / "trajectories"
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    os.environ["R2E_TRAJECTORY_DIR"] = str(trajectory_dir)
    os.environ["R2E_HINT_MODE"] = "skills" if (harness or native_skills or skills) else "no_hint"
    os.environ["R2E_EVOLVED_ENABLE_TOOLS"] = "1" if harness else "0"
    os.environ["R2E_EVOLVED_ENABLE_HOOKS"] = "1" if harness else "0"
    if harness:
        os.environ["R2E_EVOLVED_HARNESS"] = str(harness.resolve())
    else:
        os.environ.pop("R2E_EVOLVED_HARNESS", None)

    from dataclasses import replace

    changes = {"trajectory_dir": str(trajectory_dir)}
    native_reminder = None
    if hasattr(base.CONFIG, "add_turn_reminder"):
        assisted_reminder = os.environ.get("R2E_ADD_TURN_REMINDER", "0").strip().lower() in {"1", "true"}
        native_reminder = os.environ.get("R2E_NATIVE_ADD_TURN_REMINDER", "1").strip().lower() in {"1", "true"}
        changes["add_turn_reminder"] = assisted_reminder if harness else native_reminder
    base.CONFIG = replace(base.CONFIG, **changes)
    _BATCH = dict(
        harness=harness,
        skills=copy.deepcopy(skills),
        phase=phase,
        step=step,
        assisted_task_ids=selected,
        native_skills=native_skills,
        backend=backend,
    )
    if harness is not None and selected is not None:
        _install_mixed_dispatch(base, evolved, harness, selected, native_reminder)


async def generate(args, base_sample, sampling_params, evaluation: bool = False):
    if evaluation:
        from .runtime.generate import generate as original

        return await original(args, base_sample, sampling_params, evaluation=True)
    if _BATCH is None:
        raise RuntimeError("Batch must be configured before generating cases")
    sample = copy.deepcopy(base_sample)
    metadata = dict(sample.metadata or {})
    metadata["curriculum"] = dict(metadata.get("curriculum") or {})
    task_id = str(sample.label)
    selected = _BATCH["assisted_task_ids"]
    assisted = selected is None or task_id in selected
    skill = str(_BATCH["skills"].get(task_id, "") or "") if assisted else ""
    metadata["curriculum"].update(medium_hint=skill, stage2_use_hint=bool(skill))
    metadata["task_specific_skill"] = skill
    metadata["coevolve"] = {
        "phase": _BATCH["phase"],
        "step": _BATCH["step"],
        "assisted": assisted,
        "harness": "OpenHands" if _BATCH["backend"] == "openhands_sdk"
        else (str(_BATCH["harness"]) if _BATCH["harness"] else "H0"),
        "backend": _BATCH["backend"],
    }
    sample.metadata = metadata
    if _BATCH["harness"] is not None:
        from harnessadapt.training.evolved_runtime import generate as run
    else:
        from .runtime.generate import generate as run
    return await run(args, sample, sampling_params, evaluation=False)
