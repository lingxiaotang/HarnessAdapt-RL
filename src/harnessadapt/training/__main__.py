"""CLI entrypoint for preparing, launching, and evaluating HarnessAdapt-RL training runs.

Usage:
  python -m harnessadapt.training prepare --recipe harnessadapt_rl --run-dir <RUN_DIR>
  python -m harnessadapt.training preflight --run-dir <RUN_DIR>
  python -m harnessadapt.training train --run-dir <RUN_DIR> --stop-after 46
"""
from __future__ import annotations

import argparse
import datetime
import fcntl
import json
import os
import random
import shutil
import subprocess
from pathlib import Path

from .config import (
    DEFAULT_INPUT,
    DEFAULT_POOL_METADATA,
    DEFAULT_PROMPTS_DIR,
    PACKAGE_ROOT,
    WORKSPACE,
    inside,
    read_json,
    sha256,
    write_json,
)
from .recipes import ALL_RECIPES, HARNESSADAPT_RL_RECIPE, recipe_defaults, validate_recipe
from .evaluation_paths import BENCHMARKS, evaluation_directory, input_directory, resolve_portable_python


def docker(*args, **kwargs):
    return subprocess.run(["docker", *args], check=True, text=True, **kwargs)


def prepare(
    root: Path,
    *,
    recipe: str = HARNESSADAPT_RL_RECIPE,
    source: Path | None = None,
    selection_seed: int | None = None,
    fixed_harness_source: Path | None = None,
) -> dict:
    defaults = recipe_defaults(recipe)
    validate_recipe(defaults)
    selection_seed = selection_seed if selection_seed is not None else defaults["seed"]
    source = Path(source or DEFAULT_INPUT).resolve()
    harness_source = (
        str(Path(fixed_harness_source).resolve())
        if fixed_harness_source is not None
        else defaults.get("fixed_harness_source")
    )
    validate_recipe(dict(defaults, fixed_harness_source=harness_source))

    if (root / "config.json").exists():
        existing = read_json(root / "config.json")
        validate_recipe(existing)
        if existing["recipe"] != recipe:
            raise ValueError("Existing run uses a different recipe; use a fresh run directory")
        return existing
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Run directory is not empty: {root}")

    if not source.exists() and source == DEFAULT_INPUT.resolve():
        from .task_pool import build_cached_pool

        build_cached_pool(source)
    from .task_pool import read_pool

    rows = read_pool(source, require_manifest=source == DEFAULT_INPUT.resolve())
    if len(rows) != 1058:
        raise ValueError(f"Training pool must contain 1,058 tasks, found {len(rows)}")

    half_ids: set[str] = set()
    if defaults.get("epoch2_assisted_tasks", 0) > 0:
        pool_meta = read_json(DEFAULT_POOL_METADATA)
        if selection_seed == defaults["seed"] and "epoch2_assisted_task_ids" in pool_meta:
            half_ids = set(pool_meta["epoch2_assisted_task_ids"])
        else:
            half_ids = set(
                random.Random(defaults["half_assignment_seed"]).sample(
                    sorted(r["label"] for r in rows), defaults["epoch2_assisted_tasks"]
                )
            )

    from .sampling import plan_for_config

    plan = plan_for_config(
        [r["label"] for r in rows],
        dict(defaults, selection_seed=selection_seed),
        half_assisted_ids=half_ids,
    )

    root.mkdir(parents=True, exist_ok=True)
    for row in rows:
        metadata = row["metadata"]
        for key in ("task_specific_skill", "task_skill_provenance"):
            metadata.pop(key, None)
        metadata["curriculum"] = {
            "instance_id": row["label"],
            "medium_hint": "",
            "stage2_use_hint": row["label"] in half_ids,
        }
    (root / "train.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    shutil.copytree(DEFAULT_PROMPTS_DIR, root / "prompts")

    harness = root / "initial_harness"
    if defaults.get("backend") == "openhands_sdk":
        from harnessadapt.openhands import provenance

        harness.mkdir()
        write_json(harness / "upstream.json", provenance())
    elif harness_source:
        from .fixed_harness import initialize

        fixed = initialize(harness_source, harness)
        write_json(root / "fixed_harness_manifest.json", fixed)
    else:
        harness.mkdir()
        (harness / "instance_template.md").write_text(
            "<pr_description>\n{{ task }}\n</pr_description>\n\n"
            "Fix the issue in /testbed and submit the final patch with:\n"
            "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt\n",
            encoding="utf-8",
        )
        (harness / "system_prompt.md").write_text("", encoding="utf-8")
        write_json(
            harness / "release.json",
            {
                "schema_version": 1,
                "harness_release": 0,
                "bank_release": 1,
                "name": "H0_native_miniswe",
                "tools": [],
                "hooks": [],
                "meta_skills": [],
                "skill_injection_style": "natural",
                "allow_empty_meta_skills": True,
                "artifact_version": "H0",
            },
        )

    write_json(root / "sampling_plan.json", plan)
    config = dict(
        defaults,
        fixed_harness_source=harness_source,
        selection_seed=selection_seed,
        source=str(source),
        sampling_plan=str(root / "sampling_plan.json"),
        root=str(root),
        input=str(root / "train.jsonl"),
        prompts=str(root / "prompts"),
        created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )
    write_json(root / "config.json", config)
    write_json(
        root / "input_manifest.json",
        {
            "source": str(source),
            "source_sha256": sha256(source),
            "input_sha256": sha256(root / "train.jsonl"),
            "sampling_plan_sha256": sha256(root / "sampling_plan.json"),
            "pool_tasks": len(rows),
            "tasks_per_step": config["tasks"],
            "epoch1_assisted_tasks": 1058 if config.get("assisted_steps") else 0,
            "epoch2_assisted_tasks": len(half_ids),
            "selection_seed": selection_seed,
            "task_ids": [row["label"] for row in rows],
        },
    )
    return config


def preflight(root: Path) -> dict:
    config = read_json(root / "config.json")
    validate_recipe(config)
    rows = [json.loads(line) for line in Path(config["input"]).read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 1058 or config["tasks"] != 46 or config["total_steps"] != 46:
        raise ValueError("Configuration must use 1,058 tasks, 46 tasks/update, and 46 total updates (2 epochs)")
    manifest = read_json(root / "input_manifest.json")
    expected = read_json(DEFAULT_POOL_METADATA)["task_ids"]
    labels = [row["label"] for row in rows]
    if len(set(labels)) != len(labels) or set(labels) != set(expected):
        raise ValueError("Training input must contain the published 1,058 unique task IDs")
    if (
        sha256(Path(config["input"])) != manifest["input_sha256"]
        or sha256(Path(config["sampling_plan"])) != manifest["sampling_plan_sha256"]
    ):
        raise ValueError("Frozen training input or sampling plan changed")
    print(json.dumps({"preflight": "passed", "recipe": config["recipe"], "tasks": len(rows), "total_steps": config["total_steps"]}), flush=True)
    return config


def launch(
    root: Path,
    config: dict,
    *,
    stop_after: int = 46,
    preparation: bool = False,
    evaluation: bool = False,
    benchmark: str = "verified",
    multilingual_source: Path | None = None,
    swepro_tasks: Path | None = None,
    portable_python: Path | None = None,
    repetition: int = 1,
) -> None:
    validate_recipe(config)
    benchmark = benchmark.strip().lower()
    if benchmark not in BENCHMARKS or repetition < 1:
        raise ValueError("Choose a supported benchmark and a positive repetition number")
    portable_dir = resolve_portable_python(portable_python) if evaluation and benchmark == "multilingual" else None
    openhands_runtime = None
    if config.get("backend") == "openhands_sdk":
        from harnessadapt.openhands.bridge import runtime_path

        openhands_runtime = runtime_path()
        runtime_manifest = read_json(openhands_runtime / "manifest.json")
        frozen_manifest = root / "openhands_runtime_manifest.json"
        if frozen_manifest.exists() and read_json(frozen_manifest) != runtime_manifest:
            raise ValueError("Official OpenHands runtime changed; use the frozen runtime or a fresh run")
        if not frozen_manifest.exists():
            write_json(frozen_manifest, runtime_manifest)
    from .environment import check_prerequisites

    check_prerequisites(config, workspace=WORKSPACE)
    if not 1 <= stop_after <= config["total_steps"]:
        raise ValueError("Requested stop must be within the training schedule")
    # Serialize input preparation as well as execution for this training run.
    with (root / "launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        evaluation_harness = None
        if evaluation:
            from .continuation import check_checkpoint, require_optimizer_complete
            from .verified import prepare_benchmark, prepare_verified

            if benchmark == "verified":
                prepare_verified(root)
            else:
                prepare_benchmark(
                    root,
                    benchmark,
                    multilingual_source=multilingual_source,
                    swepro_tasks=swepro_tasks,
                )
            require_optimizer_complete(root, stop_after)
            check_checkpoint(root / "checkpoints", stop_after, require_latest=False)
            if config.get("verified_evolved_harness"):
                release = read_json(root / "steps" / f"{stop_after:04d}" / "release.json")
                if release.get("harness") not in (None, "H0"):
                    evaluation_harness = Path(release["harness"]).resolve(strict=True)
        mode = "preparation" if preparation else (f"{benchmark}-run{repetition}" if evaluation else "training")
        if not preparation:
            from .provenance import freeze_or_verify

            freeze_or_verify(root, manifest_name="evaluation_source_manifest.json" if evaluation else "source_manifest.json")
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        logs = root / "launches"
        logs.mkdir(exist_ok=True)
        phase_root = (evaluation_directory(root, benchmark, stop_after, repetition) if evaluation
                      else root / ("preparation_runtime" if preparation else "trainer"))
        if evaluation:
            if phase_root.exists() and any(phase_root.iterdir()):
                raise FileExistsError(
                    f"Evaluation output already exists: {phase_root}. "
                    "Choose a new --repetition; existing attempts are never mixed or overwritten."
                )
            phase_root.mkdir(parents=True, exist_ok=True)

        env = {
            "WORKSPACE": str(WORKSPACE),
            "PACKAGE_ROOT": str(PACKAGE_ROOT),
            "MODEL_DIR": config["model"],
            "REF_MODEL_PATH": config["reference"],
            "RUN_ROOT": str(phase_root),
            "CHECKPOINT_DIR": str(root / "checkpoints"),
            "NUM_GPUS": str(config["gpus"]),
            "TP_SIZE": str(config["tp_size"]),
            "CP_SIZE": str(config["cp_size"]),
            "PP_SIZE": str(config["pp_size"]),
            "NUM_ROLLOUT": str(0 if evaluation else (1 if preparation else config["total_steps"])),
            "ROLLOUT_BATCH_SIZE": str(config["tasks"]),
            "N_SAMPLES_PER_PROMPT": str(config["rollouts_per_task"]),
            "SAVE_INTERVAL": str(config.get("save_interval", 5)),
            "PROMPT_DATA": config["input"],
            "VALIDATION_DATA": str(input_directory(root, benchmark) / "verified.jsonl") if evaluation else config["input"],
            "R2E_DISABLE_EVAL": "0" if evaluation else "1",
            "R2E_HINT_MODE": "no_hint" if (evaluation or not config.get("skill_variant")) else "skills",
            "R2E_MAX_ITERATIONS": str(config["verified_turns"] if evaluation else config["turns"]),
            "R2E_MAX_CONTEXT_TOKENS": str(config["verified_context_tokens"] if evaluation else config["context_tokens"]),
            "MAX_CONTEXT_LEN": str(config["verified_context_tokens"] if evaluation else config["context_tokens"]),
            "ROLLOUT_SEED": str(config["seed"] + repetition - 1 if evaluation else config["seed"]),
            "ROLLOUT_TEMPERATURE": str(config["temperature"]),
            "R2E_EVAL_TEMPERATURE": str(config["verified_temperature"]),
            "N_SAMPLES_PER_EVAL_PROMPT": str(config.get("verified_samples", 1)),
            "LEARNING_RATE": str(config["learning_rate"]),
            "R2E_REQUIRE_SUBMISSION": "1",
            "R2E_SAVE_TRAJECTORIES": "1",
            "R2E_TRAJECTORY_DIR": str(phase_root / "trajectories"),
            "R2E_GENERATE_PATH": "harnessadapt.training.phase_runtime.generate",
            "R2E_REWARD_PATH": "harnessadapt.training.runtime.reward_processing.post_process_rewards",
            "R2E_COEVOLVE_ROOT": str(root),
            "R2E_STOP_AFTER_STEP": str(stop_after),
            "HARNESSADAPT_AGENT_BACKEND": config.get("backend", "miniswe"),
        }
        if openhands_runtime is not None:
            env["HARNESSADAPT_OPENHANDS_RUNTIME"] = str(openhands_runtime)
        if evaluation:
            # Evaluate the requested checkpoint even when a later checkpoint exists.
            env["CKPT_STEP"] = str(stop_after - 1)
            env["R2E_BENCHMARK_KIND"] = benchmark
            env["R2E_GENERATE_PATH"] = "harnessadapt.training.runtime.generate.generate"
            if evaluation_harness is not None:
                env.update(
                    R2E_GENERATE_PATH="harnessadapt.training.evolved_runtime.generate",
                    R2E_EVOLVED_HARNESS=str(evaluation_harness),
                    R2E_EVOLVED_ENABLE_TOOLS="1",
                    R2E_EVOLVED_ENABLE_HOOKS="1",
                )
            if benchmark in ("multilingual", "swepro"):
                env["R2E_GENERATE_PATH"] = "harnessadapt.training.benchmark_runtime.generate"
                if portable_dir is not None:
                    env["R2E_PORTABLE_PYTHON_PATH"] = str(portable_dir)
                    env["R2E_PORTABLE_PYTHON_TOOLS"] = "1"
                if benchmark == "swepro":
                    env.update(
                        R2E_VERIFIED_AGENT_WALL_TIME="3000",
                        R2E_VERIFIED_GRADE_TIMEOUT="3000",
                        R2E_VERIFIED_CASE_TIMEOUT="3300",
                    )
        # Pass teacher credentials through Docker's inherited environment. Only
        # variable names, never their values, enter the persisted launch command.
        child_env = dict(os.environ)
        if "HARNESSADAPT_HARNESS_TEACHER_KEY" not in child_env:
            fallback_key = child_env.get("CODEX_API_KEY") or child_env.get("OPENAI_API_KEY")
            if fallback_key:
                child_env["HARNESSADAPT_HARNESS_TEACHER_KEY"] = fallback_key
        teacher_variables = (
            "HARNESSADAPT_SKILL_TEACHER_URL", "HARNESSADAPT_SKILL_TEACHER_KEY",
            "HARNESSADAPT_HARNESS_TEACHER_URL", "HARNESSADAPT_HARNESS_TEACHER_KEY",
            # Paths refer to CLIs already installed inside the training container.
            "HARNESSADAPT_CODEX_BIN", "HARNESSADAPT_OPENCODE_BIN", "CODEX_API_KEY",
        )
        train_script = PACKAGE_ROOT / "src" / "harnessadapt" / "training" / "runtime" / "train.sh"
        command = [
            "run",
            "--rm",
            "--name",
            f"harnessadapt-{mode}-{stamp}",
            "--label",
            f"self_evolve.coevolve_root={root}",
            "--network",
            "host",
            "--ipc",
            "host",
            "--shm-size",
            "256g",
            "--runtime=nvidia",
            "-e",
            "NVIDIA_VISIBLE_DEVICES=all",
        ]
        for key, value in env.items():
            command += ["-e", f"{key}={value}"]
        if not evaluation:
            for key in teacher_variables:
                if key in child_env:
                    command += ["-e", key]
        mounted = [WORKSPACE.resolve(), PACKAGE_ROOT.resolve()]
        if portable_dir is not None:
            # Ray workers must see the same host path used by sibling task containers.
            command += ["-v", f"{portable_dir}:{portable_dir}:ro"]
            mounted.append(portable_dir)
        if openhands_runtime is not None:
            # Keep the host path visible to Ray workers. Sibling task containers
            # bind that same host path at /opt/harnessadapt-openhands.
            command += ["-v", f"{openhands_runtime}:{openhands_runtime}:ro"]
            mounted.append(openhands_runtime)
        for path in (root, Path(config["model"]), Path(config["reference"])):
            path = path.resolve()
            if not any(path.is_relative_to(parent) for parent in mounted):
                command += ["-v", f"{path}:{path}"]
                mounted.append(path)
        command += [
            "-v",
            f"{WORKSPACE}:{WORKSPACE}",
            "-v",
            f"{PACKAGE_ROOT}:{PACKAGE_ROOT}",
            "-v",
            "/var/run/docker.sock:/var/run/docker.sock",
            "-v",
            f"{PACKAGE_ROOT}/src/harnessadapt/training/runtime/patches/slime/rollout.py:{WORKSPACE}/slime/slime/ray/rollout.py:ro",
            "-v",
            f"{PACKAGE_ROOT}/src/harnessadapt/training/runtime/patches/sglang/weight_checker.py:/sgl-workspace/sglang/python/sglang/srt/utils/weight_checker.py:ro",
            "-w",
            str(WORKSPACE / "slime"),
            config["image"],
            "bash",
            str(train_script),
            "train",
        ]
        write_json(logs / f"{mode}_{stamp}.json", {"environment": env, "command": command})
        if evaluation:
            write_json(phase_root / "run.json", {
                "benchmark": benchmark, "step": stop_after, "repetition": repetition,
                "seed": int(env["ROLLOUT_SEED"]),
                "input_sha256": sha256(Path(env["VALIDATION_DATA"])),
                "portable_python": str(portable_dir) if portable_dir else None,
            })
        with (logs / f"{mode}_{stamp}.log").open("w", encoding="utf-8") as log:
            docker(*command, stdout=log, stderr=subprocess.STDOUT, env=child_env)
        if evaluation and benchmark == "swepro":
            from .swepro_scorer import score_swepro_run

            score_swepro_run(
                root,
                step=stop_after,
                repetition=repetition,
                tasks_root=swepro_tasks or Path(os.environ.get("HARNESSADAPT_SWEPRO_TASKS", str(WORKSPACE / "swebench_pro_v2_upstream_20260923/v2/tasks"))),
            )
        elif evaluation:
            from .continuation import evaluation_summary

            evaluation_summary(root, stop_after, benchmark=benchmark, repetition=repetition)


def main(argv=None) -> int:
    from .verified import BENCHMARKS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=["prepare", "preflight", "prepare-verified", "train", "resume", "status", "evaluate"],
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--recipe", choices=ALL_RECIPES, default=HARNESSADAPT_RL_RECIPE)
    parser.add_argument("--stop-after", type=int, default=46)
    parser.add_argument("--repetition", type=int, default=1,
                        help="Positive evaluation attempt number; isolates outputs and offsets the rollout seed")
    parser.add_argument("--benchmark", choices=BENCHMARKS, default="verified",
                        help="Evaluation benchmark: verified (500), multilingual (300), or swepro (642)")
    parser.add_argument("--multilingual-source", type=Path,
                        help="Optional local SWE-bench Multilingual dataset.jsonl path")
    parser.add_argument("--swepro-tasks", type=Path,
                        help="Optional local SWE-bench Pro V2 tasks directory")
    parser.add_argument("--portable-python", type=Path,
                        help="Optional portable Python directory mounted at /opt/v5python for Multilingual")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--fixed-harness-source", type=Path)
    parser.add_argument("--selection-seed", type=int)
    parser.add_argument("--check-environment", action="store_true",
                        help="With preflight, also check source revisions, converted weights, image imports and teacher CLIs")
    args = parser.parse_args(argv)
    if args.repetition < 1:
        parser.error("--repetition must be positive")
    root = inside(args.run_dir)

    if args.command == "prepare":
        config = prepare(
            root,
            recipe=args.recipe,
            source=args.source,
            selection_seed=args.selection_seed,
            fixed_harness_source=args.fixed_harness_source,
        )
        print(json.dumps(config, indent=2))
        return 0
    if args.command == "prepare-verified":
        from .verified import prepare_benchmark, prepare_verified

        if args.benchmark == "verified":
            prepare_verified(root)
        else:
            prepare_benchmark(
                root,
                args.benchmark,
                multilingual_source=args.multilingual_source,
                swepro_tasks=args.swepro_tasks,
            )
        return 0
    if args.command == "status":
        paths = sorted(root.glob("steps/*/progress.json")) + sorted(root.glob("steps/*/optimizer_complete.json"))
        print(json.dumps({str(p.relative_to(root)): read_json(p) for p in paths}, indent=2))
        return 0
    config = preflight(root)
    if args.command == "preflight":
        if args.check_environment:
            from .environment import inspect_environment

            report = inspect_environment(config, workspace=WORKSPACE)
            write_json(root / "environment_preflight.json", report)
            print(json.dumps(report, indent=2))
        return 0
    if args.command in ("train", "resume") and config.get("periodic_verified"):
        from .continuation import run

        run(root, args.stop_after)
    else:
        launch(
            root,
            config,
            stop_after=args.stop_after,
            evaluation=args.command == "evaluate",
            benchmark=args.benchmark,
            multilingual_source=args.multilingual_source,
            swepro_tasks=args.swepro_tasks,
            portable_python=args.portable_python,
            repetition=args.repetition,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
