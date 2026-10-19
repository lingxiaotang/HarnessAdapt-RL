"""Collect four production token-capturing rollouts per task (184 rollouts per arm) for harness comparison."""
from __future__ import annotations

import argparse
import asyncio
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

from .common import BASE, files, read, runtime_fingerprint, sha, write
from harnessadapt.training.diagnostics import runtime_errors


def dump(path: Path, value) -> None:
    temp = path.with_name(path.name + ".tmp")
    with gzip.open(temp, "wt", encoding="utf-8", compresslevel=3) as stream:
        json.dump(value, stream)
    temp.replace(path)


async def evaluate(root: Path, output: Path, input_path: Path, harness: Path | None) -> int:
    config = read(root / "config.json")
    output.mkdir(parents=True, exist_ok=True)
    (output / "samples").mkdir(exist_ok=True)
    rows = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    names = [row["label"] for row in rows]
    if len(names) != len(set(names)) or not rows:
        raise ValueError("Invalid frozen task batch")
    values = dict(
        R2E_CASE_CONCURRENCY=config["workers"],
        R2E_CONTAINER_START_CONCURRENCY=config["setup_workers"],
        R2E_MAX_ITERATIONS=config["turns"],
        R2E_MAX_CONTEXT_TOKENS=config["context_tokens"],
        R2E_SETUP_TIMEOUT=config["setup_timeout"],
        R2E_MODEL_REQUEST_TIMEOUT=config["request_timeout"],
        R2E_AGENT_WALL_TIME=config["wall_time"],
        R2E_CASE_TIMEOUT=config["case_timeout"],
        R2E_GRADE_TIMEOUT=config["grade_timeout"],
        R2E_REQUIRE_SUBMISSION=1,
        R2E_GRADE_UNSUBMITTED_PATCHES=0,
        R2E_ADD_TURN_REMINDER=0,
        R2E_HINT_MODE="no_hint",
        R2E_SAVE_TRAJECTORIES=1,
        R2E_TRAJECTORY_DIR=str(output / "trajectories"),
        R2E_CONTAINER_OWNER_ROOT=str(root),
        R2E_COEVOLVE_ROOT=str(root),
        R2E_CONTAINER_LIMITS_ENABLED=1,
        R2E_CONTAINER_MEMORY_GIB=config["container_memory_gib"],
        R2E_CONTAINER_PIDS_LIMIT=512,
        R2E_PRESERVE_RECOVERY_SEGMENTS=0,
        R2E_FAST_MODEL_CLIENT=1,
        MSWEA_SILENT_STARTUP=1,
        R2E_ADAPTER_BIND_HOST="127.0.0.1",
        R2E_ADAPTER_PORT=0,
        R2E_EVOLVED_ENABLE_TOOLS=1,
        R2E_EVOLVED_ENABLE_HOOKS=1,
        SLIME_ADAPTER_CPU_WORKERS=16,
        SLIME_ADAPTER_HTTP_CONNECTIONS=128,
    )
    os.environ.update({k: str(v) for k, v in values.items()})
    from harnessadapt.training.runtime import generate as base
    from harnessadapt.training.runtime.reward_processing import post_process_rewards
    from slime.utils.types import Sample

    if harness:
        from harnessadapt.training.evolved_runtime import _install

        _install(base, harness)
    args = SimpleNamespace(
        hf_checkpoint=str(BASE),
        rollout_max_context_len=config["context_tokens"],
        sglang_router_ip="127.0.0.1",
        sglang_router_port=config["server_port"],
        sglang_tool_call_parser="qwen3_coder",
        sglang_reasoning_parser="qwen3",
        reward_key=None,
        rewards_normalization=True,
        advantage_estimator="grpo",
        grpo_std_normalization=False,
    )
    server = read(root / "server.json")
    model_identity = {k: server[k] for k in ("model", "model_config_sha256")}
    protocol = dict(
        task_ids=names,
        input_sha256=sha(input_path),
        config_sha256=sha(root / "config.json"),
        runtime_sources=runtime_fingerprint(),
        harness=str(harness) if harness else None,
        harness_files=files(harness) if harness else None,
        model=model_identity,
        rollouts_per_task=config["rollouts_per_task"],
    )
    write(output / "protocol.json", protocol)
    semaphore = asyncio.Semaphore(config["workers"])
    finished = 0
    total = len(rows) * config["rollouts_per_task"]

    async def one(row, group: int, slot: int):
        nonlocal finished
        sample_path = output / "samples" / f"{group:04d}.{slot}.json.gz"
        if sample_path.exists():
            with gzip.open(sample_path, "rt", encoding="utf-8") as stream:
                result = Sample.from_dict(json.load(stream))
        else:
            async with semaphore:
                metadata = copy.deepcopy(row["metadata"])
                metadata["instance"]["instance_id"] = row["label"]
                sample = Sample(
                    group_index=group,
                    index=slot,
                    label=row["label"],
                    prompt=row["prompt"],
                    metadata=metadata,
                )
                seed = (
                    config["seed"]
                    + int(hashlib.sha256(f'{row["label"]}:{slot}'.encode()).hexdigest()[:8], 16)
                ) % (2**31)
                result = await base.generate(
                    args,
                    sample,
                    {"temperature": config["temperature"], "top_p": 1.0, "sampling_seed": seed},
                    evaluation=False,
                )
                dump(sample_path, result.to_dict())
        finished += 1
        write(output / "progress.json", dict(completed=finished, expected=total))
        return result

    samples = await asyncio.gather(
        *(
            one(row, group, slot)
            for group, row in enumerate(rows)
            for slot in range(config["rollouts_per_task"])
        )
    )
    before = {
        id(s): dict(
            eligible_before_group_mask=not s.remove_sample and bool(sum(s.loss_mask or [])),
            trainable_token_count_before_group_mask=sum(s.loss_mask or []),
        )
        for s in samples
    }
    post_process_rewards(args, samples)
    errors = []
    hook_errors = tool_exceptions = passed = gradient = effective = 0
    for s in samples:
        eligible = not s.remove_sample and sum(s.loss_mask or []) > 0
        signal = dict(
            after_generate=True,
            after_reward_processing=True,
            **before[id(s)],
            remove_sample=s.remove_sample,
            trainable_token_count=sum(s.loss_mask or []),
            enters_policy_gradient=eligible,
            zero_variance_group_masked=bool(s.metadata.get("zero_variance_group_masked")),
            optimizer_update_executed=False,
        )
        path = Path(s.metadata.get("trajectory_record_path") or "/nonexistent")
        if not path.is_file():
            errors.append(dict(task=s.label, slot=s.index, error="missing trajectory"))
            continue
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            record = json.load(stream)
        record["training_signal"] = signal
        record["evolution_sample_slot"] = s.index
        record["evolution_group_index"] = s.group_index
        dump(path, record)
        dump(
            output / "samples" / f"{s.group_index:04d}.{s.index}.postprocessed.json.gz",
            s.to_dict(),
        )
        if not s.metadata.get("reward_known") or str(s.metadata.get("masked_reason", "")).startswith("exception:"):
            errors.append(dict(task=s.label, slot=s.index, error="unknown reward or rollout exception"))
        row_hook_errors, row_tool_exceptions = runtime_errors(record)
        hook_errors += row_hook_errors
        tool_exceptions += row_tool_exceptions
        passed += s.reward == 1
        gradient += eligible
        effective += sum(s.loss_mask or [])
    summary = dict(
        complete=len(samples) == total and not errors,
        task_ids=names,
        tasks=len(rows),
        trajectories=total,
        passed=passed,
        gradient=gradient,
        pass_rate=passed / total,
        gradient_rate=gradient / total,
        trainable_rate=gradient / total,
        submitted=sum(bool(s.metadata.get("submitted")) for s in samples),
        effective_tokens=effective,
        infrastructure_errors=errors,
        hook_errors=hook_errors,
        tool_exceptions=tool_exceptions,
        optimizer_updates=False,
        production_token_masks=True,
    )
    write(output / "summary.json", summary)
    return 0 if summary["complete"] else 2


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--harness", type=Path)
    a = p.parse_args()
    raise SystemExit(asyncio.run(evaluate(a.run_dir, a.output, a.input, a.harness)))
