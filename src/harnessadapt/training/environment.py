"""Check documented training prerequisites without starting training or teacher requests."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

from .config import PACKAGE_ROOT, WORKSPACE


def teacher_backends(config: dict) -> list[str]:
    backends = []
    if config.get("skill_variant"):
        backends.append(config["skill_teacher_backend"])
    if any(config.get(k) for k in ("evolve_epoch1", "evolve_epoch2", "pre_evolve_on_base")):
        backends.append(config["harness_teacher_backend"])
    return sorted(set(backends))


def check_prerequisites(config: dict, *, workspace: Path = WORKSPACE) -> None:
    errors = []
    required = {
        "Slime model configuration": workspace / "slime/scripts/models/qwen3.5-9B.sh",
        "R2E-Gym source": workspace / "R2E-Gym/src/r2egym",
        "base model config": Path(config["model"]) / "config.json",
        "base model tokenizer": Path(config["model"]) / "tokenizer.json",
        "reference checkpoint marker": Path(config["reference"]) / "latest_checkpointed_iteration.txt",
    }
    errors.extend(f"Missing {name}: {path}" for name, path in required.items() if not path.exists())
    index = Path(config["model"]) / "model.safetensors.index.json"
    if index.is_file():
        shards = set(json.loads(index.read_text())["weight_map"].values())
        errors.extend(f"Missing base model shard: {name}" for name in shards
                      if not (index.parent / name).is_file())
    elif not (index.parent / "model.safetensors").is_file():
        errors.append(f"Missing base model weights/index: {index.parent}")
    marker = required["reference checkpoint marker"]
    if marker.is_file():
        iteration = marker.read_text().strip()
        if iteration == "release" or iteration.isdecimal():
            directory = marker.parent / ("release" if iteration == "release" else f"iter_{int(iteration):07d}")
            if not (directory / "common.pt").is_file() or not list(directory.glob("*.distcp")):
                errors.append(f"Incomplete converted reference checkpoint: {directory}")
        else:
            errors.append(f"Invalid reference checkpoint marker: {marker}")
    if shutil.which("docker") is None:
        errors.append("Missing Docker CLI (and a running daemon with NVIDIA Container Toolkit is required)")
    if errors:
        raise RuntimeError("Training environment is incomplete:\n- " + "\n- ".join(errors)
                           + "\nFollow docs/TRAINING_SETUP.md before launching.")


def inspect_environment(config: dict, *, workspace: Path = WORKSPACE, evaluation: bool = False) -> dict:
    """Explicit environment preflight: verify source pins, image imports and teacher executables."""
    import os

    check_prerequisites(config, workspace=workspace)
    integration = PACKAGE_ROOT / "integrations/training"
    sources = json.loads((integration / "sources.json").read_text())
    revisions = {}
    for name in ("slime", "R2E-Gym"):
        directory = workspace / name
        result = subprocess.run(["git", "-C", str(directory), "rev-parse", "HEAD"],
                                check=True, text=True, capture_output=True)
        revisions[name] = result.stdout.strip()
        if revisions[name] != sources[name]["revision"]:
            raise ValueError(f"{name} revision differs from integrations/training/sources.json")
        if sources[name].get("patch"):
            subprocess.run(["git", "-C", str(directory), "apply", "--reverse", "--check",
                            str(integration / sources[name]["patch"])], check=True, capture_output=True)
    image = subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", config["image"]],
                           check=True, text=True, capture_output=True).stdout.strip()
    backends = [] if evaluation else teacher_backends(config)
    command = ["docker", "run", "--rm", "--network", "none", "--entrypoint", "python"]
    for backend in backends:
        variable = f"HARNESSADAPT_{backend.upper()}_BIN"
        if variable in os.environ:
            command += ["-e", variable]
    # No credentials, task images, downloads, or model/API requests enter this probe.
    probe = '''import importlib, json, os, shutil, subprocess, sys
for name in ("torch", "ray", "sglang", "megatron.core", "minisweagent", "docker"):
    importlib.import_module(name)
versions = {}
for backend in sys.argv[1:]:
    executable = shutil.which(os.environ.get("HARNESSADAPT_" + backend.upper() + "_BIN", backend))
    if not executable:
        raise RuntimeError("Missing teacher CLI inside training image: " + backend)
    versions[backend] = subprocess.check_output([executable, "--version"], text=True).strip()
print(json.dumps({"python": sys.version.split()[0], "teacher_versions": versions}))
'''
    result = subprocess.run([*command, image, "-c", probe, *backends], check=True,
                            text=True, capture_output=True, timeout=120)
    return {"sources": revisions, "image_id": image,
            "runtime": json.loads(result.stdout.strip().splitlines()[-1]),
            "teacher_endpoints_tested": False}
