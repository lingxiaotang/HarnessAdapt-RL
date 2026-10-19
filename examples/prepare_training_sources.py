#!/usr/bin/env python3
"""Check out the pinned Slime/R2E sources and apply the recorded Slime agent patch."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

INTEGRATION = Path(__file__).resolve().parents[1] / "integrations" / "training"


def git(directory, *args, check=True):
    return subprocess.run(["git", "-C", str(directory), *args], check=check,
                          text=True, capture_output=True)


def prepare(workspace: Path):
    workspace.mkdir(parents=True, exist_ok=True)
    sources = json.loads((INTEGRATION / "sources.json").read_text())
    for name in ("slime", "R2E-Gym"):
        spec = sources[name]
        target = workspace / name
        if not target.exists():
            target.mkdir()
            git(target, "init")
            git(target, "remote", "add", "origin", spec["url"])
            git(target, "fetch", "--depth", "1", "origin", spec["revision"])
            git(target, "checkout", "--detach", "FETCH_HEAD")
        head = git(target, "rev-parse", "HEAD").stdout.strip()
        if head != spec["revision"]:
            raise ValueError(f"{target}: expected {spec['revision']}, found {head}; use a new workspace")
        if "patch" in spec:
            patch = INTEGRATION / spec["patch"]
            if hashlib.sha256(patch.read_bytes()).hexdigest() != spec["patch_sha256"]:
                raise ValueError(f"Patch checksum mismatch: {patch}")
            if git(target, "apply", "--reverse", "--check", str(patch), check=False).returncode:
                git(target, "apply", "--check", str(patch))
                git(target, "apply", str(patch))
        print(f"Prepared {name} at {head}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    prepare(parser.parse_args().workspace.expanduser().resolve())
