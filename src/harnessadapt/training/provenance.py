"""Record and verify source/config provenance for reproducible training runs."""
from __future__ import annotations

from pathlib import Path

from .config import PACKAGE_ROOT, read_json, sha256, write_json


def identities(root: Path) -> dict[str, str]:
    files = set((PACKAGE_ROOT / "src" / "harnessadapt").rglob("*.py"))
    files.update((PACKAGE_ROOT / "src" / "harnessadapt" / "training" / "runtime").glob("*.sh"))
    files.update((root / "prompts").glob("*.md"))
    files.update([root / "config.json", root / "train.jsonl", root / "sampling_plan.json", root / "input_manifest.json"])
    files.add(root / "openhands_runtime_manifest.json")
    files.update((PACKAGE_ROOT / "integrations" / "openhands").glob("*"))
    files.update((PACKAGE_ROOT / "integrations" / "training").glob("*"))
    if (root / "initial_harness").is_dir():
        files.update(p for p in (root / "initial_harness").rglob("*") if "__pycache__" not in p.parts)
    return {str(path.relative_to(root) if root in path.parents else path.relative_to(PACKAGE_ROOT)): sha256(path) for path in sorted(files) if path.is_file()}


def freeze_or_verify(root: Path, *, manifest_name: str = "source_manifest.json") -> dict[str, str]:
    target = root / manifest_name
    current = identities(root)
    if target.exists():
        before = read_json(target)
        changed = [path for path in set(before) | set(current) if before.get(path) != current.get(path)]
        if changed:
            raise ValueError("Frozen experiment source/config drift: " + ", ".join(changed[:10]))
    else:
        write_json(target, current)
    return current
