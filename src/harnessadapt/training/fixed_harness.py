"""Initialize and verify fixed or evolved harnesses (H1 / H2) across training updates."""
from __future__ import annotations

import shutil
from pathlib import Path

from .config import read_json, sha256, write_json


def hashes(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): sha256(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts
    }


def initialize(source: Path | str, target: Path) -> dict:
    source = Path(source)
    release = read_json(source / "release.json")
    for kind in ("tools", "hooks"):
        for name in release.get(kind, []):
            if not (source / kind / f"{name}.py").is_file():
                raise ValueError(f"Missing {kind} module: {name}")
    source_hashes = hashes(source)
    shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    if release.get("prompt_mode", "augment") != "replace":
        (target / "system_prompt.md").write_text("", encoding="utf-8")
    (target / "meta_bank.md").unlink(missing_ok=True)
    release.update(
        meta_skills=[],
        skill_injection_style="natural",
        allow_empty_meta_skills=release.get("prompt_mode", "augment") != "replace",
        fixed_tools_hooks=True,
    )
    write_json(target / "release.json", release)
    return {
        "source": str(source.resolve()),
        "source_hashes": source_hashes,
        "initial_hashes": hashes(target),
        "tools": release.get("tools", []),
        "hooks": release.get("hooks", []),
    }


def validate_initial(root: Path) -> None:
    manifest = read_json(root / "fixed_harness_manifest.json")
    if hashes(root / "initial_harness") != manifest["initial_hashes"]:
        raise ValueError("Frozen initial tools/hooks harness changed")


def validate_release(root: Path, harness: Path) -> None:
    from .teacher import verify_inherited_capabilities

    validate_initial(root)
    verify_inherited_capabilities(root / "initial_harness", harness)
