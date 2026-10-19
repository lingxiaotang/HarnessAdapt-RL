#!/usr/bin/env python3
"""Inventory and archive replication package files."""
from __future__ import annotations

import hashlib
import json
import os
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {"dist", ".venv", "__pycache__", "outputs", ".git", ".cache"}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory():
    files = []
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT)
        if any(part in EXCLUDED for part in relative.parts):
            continue
        if path.is_symlink():
            raise ValueError(f"Release files must be actual files, not symlinks: {relative}")
        if path.is_file() and relative.name not in {"FILE_MANIFEST.json", "SHA256SUMS"}:
            if path.suffix not in {".pyc", ".tmp"}:
                files.append(path)
    return files


def write_archive(path, files, mode):
    temporary = path.with_name(path.name + ".tmp")
    with tarfile.open(temporary, mode, format=tarfile.PAX_FORMAT) as archive:
        for file in files:
            info = archive.gettarinfo(str(file), arcname=file.relative_to(ROOT).as_posix())
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            info.mode = 0o755 if file.suffix == ".sh" else 0o644
            info.pax_headers = {}
            with file.open("rb") as stream:
                archive.addfile(info, stream)
    os.replace(temporary, path)


def main():
    ROOT.joinpath("dist").mkdir(exist_ok=True)
    files = inventory()
    entries = []
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        entries.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256(path)})
        if path.suffix == ".safetensors":
            print("Inventoried", relative, flush=True)
    manifest = ROOT / "FILE_MANIFEST.json"
    manifest.write_text(json.dumps({
        "scope": "Replication package file inventory",
        "files": entries
    }, indent=2) + "\n")
    code = ROOT / "dist/harnessadapt-code.tar.gz"
    model = ROOT / "dist/harnessadapt-update45.tar"
    write_archive(code, [p for p in files if p.suffix != ".safetensors"] + [manifest], "w:gz")
    print("Created", code.name, flush=True)
    write_archive(model, [p for p in files if p.relative_to(ROOT).parts[:2] == ("models", "update45")], "w")
    print("Created", model.name, flush=True)
    sums = "".join(f"{sha256(p)}  {p.name}\n" for p in (code, model))
    ROOT.joinpath("dist/SHA256SUMS").write_text(sums)
    print("Created archive SHA256SUMS.", flush=True)


if __name__ == "__main__":
    main()
