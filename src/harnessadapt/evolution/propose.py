"""Materialize an evidence-backed tool/hook candidate proposal for rollout evaluation."""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

from .common import files, read, sha, write


def materialize(
    root: Path,
    name: str,
    *,
    tools: list[str],
    hooks: list[str],
    evidence: Path,
    rationale: str,
    batch_index: int = 0,
    skill_titles=(),
    exact_target: bool = False,
    extra_tools=(),
    extra_hooks=(),
) -> dict:
    root = root.resolve()
    status = read(root / "status.json")
    if status["stage"] != "awaiting_proposal":
        raise ValueError("Wait for the current rollout and analysis to finish")
    branch = status["branch"]
    target = root / "targets" / branch
    evidence = evidence.resolve()
    if evidence != Path(status["evidence"]).resolve():
        raise ValueError("Analyze the latest measured round before proposing the next change")
    analysis = read(evidence)
    if not analysis.get("trajectory_count") or not rationale.strip():
        raise ValueError("A proposal requires completed solver evidence and a rationale")
    summary = read(evidence.parent / "summary.json")
    if not summary.get("complete") or summary.get("infrastructure_errors"):
        raise ValueError("Incomplete trajectory evidence cannot support a proposal")
    if not re.fullmatch(r"[0-9]{3}_[a-z0-9_]+", name):
        raise ValueError("Proposal name must begin with a three-digit round number")
    destination = root / "candidates" / branch / name
    if destination.exists() or (root / "requests" / (name + ".json")).exists():
        raise ValueError("Never overwrite an existing proposal")

    metadata = read(target / "release.json")
    available_tools = metadata["tools"]
    available_hooks = metadata["hooks"]
    if exact_target:
        shutil.copytree(target, destination)
        tools, hooks = available_tools, available_hooks
    else:
        if not set(tools) <= set(available_tools) or not set(hooks) <= set(available_hooks):
            raise ValueError("Non-target components must be supplied explicitly via --extra-tool / --extra-hook")
        destination.mkdir(parents=True)
        for category, names in (("tools", tools), ("hooks", hooks)):
            (destination / category).mkdir()
            for component in names:
                shutil.copyfile(
                    target / category / (component + ".py"),
                    destination / category / (component + ".py"),
                )
        metadata.update(
            name=f"evolution_{branch}_{name}",
            tools=list(tools),
            hooks=list(hooks),
            acceptance="unmeasured_candidate",
        )
        write(destination / "release.json", metadata)
        shutil.copyfile(target / "system_prompt.md", destination / "system_prompt.md")
        native = read(root / "native_templates.json")
        template = native["INSTANCE_TEMPLATE"]
        if "prepare_patch" in tools:
            template = (target / "instance_template.md").read_text(encoding="utf-8")
            start = template.index("## General repair skills")
            end = template.index("## Working rules")
            section = template[start:end]
            parts = re.split(r"(?=^### Skill: )", section, flags=re.M)
            selected = [
                part
                for part in parts
                if any(part.startswith("### Skill: " + title + "\n") for title in skill_titles)
            ]
            template = (
                template[:start]
                + ("## General repair skills\n\n" + "".join(selected) if selected else "")
                + template[end:]
            )
            if not {"read_file", "grep", "edit_file", "write_file", "view_diff"} <= set(tools):
                template = re.sub(
                    r"\{% if 'read_file' in available %\}.*?\{% endif %\}",
                    "",
                    template,
                    flags=re.S,
                )
        (destination / "instance_template.md").write_text(template, encoding="utf-8")
        (destination / "skills.md").write_text("\n".join(skill_titles) + "\n", encoding="utf-8")
        (destination / "README.md").write_text("Candidate harness proposal.\n", encoding="utf-8")

    for category, sources in (("tools", extra_tools), ("hooks", extra_hooks)):
        for source in sources:
            component = source.stem
            folder = destination / category
            folder.mkdir(exist_ok=True)
            shutil.copyfile(source, folder / source.name)
            metadata = read(destination / "release.json")
            metadata[category].append(component)
            write(destination / "release.json", metadata)

    from harnessadapt.runtime import LoadedHarness

    LoadedHarness.load(destination)
    parent = Path(status["incumbent"]) if status["incumbent"] else None
    old_files = files(parent) if parent else {}
    new_files = files(destination)
    changes = {
        key: dict(before=old_files.get(key), after=new_files.get(key))
        for key in sorted(old_files.keys() | new_files.keys())
        if old_files.get(key) != new_files.get(key)
    }
    request = {
        "action": "candidate",
        "branch": branch,
        "harness": str(destination),
        "parent_harness": str(parent) if parent else None,
        "evidence": str(evidence),
        "evidence_sha256": sha(evidence),
        "rationale": rationale,
        "batch_index": batch_index,
        "tools": read(destination / "release.json")["tools"],
        "hooks": read(destination / "release.json")["hooks"],
        "changes": changes,
        "candidate_sha256": new_files,
        "target_exact": exact_target,
    }
    write(root / "proposal_records" / branch / (name + ".json"), request)
    write(root / "requests" / (name + ".json"), request)
    return request


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--evidence", type=Path, required=True)
    p.add_argument("--rationale-file", type=Path, required=True)
    p.add_argument("--tools", default="")
    p.add_argument("--hooks", default="")
    p.add_argument("--batch-index", type=int, default=0)
    p.add_argument("--skill-title", action="append", default=[])
    p.add_argument("--exact-target", action="store_true")
    p.add_argument("--extra-tool", type=Path, action="append", default=[])
    p.add_argument("--extra-hook", type=Path, action="append", default=[])
    a = p.parse_args()
    result = materialize(
        a.run_dir,
        a.name,
        tools=[x for x in a.tools.split(",") if x],
        hooks=[x for x in a.hooks.split(",") if x],
        evidence=a.evidence,
        rationale=a.rationale_file.read_text(encoding="utf-8"),
        batch_index=a.batch_index,
        skill_titles=a.skill_title,
        exact_target=a.exact_target,
        extra_tools=a.extra_tool,
        extra_hooks=a.extra_hook,
    )
    print(result["harness"])
