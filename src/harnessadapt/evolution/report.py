"""Audit and summarize measured tool and hook evolution rounds (H0 -> H1 -> H2; Table 15)."""
from __future__ import annotations

import argparse
from pathlib import Path

from .audit_results import verify_results
from .common import decision, files, read, write


def verify_pair(parent_output: Path, candidate_output: Path) -> tuple[dict, dict]:
    parent = read(parent_output / "protocol.json")
    candidate = read(candidate_output / "protocol.json")
    for key in ("task_ids", "input_sha256", "config_sha256", "model", "runtime_sources", "rollouts_per_task"):
        if parent[key] != candidate[key]:
            raise ValueError(f"Unpaired comparison {key}: {candidate_output}")
    return parent, candidate


def report(root: Path) -> dict:
    status = read(root / "status.json")
    records = []
    incumbents = {"h1": None, "h2": None}
    audited_outputs = set()
    for path_str in status["rounds"]:
        path = Path(path_str)
        stored = read(path)
        folder = path.parent
        p = (
            read(folder / "parent/summary.json")
            if (folder / "parent/summary.json").exists()
            else read(root / "h1/baseline/summary.json")
        )
        c = read(folder / "candidate/summary.json")
        recomputed = decision(p, c)
        if recomputed["accepted"] != stored["accepted"] or recomputed["status"] != stored["status"]:
            raise ValueError(f"Stored decision disagrees with measured metrics: {path}")
        parent_output = folder / "parent" if (folder / "parent/protocol.json").exists() else root / "h1/baseline"
        verify_pair(parent_output, folder / "candidate")
        for output in (parent_output, folder / "candidate"):
            if output not in audited_outputs:
                verify_results(output)
                audited_outputs.add(output)
        branch = stored["proposal"]["branch"]
        if branch == "h2" and incumbents["h2"] is None:
            incumbents["h2"] = incumbents["h1"]
        if stored["accepted"]:
            incumbents[branch] = stored["candidate_harness"]
        records.append(
            dict(
                branch=branch,
                round=folder.name,
                status=stored["status"],
                parent_pass=p["passed"],
                candidate_pass=c["passed"],
                parent_trainable=p["gradient"],
                candidate_trainable=c["gradient"],
                pass_rel_change_pct=recomputed["pass_rel_change_pct"],
                trainable_rel_change_pct=recomputed["trainable_rel_change_pct"],
                n=p["trajectories"],
                rationale=stored["proposal"]["rationale"],
                decision=str(path),
            )
        )
    endpoints = {b: (root / b / "endpoint_verified.json").exists() for b in ("h1", "h2")}
    result = dict(stage=status["stage"], rounds=records, endpoints=endpoints, complete=all(endpoints.values()))
    write(root / "audit.json", result)
    text = [
        "# Tool and Hook Evolution Summary (H0 -> H1 -> H2)",
        "",
        "Gate: both pass rate and trainable rate nondecreasing, with at least one strictly increasing.",
        "",
        "| Epoch / Target | Proposal | Decision | Pass (parent -> candidate) | Trainable (parent -> candidate) | N |",
        "|---|---|---|---|---|---:|",
    ]
    text += [
        f'| {r["branch"].upper()} | {r["round"]} | {r["status"]} | {r["parent_pass"]} -> {r["candidate_pass"]} | '
        f'{r["parent_trainable"]} -> {r["candidate_trainable"]} | {r["n"]} |'
        for r in records
    ]
    (root / "REPORT.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True, type=Path)
    print(report(p.parse_args().run_dir.resolve()))
