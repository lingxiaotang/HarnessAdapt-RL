"""Isolated official SWE-bench Multilingual grading (300 tasks, 19 log parsers).

Ported from `trajectory_meta_skill_teacher/eval_*_multilingual_*/score_output.py`
and `check_parsers.py`. Uses the pinned official `vendor/swebench` log parsers
(`c`, `go`, `java`, `javascript`, `php`, `python`, `ruby`, `rust`) and grading
functions (`get_logs_eval`, `get_eval_tests_report`, `get_resolution_status`).
"""
from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import tempfile
import types
import uuid
from pathlib import Path
from typing import Any

VENDOR_ROOT = Path(__file__).resolve().parent / "vendor"


def _load_vendored_swebench():
    """Load the vendored multilingual swebench harness without mutating global swebench imports."""
    vendor_str = str(VENDOR_ROOT)
    if vendor_str not in sys.path:
        sys.path.insert(0, vendor_str)
    # Bypass any installed Python-only swebench.harness package initializer
    for mod_name in list(sys.modules):
        if mod_name == "swebench" or mod_name.startswith("swebench."):
            mod = sys.modules[mod_name]
            mod_file = getattr(mod, "__file__", None)
            if mod_file and not Path(mod_file).resolve().is_relative_to(VENDOR_ROOT.resolve()):
                sys.modules.pop(mod_name, None)
    harness = types.ModuleType("swebench.harness")
    harness.__path__ = [str(VENDOR_ROOT / "swebench" / "harness")]
    sys.modules["swebench.harness"] = harness

    from swebench.harness.constants import EvalType
    from swebench.harness.grading import (
        get_eval_tests_report,
        get_logs_eval,
        get_resolution_status,
    )
    from swebench.harness.log_parsers import PARSER_REGISTRY
    from swebench.types import TestSpec

    return TestSpec, PARSER_REGISTRY, get_logs_eval, get_eval_tests_report, get_resolution_status, EvalType


def score(instance: dict[str, Any], output: str) -> tuple[float, dict[str, Any]]:
    """Score a single SWE-bench Multilingual evaluation log output."""
    (
        TestSpec,
        PARSER_REGISTRY,
        get_logs_eval,
        get_eval_tests_report,
        get_resolution_status,
        EvalType,
    ) = _load_vendored_swebench()

    spec = TestSpec(
        instance_id=instance["instance_id"],
        image=instance.get("image") or instance["image_name"],
        eval_script_list=[],
        repo=instance["repo"],
        version=instance["version"],
        FAIL_TO_PASS=instance["FAIL_TO_PASS"],
        PASS_TO_PASS=instance["PASS_TO_PASS"],
        log_parser=instance["log_parser"],
        eval_type=instance.get("eval_type", "pass_and_fail"),
    )
    if not spec.FAIL_TO_PASS or spec.log_parser not in PARSER_REGISTRY:
        raise ValueError(
            f"Invalid Multilingual TestSpec for {spec.instance_id}: "
            f"log_parser={spec.log_parser!r}, FAIL_TO_PASS={len(spec.FAIL_TO_PASS)}"
        )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".log", encoding="utf-8") as handle:
        handle.write(output)
        handle.flush()
        statuses, valid = get_logs_eval(spec, handle.name)
    report = get_eval_tests_report(
        statuses if valid else {},
        {"FAIL_TO_PASS": spec.FAIL_TO_PASS, "PASS_TO_PASS": spec.PASS_TO_PASS},
        eval_type=EvalType(spec.eval_type),
    )
    resolution = get_resolution_status(report) if valid else "RESOLVED_NO"
    result = {
        "resolution_status": resolution,
        "valid_test_output": valid,
        "parsed_tests": len(statuses),
        "log_parser": spec.log_parser,
    }
    for category in ("FAIL_TO_PASS", "PASS_TO_PASS"):
        cat_report = report[category]
        result[category] = {
            "success_count": len(cat_report["success"]),
            "failure_count": len(cat_report["failure"]),
            "failure": cat_report["failure"][:20],
            "failure_truncated": len(cat_report["failure"]) > 20,
        }
    return float(valid and resolution == "RESOLVED_FULL"), result


def verify_multilingual_parsers(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Verify that all 300 Multilingual tasks map to supported parsers and score 0 on empty logs."""
    _, PARSER_REGISTRY, *_ = _load_vendored_swebench()
    parsers_used = set()
    for row in rows:
        instance = row["metadata"]["instance"] if "metadata" in row else row
        parser_name = instance.get("log_parser", "")
        if parser_name not in PARSER_REGISTRY:
            raise ValueError(f"Unsupported Multilingual log_parser {parser_name!r} for {instance.get('instance_id')}")
        parsers_used.add(parser_name)
        reward, _ = score(instance, "")
        if reward != 0.0:
            raise ValueError(f"Empty log scored non-zero ({reward}) for {instance.get('instance_id')}")
    return {"tasks": len(rows), "unique_parsers": len(parsers_used), "parsers": sorted(parsers_used)}


def score_multilingual_subprocess(instance: dict[str, Any], output: str) -> tuple[float, dict[str, Any]]:
    """Run official Multilingual scoring in an isolated subprocess and save grading logs."""
    run_root = Path(os.environ.get("R2E_COEVOLVE_ROOT") or os.environ.get("RUN_ROOT") or ".")
    folder = run_root / "grading_logs" / str(instance["instance_id"])
    folder.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    (folder / f"{token}.log").write_text(output, encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve())],
        input=json.dumps({"instance": instance, "output": output}),
        text=True,
        capture_output=True,
        timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError("Multilingual parser failed: " + proc.stderr[-2000:])
    reward, summary = json.loads(proc.stdout)
    (folder / f"{token}.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return float(reward), summary


if __name__ == "__main__":
    payload = json.load(sys.stdin)
    with contextlib.redirect_stdout(sys.stderr):
        scored = score(payload["instance"], payload["output"])
    print(json.dumps(scored))
