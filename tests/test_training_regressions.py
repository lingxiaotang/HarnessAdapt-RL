"""CPU-only regression checks for orchestration and harness acceptance.

These tests mock training, dataset preparation, and checkpoint I/O; they never
start Docker, request a teacher, or download a model.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from harnessadapt.runtime import LoadedHarness
from harnessadapt.training import __main__ as cli, continuation, verified
from harnessadapt.training.config import acceptance, read_json, write_json
from harnessadapt.training.evaluation_paths import input_directory
from harnessadapt.training.diagnostics import runtime_errors
from harnessadapt.training.fixed_harness import hashes
from harnessadapt.training.pre_evolution import prepare_harness
from harnessadapt.training.recipes import recipe_defaults, validate_recipe
from harnessadapt.training.rollout import collection


class ContinuationTests(unittest.TestCase):
    def run_schedule(self, stop_after):
        calls = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def launch(root, config, *, stop_after, evaluation=False, repetition=1):
                calls.append((stop_after, evaluation))
                if not evaluation:
                    write_json(root / "steps" / f"{stop_after:04d}" / "optimizer_complete.json", {
                        "step": stop_after,
                        "optimizer_call_completed": True,
                        "inference_weights_synchronized": True,
                        "nonzero_parameter_update_observed": True,
                    })

            with (
                patch.object(cli, "preflight", return_value=recipe_defaults()),
                patch.object(cli, "launch", side_effect=launch),
                patch.object(verified, "prepare_verified"),
                patch.object(continuation, "check_checkpoint"),
                patch.object(continuation, "evaluation_summary", return_value={"complete": True}),
            ):
                continuation.run(root, stop_after)
            status = read_json(root / "continuation_status.json")
        return calls, status

    def test_tail_update_runs_without_extra_evaluation(self):
        calls, status = self.run_schedule(46)
        self.assertEqual([step for step, evaluation in calls if not evaluation],
                         [5, 10, 15, 20, 25, 30, 35, 40, 45, 46])
        self.assertEqual([step for step, evaluation in calls if evaluation],
                         [5, 10, 15, 20, 25, 30, 35, 40, 45])
        self.assertEqual(status, {"stage": "complete", "completed_step": 46})

    def test_stop_before_first_checkpoint_still_trains(self):
        calls, status = self.run_schedule(3)
        self.assertEqual(calls, [(3, False)])
        self.assertEqual(status["completed_step"], 3)

    def test_extra_evaluations_cannot_exceed_requested_stop(self):
        self.assertEqual(continuation.evaluation_steps(10, extra_steps=(0, 7, 45)), [5, 7, 10])

    def test_failed_optimizer_cannot_claim_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch.object(cli, "preflight", return_value=recipe_defaults()),
                patch.object(cli, "launch"),
                patch.object(verified, "prepare_verified"),
            ):
                with self.assertRaises(FileNotFoundError):
                    continuation.run(root, 1)
            self.assertNotEqual(read_json(root / "continuation_status.json")["stage"], "complete")


class HarnessAcceptanceTests(unittest.TestCase):
    def test_duplicate_diagnostics_are_not_double_counted(self):
        counters = {"harness_hook_errors": {"broken": 2}, "harness_tool_exceptions": {"bad": 1}}
        self.assertEqual(runtime_errors({**counters, "outcome": counters}), (2, 1))

    def test_legacy_tool_errors_are_counted(self):
        record = {"messages": [{"role": "tool", "content": '{"code":"tool_runtime_error"}'}]}
        self.assertEqual(runtime_errors(record), (0, 1))

    def test_invalid_tool_result_is_recorded(self):
        harness = LoadedHarness(
            directory=Path("."), release=1, bank_release=1, profile="test", system_prompt="",
            tools={"broken": SimpleNamespace(execute=lambda arguments, environment: None)}, hooks={},
        )
        harness.execute_tool({"tool_name": "broken"}, None)
        self.assertEqual(harness.diagnostics()["harness_tool_exceptions"], {"broken": 1})

    def test_collection_rejects_improving_candidate_with_hook_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "trajectories" / "train"
            target.mkdir(parents=True)
            for slot, reward in enumerate((0, 1, 0, 1)):
                record = {
                    "instance_id": "task", "official_reward": reward,
                    "official_submitted": True,
                    "harness_hook_errors": {"broken": 1},
                    "outcome": {"reward_known": True},
                    "training_signal": {
                        "eligible_before_group_mask": True,
                        "trainable_token_count_before_group_mask": 10,
                    },
                }
                with gzip.open(target / f"{slot}.json.gz", "wt", encoding="utf-8") as stream:
                    json.dump(record, stream)
            candidate = collection(root, {"tasks": 1, "rollouts_per_task": 4, "task_ids": ["task"]})
        current = dict(candidate, passed=1, hook_errors=0)
        self.assertEqual(candidate["hook_errors"], 4)
        self.assertFalse(acceptance(current, candidate)["accepted"])
        self.assertTrue(acceptance(current, dict(candidate, hook_errors=0))["accepted"])

    def test_openhands_cannot_silently_use_native_agent(self):
        config = recipe_defaults("openhands_grpo")
        validate_recipe(config)
        with self.assertRaises(ValueError):
            validate_recipe(dict(config, backend="miniswe"))


class PreEvolutionTests(unittest.TestCase):
    def test_missing_pre_evolution_cannot_start_after_first_update(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RuntimeError):
                prepare_harness(None, Path(directory), recipe_defaults("pre_evolved"), 1)

    def test_frozen_harness_is_reused_and_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            harness = root / "harness"
            harness.mkdir()
            source = harness / "release.json"
            source.write_text('{"harness_release": 1}', encoding="utf-8")
            write_json(root / "pre_evolution" / "release.json", {
                "harness": str(harness), "harness_hashes": hashes(harness),
            })
            config = recipe_defaults("pre_evolved")
            self.assertEqual(prepare_harness(None, root, config, 10), harness)
            source.write_text('{"harness_release": 2}', encoding="utf-8")
            with self.assertRaises(ValueError):
                prepare_harness(None, root, config, 10)


class CrossBenchmarkTests(unittest.TestCase):
    def test_multilingual_parser_registry_and_scoring(self):
        from harnessadapt.training.multilingual_scorer import score, _load_vendored_swebench

        _, registry, *_ = _load_vendored_swebench()
        for parser_name in ("parse_log_maven", "parse_log_cargo", "parse_log_gotest", "parse_log_jest", "parse_log_phpunit", "parse_log_redis"):
            self.assertIn(parser_name, registry)
        sample_instance = {
            "instance_id": "tokio-rs__tokio-4898",
            "image": "swebench/sweb.eval.x86_64.tokio-rs_1776_tokio-4898:latest",
            "repo": "tokio-rs/tokio",
            "version": "4898",
            "FAIL_TO_PASS": ["sync_watch::drop_rx"],
            "PASS_TO_PASS": [],
            "log_parser": "parse_log_cargo",
            "eval_type": "pass_and_fail",
        }
        self.assertEqual(score(sample_instance, "")[0], 0.0)
        passing_log = (
            ">>>>> Start Test Output\n"
            "test sync_watch::drop_rx ... ok\n"
            ">>>>> End Test Output\n"
            ">>>>> Test Exit Code: 0\n"
        )
        self.assertEqual(score(sample_instance, passing_log)[0], 1.0)

    def test_portable_python_tool_adaptation_keeps_disk_files_unchanged(self):
        import os
        from harnessadapt.runtime import PROJECT_DIR, LoadedHarness

        h2_dir = PROJECT_DIR / "harnesses" / "h2"
        before = hashes(h2_dir)
        with patch.dict(os.environ, {"R2E_PORTABLE_PYTHON_TOOLS": "1"}):
            loaded = LoadedHarness.load(h2_dir)
            self.assertIn(
                "/opt/v5python/bin/python3",
                loaded.tools["run_check"].SMOKE_ARGUMENTS["command"],
            )
        self.assertEqual(hashes(h2_dir), before)

    @unittest.skipUnless(
        verified.DEFAULT_MULTILINGUAL_SOURCE.is_file(),
        "Requires local Multilingual dataset.jsonl cache",
    )
    def test_prepare_and_validate_multilingual_300(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            verified.prepare_multilingual(root)
            rows = verified.validate_multilingual(root)
            self.assertEqual(len(rows), 300)
            manifest = read_json(input_directory(root, "multilingual") / "verified_manifest.json")
            self.assertEqual(
                manifest["sha256"],
                "95ddb9bb9429b8eed1c0c856d83e1ac4e404002c81bb4893c651fb97201f1df1",
            )

    @unittest.skipUnless(
        verified.DEFAULT_SWEPRO_TASKS_DIR.is_dir(),
        "Requires local SWE-bench Pro V2 task directory",
    )
    def test_prepare_and_validate_swepro_642(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            verified.prepare_swepro(root)
            rows = verified.validate_swepro(root)
            self.assertEqual(len(rows), 642)
            manifest = read_json(input_directory(root, "swepro") / "verified_manifest.json")
            self.assertEqual(
                manifest["sha256"],
                "b52253853ead2e6c0d6e7404a44543a20bbdcd40c01317b9001adf5d1077807a",
            )


if __name__ == "__main__":
    unittest.main()
