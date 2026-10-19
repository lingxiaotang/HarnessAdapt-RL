"""CPU regression checks for setup, benchmark isolation, and portable Python."""
from contextlib import ExitStack, contextmanager, redirect_stdout
import gzip
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from harnessadapt.training import __main__ as cli, continuation, environment, provenance, verified, swepro_scorer
from harnessadapt.training.config import read_json, write_json
from harnessadapt.training.evaluation_paths import input_directory, evaluation_directory, resolve_portable_python
from harnessadapt.training.recipes import recipe_defaults


def record(path, instance_id="task", reward=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as stream:
        json.dump({"instance_id": instance_id, "official_reward": reward,
                   "official_submitted": False}, stream)


def portable_runtime(root):
    executable = root / "bin/python3"
    executable.parent.mkdir(parents=True)
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)
    stdlib = root / "lib/python3.11/encodings"
    stdlib.mkdir(parents=True)
    (stdlib / "__init__.py").touch()
    return root


class PortablePythonTests(unittest.TestCase):
    def test_unset_and_empty_never_resolve_to_current_directory(self):
        for env in ({}, {"HARNESSADAPT_PORTABLE_PYTHON": ""}, {"HARNESSADAPT_PORTABLE_PYTHON": "  "}):
            with patch.dict(os.environ, env, clear=True):
                self.assertIsNone(resolve_portable_python())

    def test_invalid_runtime_fails_instead_of_silently_disabling(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for path in (root, root / "missing"):
                with self.assertRaisesRegex(ValueError, "bin/python3"):
                    resolve_portable_python(path)
            portable_runtime(root)
            (root / "lib/python3.11/encodings/__init__.py").unlink()
            with self.assertRaisesRegex(ValueError, "standard library"):
                resolve_portable_python(root)

    def test_explicit_path_overrides_environment(self):
        with tempfile.TemporaryDirectory() as temp:
            root = portable_runtime(Path(temp))
            with patch.dict(os.environ, HARNESSADAPT_PORTABLE_PYTHON="/missing"):
                self.assertEqual(resolve_portable_python(root), root)
            with patch.dict(os.environ, HARNESSADAPT_PORTABLE_PYTHON=str(root)):
                self.assertEqual(resolve_portable_python(), root)


class LaunchTests(unittest.TestCase):
    @contextmanager
    def launcher(self):
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            root = Path(temp)
            config = recipe_defaults("grpo")
            config.update(input=str(root / "train.jsonl"))

            def prepare(root, benchmark="verified", **kwargs):
                path = input_directory(root, benchmark) / "verified.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"label": benchmark}) + "\n")

            stack.enter_context(patch.dict(os.environ, {}, clear=True))
            stack.enter_context(patch.object(environment, "check_prerequisites"))
            stack.enter_context(patch.object(verified, "prepare_verified", side_effect=prepare))
            stack.enter_context(patch.object(verified, "prepare_benchmark", side_effect=prepare))
            stack.enter_context(patch.object(continuation, "require_optimizer_complete"))
            stack.enter_context(patch.object(continuation, "check_checkpoint"))
            summary = stack.enter_context(patch.object(continuation, "evaluation_summary"))
            scorer = stack.enter_context(patch.object(swepro_scorer, "score_swepro_run"))
            stack.enter_context(patch.object(provenance, "freeze_or_verify"))
            docker = stack.enter_context(patch.object(cli, "docker"))
            yield root, config, docker, scorer, summary

    @staticmethod
    def launch_env(docker):
        command = docker.call_args.args
        return {command[i + 1].split("=", 1)[0]: command[i + 1].split("=", 1)[1]
                for i, arg in enumerate(command[:-1]) if arg == "-e" and "=" in command[i + 1]}

    def test_benchmarks_and_repetitions_have_disjoint_outputs(self):
        with self.launcher() as (root, config, docker, scorer, summary):
            outputs, inputs = set(), set()
            for benchmark in ("verified", "multilingual", "swepro"):
                for repetition in (1, 2):
                    cli.launch(root, config, evaluation=True, stop_after=45,
                               benchmark=benchmark, repetition=repetition)
                    env = self.launch_env(docker)
                    outputs.add(env["R2E_TRAJECTORY_DIR"])
                    inputs.add(env["VALIDATION_DATA"])
                    self.assertEqual(int(env["ROLLOUT_SEED"]), config["seed"] + repetition - 1)
                    self.assertEqual(env["CKPT_STEP"], "44")
                    self.assertNotIn("R2E_PORTABLE_PYTHON_PATH", env)
                    meta = read_json(Path(env["RUN_ROOT"]) / "run.json")
                    self.assertEqual((meta["benchmark"], meta["repetition"]), (benchmark, repetition))
            self.assertEqual(len(outputs), 6)
            self.assertEqual(len(inputs), 3)
            self.assertEqual(scorer.call_args.kwargs["repetition"], 2)
            self.assertEqual(summary.call_args.kwargs, {"benchmark": "multilingual", "repetition": 2})

    def test_nonempty_attempt_is_not_reused_and_next_number_succeeds(self):
        with self.launcher() as (root, config, docker, _, __):
            cli.launch(root, config, evaluation=True, stop_after=45)
            with self.assertRaisesRegex(FileExistsError, "--repetition"):
                cli.launch(root, config, evaluation=True, stop_after=45)
            self.assertEqual(docker.call_count, 1)
            cli.launch(root, config, evaluation=True, stop_after=45, repetition=2)
            self.assertEqual(docker.call_count, 2)

    def test_supplied_runtime_is_mounted_for_ray_and_task_containers(self):
        with self.launcher() as (root, config, docker, _, __):
            portable = portable_runtime(root / "python")
            cli.launch(root, config, evaluation=True, stop_after=45,
                       benchmark="multilingual", portable_python=portable)
            env = self.launch_env(docker)
            self.assertEqual(env["R2E_PORTABLE_PYTHON_PATH"], str(portable))
            self.assertEqual(env["R2E_PORTABLE_PYTHON_TOOLS"], "1")
            self.assertIn(f"{portable}:{portable}:ro", docker.call_args.args)

    def test_invalid_portable_path_fails_before_docker_or_dataset_preparation(self):
        with self.launcher() as (root, config, docker, _, __):
            with self.assertRaises(ValueError):
                cli.launch(root, config, evaluation=True, benchmark="multilingual", portable_python=root)
            docker.assert_not_called()
            verified.prepare_benchmark.assert_not_called()

    def test_cli_forwards_repetition(self):
        with patch.object(cli, "preflight", return_value=recipe_defaults()), patch.object(cli, "launch") as launch:
            cli.main(["evaluate", "--run-dir", "/tmp/test-run", "--stop-after", "45",
                      "--benchmark", "multilingual", "--repetition", "3"])
        self.assertEqual(launch.call_args.kwargs["repetition"], 3)


class ScoringAndResumeTests(unittest.TestCase):
    def test_verified_summary_ignores_other_benchmarks_and_repetitions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inputs = input_directory(root, "verified")
            inputs.mkdir(parents=True)
            (inputs / "verified.jsonl").write_text('{"label": "task"}\n')
            expected = evaluation_directory(root, "verified", 45)
            record(expected / "trajectories/task.json.gz", reward=1)
            record(evaluation_directory(root, "swepro", 45) / "trajectories/other.json.gz")
            record(evaluation_directory(root, "verified", 45, 2) / "trajectories/task.json.gz")
            record(root / "verified/step_0045/trajectories/legacy.json.gz")
            self.assertEqual(continuation.evaluation_summary(root, 45)["pass_at_1"], 1)
            self.assertEqual(continuation.evaluation_summary(root, 45, repetition=2)["pass_at_1"], 0)

    def test_swepro_scoring_isolates_inputs_results_and_summary(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(swepro_scorer, "docker") as docker:
            root = Path(temp)
            directory = evaluation_directory(root, "swepro", 45, 2)
            for i in range(642):
                record(directory / f"trajectories/{i}.json.gz", str(i))
            record(evaluation_directory(root, "verified", 45) / "trajectories/unrelated.json.gz")
            record(evaluation_directory(root, "swepro", 45, 1) / "trajectories/old.json.gz")
            with redirect_stdout(io.StringIO()):
                result = swepro_scorer.score_swepro_run(root, step=45, repetition=2,
                                                       tasks_root=root / "absent-tasks", workers=1)
            self.assertTrue(result["complete"])
            self.assertEqual(result["graded"], 642)
            self.assertEqual(read_json(directory / "summary.json"), result)
            self.assertEqual(len(list((directory / "official_v2_results").glob("*/result.json"))), 642)
            self.assertFalse((root / "official_v2_summary.json").exists())
            docker.assert_not_called()  # Empty submissions are protocol-defined failures.

    def test_resume_preserves_failed_attempt_and_selects_fresh_repetition(self):
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            root = Path(temp)
            failed = evaluation_directory(root, "verified", 5)
            record(failed / "trajectories/partial.json.gz")
            write_json(root / "steps/0005/optimizer_complete.json", {
                "step": 5, "optimizer_call_completed": True,
                "inference_weights_synchronized": True, "nonzero_parameter_update_observed": True,
            })
            stack.enter_context(patch.object(cli, "preflight", return_value=recipe_defaults()))
            stack.enter_context(patch.object(verified, "prepare_verified"))
            stack.enter_context(patch.object(continuation, "check_checkpoint"))
            stack.enter_context(patch.object(continuation, "evaluation_summary"))
            launch = stack.enter_context(patch.object(cli, "launch"))
            continuation.run(root, 5)
            self.assertEqual(launch.call_args.kwargs["repetition"], 2)
            self.assertTrue((failed / "trajectories/partial.json.gz").exists())


class EnvironmentTests(unittest.TestCase):
    def test_missing_setup_reports_all_prerequisites_and_instructions(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(environment.shutil, "which", return_value=None):
            root = Path(temp)
            with self.assertRaises(RuntimeError) as error:
                environment.check_prerequisites({"model": str(root / "model"), "reference": str(root / "ref")},
                                                workspace=root)
            message = str(error.exception)
            for text in ("Slime", "R2E-Gym", "base model", "reference checkpoint", "Docker", "TRAINING_SETUP.md"):
                self.assertIn(text, message)

    def test_teacher_requirements_follow_recipe(self):
        self.assertEqual(environment.teacher_backends(recipe_defaults("grpo")), [])
        self.assertEqual(environment.teacher_backends(recipe_defaults("skill_mechanism")), ["opencode"])
        self.assertEqual(environment.teacher_backends(recipe_defaults()), ["codex", "opencode"])


if __name__ == "__main__":
    unittest.main()
