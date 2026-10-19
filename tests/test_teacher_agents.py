"""Teacher orchestration tests; no installed CLI, model, or network is used."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from harnessadapt.training import teacher, teacher_agents
from harnessadapt.training.config import read_json


class AgentInvocationTests(unittest.TestCase):
    def invoke(self, backend="codex", *, answer="final answer", events=None, returncode=0):
        seen = {}

        def popen(command, **kwargs):
            seen.update(command=command, **kwargs)
            if "--output-last-message" in command and answer is not None:
                Path(command[command.index("--output-last-message") + 1]).write_text(answer)
            for event in events or []:
                kwargs["stdout"].write(json.dumps(event) + "\n")
            process = Mock(returncode=returncode)
            process.communicate.return_value = (None, None)
            seen["process"] = process
            return process

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch.object(teacher_agents, "agent_executable", return_value="/external/agent"),
                patch.object(teacher_agents.subprocess, "Popen", side_effect=popen),
            ):
                result = teacher_agents.call_teacher_agent(
                    "large trajectory input", backend=backend, model="requested-model",
                    base_url="https://teacher.example/v1", api_key="test-secret",
                    run_dir=root / "logs", workspace=root / "workspace" if backend == "codex" else None,
                )
            self.assertTrue((root / "logs/events.jsonl").is_file())
            self.assertTrue((root / "logs/stderr.log").is_file())
        return result, seen

    def test_codex_uses_agent_exec_and_stdin_with_environment_auth(self):
        result, seen = self.invoke(events=[
            {"type": "item.completed", "item": {"type": "command_execution", "output": "tool output"}},
            {"type": "turn.completed"},
        ])
        self.assertEqual(result, "final answer")
        self.assertIn("exec", seen["command"])
        self.assertIn("workspace-write", seen["command"])
        self.assertIn('model_providers.harnessadapt.wire_api="responses"', seen["command"])
        self.assertNotIn("test-secret", " ".join(seen["command"]))
        self.assertEqual(seen["env"]["HARNESSADAPT_AGENT_API_KEY"], "test-secret")
        seen["process"].communicate.assert_called_once_with("large trajectory input", timeout=600)

    def test_opencode_disables_tools_and_extracts_only_text_events(self):
        result, seen = self.invoke("opencode", events=[
            {"type": "step_start"},
            {"type": "reasoning", "part": {"text": "private reasoning"}},
            {"type": "text", "part": {"text": "skill answer"}},
        ])
        self.assertEqual(result, "skill answer")
        config = json.loads(seen["env"]["OPENCODE_CONFIG_CONTENT"])
        self.assertEqual(config["agent"]["teacher"]["permission"], {"*": "deny"})
        self.assertIn("harnessadapt/requested-model", seen["command"])
        self.assertNotIn("test-secret", json.dumps(config))

    def test_failure_exit_and_failed_turn_are_not_answers(self):
        for kwargs in ({"returncode": 2}, {"events": [{"type": "turn.failed"}]}, {"answer": None}):
            with self.subTest(kwargs=kwargs), self.assertRaises(RuntimeError):
                self.invoke(**kwargs)

    def test_missing_cli_fails_without_http_fallback(self):
        with patch.object(teacher_agents.shutil, "which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "HARNESSADAPT_CODEX_BIN"):
                teacher_agents.agent_executable("codex")

    def test_timeout_kills_process_group_and_retains_logs(self):
        process = Mock(pid=12345)
        process.communicate.side_effect = [subprocess.TimeoutExpired("agent", 1), (None, None)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch.object(teacher_agents.subprocess, "Popen", return_value=process),
                patch.object(teacher_agents.os, "killpg") as kill,
                self.assertRaises(TimeoutError),
            ):
                teacher_agents._run(["agent"], "prompt", cwd=root, env={}, run_dir=root, timeout=1)
            kill.assert_called_once_with(12345, teacher_agents.signal.SIGKILL)
            self.assertTrue((root / "events.jsonl").exists())


class HarnessAgentTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.harness = self.root / "incumbent"
        shutil.copytree(ROOT / "harnesses/h1", self.harness)
        self.markdown = self.root / "trajectories"
        self.markdown.mkdir()
        (self.markdown / "task.md").write_text("Historical tool observations")

    def propose(self, action, answer="Implemented and checked candidate"):
        def agent(prompt, **kwargs):
            self.assertEqual(kwargs["backend"], "codex")
            self.assertIn("Historical tool observations", prompt)
            self.assertEqual(kwargs["model"], "gpt-5.6-sol")
            action(kwargs["workspace"] / "candidate_harness")
            return answer

        with patch.object(teacher, "call_teacher_agent", side_effect=agent):
            return teacher.propose_harness_change(
                self.root / "proposal", self.markdown, self.harness, [], {}, epoch=1, round_index=1,
            )

    def test_actual_agent_edit_becomes_validated_candidate(self):
        original = (self.harness / "instance_template.md").read_bytes()
        result = self.propose(lambda candidate: (candidate / "instance_template.md").write_bytes(original + b"\nCheck observations.\n"))
        self.assertIsNotNone(result)
        self.assertEqual((self.harness / "instance_template.md").read_bytes(), original)
        self.assertEqual((result / "instance_template.md").read_bytes(), original + b"\nCheck observations.\n")
        self.assertEqual(read_json(result / "release.json")["harness_release"],
                         read_json(self.harness / "release.json")["harness_release"] + 1)
        self.assertEqual(read_json(self.root / "proposal/proposal_result.json")["proposed_files"], ["instance_template.md"])

    def test_no_change_does_not_create_candidate(self):
        self.assertIsNone(self.propose(lambda _: None, "no_change"))
        self.assertEqual(read_json(self.root / "proposal/proposal_result.json")["status"], "no_change")
        self.assertFalse((self.root / "proposal/candidate_harness").exists())

    def test_markdown_code_without_edits_is_not_a_candidate(self):
        self.assertIsNone(self.propose(lambda _: None, "### instance_template.md\n```text\nnew prompt\n```"))
        self.assertEqual(read_json(self.root / "proposal/proposal_result.json")["status"], "no_files_changed")

    def test_skill_bank_edit_is_rejected(self):
        self.assertIsNone(self.propose(lambda candidate: (candidate / "system_prompt.md").write_text("unapproved change")))
        self.assertEqual(read_json(self.root / "proposal/proposal_result.json")["status"], "validation_failed")

    def test_deletion_is_rejected(self):
        self.assertIsNone(self.propose(lambda candidate: (candidate / "instance_template.md").unlink()))
        self.assertEqual(read_json(self.root / "proposal/proposal_result.json")["status"], "validation_failed")

    def test_symlink_is_rejected(self):
        self.assertIsNone(self.propose(lambda candidate: (candidate / "hooks/linked.py").symlink_to(self.harness / "instance_template.md")))
        self.assertEqual(read_json(self.root / "proposal/proposal_result.json")["status"], "validation_failed")


if __name__ == "__main__":
    unittest.main()
