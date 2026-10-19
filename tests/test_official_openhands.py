"""Official backend boundary tests. Authored without running them.

The integration test requires the pinned SDK in Python 3.12 and uses a local
scripted HTTP model, a disposable repository, and the REAL official SDK/tools.
It does not download packages/models or start training/Docker.
"""
from __future__ import annotations

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from harnessadapt.openhands import BACKEND, VERSION, provenance
from harnessadapt.openhands.bridge import ModelConnection, OfficialAgent, runtime_path
from harnessadapt.openhands.worker import execute, git_patch
from harnessadapt.training.recipes import recipe_defaults, validate_recipe


class BoundaryTests(unittest.TestCase):
    def agent(self):
        return OfficialAgent(ModelConnection("http://127.0.0.1/v1", "secret-test-key"), Mock())

    def test_recipe_rejects_legacy_backend_and_local_harness(self):
        config = recipe_defaults("openhands_grpo")
        validate_recipe(config)
        for overrides in ({"backend": "openhands_codeact_v1"}, {"backend": "miniswe"},
                          {"openhands_version": "0.0.0"}, {"fixed_harness_source": "/tmp/local"},
                          {"skill_variant": "mechanism"}, {"verified_evolved_harness": True}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                validate_recipe(dict(config, **overrides))

    def test_unfinished_empty_or_truncated_results_cannot_submit(self):
        good = {"provenance": provenance(), "exit_status": "Submitted",
                "sdk_status": "finished", "submission": "diff --git a/x b/x\n"}
        self.assertEqual(self.agent()._accept_result(good)["submission"], good["submission"])
        for overrides in ({"sdk_status": "error"}, {"submission": ""},
                          {"diagnostics": {"generation_length_exit": True}},
                          {"diagnostics": {"context_budget_exit": True}}):
            with self.subTest(overrides=overrides), self.assertRaises(RuntimeError):
                self.agent()._accept_result(dict(good, **overrides))

    def test_sdk_failure_is_not_converted_to_known_reward(self):
        with self.assertRaisesRegex(RuntimeError, "Official OpenHands failed"):
            self.agent()._accept_result({"provenance": provenance(), "exit_status": "Error",
                                         "error": "provider unavailable"})

    def test_budget_exit_preserves_diagnostics_and_discards_patch(self):
        agent = self.agent()
        result = agent._accept_result({"provenance": provenance(), "exit_status": "GenerationLengthExceeded",
                                       "submission": "must not be graded", "n_calls": 2,
                                       "diagnostics": {"generation_length_exit": True}})
        self.assertEqual(result["submission"], "")
        self.assertTrue(agent.model.generation_length_exit)
        self.assertEqual(agent.n_calls, 2)

    def test_transport_failure_does_not_expose_upload_or_credential(self):
        agent = self.agent()
        agent.environment.execute.return_value = {"returncode": 1, "output": "secret-test-key"}
        with self.assertRaises(RuntimeError) as error:
            agent._exec("printf secret-test-key")
        self.assertNotIn("secret-test-key", str(error.exception))

    def test_runtime_version_mismatch_is_rejected_before_container_start(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "manifest.json").write_text(json.dumps({"backend": BACKEND, "packages": {}}))
            with patch.dict(os.environ, HARNESSADAPT_OPENHANDS_RUNTIME=directory):
                with self.assertRaises(ValueError):
                    runtime_path()


@contextmanager
def repository():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)

        def git(*args):
            return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True).stdout

        git("init", "--quiet")
        (root / "module.py").write_text("value = 1\n")
        git("add", "module.py")
        git("-c", "user.name=Test", "-c", "user.email=test@example.invalid",
            "-c", "commit.gpgsign=false", "commit", "--quiet", "-m", "fixture")
        yield root, git, git("rev-parse", "HEAD").decode().strip()


@unittest.skipUnless(shutil.which("git"), "Requires Git for disposable repositories")
class PatchTests(unittest.TestCase):
    def test_collects_staged_new_and_deleted_changes_without_touching_real_index(self):
        with repository() as (root, git, base):
            (root / "r2e_tests").write_text("benchmark fixture")
            (root / "module.py").unlink()
            (root / "new.py").write_text("value = 2\n")
            git("add", "new.py")
            index_before = git("ls-files", "--stage")
            result = git_patch(str(root), base, ["r2e_tests"])
            self.assertIn("deleted file mode", result)
            self.assertIn("new file mode", result)
            self.assertNotIn("r2e_tests", result)
            self.assertEqual(index_before, git("ls-files", "--stage"))


def sdk_available():
    try:
        return (sys.version_info >= (3, 12) and shutil.which("git")
                and version("openhands-sdk") == VERSION and version("openhands-tools") == VERSION)
    except PackageNotFoundError:
        return False


@unittest.skipUnless(sdk_available(), "Requires pinned official OpenHands SDK/tools in Python 3.12")
class OfficialSDKIntegrationTests(unittest.TestCase):
    def test_real_sdk_edits_and_finishes_via_openai_session_endpoint(self):
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests.append((self.path, self.headers.get("Authorization"), body))
                if len(requests) == 1:
                    name, arguments = "terminal", {"command": "printf 'value = 2\\n' > module.py"}
                else:
                    name, arguments = "finish", {"message": "Fixed module.py"}
                payload = {"id": f"completion-{len(requests)}", "object": "chat.completion",
                           "created": 1, "model": "test-model", "choices": [{"index": 0,
                           "finish_reason": "tool_calls", "message": {"role": "assistant", "content": "",
                           "tool_calls": [{"id": f"call-{len(requests)}", "type": "function",
                                           "function": {"name": name, "arguments": json.dumps(arguments)}}]}}],
                           "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}}
                raw = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, *_args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with repository() as (root, _git, _base), tempfile.TemporaryDirectory() as state:
                with patch.dict(os.environ, OH_PERSISTENCE_DIR=state):
                    result = execute({"version": VERSION, "workspace": str(root),
                                      "task": "Change module.py to value = 2, then finish.",
                                      "model": "openai/test-model", "base_url": f"http://127.0.0.1:{server.server_port}/v1",
                                      "api_key": "test-session-123", "temperature": 1.0, "top_p": 1.0,
                                      "context_tokens": 32768, "max_output_tokens": 1024,
                                      "request_timeout": 10, "turns": 3}, 60)
                self.assertEqual(result["exit_status"], "Submitted", result.get("error"))
                self.assertIn("+value = 2", result["submission"])
                self.assertEqual(result["provenance"], provenance())
                self.assertEqual(result["n_calls"], 2)
                self.assertTrue(all(path == "/v1/chat/completions" and auth == "Bearer test-session-123"
                                    for path, auth, _body in requests))
                names = {tool["function"]["name"] for tool in requests[0][2]["tools"]}
                self.assertIn("terminal", names)
                self.assertIn("finish", names)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
