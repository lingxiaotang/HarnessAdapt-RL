"""Standalone write_file tool for the OpenCode-inspired Mini-SWE harness."""

import base64
import json
import shlex

TOOL_SCHEMA = json.loads(r'''{
  "type": "function",
  "function": {
    "name": "write_file",
    "description": "Create a new UTF-8 source file without overwriting an existing path. Use edit_file for existing files. Parent directories must already exist; create them with bash when needed. Relative paths start in /testbed, and /tmp is available for reproduction scripts. Repository tests/configuration are protected. A created source file remains untracked until you explicitly include it in your final patch. This never stages or submits.",
    "parameters": {
      "type": "object",
      "properties": {
        "path": {
          "type": "string",
          "description": "New source file under /testbed, or scratch file under /tmp."
        },
        "content": {
          "type": "string",
          "description": "Complete UTF-8 file content, at most 250000 characters."
        }
      },
      "required": [
        "path",
        "content"
      ],
      "additionalProperties": false
    }
  }
}''')
SMOKE_ARGUMENTS = json.loads(r'''{"path":"new_helper.py","content":"def helper():\n    return True\n"}''')

_SCRIPT = r'''import codecs
import difflib
import fnmatch
import io
import json
import os
from pathlib import Path
import re
import signal
import stat
import sys
import tempfile
import tokenize

ROOT = Path("/testbed")
SCRATCH = Path("/tmp")
MAX_BYTES = 4 * 1024 * 1024
MAX_OUTPUT = 9000


def fail(code, message):
    print("ERROR [" + code + "]: " + str(message)[:1400])
    raise SystemExit(2)


def timeout_handler(signum, frame):
    fail("timeout", "Operation exceeded 8 seconds. Narrow the scope or use a bounded bash command.")


def arguments(required, optional):
    value = json.loads(sys.argv[1])
    if not isinstance(value, dict):
        fail("arguments", "Provide a JSON object.")
    missing = sorted(set(required) - set(value))
    extra = sorted(set(value) - set(required) - set(optional))
    if missing or extra:
        fail("arguments", "Missing fields: %s; unknown fields: %s." % (missing, extra))
    return value


def integer(value, name, minimum, maximum):
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        fail("arguments", "%s must be an integer from %s to %s." % (name, minimum, maximum))
    return value


def inside(path, root):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def checked_path(raw, mutate=False):
    if not isinstance(raw, str) or not raw or len(raw) > 1024 or any(c in raw for c in "\x00\n\r"):
        fail("path", "Provide a nonempty path of at most 1024 characters without NUL/newlines.")
    path = Path(raw)
    if ".." in path.parts or ".git" in path.parts:
        fail("path", "Parent traversal and .git paths are unavailable. Use an explicit /testbed or /tmp path.")
    if not path.is_absolute():
        path = ROOT / path
    root = ROOT.resolve()
    scratch = SCRATCH.resolve()
    if not (inside(path, root) or inside(path, scratch)):
        fail("path", "Use a path inside /testbed or /tmp.")
    resolved = path.resolve()
    if not (inside(resolved, root) or inside(resolved, scratch)) or ".git" in resolved.parts:
        fail("path", "The resolved path must remain inside /testbed or /tmp, outside .git.")
    if mutate:
        if any(part.is_symlink() for part in (path,) + tuple(path.parents)):
            fail("path", "Editing through symlinks is unsupported; use a direct source path.")
        for candidate in (path, resolved):
            if inside(candidate, root):
                relative = candidate.relative_to(root)
                parts = [p.lower() for p in relative.parts]
                name = parts[-1] if parts else ""
                forbidden_dirs = {"test", "tests", "testing", "__tests__", "r2e_tests", "config", "configs", ".github", ".gitlab"}
                forbidden_names = {
                    "setup.py", "setup.cfg", "pyproject.toml", "tox.ini", "pytest.ini", "conftest.py",
                    "package.json", "package-lock.json", "npm-shrinkwrap.json", "yarn.lock",
                    "pnpm-lock.yaml", "poetry.lock", "pipfile", "pipfile.lock",
                    "environment.yml", "environment.yaml", "dockerfile", "makefile",
                    "patch.txt", "test.py", "tests.py", ".gitignore", ".gitattributes", ".gitmodules",
                }
                if (any(p in forbidden_dirs and not (i == 1 and tuple(parts[:2]) in {("numpy", "testing"), ("pyramid", "config"), ("django", "test"), ("pylint", "config")}) for i, p in enumerate(parts))
                    or name in forbidden_names or name.startswith("test_")
                    or re.search(r"(^test\.|[._]tests?\.[^.]+$)", name)
                    or name.startswith(("requirements", "repro", "debug_", "verify_fix", "check_fix"))
                    or re.search(r"(?:_tests?|\.test|\.spec)\.[^.]+$", name)
                    or Path(name).suffix in {".toml", ".ini", ".cfg", ".yaml", ".yml", ".lock", ".json", ".txt", ".md", ".rst", ".log", ".patch", ".diff", ".pyc", ".pyo", ".csv", ".png", ".jpg", ".zip", ".gz"}):
                    fail("source_only", "This path is excluded by the conservative source-file filter. Use /tmp for reproduction scripts; handle unusual genuine source files with bash while respecting task restrictions.")
        if resolved.exists() and resolved.stat().st_nlink > 1:
            fail("path", "Hard-linked files are unsupported; inspect the path with bash.")
    return resolved


def read_text(path):
    if not path.is_file():
        fail("not_file", "No regular file at %s. Locate the file with grep or bash." % path)
    if path.stat().st_size > MAX_BYTES:
        fail("file_too_large", "File exceeds 4 MiB. Use a focused, bounded bash command.")
    data = path.read_bytes()
    if len(data) > MAX_BYTES:
        fail("file_too_large", "File exceeds 4 MiB.")
    if b"\x00" in data:
        fail("binary", "Binary/UTF-16 files are unsupported; inspect with bash.")
    encoding = "utf-8-sig" if data.startswith(codecs.BOM_UTF8) else "utf-8"
    if path.suffix in {".py", ".pyi"}:
        try:
            encoding, unused = tokenize.detect_encoding(io.BytesIO(data).readline)
        except (SyntaxError, UnicodeError) as error:
            fail("encoding", str(error))
    try:
        return data, data.decode(encoding), encoding
    except UnicodeError:
        fail("encoding", "Cannot decode as %s. Inspect the file encoding with bash." % encoding)


signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(8)

def main():
    a = arguments({"path", "content"}, set())
    content = a["content"]
    if not isinstance(content, str) or len(content) > 250000 or "\x00" in content:
        fail("arguments", "content must be text without NUL, at most 250000 characters.")
    path = checked_path(a["path"], mutate=True)
    if path.exists():
        fail("exists", "File already exists; read it and use edit_file. Nothing was overwritten.")
    if not path.parent.is_dir():
        fail("parent", "Parent directory does not exist; create it with bash, then retry.")
    encoded = content.encode("utf-8")
    descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink()
        raise
    print("CREATED %s | bytes=%s | UTF-8. Not staged or submitted; include source changes explicitly in your final patch." % (str(path)[:500], len(encoded)))

try:
    main()
except (OSError, ValueError, TypeError, UnicodeError) as error:
    fail("operation", "%s: %s" % (type(error).__name__, error))
finally:
    signal.alarm(0)
'''

def execute(arguments, environment):
    """Execute only inside the task environment; no host file access."""
    payload = base64.b64encode(json.dumps({"script": _SCRIPT, "arguments": json.dumps(arguments, ensure_ascii=False)}, ensure_ascii=False).encode("utf-8")).decode("ascii")
    launcher = "import base64,json,sys; p=json.loads(base64.b64decode(sys.argv[1])); sys.argv=['harness',p['arguments']]; exec(compile(p['script'],'<harness-tool>','exec'))"
    command = "timeout 12s python3 -c " + shlex.quote(launcher) + " " + shlex.quote(payload)
    if len(command.encode("utf-8")) > 100000:
        return {
            "output": "ERROR [arguments_too_large]: Split this operation into smaller edits; the encoded command exceeds 100000 bytes.",
            "returncode": 2,
            "exception_info": "",
        }
    result = environment.execute({"command": command})
    result = dict(result)
    output = str(result.get("output") or "")
    if len(output) > 9400:
        output = output[:9100] + "\n[Output truncated. Narrow the request or use bounded bash output.]"
    if result.get("returncode") == 124 and not output:
        output = "ERROR [timeout]: Operation exceeded 12 seconds. Narrow the scope or use bounded bash."
    result["output"] = output
    result.setdefault("exception_info", "")
    return result
