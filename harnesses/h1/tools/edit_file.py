"""Standalone edit_file tool for the OpenCode-inspired Mini-SWE harness."""

import base64
import json
import shlex

TOOL_SCHEMA = json.loads(r'''{
  "type": "function",
  "function": {
    "name": "edit_file",
    "description": "Replace exactly one occurrence of old_text in an existing source file. Read the region first; copy exact text without displayed line numbers and include enough context to make the anchor unique. Zero or multiple matches leave the file unchanged. Preserves supported encoding, newline style and permissions, and returns a short diff. Repository tests/configuration are protected; /tmp may hold reproduction scripts. This edits only; it never submits.",
    "parameters": {
      "type": "object",
      "properties": {
        "path": {
          "type": "string",
          "description": "Existing source file, relative to /testbed or absolute /testbed or /tmp."
        },
        "old_text": {
          "type": "string",
          "description": "Nonempty exact anchor occurring once; at most 250000 characters."
        },
        "new_text": {
          "type": "string",
          "description": "Replacement text; empty string deletes the anchor; at most 250000 characters."
        }
      },
      "required": [
        "path",
        "old_text",
        "new_text"
      ],
      "additionalProperties": false
    }
  }
}''')
SMOKE_ARGUMENTS = json.loads(r'''{"path":"example.py","old_text":"return False","new_text":"return True"}''')

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
MAX_OUTPUT = 4500


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
    a = arguments({"path", "old_text", "new_text"}, set())
    old = a["old_text"]
    new = a["new_text"]
    if not isinstance(old, str) or not old or len(old) > 250000:
        fail("arguments", "old_text must contain 1–250000 characters.")
    if not isinstance(new, str) or len(new) > 250000:
        fail("arguments", "new_text must be a string of at most 250000 characters.")
    path = checked_path(a["path"], mutate=True)
    data, text, encoding = read_text(path)
    before = path.stat()
    # Normalize argument newlines only when the file has one unambiguous style.
    if "\r\n" in text and "\n" not in text.replace("\r\n", ""):
        old = old.replace("\r\n", "\n").replace("\n", "\r\n")
        new = new.replace("\r\n", "\n").replace("\n", "\r\n")
    elif "\r" not in text:
        old = old.replace("\r\n", "\n")
        new = new.replace("\r\n", "\n")
    positions = []
    start = 0
    while len(positions) < 6:
        at = text.find(old, start)
        if at < 0:
            break
        positions.append(at)
        start = at + 1
    count = len(positions)
    if count != 1:
        message = "Anchor matches %s%s times; file unchanged. " % ("at least " if count == 6 else "", count)
        if count == 0:
            message += "Re-read the region, preserve whitespace, and omit displayed line numbers."
        else:
            lines = [str(text.count("\n", 0, at) + 1) for at in positions[:5]]
            message += "Matches start at lines %s; add surrounding context to old_text." % ", ".join(lines)
        fail("anchor", message)
    changed = text.replace(old, new, 1)
    if changed == text:
        print("UNCHANGED: old_text equals new_text; no file write.")
        return
    encoded = changed.encode(encoding)
    if len(encoded) > MAX_BYTES:
        fail("file_too_large", "Result exceeds 4 MiB; file unchanged.")
    # Prepare bounded feedback before writing: a slow diff must not mutate then report failure.
    header = "EDITED %s | one replacement | encoding=%s\n" % (str(path)[:500], encoding)
    rows = []
    size = len(header)
    clipped = False
    diff = difflib.unified_diff(text.splitlines(), changed.splitlines(), fromfile="before", tofile="after", lineterm="", n=3)
    for row in diff:
        if len(row) > 1800:
            row = row[:1800] + " [LINE TRUNCATED]"
            clipped = True
        if size + len(row) + 1 > MAX_OUTPUT - 180:
            clipped = True
            break
        rows.append(row)
        size += len(row) + 1
    trailer = "\n[Diff truncated; use read_file/view_diff to inspect.]" if clipped else ""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".harness-edit-", dir=str(path.parent), delete=False) as handle:
            temporary = handle.name
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, stat.S_IMODE(before.st_mode))
        current = path.stat()
        if (current.st_ino, current.st_mtime_ns, current.st_size) != (before.st_ino, before.st_mtime_ns, before.st_size) or path.read_bytes() != data:
            fail("changed_during_edit", "File changed while editing; re-read and retry. This tool did not replace it.")
        os.replace(temporary, str(path))
        temporary = None
    finally:
        if temporary is not None:
            os.unlink(temporary)
    print(header + "\n".join(rows) + trailer)

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
