"""Standalone grep tool for the OpenCode-inspired Mini-SWE harness."""

import base64
import json
import shlex

TOOL_SCHEMA = json.loads(r'''{
  "type": "function",
  "function": {
    "name": "grep",
    "description": "Search text with a regex or literal string, returning file paths and line numbers. Narrow path and include before broad searches; use read_file around a match. Results, file sizes, and runtime are bounded, and incomplete searches are marked. Git metadata, symlinks, caches, and binary files are skipped. If a search times out, simplify it or use a bounded bash search.",
    "parameters": {
      "type": "object",
      "properties": {
        "pattern": {
          "type": "string",
          "description": "Regex by default; at most 1000 characters."
        },
        "path": {
          "type": "string",
          "description": "File/directory under /testbed or /tmp; default '.'."
        },
        "include": {
          "type": "string",
          "description": "Filename or relative-path glob, e.g. '*.py' or 'src/*.py'; default '*'."
        },
        "literal": {
          "type": "boolean",
          "description": "Match the exact pattern text instead of regex; default false."
        },
        "max_results": {
          "type": "integer",
          "description": "Maximum matching lines, 1–100; default 60.",
          "minimum": 1,
          "maximum": 100
        }
      },
      "required": [
        "pattern"
      ],
      "additionalProperties": false
    }
  }
}''')
SMOKE_ARGUMENTS = json.loads(r'''{"pattern":"class ","path":".","include":"*.py","literal":true,"max_results":10}''')

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
                if (any(p in forbidden_dirs for p in parts)
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
    a = arguments({"pattern"}, {"path", "include", "literal", "max_results"})
    pattern = a["pattern"]
    include = a.get("include", "*")
    literal = a.get("literal", False)
    if not isinstance(pattern, str) or not pattern or len(pattern) > 1000:
        fail("arguments", "pattern must contain 1–1000 characters.")
    if not isinstance(include, str) or not include or len(include) > 500:
        fail("arguments", "include must be a nonempty glob of at most 500 characters.")
    if not isinstance(literal, bool):
        fail("arguments", "literal must be true or false.")
    maximum = integer(a.get("max_results", 60), "max_results", 1, 100)
    path = checked_path(a.get("path", "."))
    if not path.exists() or not (path.is_dir() or path.is_file()):
        fail("not_found", "Search path must be an existing file or directory.")
    try:
        expression = re.compile(re.escape(pattern) if literal else pattern)
    except re.error as error:
        fail("regex", "%s. Fix the regex or set literal=true." % error)
    rows = []
    size = 0
    scanned = 0
    skipped = 0
    truncated = False
    excluded = {".git", ".hg", ".svn", "__pycache__", ".pytest_cache", ".mypy_cache", ".venv", "venv", "node_modules"}
    def candidates():
        if path.is_file():
            yield path
            return
        def walk_error(error):
            nonlocal skipped
            skipped += 1
        for directory, dirs, files in os.walk(str(path), followlinks=False, onerror=walk_error):
            dirs[:] = sorted(d for d in dirs if d not in excluded and not (Path(directory) / d).is_symlink())
            for name in sorted(files):
                if name in excluded:
                    continue
                candidate = Path(directory) / name
                relative = str(candidate.relative_to(path))
                if fnmatch.fnmatch(name, include) or fnmatch.fnmatch(relative, include):
                    yield candidate
    for candidate in candidates():
        if candidate.is_symlink() or not candidate.is_file():
            skipped += 1
            continue
        if scanned >= 10000:
            truncated = True
            break
        scanned += 1
        try:
            if candidate.stat().st_size > MAX_BYTES:
                skipped += 1
                continue
            data = candidate.read_bytes()
            if len(data) > MAX_BYTES or b"\x00" in data:
                skipped += 1
                continue
            text = data.decode("utf-8-sig")
        except (OSError, UnicodeError):
            skipped += 1
            continue
        for number, line in enumerate(text.splitlines(), 1):
            if not expression.search(line):
                continue
            fragment = line[:1000]
            if len(line) > 1000:
                fragment += " [LINE TRUNCATED]"
            row = "%s:%s: %s" % (str(candidate)[:600], number, fragment)
            if len(rows) >= maximum or size + len(row) + 1 > MAX_OUTPUT - 450:
                truncated = True
                break
            rows.append(row)
            size += len(row) + 1
        if truncated:
            break
    summary = "SEARCH matches_shown=%s files_scanned=%s skipped=%s." % (len(rows), scanned, skipped)
    if truncated:
        summary += " TRUNCATED: narrow path/include/pattern or use bounded bash."
    elif skipped:
        summary += " INCOMPLETE: skipped unreadable, oversized, non-UTF-8 or binary files."
    else:
        summary += " Complete within scope (Git metadata, symlinks and caches excluded)."
    print("\n".join(rows + [summary]))

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
