"""Read a bounded page of selected unstaged source changes without mutation."""

import json
import re
import shlex

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "view_diff",
        "description": (
            "Inspect a bounded page of current changes for exact source files, including "
            "untracked new files and deletions. line_offset is one-based; continue with "
            "next_line_offset. Long lines are visibly abbreviated. Read-only; requires "
            "unstaged selected paths and never submits or checks correctness. Use prepare_patch "
            "to create the complete patch before reviewing and explicitly submitting it."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 32,
                    "description": "Exact source files, relative to /testbed or absolute beneath it. No directories or patterns.",
                },
                "line_offset": {"type": "integer", "minimum": 1, "description": "First diff line to show, starting at 1. Default 1."},
                "max_lines": {"type": "integer", "minimum": 1, "maximum": 200, "description": "Maximum diff lines to show, default 120. Output may end sooner at the character limit."},
            },
            "required": ["files"],
            "additionalProperties": False,
        },
    },
}
SMOKE_ARGUMENTS = {"files": ["package/module.py"]}


def _error(message):
    return {
        "output": "view_diff: " + str(message)[:8000],
        "returncode": 2,
        "exception_info": "",
    }


def _paths(value):
    if not isinstance(value, list) or not 1 <= len(value) <= 32:
        raise ValueError("files must contain 1 to 32 explicit source paths")
    result = []
    for path in value:
        if not isinstance(path, str) or not path or len(path) > 240:
            raise ValueError("each path must be a nonempty string of at most 240 characters")
        if path.startswith("/testbed/"):
            path = path[len("/testbed/"):]
        if path != path.strip() or path.startswith(("/", "-", ":")):
            raise ValueError("use explicit paths beneath /testbed")
        if any(ord(c) < 32 or ord(c) == 127 for c in path) or any(c in path for c in "\\*?[]"):
            raise ValueError("control characters, backslashes and path patterns are not allowed")
        parts = path.split("/")
        if any(p in ("", ".", "..") for p in parts):
            raise ValueError("empty components and path traversal are not allowed")
        lowered = [p.lower() for p in parts]
        # These are public library implementations, not test suites/config artifacts.
        library_prefixes = {("numpy", "testing"), ("pyramid", "config"), ("django", "test"), ("pylint", "config")}
        if any((p in {".git", ".github", ".gitlab", "test", "tests", "testing", "__tests__", "r2e_tests", "config", "configs", "__pycache__", ".pytest_cache", "artifacts", "node_modules"}
                and not (i == 1 and tuple(lowered[:2]) in library_prefixes))
               or p.endswith(".git") for i, p in enumerate(lowered)):
            raise ValueError("tests, configuration, Git metadata and artifacts are not source targets")
        name = lowered[-1]
        if re.match(r"^[a-z]:", path.lower()):
            raise ValueError("drive paths are not allowed")
        if name.startswith(("test_", "repro", "debug_", "verify_fix", "check_fix", "requirements")) or re.search(r"(?:^test\.|[._]tests?\.[^.]+$|\.spec\.[^.]+$)", name):
            raise ValueError("tests and scratch verification scripts cannot be exported")
        forbidden_names = {
            "setup.py", "setup.cfg", "pyproject.toml", "tox.ini", "pytest.ini", "conftest.py",
            "package.json", "package-lock.json", "npm-shrinkwrap.json", "yarn.lock",
            "pnpm-lock.yaml", "poetry.lock", "pipfile", "pipfile.lock",
            "environment.yml", "environment.yaml", "dockerfile", "makefile",
            "patch.txt", "test.py", "tests.py", ".gitignore", ".gitattributes", ".gitmodules",
        }
        if name in forbidden_names or name.endswith((".toml", ".cfg", ".ini", ".yaml", ".yml", ".json", ".lock", ".txt", ".md", ".rst", ".log", ".patch", ".diff", ".pyc", ".pyo", ".csv", ".png", ".jpg", ".zip", ".gz")):
            raise ValueError("configuration and generated artifacts cannot be exported")
        if path not in result:
            result.append(path)
    if sum(len(path) for path in result) > 2048:
        raise ValueError("combined path length exceeds 2048 characters; select fewer files")
    return result


# This program runs only inside the task container. It reads the current
# worktree/index diff and status, never a revision or history. Selected paths
# must have no staged/conflicted changes; unrelated index entries are preserved.
# New files use Git's regular no-index diff against /dev/null, without staging.
_REMOTE = r'''
import json, os, stat, subprocess, sys

data = json.loads(sys.argv[1])
root = os.getcwd()
git_env = dict(os.environ, GIT_OPTIONAL_LOCKS="0", GIT_LITERAL_PATHSPECS="1")
limit = 8 * 1024 * 1024

def run_git(args, ok=(0,)):
    result = subprocess.run(["git"] + args, cwd=root, env=git_env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            timeout=25)
    if result.returncode not in ok:
        raise ValueError("Git failed: " + result.stderr.decode("utf-8", "replace")[:700])
    if len(result.stdout) > limit:
        raise ValueError("selected changes exceed the 8 MiB patch limit; narrow the file selection")
    return result.stdout

def inspect_path(path):
    candidate = root
    for component in path.split("/"):
        candidate = os.path.join(candidate, component)
        if os.path.islink(candidate):
            raise ValueError("symbolic links are not accepted: " + path)
    if os.path.lexists(candidate):
        info = os.stat(candidate, follow_symlinks=False)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("provide regular files, not directories/devices: " + path)
        if info.st_size > 2 * 1024 * 1024:
            raise ValueError("source file exceeds the 2 MiB limit: " + path)
    return candidate

def parse_status(raw):
    entries = raw.split(b"\0")
    status = {}
    blocked = set()
    position = 0
    while position < len(entries):
        entry = entries[position]
        position += 1
        if not entry:
            continue
        if len(entry) < 4 or entry[2:3] != b" ":
            raise ValueError("cannot safely parse Git status")
        code = entry[:2]
        path = os.fsdecode(entry[3:])
        related = [path]
        # Porcelain v1 -z emits destination NUL original NUL for R/C.
        # The second path is not another status record, even when it starts
        # with characters that happen to resemble an XY status prefix.
        if b"R" in code or b"C" in code:
            if position >= len(entries) or not entries[position]:
                raise ValueError("incomplete rename/copy record in Git status")
            related.append(os.fsdecode(entries[position]))
            position += 1
        status[path] = code
        if code[:1] not in (b" ", b"?") or b"U" in code:
            blocked.update(related)
    return status, blocked

try:
    raw_status = run_git(["status", "--porcelain=v1", "--untracked-files=all", "-z"])
    status, blocked = parse_status(raw_status)
    selected_blocked = [path for path in data["files"] if path in blocked]
    if selected_blocked:
        raise ValueError("selected paths have staged or conflicted changes: " + ", ".join(selected_blocked) + ". Only unstaged selected paths can be exported; Git state was not changed")
    chunks = []
    changed = []
    for path in data["files"]:
        absolute = inspect_path(path)
        code = status.get(path)
        if not os.path.lexists(absolute) and code != b" D":
            raise ValueError("file is missing and is not a tracked deletion: " + path)
        if code is None and os.path.lexists(absolute):
            selected_status = run_git(["status", "--porcelain=v1", "--untracked-files=all", "--ignored", "-z", "--", path])
            if any(entry.startswith(b"!! ") for entry in selected_status.split(b"\0")):
                raise ValueError("selected new file is ignored by Git and cannot be silently omitted: " + path)
        flags = ["--no-ext-diff", "--no-textconv", "--no-color", "--binary",
                 "--no-renames", "--src-prefix=a/", "--dst-prefix=b/"]
        if code == b"??":
            chunk = run_git(["diff", "--no-index"] + flags + ["--", "/dev/null", path], ok=(0, 1))
        else:
            chunk = run_git(["diff"] + flags + ["--", path])
        if chunk:
            chunks.append(chunk)
            changed.append(path)
    patch = b"".join(chunks)
    if len(patch) > limit:
        raise ValueError("selected changes exceed the 8 MiB patch limit")
    lines = patch.decode("utf-8", "replace").splitlines()
    offset = data["line_offset"]
    stop = min(len(lines), offset - 1 + data["max_lines"])
    rendered = []
    used = 0
    cursor = offset - 1
    abbreviated = False
    while cursor < stop:
        line = lines[cursor]
        if len(line) > 1000:
            line = line[:1000] + " [long line abbreviated; prepare/read patch.txt for complete bytes]"
            abbreviated = True
        encoded_length = len(json.dumps(line, ensure_ascii=False)) + 2
        if rendered and used + encoded_length > 6500:
            break
        rendered.append(line)
        used += encoded_length
        cursor += 1
    print(json.dumps({"ok": True, "total_lines": len(lines), "line_offset": offset,
                      "returned_lines": len(rendered), "next_line_offset": cursor + 1 if cursor < len(lines) else None,
                      "abbreviated_lines": abbreviated, "diff": "\n".join(rendered),
                      "note": "No current changes for selected files." if not lines else "Review only; no files were written and no submission occurred."}, ensure_ascii=False))
except Exception as error:
    print("view_diff: " + str(error)[:1500])
    sys.exit(2)
'''


def execute(arguments, environment):
    if not isinstance(arguments, dict) or "files" not in arguments or set(arguments) - {"files", "line_offset", "max_lines"}:
        return _error("provide files, and optionally line_offset/max_lines; example {\"files\":[\"package/module.py\"]}")
    try:
        files = _paths(arguments["files"])
    except ValueError as exc:
        return _error(exc)
    offset = arguments.get("line_offset", 1)
    maximum = arguments.get("max_lines", 120)
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 1:
        return _error("line_offset must be an integer of at least 1")
    if isinstance(maximum, bool) or not isinstance(maximum, int) or not 1 <= maximum <= 200:
        return _error("max_lines must be an integer from 1 to 200")
    payload = {"files": files, "line_offset": offset, "max_lines": maximum}
    command = "cd /testbed && python3 -c " + shlex.quote(_REMOTE) + " " + shlex.quote(json.dumps(payload))
    try:
        result = environment.execute({"command": command})
    except Exception as exc:
        return _error("container execution failed: " + str(exc))
    if not isinstance(result, dict):
        return _error("container returned an invalid observation")
    return {"output": str(result.get("output", ""))[:9500],
            "returncode": int(result.get("returncode", 1)),
            "exception_info": result.get("exception_info", "")}
