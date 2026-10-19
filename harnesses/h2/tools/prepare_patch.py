"""Prepare unstaged changes in selected files; native bash still submits them."""

import json
import re
import shlex

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "prepare_patch",
        "description": (
            "Write selected source changes to patch.txt for inspection. Includes "
            "modified, deleted and untracked new files; leaves Git's index unchanged. "
            "Rejects staged/conflicted selected paths, tests, configuration and scratch artifacts. "
            "Returns a bounded preview and checks reverse applicability to the current worktree. Also reports added/removed line counts, finds existing repository tests that reference the changed code and runs up to two of them for at most 25 seconds (related_test_run). Review the preview (read omitted hunks if needed), then submit separately using native bash. This action "
            "never submits, ends the task or officially grades the patch. On failure, an "
            "older patch.txt may remain and must not be mistaken for a new patch."
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
                }
            },
            "required": ["files"],
            "additionalProperties": False,
        },
    },
}
SMOKE_ARGUMENTS = {"files": ["package/module.py"]}


def _error(message):
    return {
        "output": "prepare_patch: " + str(message)[:8000] + "\nNo new patch was prepared. Any previous patch.txt is unchanged; do not submit it as this result.",
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
import hashlib, importlib.util, json, os, re, signal, stat, subprocess, sys, tempfile, time

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

def line_changes(chunks, changed):
    counts = []
    for path, chunk in zip(changed, chunks):
        added = removed = 0
        for line in chunk.decode("utf-8", "replace").splitlines():
            if line.startswith("+") and not line.startswith("+++"):
                added += 1
            elif line.startswith("-") and not line.startswith("---"):
                removed += 1
        counts.append({"file": path, "added": added, "removed": removed})
    return counts

def is_test_file(path):
    parts = path.lower().split("/")
    name = parts[-1]
    if not name.endswith(".py") or name in ("conftest.py", "__init__.py", "runtests.py"):
        return False
    return (name.startswith("test_") or name.endswith("_test.py") or name == "tests.py"
            or any(part in ("tests", "test", "testing") for part in parts[:-1]))

# Tracked test files ranked by name, package and references to changed symbols.
def related_tests(changed, patch_text):
    try:
        listed = run_git(["ls-files", "-z"])
    except Exception:
        return []
    tests = [p for p in os.fsdecode(listed).split("\0") if p and is_test_file(p) and p not in changed]
    if not tests:
        return []
    scores = {}
    for source in changed:
        parts = source.split("/")
        base = parts[-1][:-3] if parts[-1].endswith(".py") else parts[-1]
        if base == "__init__" and len(parts) > 1:
            base = parts[-2]
        package = parts[-2] if len(parts) > 1 else ""
        named = re.compile("(?:^|_)" + re.escape(base.lower()) + "s?(?:_|$)") if len(base) > 2 else None
        for test in tests:
            test_parts = test.split("/")
            score = 0
            if named is not None and named.search(test_parts[-1][:-3].lower()):
                score += 3
            if package and package in test_parts[:-1]:
                score += 1
            if score:
                scores[test] = scores.get(test, 0) + score
    symbols = []
    for pattern in (r"^@@[^@\n]*@@[ \t]*(?:async[ \t]+)?(?:def|class)[ \t]+(\w+)",
                    r"^[+-][ \t]*(?:async[ \t]+)?(?:def|class)[ \t]+(\w+)"):
        for symbol in re.findall(pattern, patch_text, re.M):
            if len(symbol) > 3 and not symbol.startswith("__") and symbol not in symbols:
                symbols.append(symbol)
    symbols = symbols[:8]
    if symbols:
        arguments = ["grep", "-l", "-w", "-F", "-I"]
        for symbol in symbols:
            arguments += ["-e", symbol]
        try:
            found = subprocess.run(["git"] + arguments, cwd=root, env=git_env, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, timeout=5).stdout
            wanted = set(tests)
            for test in os.fsdecode(found).splitlines():
                if test in wanted:
                    scores[test] = scores.get(test, 0) + 2
        except Exception:
            pass
    return sorted((p for p in scores if scores[p] >= 2), key=lambda p: (-scores[p], len(p), p))[:5]

def related_command(tests):
    runtests = os.path.join(root, "tests", "runtests.py")
    if os.path.isfile(runtests):
        labels = [t[len("tests/"):-3].replace("/", ".") for t in tests if t.startswith("tests/")][:2]
        if not labels:
            return None, ""
        arguments = [sys.executable, "tests/runtests.py"]
        try:
            with open(runtests, "rb") as handle:
                if b"--parallel" in handle.read():
                    arguments += ["--parallel", "1"]
        except OSError:
            pass
        return arguments + labels, "python tests/runtests.py " + " ".join(labels)
    files = tests[:2]
    if not files:
        return None, ""
    try:
        has_pytest = importlib.util.find_spec("pytest") is not None
    except Exception:
        has_pytest = False
    if has_pytest:
        return ([sys.executable, "-m", "pytest", "-q", "-rf", "-p", "no:cacheprovider"] + files,
                "python -m pytest -q " + " ".join(files))
    if os.path.isfile(os.path.join(root, "bin", "test")):
        return [sys.executable, "bin/test", "-C"] + files, "python bin/test -C " + " ".join(files)
    return None, "python -m pytest -q " + " ".join(files)

FAILED_ID = re.compile(r"^(?:FAILED|FAIL|ERROR)\s+(\S+::\S+)|^(?:FAIL|ERROR): (\S+) \(([^)]+)\)|^_{3,} (\S+\.py:\w+) _{3,}$", re.M)
ZERO_TESTS = re.compile(r"collected 0 items|no tests ran|Ran 0 tests", re.I)

# Bounded run of existing tests that reference the changed code; evidence, never grading.
def run_related(arguments, display, digest):
    if arguments is None:
        return {"status": "unavailable", "command": display}
    key = hashlib.sha256((digest + "\0" + root + "\0" + " ".join(arguments)).encode("utf-8")).hexdigest()[:16]
    cache = "/tmp/harness-related-" + key + ".json"
    log = "/tmp/harness-related-" + key + ".log"
    try:
        with open(cache, encoding="utf-8") as handle:
            result = json.load(handle)
        result["cached"] = True
        return result
    except (OSError, ValueError):
        pass
    try:
        seconds = max(1, min(25, int(os.environ.get("HARNESS_RELATED_TEST_TIMEOUT", "25"))))
    except ValueError:
        seconds = 25
    started = time.time()
    process = subprocess.Popen(arguments, cwd=root, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
                               stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               start_new_session=True)
    timed_out = False
    try:
        output = process.communicate(timeout=seconds)[0]
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except OSError:
            pass
        try:
            output = process.communicate(timeout=3)[0]
        except subprocess.TimeoutExpired:
            process.kill()
            output = b""
    text = output.decode("utf-8", "replace")
    try:
        with open(log, "w", encoding="utf-8", errors="replace") as handle:
            handle.write(text)
    except OSError:
        log = ""
    failing = []
    for match in FAILED_ID.finditer(text):
        if match.group(1) or match.group(4):
            identifier = match.group(1) or match.group(4)
        else:
            identifier = match.group(3) if match.group(3).endswith(match.group(2)) else match.group(3) + "." + match.group(2)
        if identifier not in failing:
            failing.append(identifier)
    code = process.returncode
    if timed_out:
        status = "timeout"
    elif ZERO_TESTS.search(text):
        status = "no_tests"
    elif code == 0:
        status = "passed"
    elif failing:
        status = "failed"
    else:
        status = "error"
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    result = {"status": status, "command": display, "exit_code": None if timed_out else code,
              "seconds": round(time.time() - started, 1), "failing": failing[:5],
              "summary": " | ".join(lines[-3:])[-400:], "log": log}
    try:
        with open(cache, "w", encoding="utf-8") as handle:
            json.dump(result, handle)
    except OSError:
        pass
    return result

def evidence(chunks, changed, patch, digest):
    counts = line_changes(chunks, changed)
    large = [c["file"] for c in counts if c["removed"] >= 100 and c["removed"] >= 2 * c["added"]]
    extra = {"line_changes": counts, "large_deletions": large, "related_tests": [], "suggested_check": "",
             "related_test_run": {"status": "unavailable", "command": ""}}
    try:
        extra["related_tests"] = related_tests(changed, patch.decode("utf-8", "replace"))
        arguments, extra["suggested_check"] = related_command(extra["related_tests"])
        extra["related_test_run"] = run_related(arguments, extra["suggested_check"], digest)
    except Exception as error:
        extra["related_test_run"] = {"status": "unavailable", "command": extra["suggested_check"],
                                     "summary": type(error).__name__}
    return extra

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
    if not patch:
        raise ValueError("no changes in selected files; ignored untracked files are not included")
    # Read-only structural check against the current worktree, not the grader base.
    reverse_check = subprocess.run(["git", "apply", "--reverse", "--check", "-"],
                                  input=patch, cwd=root, env=git_env,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=25)
    if reverse_check.returncode != 0:
        raise ValueError("generated patch failed reverse applicability check: " +
                         reverse_check.stderr.decode("utf-8", "replace")[:700])
    destination = os.path.join(root, "patch.txt")
    if os.path.islink(destination) or (os.path.lexists(destination) and not stat.S_ISREG(os.stat(destination, follow_symlinks=False).st_mode)):
        raise ValueError("patch.txt must be absent or a regular file, never a symbolic link or directory")
    temporary = None
    try:
        descriptor, temporary = tempfile.mkstemp(prefix=".prepared-patch-", dir=root)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(patch)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            os.unlink(temporary)
    digest = hashlib.sha256(patch).hexdigest()
    report = {"ok": True, "artifact": "patch.txt", "bytes": len(patch), "sha256": digest, "files": changed,
              "reverse_apply_check": True, "preview_complete": len(patch.decode("utf-8", "replace")) <= 2400}
    report.update(evidence(chunks, changed, patch, digest))
    report["next_action"] = ("Review this preview; read omitted hunks if truncated. related_test_run is a bounded run of existing tests that reference the changed code: "
                             "inspect any failure before submitting and do not dismiss it without evidence. If it did not run and budget allows, run suggested_check (or a closer existing test) with run_check. "
                             "If ready, use bash: echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt. Never rewrite patch.txt by hand. Source edits invalidate this artifact; prepare again. "
                             "This is not official grading.")
    print(json.dumps(report))
    print("PATCH PREVIEW (complete)" if len(patch.decode("utf-8", "replace")) <= 2400 else "PATCH PREVIEW (truncated; inspect omitted hunks if needed)")
    print(patch.decode("utf-8", "replace")[:2400])
except Exception as error:
    print("prepare_patch: " + str(error)[:1500])
    print("No new patch was prepared. Any previous patch.txt is unchanged; do not submit it as this result.")
    sys.exit(2)
'''


def execute(arguments, environment):
    if not isinstance(arguments, dict) or set(arguments) != {"files"}:
        return _error("provide only files, for example {\"files\":[\"package/module.py\"]}")
    try:
        files = _paths(arguments["files"])
    except ValueError as exc:
        return _error(exc)
    command = "cd /testbed && python3 -c " + shlex.quote(_REMOTE) + " " + shlex.quote(json.dumps({"files": files}))
    try:
        result = environment.execute({"command": command})
    except Exception as exc:
        return _error("container execution failed: " + str(exc))
    if not isinstance(result, dict):
        return _error("container returned an invalid observation")
    return {"output": str(result.get("output", ""))[:9500],
            "returncode": int(result.get("returncode", 1)),
            "exception_info": result.get("exception_info", "")}
