"""Restore one tracked file to the task's starting content from the Git index; never reads history."""

import json
import shlex

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "restore_file",
        "description": (
            "Discard your edits to one tracked file under /testbed by rewriting it from the task's starting version "
            "(the unchanged Git index). Use it when an edit went wrong instead of rebuilding the file by hand; "
            "Git checkout, restore, stash and show are unavailable here. Untracked new files have no starting "
            "version; delete them with bash. Other files, the index and patch.txt are left unchanged."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "One tracked file, relative to /testbed or absolute beneath it.",
                }
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
}
SMOKE_ARGUMENTS = {"path": "package/module.py"}


def _error(message):
    return {"output": "restore_file: " + str(message)[:2000] + "\nThe file was not changed.",
            "returncode": 2, "exception_info": ""}


def _path(value):
    if not isinstance(value, str) or not value or len(value) > 240:
        raise ValueError("path must be a nonempty string of at most 240 characters")
    if value.startswith("/testbed/"):
        value = value[len("/testbed/"):]
    if value != value.strip() or value.startswith(("/", "-", ":")):
        raise ValueError("use one explicit path beneath /testbed")
    if any(ord(c) < 32 or ord(c) == 127 for c in value) or any(c in value for c in "\\*?[]"):
        raise ValueError("control characters, backslashes and path patterns are not allowed")
    parts = value.split("/")
    if any(p in ("", ".", "..") for p in parts) or any(p.lower() == ".git" or p.lower().endswith(".git") for p in parts):
        raise ValueError("empty components, traversal and Git metadata are not allowed")
    if value == "patch.txt":
        raise ValueError("patch.txt is an artifact; regenerate it with prepare_patch")
    return value


# Runs only inside the task container. The index is never staged by the agent
# (git add is unavailable), so its blob is the file as the task started.
_REMOTE = r'''
import json, os, stat, subprocess, sys, tempfile

path = json.loads(sys.argv[1])["path"]
root = os.getcwd()
env = dict(os.environ, GIT_OPTIONAL_LOCKS="0", GIT_LITERAL_PATHSPECS="1")

def git(args):
    return subprocess.run(["git"] + args, cwd=root, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, timeout=20)

try:
    listed = git(["ls-files", "-s", "-z", "--", path])
    entries = [e for e in listed.stdout.split(b"\0") if e]
    if listed.returncode != 0 or not entries:
        raise ValueError("not a tracked file: " + path + ". Untracked new files have no starting version; remove them with bash if unwanted")
    if len(entries) != 1:
        raise ValueError("path must name exactly one tracked file, not a directory")
    info, name = entries[0].split(b"\t", 1)
    mode, blob, stage = info.split(b" ")
    if os.fsdecode(name) != path:
        raise ValueError("path must name a tracked file exactly")
    if stage != b"0":
        raise ValueError("the file has a conflicted index entry")
    if mode not in (b"100644", b"100755"):
        raise ValueError("only regular tracked files can be restored")
    candidate = root
    for component in path.split("/")[:-1]:
        candidate = os.path.join(candidate, component)
        if os.path.islink(candidate) or (os.path.lexists(candidate) and not os.path.isdir(candidate)):
            raise ValueError("a parent of the path is not a regular directory")
    target = os.path.join(root, path)
    if os.path.islink(target) or (os.path.lexists(target) and not stat.S_ISREG(os.lstat(target).st_mode)):
        raise ValueError("the target exists but is not a regular file")
    content = git(["cat-file", "blob", blob.decode("ascii")])
    if content.returncode != 0:
        raise ValueError("could not read the starting version")
    data = content.stdout
    previous = None
    if os.path.exists(target):
        with open(target, "rb") as handle:
            previous = handle.read()
        if previous == data:
            print(json.dumps({"ok": True, "restored": path, "changed": False,
                              "next_action": "The file already matches its starting version."}))
            sys.exit(0)
    directory = os.path.dirname(target)
    if not os.path.isdir(directory):
        os.makedirs(directory)
    descriptor, temporary = tempfile.mkstemp(prefix=".harness-restore-", dir=directory)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o755 if mode == b"100755" else 0o644)
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            os.unlink(temporary)
    print(json.dumps({"ok": True, "restored": path, "changed": True, "bytes": len(data),
                      "previous_bytes": None if previous is None else len(previous),
                      "next_action": "Your edits to this file are discarded. Re-apply only the intended change with edit_file; if a patch was prepared, run prepare_patch again."}))
except Exception as error:
    print("restore_file: " + str(error)[:1500])
    print("The file was not changed.")
    sys.exit(2)
'''


def execute(arguments, environment):
    if not isinstance(arguments, dict) or set(arguments) != {"path"}:
        return _error("provide only path, for example {\"path\":\"package/module.py\"}")
    try:
        path = _path(arguments["path"])
    except ValueError as exc:
        return _error(exc)
    command = "cd /testbed && python3 -c " + shlex.quote(_REMOTE) + " " + shlex.quote(json.dumps({"path": path}))
    try:
        result = environment.execute({"command": command})
    except Exception as exc:
        return _error("container execution failed: " + str(exc))
    if not isinstance(result, dict):
        return _error("container returned an invalid observation")
    return {"output": str(result.get("output", ""))[:4000],
            "returncode": int(result.get("returncode", 1)),
            "exception_info": result.get("exception_info", "")}
