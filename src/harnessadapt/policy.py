"""Command policy and environment extracted from the existing Mini-SWE runner."""
from __future__ import annotations
import argparse
import json
import os
import posixpath
import re
import shlex
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from ipaddress import ip_address
from pathlib import Path
from urllib.parse import urlsplit

_GIT_POLICY_HELP = (
    "Allowed Git commands are: 'git status --short' and current-worktree "
    "'git diff' (optionally with safe display flags and '--' followed by "
    "explicit non-test source paths). Git history, revisions, remotes, index "
    "writes, worktree writes, and test paths are forbidden."
)

_NETWORK_SOURCE_POLICY_HELP = (
    "Remote file/package fetches are disabled during agent rollouts so network "
    "content cannot be staged and copied over repository source. Use the "
    "checked-out worktree and already-installed dependencies. Loopback requests "
    "used to test a local service remain allowed."
)

_GIT_DIFF_FLAGS = {
    "--binary",
    "--check",
    "--exit-code",
    "--name-only",
    "--name-status",
    "--no-color",
    "--no-ext-diff",
    "--numstat",
    "--quiet",
    "--shortstat",
    "--stat",
    "--color=never",
}

_GIT_STATUS_FLAGS = {"--short", "-s", "--porcelain", "--porcelain=v1", "--porcelain=v2"}

_SHELL_PUNCTUATION = ";&|()<>"

_SHELL_COMMAND_SEPARATORS = {"&&", "||", ";", "|", "&", "("}

_TEST_PATH_PARTS = {"test", "tests", "testing", "__tests__"}

_COMMAND_WRAPPERS = {
    "command",
    "env",
    "exec",
    "nice",
    "nohup",
    "stdbuf",
    "sudo",
    "timeout",
    "xargs",
}

_REMOTE_FILE_CLIENTS = {"aria2c", "curl", "fetch", "scp", "sftp", "wget"}

_PACKAGE_CLIENTS = {"pip", "pip2", "pip3", "uv"}

_REMOTE_URL = re.compile(r"(?i)\b(?:https?|ftps?|sftp)://[^\s'\"<>]+")

_LOCAL_HOST_TARGET = re.compile(
    r"(?i)^(?:https?://)?(?:localhost|127(?:\.\d{1,3}){3}|\[?::1\]?)"
    r"(?::\d+)?(?:/|$)"
)

_INTERPRETER_NETWORK_FETCH = re.compile(
    r"(?:urllib(?:\.request)?\.(?:urlopen|urlretrieve)|"
    r"\b(?:urlopen|urlretrieve)\s*\(|"
    r"requests\.(?:get|post|request|stream)|"
    r"httpx\.(?:get|post|request|stream)|"
    r"aiohttp\.ClientSession|urllib3\.PoolManager|\bfetch\s*\()",
    re.I,
)

def _shell_tokens(command: str) -> list[str]:
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=_SHELL_PUNCTUATION)
        lexer.whitespace_split = True
        lexer.commenters = ""
        return list(lexer)
    except ValueError:
        # Mini-SWE will report malformed shell quoting itself. The Git policy
        # remains fail-closed when the raw command appears to invoke Git.
        return []

def _is_shell_control(token: str) -> bool:
    return bool(token) and all(char in _SHELL_PUNCTUATION for char in token)

def _is_git_executable(token: str) -> bool:
    return posixpath.basename(token.rstrip("/")) == "git"

def _is_git_command_position(tokens: list[str], index: int) -> bool:
    start = index
    while start > 0 and tokens[start - 1] not in _SHELL_COMMAND_SEPARATORS:
        start -= 1
    prefix = tokens[start:index]
    if not prefix:
        return True
    if all(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", token) for token in prefix):
        return True
    first = posixpath.basename(prefix[0].rstrip("/"))
    return first in _COMMAND_WRAPPERS or prefix[-1] in {"-exec", "-execdir"}

def _invocation_args(tokens: list[str], index: int) -> list[str]:
    """Return one simple command's arguments, stopping before shell control."""
    args: list[str] = []
    for arg_index in range(index + 1, len(tokens)):
        argument = tokens[arg_index]
        if _is_shell_control(argument):
            break
        # ``2>/dev/null`` is tokenized as ``2``, ``>``, ``/dev/null``.
        if (
            argument.isdigit()
            and arg_index + 1 < len(tokens)
            and _is_shell_control(tokens[arg_index + 1])
        ):
            break
        args.append(argument)
    return args

def _git_invocations(command: str) -> list[list[str]]:
    tokens = _shell_tokens(command)
    invocations: list[list[str]] = []
    for index, token in enumerate(tokens):
        if not _is_git_executable(token) or not _is_git_command_position(tokens, index):
            continue
        invocations.append(_invocation_args(tokens, index))

    # shlex keeps the program passed to ``bash -c`` as one token. Recursively
    # inspect shell/eval command strings without treating ordinary text such as
    # ``grep git README`` as an executable invocation.
    for index, token in enumerate(tokens):
        if index == 0 or tokens[index - 1] != "-c":
            continue
        segment_start = index - 1
        while segment_start > 0 and not _is_shell_control(tokens[segment_start - 1]):
            segment_start -= 1
        executable = posixpath.basename(tokens[segment_start].rstrip("/"))
        if executable in {"bash", "dash", "sh", "zsh"}:
            invocations.extend(_git_invocations(token))
    for index, token in enumerate(tokens):
        if index > 0 and tokens[index - 1] == "eval":
            invocations.extend(_git_invocations(token))
    return invocations

def _is_loopback_url(url: str) -> bool:
    try:
        host = urlsplit(url).hostname
    except ValueError:
        return False
    if not host:
        return False
    if host.lower() == "localhost":
        return True
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return False

def _has_only_loopback_targets(args: list[str]) -> bool:
    text = " ".join(args)
    urls = _REMOTE_URL.findall(text)
    if urls:
        return all(_is_loopback_url(url) for url in urls)
    # Curl accepts ``localhost:8000/path`` without a URL scheme.
    operands = [arg for arg in args if not arg.startswith("-")]
    return bool(operands) and any(_LOCAL_HOST_TARGET.match(arg) for arg in operands)

def _informational_client_call(args: list[str]) -> bool:
    return bool(args) and all(
        argument in {"--help", "--version", "-V", "-h"} for argument in args
    )

def _pip_invocation(executable: str, args: list[str]) -> tuple[str, list[str]] | None:
    """Return a network-capable pip operation and its remaining arguments."""
    if executable == "uv":
        if len(args) < 2 or args[0] != "pip":
            return None
        operation_index = 1
    else:
        operation_index = 0
    if operation_index >= len(args):
        return None
    operation = args[operation_index]
    if operation not in {"download", "install", "wheel"}:
        return None
    return operation, args[operation_index + 1 :]

def _local_pip_install(args: list[str]) -> bool:
    """Recognize common worktree/local-artifact installs that need no source fetch."""
    urls = _REMOTE_URL.findall(" ".join(args))
    if any(not _is_loopback_url(url) for url in urls):
        return False
    if "--no-index" in args:
        return True
    for index, argument in enumerate(args):
        if argument in {"-e", "--editable"} and index + 1 < len(args):
            target = args[index + 1]
            if target == "." or target.startswith(("./", "../", "/", "file://")):
                return True
    operands = [argument for argument in args if not argument.startswith("-")]
    return bool(operands) and all(
        argument == "."
        or argument.startswith(("./", "../", "/", "file://"))
        or argument.endswith((".whl", ".tar.gz", ".zip"))
        for argument in operands
    )

def _nested_shell_commands(tokens: list[str]) -> list[str]:
    nested: list[str] = []
    for index, token in enumerate(tokens):
        executable = posixpath.basename(token.rstrip("/"))
        if executable not in {
            "bash",
            "dash",
            "sh",
            "zsh",
        } or not _is_git_command_position(tokens, index):
            continue
        args = _invocation_args(tokens, index)
        for arg_index, argument in enumerate(args[:-1]):
            if re.fullmatch(r"-[A-Za-z]*c[A-Za-z]*", argument):
                nested.append(args[arg_index + 1])
                break
    for index, token in enumerate(tokens[:-1]):
        if posixpath.basename(token.rstrip("/")) == "eval" and _is_git_command_position(
            tokens, index
        ):
            nested.append(tokens[index + 1])
    return nested

def _network_source_policy_violation(command: str, *, _depth: int = 0) -> str | None:
    """Block agent-side remote content retrieval before it can replace source.

    The rule deliberately blocks staging into ``/tmp`` as well as direct writes
    under ``/testbed``. Otherwise ``curl URL -o /tmp/x`` followed by ``cp``
    trivially bypasses a destination-only check. Harness setup runs outside this
    proxy, so image preparation is unaffected.
    """
    if _depth > 3:
        return "nested shell command exceeded the network-policy parse depth"
    tokens = _shell_tokens(command)
    if not tokens:
        # A malformed command is handled by the shell unless it visibly names a
        # remote fetcher, in which case fail closed rather than risk a bypass.
        if re.search(
            r"(?:^|[\s;&|()])(?:curl|wget|pip\s+download)(?:[\s;&|()]|$)",
            command,
        ):
            return "could not safely parse a remote-content command"
        return None

    for index, token in enumerate(tokens):
        if not _is_git_command_position(tokens, index):
            continue
        executable = posixpath.basename(token.rstrip("/"))
        args = _invocation_args(tokens, index)

        if executable in _REMOTE_FILE_CLIENTS:
            if _informational_client_call(args) or _has_only_loopback_targets(args):
                continue
            return f"remote file client '{executable}' is forbidden"

        pip_command = (
            _pip_invocation(executable, args)
            if executable in _PACKAGE_CLIENTS
            else None
        )
        if (
            pip_command is None
            and re.fullmatch(r"(?:python|python\d+(?:\.\d+)?|pypy\d*)", executable)
            and len(args) >= 3
            and args[0] == "-m"
            and re.fullmatch(r"pip\d*", posixpath.basename(args[1]))
        ):
            pip_command = _pip_invocation(posixpath.basename(args[1]), args[2:])
        if pip_command is not None:
            operation, operation_args = pip_command
            if _local_pip_install(operation_args):
                continue
            return f"network-capable package command '{executable} {operation}' is forbidden"

        if re.fullmatch(
            r"(?:python|python\d+(?:\.\d+)?|pypy\d*|node|ruby|perl)", executable
        ):
            invocation_text = " ".join(args)
            # Heredoc bodies remain in the raw command rather than args on some
            # shlex versions, hence inspect both representations.
            if _INTERPRETER_NETWORK_FETCH.search(
                invocation_text
            ) or _INTERPRETER_NETWORK_FETCH.search(command):
                urls = _REMOTE_URL.findall(command)
                if not urls or any(not _is_loopback_url(url) for url in urls):
                    return f"remote content fetch through '{executable}' is forbidden"

    for nested in _nested_shell_commands(tokens):
        if reason := _network_source_policy_violation(nested, _depth=_depth + 1):
            return reason
    return None

def _unsafe_diff_path(path: str) -> str | None:
    if not path or path in {".", ".."}:
        return "broad or parent-directory pathspecs are forbidden"
    if path.startswith(("/", ":")) or any(char in path for char in "*?[]"):
        return "absolute, magic, and wildcard pathspecs are forbidden"
    normalized = posixpath.normpath(path)
    parts = [part.lower() for part in normalized.split("/")]
    if normalized == ".." or normalized.startswith("../") or ".git" in parts:
        return "paths outside the worktree and .git metadata are forbidden"
    basename = parts[-1]
    if (
        any(part in _TEST_PATH_PARTS for part in parts)
        or basename.startswith("test_")
        or basename.endswith(("_test.py", "_tests.py", "tests.py"))
    ):
        return "test paths are forbidden"
    return None

def _git_policy_violation(command: str) -> str | None:
    """Return a user-facing reason when an agent shell command is forbidden."""
    # Git metadata can expose refs/reflogs even without invoking the git binary.
    if re.search(r"(?:^|[\s/'\"])[.]git(?:[/\s'\"]|$)", command):
        return "direct access to .git metadata is forbidden"

    invocations = _git_invocations(command)
    if not invocations:
        # Fail closed for malformed quoting that still visibly invokes Git.
        if not _shell_tokens(command) and re.search(
            r"(?:^|[\s;&|()])(?:/[^\s;&|()]*/)?git(?:[\s;&|()]|$)", command
        ):
            return "could not safely parse the Git command"
        return None

    for args in invocations:
        if not args:
            return "Git requires an explicitly allowed subcommand"
        subcommand, *rest = args
        if subcommand == "status":
            if not rest or any(argument not in _GIT_STATUS_FLAGS for argument in rest):
                return "only the short/porcelain form of 'git status' is allowed"
            continue
        if subcommand != "diff":
            return f"Git subcommand '{subcommand}' is forbidden"

        paths: list[str] = []
        saw_separator = False
        for argument in rest:
            if argument == "--":
                if saw_separator:
                    return "multiple '--' separators are forbidden"
                saw_separator = True
                continue
            if saw_separator:
                paths.append(argument)
                continue
            if argument in _GIT_DIFF_FLAGS:
                continue
            if re.fullmatch(r"-U\d+", argument) or re.fullmatch(
                r"--unified=\d+", argument
            ):
                continue
            # A bare value before ``--`` is ambiguous with a revision and is
            # therefore never treated as a path.
            return f"Git diff revision or unsafe option '{argument}' is forbidden"
        for path in paths:
            if reason := _unsafe_diff_path(path):
                return f"Git diff path '{path}' is forbidden: {reason}"
    return None

def _git_policy_observation(command: str) -> dict | None:
    if not (reason := _git_policy_violation(command)):
        return None
    message = f"SKYRL_GIT_POLICY_BLOCKED: {reason}. {_GIT_POLICY_HELP}"
    return {"output": message, "returncode": 126, "exception_info": message}

def _command_policy_observation(command: str) -> dict | None:
    if observation := _git_policy_observation(command):
        return observation
    if not (reason := _network_source_policy_violation(command)):
        return None
    message = (
        f"SKYRL_NETWORK_SOURCE_POLICY_BLOCKED: {reason}. "
        f"{_NETWORK_SOURCE_POLICY_HELP}"
    )
    return {"output": message, "returncode": 126, "exception_info": message}

class _GitPolicyEnvironment:
    """Proxy a Mini-SWE environment and enforce shared agent command policy."""

    def __init__(self, environment):
        self._environment = environment

    def __getattr__(self, name):
        return getattr(self._environment, name)

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None):
        command = str(action.get("command", ""))
        if observation := _command_policy_observation(command):
            return observation
        return self._environment.execute(action, cwd=cwd, timeout=timeout)


def _install_git_command_policy(environment):
    if isinstance(environment, _GitPolicyEnvironment):
        return environment
    return _GitPolicyEnvironment(environment)


SYSTEM_TEMPLATE = (
    "You are a helpful assistant that can interact with a computer shell to "
    "solve programming tasks.\n"
    "{{ opsd_note | default('') }}"
)

INSTANCE_TEMPLATE = """\
<pr_description>
Consider the following PR description:
{{task}}
</pr_description>

<instructions>
# Task Instructions

## Overview
You're a software engineer interacting continuously with a computer by \
submitting commands. You'll be helping implement necessary changes to meet \
requirements in the PR description. Your task is specifically to make changes to \
non-test files in the current directory in order to fix the issue described in \
the PR description in a way that is general and consistent with the codebase.
<IMPORTANT>This is an interactive process where you will think and issue AT LEAST \
ONE command, see the result, then think and issue your next command(s).</IMPORTANT>

## Important Boundaries
- MODIFY: Regular source code files in /testbed (this is the working directory).
- DO NOT MODIFY: Tests, configuration files (pyproject.toml, setup.cfg, etc.)

## Recommended Workflow
1. Analyze the codebase by finding and reading relevant files
2. Create a script to reproduce the issue
3. Edit the source code to resolve the issue
4. Verify your fix works by running your script again
5. Test edge cases to ensure your fix is robust
{{ rigor_user_protocol | default('') }}
{{ opsd_priority_block | default('') }}
## Command Execution Rules
- Directory or environment variable changes are not persistent. Every action is \
executed in a new subshell. Prefix any action with `cd /path && ...` if needed.
- Always use non-interactive flags (-y, -f). Avoid interactive tools (vi, nano).
- Git is restricted to `git status --short` and current-worktree `git diff`. Do \
not inspect Git history/revisions/remotes, run Git write commands, access `.git`, \
or use Git to inspect test paths. Use ordinary source-file tools for code search.
- Every response MUST include AT LEAST ONE available tool call. Use the tool
  required by the current workflow step rather than substituting another tool.

## Submission
When you've completed your work, you MUST submit your changes as a git patch.
Follow these steps IN ORDER, with SEPARATE commands:

Step 1: Create the patch file
Run `git diff -- path/to/file1 path/to/file2 > patch.txt` listing only the source \
files you modified. Do NOT commit your changes.

<IMPORTANT>
The patch must only contain changes to the specific source files you modified to \
fix the issue. Do not submit test/reproduction files, helper scripts you created, \
or build/config files.
</IMPORTANT>

Step 2: Verify your patch
Inspect patch.txt to confirm it only contains your intended changes.

Step 3: Submit (EXACT command required)
You MUST use this EXACT command to submit:

```bash
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt
```

If the command fails (nonzero exit status), it will not submit.

<CRITICAL>
- Creating/viewing the patch and submitting it MUST be separate commands.
- You CANNOT continue working on this task after submitting.
</CRITICAL>
</instructions>"""


def _build_opsd_note(hint: str) -> str:
    if not hint:
        return ""
    return (
        "\n\n## Skills for this task\n\n"
        "Apply these task-specific skills when solving the current issue:\n\n"
        + hint.strip()
        + "\n"
    )

def _parse_returncode(code) -> int:
    """R2E-Gym runtime.run returns '0' | 'Error: Exit code N' | '-1'."""
    s = str(code).strip()
    if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
        return int(s)
    if "Exit code" in s:
        try:
            return int(s.rsplit(" ", 1)[-1])
        except ValueError:
            return 1
    return 1

def _runtime_shell_command(command: str) -> str:
    """Make R2E-Gym's outer ``timeout`` execute one complete shell program.

    ``DockerRuntime.run`` constructs ``timeout N <code>``. Passing a compound
    command directly makes shell builtins such as ``cd`` become the executable
    handed to ``timeout``. Wrapping the complete action in ``sh -lc`` preserves
    normal Bash-tool semantics while retaining the runtime's timeout.
    """

    return f"/bin/sh -lc {shlex.quote(command)}"

def _make_environment_cls():
    """Build the adapter lazily so importing this module never requires mini."""
    from minisweagent.exceptions import Submitted

    class R2EGymEnvironment:
        """mini-swe-agent Environment backed by an R2E-Gym DockerRuntime."""

        def __init__(self, runtime, cwd: str = "/testbed", timeout: int = 60,
                     opsd_note: str = "", opsd_priority_block: str = "",
                     hard_step_timeout: int | None = None):
            self.runtime = runtime
            self._cwd = cwd
            self._timeout = timeout
            # A single agent command may not block the run for longer than this;
            # on overshoot the container is force-killed (see execute()).
            self._hard_step_timeout = hard_step_timeout or (int(timeout) + 30)
            self._opsd_note = opsd_note
            self._opsd_priority_block = opsd_priority_block

        def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None):
            command = action.get("command", "")
            if observation := _command_policy_observation(command):
                return observation
            workdir = cwd or self._cwd
            eff = timeout or self._timeout
            # HARD per-command wall clock. R2E-Gym enforces step timeout by
            # prefixing `timeout {eff} <cmd>`, but that only bounds the FIRST
            # simple command -- a compound command (e.g. a heredoc `cat` followed
            # by `python3 script.py`) lets the trailing process escape and run
            # unbounded, wedging the docker exec (and thus the whole run) forever.
            # `_run_with_deadline` runs the call in a daemon thread and, if it
            # overshoots, force-removes the container so nothing can hang past the
            # ceiling no matter what is running inside.
            hard = max(eff, self._hard_step_timeout)
            try:
                out, code = _run_with_deadline(
                    lambda: self.runtime.run(
                        _runtime_shell_command(command),
                        timeout=eff,
                        workdir=workdir,
                    ),
                    hard,
                    self.runtime,
                    "exec",
                )
                output = {
                    "output": out,
                    "returncode": _parse_returncode(code),
                    "exception_info": "",
                }
            except Exception as e:  # noqa: BLE001 (incl. TimeoutError from the deadline)
                output = {
                    "output": str(e),
                    "returncode": -1,
                    "exception_info": f"error executing command: {e}",
                }
            self._check_finished(output)
            return output

        def _check_finished(self, output: dict) -> None:
            lines = output.get("output", "").lstrip().splitlines(keepends=True)
            if (
                lines
                and lines[0].strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
                and output["returncode"] == 0
            ):
                submission = "".join(lines[1:])
                raise Submitted(
                    {
                        "role": "exit",
                        "content": submission,
                        "extra": {"exit_status": "Submitted", "submission": submission},
                    }
                )

        def get_template_vars(self, **kwargs) -> dict:
            return {
                "cwd": self._cwd,
                "opsd_note": self._opsd_note,
                "opsd_priority_block": self._opsd_priority_block,
                **kwargs,
            }

        def serialize(self) -> dict:
            return {
                "info": {
                    "config": {
                        "environment_type": "R2EGymEnvironment",
                        "cwd": self._cwd,
                    }
                }
            }

    return R2EGymEnvironment

def _force_kill_container(runtime) -> None:
    """Best-effort force-remove an instance's docker container.

    Used to unblock a wedged docker call: R2E-Gym's grading/patch helpers run
    commands *inside* the container, and if the container hangs at the docker
    layer the Python call blocks forever (the ``timeout=`` passed to
    ``_calculate_reward`` is only an in-container soft timeout). Removing the
    container makes the pending docker exec raise so the worker thread unwinds.
    """
    try:
        c = getattr(runtime, "container", None)
        if c is not None:
            c.remove(force=True)
            return
    except Exception:  # noqa: BLE001
        pass
    try:
        name = getattr(runtime, "container_name", None)
        if name:
            subprocess.run(["docker", "rm", "-f", name],
                           capture_output=True, timeout=60)
    except Exception:  # noqa: BLE001
        pass

def _run_with_deadline(fn, timeout: float, runtime, label: str):
    """Run ``fn()`` under a HARD wall-clock deadline (works in worker threads).

    On timeout, force-remove the instance container (to unblock a wedged docker
    exec) and raise ``TimeoutError``. The worker runs in a daemon thread, so a
    genuinely stuck call can never block the thread pool or process exit -- this
    is the safeguard that bounds the per-instance grading step (the previous
    code could hang indefinitely on a wedged container).
    """
    box: dict = {}

    def _target():
        try:
            box["value"] = fn()
        except BaseException as e:  # noqa: BLE001
            box["error"] = e

    t = threading.Thread(target=_target, name=f"deadline-{label}", daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        _force_kill_container(runtime)
        raise TimeoutError(f"{label} exceeded hard deadline of {timeout}s")
    if "error" in box:
        raise box["error"]
    return box.get("value")
