"""Render saved Mini-SWE/R2E trajectory records as complete Markdown files."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable


def _text(value: Any) -> str:
    """Flatten common chat-content encodings without dropping any text."""

    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        if "text" in value:
            return _text(value["text"])
        return json.dumps(value, ensure_ascii=False, indent=2, default=str)
    if isinstance(value, list):
        return "\n".join(_text(item) for item in value)
    return str(value)


def _fence(text: str, language: str = "text") -> str:
    """Use a fence that cannot be closed by the recorded content."""

    body = text.rstrip()
    longest = max((len(match) for match in re.findall(r"`+", body)), default=0)
    marker = "`" * max(3, longest + 1)
    return f"{marker}{language}\n{body}\n{marker}"


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "trajectory"


def _value(record: dict[str, Any], key: str, default: Any = "") -> Any:
    value = record.get(key, default)
    return default if value is None else value


def _render_tool_calls(message: dict[str, Any]) -> list[str]:
    calls = message.get("tool_calls") or []
    if not calls:
        return []
    lines = ["", "Tool calls:"]
    for index, call in enumerate(calls, start=1):
        function = call.get("function") if isinstance(call, dict) else {}
        function = function if isinstance(function, dict) else {}
        name = str(function.get("name") or "unknown")
        arguments = _text(function.get("arguments")).strip()
        lines.extend(["", f"#### Tool call {index}: `{name}`"])
        if arguments:
            lines.extend(["", _fence(arguments, "json")])
    return lines


def render_trajectory_markdown(record: dict[str, Any]) -> str:
    """Return an untruncated, human-readable Markdown transcript."""

    instance_id = str(_value(record, "instance_id", "unknown"))
    outcome = record.get("outcome") if isinstance(record.get("outcome"), dict) else {}
    exit_info = record.get("exit_info") if isinstance(record.get("exit_info"), dict) else {}
    messages = record.get("messages") if isinstance(record.get("messages"), list) else []
    budgets = record.get("budgets") if isinstance(record.get("budgets"), dict) else {}

    lines = [f"# Trajectory: `{instance_id}`", "", "## Outcome", ""]
    lines.extend(
        [
            "| Field | Value |",
            "| --- | --- |",
            f"| Official reward | `{_value(record, 'official_reward', 0)}` |",
            f"| Explicitly submitted | `{_value(record, 'official_submitted', False)}` |",
            f"| Exit status | `{_value(outcome, 'exit_status', exit_info.get('exit_status', ''))}` |",
            f"| Trainable | `{_value(outcome, 'trainable', False)}` |",
            f"| External stop | `{_value(outcome, 'external_stop', False)}` |",
            f"| Timed out | `{_value(outcome, 'timed_out', False)}` |",
            f"| Agent calls | `{_value(outcome, 'agent_calls', '')}` |",
            f"| Elapsed seconds | `{_value(outcome, 'elapsed_seconds', '')}` |",
        ]
    )
    error = _text(_value(outcome, "error", "")).strip()
    if error:
        lines.extend(["", "### Error", "", _fence(error)])

    lines.extend(["", "## Run context", ""])
    lines.append(f"- Session: `{_value(record, 'session_id', '')}`")
    lines.append(f"- Hint used: `{_value(record, 'hint_used', False)}`")
    if budgets:
        lines.extend(["- Budgets:", "", _fence(json.dumps(budgets, ensure_ascii=False, indent=2), "json")])
    hint = _text(_value(record, "hint", "")).strip()
    if hint:
        lines.extend(["", "### Hint", "", _fence(hint)])

    lines.extend(["", "## Full conversation"])
    if not messages:
        lines.extend(["", "(No messages were captured in this record.)"])
    for index, message in enumerate(messages, start=1):
        if not isinstance(message, dict):
            lines.extend(["", f"### Event {index}", "", _fence(_text(message))])
            continue
        role = str(message.get("role") or "unknown")
        lines.extend(["", f"### {index}. `{role}`"])
        name = _text(message.get("name")).strip()
        if name:
            lines.extend(["", f"Name: `{name}`"])
        content = _text(message.get("content")).strip()
        if content:
            lines.extend(["", _fence(content)])
        lines.extend(_render_tool_calls(message))
        function_call = message.get("function_call")
        if function_call:
            lines.extend(["", "Function call:", "", _fence(_text(function_call), "json")])

    for heading, key in (
        ("Explicit submission patch", "explicit_submission_patch"),
        ("Worktree patch", "worktree_patch"),
        ("Unsubmitted worktree patch", "unsubmitted_worktree_patch"),
    ):
        patch = _text(_value(record, key, "")).strip()
        if patch:
            lines.extend(["", f"## {heading}", "", _fence(patch, "diff")])

    if exit_info:
        lines.extend(
            [
                "",
                "## Exit metadata",
                "",
                _fence(json.dumps(exit_info, ensure_ascii=False, indent=2, default=str), "json"),
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def markdown_path_for_record(record: dict[str, Any], output_dir: Path) -> Path:
    instance_id = _safe_name(str(_value(record, "instance_id", "unknown")))
    session_id = _safe_name(str(_value(record, "session_id", "session")))
    return output_dir / f"{instance_id}.{session_id}.md"


def write_trajectory_markdown(record: dict[str, Any], output_dir: Path) -> Path:
    """Atomically write one record's Markdown transcript and return its path."""

    output_dir.mkdir(parents=True, exist_ok=True)
    target = markdown_path_for_record(record, output_dir)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(render_trajectory_markdown(record), encoding="utf-8")
    temporary.replace(target)
    return target


def write_index(records: Iterable[dict[str, Any]], output_dir: Path) -> Path:
    """Write a compact index for a set of records already rendered to Markdown."""

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = list(records)
    lines = ["# Trajectory index", "", "| Instance | Reward | Submitted | Exit | Transcript |", "| --- | ---: | --- | --- | --- |"]
    for record in sorted(rows, key=lambda value: str(_value(value, "instance_id", ""))):
        outcome = record.get("outcome") if isinstance(record.get("outcome"), dict) else {}
        path = markdown_path_for_record(record, output_dir)
        lines.append(
            "| `{instance}` | {reward} | {submitted} | `{status}` | [{name}]({name}) |".format(
                instance=_value(record, "instance_id", "unknown"),
                reward=_value(record, "official_reward", 0),
                submitted=_value(record, "official_submitted", False),
                status=_value(outcome, "exit_status", ""),
                name=path.name,
            )
        )
    target = output_dir / "_index.md"
    temporary = target.with_suffix(".md.tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(target)
    return target

