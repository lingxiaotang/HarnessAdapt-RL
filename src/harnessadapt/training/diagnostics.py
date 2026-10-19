"""Normalize harness failures recorded by both rollout collectors."""
from __future__ import annotations


def _count(value) -> int:
    if isinstance(value, dict):
        return sum(_count(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_count(item) for item in value)
    if value is None:
        return 0
    return max(0, int(value))


def runtime_errors(record: dict) -> tuple[int, int]:
    outcome = record.get("outcome") or {}

    def counter(*names):
        # Diagnostics may be duplicated at the top level and in outcome.
        return max(
            (_count(source[name]) for source in (record, outcome)
             for name in names if name in source),
            default=0,
        )

    hooks = counter("harness_hook_errors", "hook_errors")
    tools = counter("harness_tool_exceptions", "tool_exceptions")
    # Older trajectories contain tool failures only in tool observations.
    observations = sum(
        message.get("role") == "tool" and any(
            code in str(message.get("content", ""))
            for code in ('"tool_runtime_error"', '"invalid_tool_result"')
        )
        for message in record.get("messages", [])
    )
    return hooks, max(tools, observations)
