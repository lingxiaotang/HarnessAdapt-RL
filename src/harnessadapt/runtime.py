"""Publication adapter derived from the existing harness runtime; inference only."""

from __future__ import annotations

import copy
import functools
import gzip
import importlib.util
import json
import os
import re
import secrets
import types
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parents[2]
HARNESS_ENV = "R2E_EVOLVED_HARNESS"
ENABLE_TOOLS_ENV = "R2E_EVOLVED_ENABLE_TOOLS"
ENABLE_HOOKS_ENV = "R2E_EVOLVED_ENABLE_HOOKS"
PROFILE_ENV = "R2E_EVOLVED_PROFILE"


def _enabled(name: str, *, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {raw!r}")


def _adapt_portable_python_source(source: str) -> str:
    """Route internal harness tool helpers through mounted `/opt/v5python` without altering solver shell."""
    return (
        source.replace(
            '"timeout 12s python3 -c "',
            '"PYTHONHOME=/opt/v5python timeout 12s /opt/v5python/bin/python3 -c "',
        )
        .replace(
            '"cd /testbed && python3 -c "',
            '"cd /testbed && PYTHONHOME=/opt/v5python /opt/v5python/bin/python3 -c "',
        )
        .replace(
            r"""SMOKE_ARGUMENTS = {'command':"python3 -c 'assert 2 + 2 == 4; print(\"assertions passed\")'"}""",
            r"""SMOKE_ARGUMENTS = {'command':"PYTHONHOME=/opt/v5python /opt/v5python/bin/python3 -c 'assert 2 + 2 == 4; print(\"assertions passed\")'"}""",
        )
        .replace(
            r"""+' >"$harness_check_log" 2>&1\nharness_check_rc=$?\npython3 -c '""",
            r"""+' >"$harness_check_log" 2>&1\nharness_check_rc=$?\nPYTHONHOME=/opt/v5python /opt/v5python/bin/python3 -c '""",
        )
    )


def _load_module(path: Path, prefix: str) -> ModuleType:
    name = f"{prefix}_{path.stem}_{secrets.token_hex(6)}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load harness module: {path}")
    module = importlib.util.module_from_spec(spec)
    import sys

    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        if prefix == "runtime_tool" and _enabled("R2E_PORTABLE_PYTHON_TOOLS", default=False):
            source = _adapt_portable_python_source(path.read_text(encoding="utf-8"))
            module.__file__ = str(path)
            exec(compile(source, str(path), "exec"), module.__dict__)
        else:
            spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


@dataclass
class LoadedHarness:
    directory: Path
    release: int
    bank_release: int
    profile: str
    system_prompt: str
    tools: dict[str, ModuleType]
    hooks: dict[str, ModuleType]
    hook_states: dict[str, dict[str, Any]] = field(default_factory=dict)
    tool_calls: dict[str, int] = field(default_factory=dict)
    hook_prompts: dict[str, int] = field(default_factory=dict)
    hook_errors: dict[str, int] = field(default_factory=dict)
    tool_exceptions: dict[str, int] = field(default_factory=dict)
    include_native_bash: bool = True
    prompt_mode: str = "augment"
    skill_injection_style: str = "natural"
    instance_template: str = ""

    @classmethod
    def load(cls, directory: Path) -> "LoadedHarness":
        resolved = directory.resolve(strict=True)
        if not resolved.is_dir() or not (resolved / "release.json").is_file():
            raise ValueError(f"harness directory is missing release.json: {resolved}")
        metadata = json.loads((resolved / "release.json").read_text(encoding="utf-8"))
        release = int(metadata.get("harness_release") if metadata.get("harness_release") is not None else -1)
        bank_release = int(metadata.get("bank_release") or 0)
        if release < 0 or bank_release <= 0:
            raise ValueError("harness release metadata is incomplete")
        system_prompt = (
            (resolved / "system_prompt.md").read_text(encoding="utf-8").strip()
        )
        if not system_prompt and not metadata.get('allow_empty_meta_skills', False):
            raise ValueError("harness system prompt is empty")
        expected_tools = tuple(str(name) for name in metadata.get("tools") or ())
        expected_hooks = tuple(str(name) for name in metadata.get("hooks") or ())
        style = metadata.get("skill_injection_style", "natural")
        if style not in {"natural", "legacy"}:
            raise ValueError(f"unknown skill injection style: {style}")
        prompt_mode = metadata.get("prompt_mode", "augment")
        if prompt_mode not in {"augment", "replace"}:
            raise ValueError(f"unknown harness prompt mode: {prompt_mode}")
        include_native_bash = metadata.get("include_native_bash", True)
        if not isinstance(include_native_bash, bool):
            raise ValueError("include_native_bash must be a boolean")
        template_file = resolved / "instance_template.md"
        tools = {
            name: _load_module(resolved / "tools" / f"{name}.py", "runtime_tool")
            for name in expected_tools
        }
        hooks = {
            name: _load_module(resolved / "hooks" / f"{name}.py", "runtime_hook")
            for name in expected_hooks
        }
        for name, module in tools.items():
            schema = getattr(module, "TOOL_SCHEMA", None)
            if (
                not isinstance(schema, dict)
                or (schema.get("function") or {}).get("name") != name
            ):
                raise ValueError(f"invalid runtime Tool schema: {name}")
            if not callable(getattr(module, "execute", None)):
                raise ValueError(f"runtime Tool has no execute function: {name}")
        for name, module in hooks.items():
            if getattr(module, "HOOK_NAME", None) != name:
                raise ValueError(f"invalid runtime Hook name: {name}")
            if not callable(getattr(module, "observe", None)):
                raise ValueError(f"runtime Hook has no observe function: {name}")
        return cls(
            directory=resolved,
            release=release,
            bank_release=bank_release,
            profile=str(metadata.get("profile") or f"harness_h{release}"),
            system_prompt=system_prompt,
            tools=tools,
            hooks=hooks,
            hook_states={name: {} for name in hooks},
            skill_injection_style=style,
            include_native_bash=include_native_bash,
            prompt_mode=prompt_mode,
            instance_template=template_file.read_text() if template_file.is_file() else "",
        )

    def schemas(self) -> list[dict[str, Any]]:
        schemas = []
        if self.include_native_bash:
            from minisweagent.models.utils.actions_toolcall import BASH_TOOL

            schemas.append(BASH_TOOL)
        return schemas + [self.tools[name].TOOL_SCHEMA for name in self.tools]

    def execute_tool(self, action: dict[str, Any], environment: Any) -> dict[str, Any]:
        name = str(action.get("tool_name") or "")
        module = self.tools[name]
        arguments = {
            key: copy.deepcopy(value)
            for key, value in action.items()
            if key not in {"tool_name", "tool_call_id"}
        }
        self.tool_calls[name] = self.tool_calls.get(name, 0) + 1
        try:
            result = module.execute(arguments, environment)
        except Exception as error:
            from minisweagent.exceptions import Submitted

            if isinstance(error, Submitted):
                # An explicit submission tool uses the same terminal event as
                # bash submission; do not turn successful delivery into error.
                raise
            self.tool_exceptions[name] = self.tool_exceptions.get(name, 0) + 1
            return {
                "output": json.dumps(
                    {
                        "ok": False,
                        "tool": name,
                        "error": {
                            "code": "tool_runtime_error",
                            "message": f"{type(error).__name__}: {error}"[:1000],
                        },
                        "next_action": "inspect the error and use an existing focused action",
                    },
                    ensure_ascii=False,
                ),
                "returncode": 2,
                "exception_info": "",
            }
        if not isinstance(result, dict) or not {
            "output",
            "returncode",
            "exception_info",
        }.issubset(result):
            self.tool_exceptions[name] = self.tool_exceptions.get(name, 0) + 1
            return {
                "output": json.dumps(
                    {
                        "ok": False,
                        "tool": name,
                        "error": {
                            "code": "invalid_tool_result",
                            "message": "The generated Tool returned an invalid observation.",
                        },
                    }
                ),
                "returncode": 2,
                "exception_info": "",
            }
        bounded = dict(result)
        if bounded.get("exception_info"):
            self.tool_exceptions[name] = self.tool_exceptions.get(name, 0) + 1
        output = str(bounded.get("output") or "")
        if len(output) > 10_000:
            output = output[:5_000] + "\n<truncated>\n" + output[-5_000:]
        bounded["output"] = output
        return bounded

    def observe(
        self,
        action: dict[str, Any],
        output: dict[str, Any],
        *,
        step: int,
        worktree_status_before: str = "",
        worktree_status_after: str = "",
        step_limit: int = 0,
        worktree_snapshot: dict[str, Any] | None = None,
        initial_worktree_snapshot: dict[str, Any] | None = None,
        last_action_in_turn: bool = True,
        context_remaining_tokens: int | None = None,
    ) -> tuple[str, ...]:
        arguments = {
            key: copy.deepcopy(value)
            for key, value in action.items()
            if key not in {"tool_name", "tool_call_id"}
        }
        event = {
            "tool_name": str(action.get("tool_name") or "bash"),
            "arguments": arguments,
            "output": str(output.get("output") or "")[:10_000],
            "returncode": output.get("returncode"),
            "exception_info": str(output.get("exception_info") or "")[:2_000],
            "normalized_call": json.dumps(
                {
                    "tool_name": str(action.get("tool_name") or "bash"),
                    "arguments": arguments,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ),
            "step": int(step),
            "context_remaining_tokens": context_remaining_tokens,
            "step_limit": int(step_limit),
            "remaining_turns": max(0, int(step_limit) - int(step)) if step_limit else None,
            "worktree_snapshot": copy.deepcopy(worktree_snapshot or {}),
            "initial_worktree_snapshot": copy.deepcopy(initial_worktree_snapshot or {}),
            "last_action_in_turn": bool(last_action_in_turn),
            "worktree_status_before": worktree_status_before,
            "worktree_status_after": worktree_status_after,
            "worktree_delta": _worktree_delta(
                worktree_status_before, worktree_status_after
            ),
        }
        messages: list[str] = []
        for name, module in self.hooks.items():
            try:
                message = module.observe(copy.deepcopy(event), self.hook_states[name])
            except Exception:
                self.hook_errors[name] = self.hook_errors.get(name, 0) + 1
                continue
            if isinstance(message, str) and message.strip():
                prompt = str(getattr(module, "PROMPT", ""))
                if not _hook_prompt_matches(prompt, message):
                    self.hook_errors[name] = self.hook_errors.get(name, 0) + 1
                    continue
                self.hook_prompts[name] = self.hook_prompts.get(name, 0) + 1
                messages.append(f"[{name}] {message.strip()}")
        return tuple(messages)

    def diagnostics(self) -> dict[str, Any]:
        return {
            "harness_profile": self.profile,
            "harness_prompt_mode": self.prompt_mode,
            "harness_include_native_bash": self.include_native_bash,
            "harness_release": self.release,
            "harness_bank_release": self.bank_release,
            "harness_tool_calls": dict(self.tool_calls),
            "harness_hook_prompts": dict(self.hook_prompts),
            "harness_hook_errors": dict(self.hook_errors),
            "harness_tool_exceptions": dict(self.tool_exceptions),
            "harness_tools_enabled": bool(self.tools),
            "harness_hooks_enabled": bool(self.hooks),
        }


def _runtime_view(harness: LoadedHarness) -> LoadedHarness:
    """Apply explicit experiment toggles without changing a published release."""

    if not _enabled(ENABLE_TOOLS_ENV):
        harness.tools = {}
        harness.tool_calls = {}
        harness.tool_exceptions = {}
    if not _enabled(ENABLE_HOOKS_ENV):
        harness.hooks = {}
        harness.hook_states = {}
        harness.hook_prompts = {}
        harness.hook_errors = {}
    profile = os.environ.get(PROFILE_ENV, "").strip()
    if profile:
        harness.profile = profile
    return harness


def _remaining_context_after_tools(model, messages, tools, base_module):
    return model.remaining_context(messages, tools)


def _hook_prompt_matches(template: str, message: str) -> bool:
    if "<factual delta>" not in template:
        return message.strip() == template.strip()
    before, after = template.split("<factual delta>", 1)
    stripped = message.strip()
    return (
        stripped.startswith(before)
        and stripped.endswith(after)
        and bool(stripped[len(before) : len(stripped) - len(after) if after else None])
    )


def _worktree_delta(before: str, after: str) -> str:
    before_lines = set(before.splitlines())
    after_lines = set(after.splitlines())
    changes = [f"+ {line}" for line in sorted(after_lines - before_lines)]
    changes.extend(f"- {line}" for line in sorted(before_lines - after_lines))
    return "; ".join(changes)


def _worktree_status(environment: Any) -> str:
    return _worktree_snapshot(environment)["status"]


def _worktree_snapshot(environment: Any) -> dict[str, Any]:
    try:
        result = environment.execute(
            {"command": "git status --porcelain=v1"}
        )
    except Exception:
        return {"available": False, "status": ""}
    if not isinstance(result, dict) or result.get("returncode") != 0:
        return {"available": False, "status": ""}
    return {"available": True, "status": str(result.get("output") or "")[:20_000]}


def _instance_id_from_session(session_id: str) -> str:
    match = re.fullmatch(r"r2e-(.+)-\d+-\d+", session_id)
    return match.group(1) if match else ""


def _tool_value(value: Any, field: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(field, default)
    return getattr(value, field, default)


def _validate_arguments(arguments: Any, schema: dict[str, Any]) -> str:
    if not isinstance(arguments, dict):
        return "arguments must be a JSON object"
    properties = schema.get("properties") or {}
    required = set(schema.get("required") or ())
    missing = sorted(required - set(arguments))
    if missing:
        return f"missing required arguments: {missing}"
    if schema.get("additionalProperties") is False:
        extra = sorted(set(arguments) - set(properties))
        if extra:
            return f"unknown arguments: {extra}"
    python_types = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    for name, value in arguments.items():
        definition = properties.get(name) or {}
        expected = definition.get("type")
        expected_type = python_types.get(expected)
        if expected_type is not None and not isinstance(value, expected_type):
            return f"argument {name!r} must have type {expected}"
        if expected == "integer" and isinstance(value, bool):
            return f"argument {name!r} must have type integer"
        if expected == "number" and isinstance(value, bool):
            return f"argument {name!r} must have type number"
        if "enum" in definition and value not in definition["enum"]:
            return f"argument {name!r} must be one of {definition['enum']}"
    return ""


def _parse_actions(
    harness: LoadedHarness, model: Any, response: Any
) -> list[dict[str, Any]]:
    from jinja2 import StrictUndefined, Template
    from minisweagent.exceptions import FormatError

    tool_calls = response.choices[0].message.tool_calls or []
    if not tool_calls:
        raise FormatError(
            {
                "role": "user",
                "content": Template(
                    model.config.format_error_template, undefined=StrictUndefined
                ).render(
                    error="No tool calls found in the response. Every response MUST include at least one tool call.",
                    actions=[],
                    has_tool_calls=False,
                    finish_reason=response.choices[0].finish_reason,
                ),
                "extra": {"interrupt_type": "FormatError"},
            }
        )
    schemas = {
        schema["function"]["name"]: schema["function"]["parameters"]
        for schema in harness.schemas()
    }
    actions: list[dict[str, Any]] = []
    for call in tool_calls:
        function = _tool_value(call, "function", {})
        name = str(_tool_value(function, "name", ""))
        raw = _tool_value(function, "arguments", "")
        error = ""
        try:
            arguments = json.loads(raw) if isinstance(raw, str) else raw
        except Exception as exception:
            arguments = {}
            error = f"Error parsing tool call arguments: {exception}."
        if name not in schemas:
            error += f" Unknown tool {name!r}."
        else:
            validation_error = _validate_arguments(arguments, schemas[name])
            if validation_error:
                error += " " + validation_error
            if getattr(harness.tools.get(name), "TERMINAL", False) and len(tool_calls) != 1:
                error += " A terminal tool must be the only tool call in its response."
        if error:
            raise FormatError(
                {
                    "role": "user",
                    "content": Template(
                        model.config.format_error_template, undefined=StrictUndefined
                    ).render(
                        error=error.strip(),
                        actions=[],
                        has_tool_calls=True,
                        finish_reason=response.choices[0].finish_reason,
                    ),
                    "extra": {"interrupt_type": "FormatError"},
                }
            )
        action = {
            "tool_name": name,
            "tool_call_id": _tool_value(call, "id", ""),
            **arguments,
        }
        if name == "bash":
            action["command"] = arguments["command"]
        actions.append(action)
    return actions


def _install_model_tools(
    model: Any, harness: LoadedHarness, base_module: ModuleType
) -> Any:
    inner = getattr(model, "_model", model)

    def stock_query(
        self: Any, messages: list[dict[str, Any]], **kwargs: Any
    ) -> Any:
        import litellm

        return litellm.completion(
            model=self.config.model_name,
            messages=messages,
            tools=harness.schemas(),
            **(self.config.model_kwargs | kwargs),
        )

    inner._query = types.MethodType(stock_query, inner)


    inner._parse_actions = types.MethodType(
        lambda self, response: _parse_actions(harness, self, response), inner
    )
    inner._slime_format_error_history_recovery = False
    base_module._install_format_error_history_recovery(inner)
    return model


def _install(base_module: ModuleType, release_directory: Path) -> LoadedHarness:
    release = _runtime_view(LoadedHarness.load(release_directory))
    original_case_model = base_module._case_model
    original_agent_class = base_module._agent_class
    original_run_case = base_module._run_case

    @functools.wraps(original_run_case)
    def run_case(**kwargs: Any) -> Any:
        # Reuse the existing per-task carrier without its old hint/persona
        # framing. Preserve explicit carrier overrides from other callers.
        if release.skill_injection_style != "legacy" and kwargs.get("opsd_note_override") is None:
            skill = str(kwargs.get("hint") or "").strip()
            kwargs["opsd_note_override"] = (
                "\n\n## Skills for this task\n\n"
                "Apply these task-specific skills when solving the current issue:\n\n"
                + skill + "\n"
                if skill else ""
            )
        return original_run_case(**kwargs)

    @functools.wraps(original_case_model)
    def case_model(*args: Any, **kwargs: Any) -> Any:
        model = original_case_model(*args, **kwargs)
        case_harness = _runtime_view(LoadedHarness.load(release.directory))
        model._evolved_harness = case_harness
        if any(getattr(h, "USE_CONTEXT_BUDGET", False) for h in case_harness.hooks.values()):
            model._budget_tokenizer = kwargs.get("tokenizer", args[3] if len(args) > 3 else None)
        return _install_model_tools(model, case_harness, base_module)

    def agent_class() -> type:
        parent = original_agent_class()

        class EvolvedHarnessAgent(parent):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                model = args[0] if args else kwargs.get("model")
                self._evolved_harness = getattr(model, "_evolved_harness")
                system = str(kwargs.get("system_template") or "")
                original_system = system
                # Put general skills before the task-specific carrier. Render
                # skill text as variable values, never as Jinja source code.
                system = system.replace("{{ opsd_note | default('') }}", "")
                kwargs["system_template"] = (
                    system.rstrip()
                    + "\n\n## General problem-solving skills\n\n"
                    + "Apply these general skills while investigating, modifying, "
                    + "and completing the task:\n\n"
                    + "{{ general_problem_solving_skills }}"
                    + "{{ opsd_note | default('') }}"
                )
                if self._evolved_harness.skill_injection_style == "legacy":
                    # Exact old carrier for comparisons against archived runs.
                    kwargs["system_template"] = (
                        str(original_system)
                        + "\n\n<EVOLVED_HARNESS>\n"
                        + self._evolved_harness.system_prompt
                        + "\n</EVOLVED_HARNESS>"
                    )
                if self._evolved_harness.prompt_mode == "replace":
                    # Render the independent backend prompt as data, not Jinja.
                    kwargs["system_template"] = "{{ general_problem_solving_skills }}"
                if self._evolved_harness.instance_template:
                    kwargs["instance_template"] = self._evolved_harness.instance_template
                super().__init__(*args, **kwargs)
                self._budget_hook_snapshots = any(
                    hasattr(module, "WORKTREE_REMAINING_TURNS")
                    for module in self._evolved_harness.hooks.values()
                )
                self._initial_worktree_snapshot = (
                    _worktree_snapshot(self.env) if self._budget_hook_snapshots else {}
                )

            def get_template_vars(self, **kwargs: Any) -> dict[str, Any]:
                variables = super().get_template_vars(**kwargs)
                variables["general_problem_solving_skills"] = self._evolved_harness.system_prompt
                # Keep optional-tool instructions consistent with ablation flags.
                variables["harness_tool_names"] = tuple(self._evolved_harness.tools)
                return variables

            def execute_actions(self, message: dict[str, Any]) -> list[dict[str, Any]]:
                actions = message.get("extra", {}).get("actions", [])
                outputs: list[dict[str, Any]] = []
                status_pairs: list[tuple[str, str]] = []
                step_limit = int(getattr(self.config, "step_limit", 0) or 0)
                remaining = max(0, step_limit - int(self.n_calls))
                hooks = self._evolved_harness.hooks.values()
                continuous_status = any(
                    not hasattr(module, "WORKTREE_REMAINING_TURNS") for module in hooks
                )
                for action in actions:
                    status_before = (
                        _worktree_status(self.env)
                        if continuous_status
                        else ""
                    )
                    name = str(action.get("tool_name") or "bash")
                    if name == "bash":
                        outputs.append(self.env.execute(action))
                    else:
                        outputs.append(
                            self._evolved_harness.execute_tool(action, self.env)
                        )
                    status_pairs.append(
                        (
                            status_before,
                            _worktree_status(self.env)
                            if continuous_status
                            else "",
                        )
                    )
                budget_snapshot = (
                    _worktree_snapshot(self.env)
                    if actions and step_limit and any(
                        remaining in getattr(module, "WORKTREE_REMAINING_TURNS", ())
                        for module in self._evolved_harness.hooks.values()
                    ) else {}
                )
                observations = self.model.format_observation_messages(
                    message, outputs, self.get_template_vars()
                )
                context_remaining = None
                if any(getattr(h, "USE_CONTEXT_BUDGET", False) for h in self._evolved_harness.hooks.values()):
                    context_remaining = _remaining_context_after_tools(
                        self.model, self.messages + observations, self._evolved_harness.schemas(), base_module)
                for action_index, (action, output, observation, statuses) in enumerate(zip(
                    actions, outputs, observations, status_pairs
                )):
                    prompts = self._evolved_harness.observe(
                        action,
                        output,
                        step=int(self.n_calls),
                        worktree_status_before=statuses[0],
                        worktree_status_after=statuses[1],
                        step_limit=step_limit,
                        worktree_snapshot=budget_snapshot,
                        initial_worktree_snapshot=self._initial_worktree_snapshot,
                        last_action_in_turn=action_index == len(actions) - 1,
                        context_remaining_tokens=context_remaining,
                    )
                    if prompts:
                        observation["content"] = (
                            str(observation.get("content") or "")
                            + "\n<HARNESS_GUIDANCE>\n"
                            + "\n".join(prompts)
                            + "\n</HARNESS_GUIDANCE>"
                        )
                return self.add_messages(*observations)

        EvolvedHarnessAgent.__name__ = f"EvolvedHarnessAgentH{release.release}"
        return EvolvedHarnessAgent

    base_module._case_model = case_model
    base_module._agent_class = agent_class
    base_module._run_case = run_case
    return release

