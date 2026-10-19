"""Training-time runtime adapter for evolved harnesses (H1 / H2) with Slime token/logprob capture."""
from __future__ import annotations

import functools
import gzip
import json
import os
import secrets
import types
from pathlib import Path
from types import ModuleType
from typing import Any

from harnessadapt.runtime import (
    ENABLE_HOOKS_ENV,
    ENABLE_TOOLS_ENV,
    HARNESS_ENV,
    PROFILE_ENV,
    LoadedHarness,
    _instance_id_from_session,
    _parse_actions,
    _runtime_view,
    _worktree_snapshot,
    _worktree_status,
)


def _remaining_context_after_tools(model, messages, tools, base_module):
    from slime.agent.adapters.common import _render_token_ids
    from slime.agent.adapters.openai import _tools_to_chat_tools, _translate_messages

    tokenizer = model._budget_tokenizer
    if tokenizer is None:
        raise ValueError("Context-budget hook requires the served tokenizer")
    prepare = getattr(model, "_prepare_messages_for_api", None)
    prepared = prepare(messages) if prepare else messages
    prepared = base_module._normalize_tool_arguments_for_template(prepared)
    ids = _render_token_ids(
        _translate_messages(prepared), tokenizer, tools=_tools_to_chat_tools(tools)
    )
    return max(0, int(model.model_context_length) - int(model.safety_margin) - len(ids))


def _install_model_tools(
    model: Any, harness: LoadedHarness, base_module: ModuleType
) -> Any:
    inner = getattr(model, "_model", model)

    if inner.__class__.__name__ == "FastOpenAIModel":

        def fast_query(self: Any, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
            from litellm import exceptions as E
            from harnessadapt.training.runtime import fast_model

            params = self.config.model_kwargs | kwargs
            api_base = str(params.get("api_base") or "").rstrip("/")
            if not api_base:
                raise E.BadRequestError(
                    message="fast model client requires api_base",
                    model=self.config.model_name,
                    llm_provider="openai",
                )
            body = {
                key: value
                for key, value in params.items()
                if key not in fast_model._TRANSPORT_KEYS and value is not None
            }
            body.update(
                {
                    "model": fast_model._wire_model_name(self.config.model_name),
                    "messages": messages,
                    "tools": harness.schemas(),
                }
            )
            headers = {"Content-Type": "application/json"}
            if params.get("api_key"):
                headers["Authorization"] = f"Bearer {params['api_key']}"
            url = f"{api_base}/chat/completions"
            try:
                response = fast_model._session().post(
                    url,
                    data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                    headers=headers,
                    timeout=params.get("timeout"),
                )
            except Exception as error:
                raise E.APIConnectionError(
                    message=f"{type(error).__name__}: {error}",
                    model=self.config.model_name,
                    llm_provider="openai",
                ) from error
            if response.status_code >= 400:
                fast_model._raise_for_status(
                    response.status_code, response.text, self.config.model_name, url
                )
            return fast_model._to_model_response(response.json())

        inner._query = types.MethodType(fast_query, inner)
    else:

        def stock_query(self: Any, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
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
    original_write_record = base_module._write_training_trajectory_record
    original_run_case = base_module._run_case

    @functools.wraps(original_run_case)
    def run_case(**kwargs: Any) -> Any:
        if release.skill_injection_style != "legacy" and kwargs.get("opsd_note_override") is None:
            skill = str(kwargs.get("hint") or "").strip()
            kwargs["opsd_note_override"] = (
                "\n\n## Skills for this task\n\n"
                "Apply these task-specific skills when solving the current issue:\n\n"
                + skill
                + "\n"
                if skill
                else ""
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
                system = system.replace("{{ opsd_note | default('') }}", "")
                kwargs["system_template"] = (
                    system.rstrip()
                    + "\n\n## General problem-solving skills\n\n"
                    + "Apply these general skills while investigating, modifying, "
                    + "and completing the task:\n\n"
                    + "{{ general_problem_solving_skills }}"
                    + "{{ opsd_note | default('') }}"
                )
                if self._evolved_harness.prompt_mode == "replace":
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
                    status_before = _worktree_status(self.env) if continuous_status else ""
                    name = str(action.get("tool_name") or "bash")
                    if name == "bash":
                        outputs.append(self.env.execute(action))
                    else:
                        outputs.append(self._evolved_harness.execute_tool(action, self.env))
                    status_pairs.append(
                        (status_before, _worktree_status(self.env) if continuous_status else "")
                    )
                budget_snapshot = (
                    _worktree_snapshot(self.env)
                    if actions
                    and step_limit
                    and any(
                        remaining in getattr(module, "WORKTREE_REMAINING_TURNS", ())
                        for module in self._evolved_harness.hooks.values()
                    )
                    else {}
                )
                observations = self.model.format_observation_messages(
                    message, outputs, self.get_template_vars()
                )
                context_remaining = None
                if any(
                    getattr(h, "USE_CONTEXT_BUDGET", False)
                    for h in self._evolved_harness.hooks.values()
                ):
                    context_remaining = _remaining_context_after_tools(
                        self.model,
                        self.messages + observations,
                        self._evolved_harness.schemas(),
                        base_module,
                    )
                for action_index, (action, output, observation, statuses) in enumerate(
                    zip(actions, outputs, observations, status_pairs)
                ):
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

    @functools.wraps(original_write_record)
    def write_record(*args: Any, **kwargs: Any) -> None:
        instance = kwargs.get("instance")
        if isinstance(instance, dict) and not str(instance.get("instance_id") or ""):
            fallback = _instance_id_from_session(str(kwargs.get("session_id") or ""))
            if fallback:
                kwargs = dict(kwargs)
                kwargs["instance"] = {**instance, "instance_id": fallback}
        original_write_record(*args, **kwargs)
        outcome = kwargs.get("outcome")
        agent = kwargs.get("agent")
        path_text = str(getattr(outcome, "trajectory_record_path", "") or "")
        if not path_text:
            return
        target = Path(path_text)
        try:
            with gzip.open(target, "rt", encoding="utf-8") as handle:
                payload = json.load(handle)
            harness = getattr(agent, "_evolved_harness", None)
            diagnostics = (
                harness.diagnostics() if harness is not None else release.diagnostics()
            )
            payload.update(diagnostics)
            payload.setdefault("outcome", {}).update(diagnostics)
            temporary = target.with_suffix(target.suffix + f".{secrets.token_hex(4)}.tmp")
            with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=3) as handle:
                json.dump(payload, handle, ensure_ascii=False, default=str)
            os.replace(temporary, target)
            from harnessadapt.training.runtime.trajectory_markdown import (
                write_trajectory_markdown,
            )

            write_trajectory_markdown(payload, target.parent.parent / "train_markdown")
        except Exception:
            return

    base_module._case_model = case_model
    base_module._agent_class = agent_class
    base_module._write_training_trajectory_record = write_record
    base_module._run_case = run_case
    return release


def _configured_release() -> Path:
    raw = os.environ.get(HARNESS_ENV, "").strip()
    if not raw:
        raise ValueError(f"{HARNESS_ENV} must point to a published harness release")
    return Path(raw)


_ACTIVE_HARNESS: LoadedHarness | None = None
_BASE_GENERATE_MODULE: ModuleType | None = None


def _ensure_installed() -> LoadedHarness:
    global _ACTIVE_HARNESS, _BASE_GENERATE_MODULE
    if _ACTIVE_HARNESS is None:
        from harnessadapt.training.runtime import generate as base_generate_module

        _BASE_GENERATE_MODULE = base_generate_module
        _ACTIVE_HARNESS = _install(base_generate_module, _configured_release())
    return _ACTIVE_HARNESS


async def generate(
    args: Any,
    base_sample: Any,
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> Any:
    _ensure_installed()
    if _BASE_GENERATE_MODULE is None:
        raise RuntimeError("Base generate module was not installed")
    return await _BASE_GENERATE_MODULE.generate(
        args, base_sample, sampling_params, evaluation=evaluation
    )


__all__ = [
    "ENABLE_HOOKS_ENV",
    "ENABLE_TOOLS_ENV",
    "HARNESS_ENV",
    "LoadedHarness",
    "PROFILE_ENV",
    "generate",
]
