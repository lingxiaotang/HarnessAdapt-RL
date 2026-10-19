"""Slime custom rollout function for Mini-SWE-Agent on local R2E Docker tasks.

The model-facing HTTP adapter is Slime's own ``OpenAIAdapter``.  It records the
exact SGLang token ids and rollout log probabilities for every assistant turn;
Mini-SWE-Agent only supplies the control loop and bash tool execution.

Failure handling is deliberately conservative:

* one prompt execution always returns exactly one ``Sample`` so an eight-sample
  GRPO group can never be reshaped incorrectly;
* only trajectories that finish through the explicit R2E submission protocol
  are trainable; every other exit receives an all-zero loss mask;
* each Mini-SWE turn gets the model context that remains at that turn instead
  of a fixed response cap.
"""

from __future__ import annotations

import asyncio
import functools
import gzip
import hashlib
import json
import logging
import math
import os
import re
import secrets
import subprocess
import tempfile
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

from slime.agent.adapters import OpenAIAdapter
from slime.agent.aiohttp_threaded import FilteredAccessLogger, run_app_in_thread
from slime.utils.misc import SingletonMeta
from slime.utils.processing_utils import load_tokenizer
from slime.utils.types import Sample

from .curriculum import hint_for_sample

logger = logging.getLogger(__name__)

os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")
os.environ.setdefault("MSWEA_COST_TRACKING", "ignore_errors")


HEAD_TAIL_OBSERVATION_TEMPLATE = """\
{% if output.exception_info %}<exception>{{ output.exception_info }}</exception>
{% endif %}<returncode>{{ output.returncode }}</returncode>
{% if output.output | length < 10000 -%}
<output>
{{ output.output -}}
</output>
{%- else -%}
<warning>
The output of your last command was too long.
Please try a different command that produces less output.
If you're looking at a file you can try use head, tail or sed to view a smaller number of lines selectively.
If you're using grep or find and it produced too much output, you can use a more selective search pattern.
If you really need to see something from the full command's output, you can redirect output to a file and then search in that file.
</warning>
{%- set elided_chars = output.output | length - 10000 -%}
<output_head>
{{ output.output[:5000] }}
</output_head>
<elided_chars>
{{ elided_chars }} characters elided
</elided_chars>
<output_tail>
{{ output.output[-5000:] }}
</output_tail>
{%- endif -%}
"""

_EXTERNAL_STOP_STATUSES = {
    "timeexceeded",
    "limitsexceeded",
    "generationlengthexceeded",
    "contextbudgetexceeded",
}
_FORMAT_ERROR_ARGUMENTS_KEY = "_slime_original_tool_arguments"
_FORMAT_ERROR_KIND_KEY = "format_error_kind"
_FORMAT_ERROR_MESSAGE_NAME = "slime_format_error_recovery"
_NO_TOOL_FEEDBACK_MARKER = "No tool calls found in the response"
_NON_TARGET_TOKEN_CACHE: dict[int, bool] = {}


class _GenerationLengthStop(Exception):
    """Internal signal raised before Mini-SWE can label truncation as no-tool."""

    def __init__(self, response: Any | None = None) -> None:
        super().__init__("generation length exceeded")
        self.prompt_tokens = _response_prompt_tokens(response)


@dataclass(frozen=True)
class R2EConfig:
    adapter_bind_host: str
    adapter_port: int
    case_concurrency: int
    container_start_concurrency: int
    container_limits_enabled: bool
    container_memory_gib: int
    container_pids_limit: int
    max_context_tokens: int
    context_safety_margin: int
    min_generation_tokens: int
    max_iterations: int
    verified_max_iterations: int
    max_consecutive_format_errors: int
    step_timeout: int
    hard_step_timeout: int
    setup_timeout: int
    patch_timeout: int
    grade_timeout: int
    agent_wall_time: int
    case_timeout: int
    verified_grade_timeout: int
    verified_agent_wall_time: int
    verified_case_timeout: int
    model_request_timeout: int
    drop_session_timeout: int
    require_submission: bool
    add_turn_reminder: bool
    grade_unsubmitted_patches: bool
    save_trajectory_records: bool
    trajectory_dir: str

    def __post_init__(self) -> None:
        if self.case_concurrency < 1:
            raise ValueError("R2E_CASE_CONCURRENCY must be positive")
        if not 1 <= self.container_start_concurrency <= self.case_concurrency:
            raise ValueError(
                "R2E_CONTAINER_START_CONCURRENCY must be between 1 and "
                f"R2E_CASE_CONCURRENCY={self.case_concurrency}, got "
                f"{self.container_start_concurrency}"
            )
        if self.container_memory_gib < 1:
            raise ValueError("R2E_CONTAINER_MEMORY_GIB must be positive")
        if self.container_pids_limit < 1:
            raise ValueError("R2E_CONTAINER_PIDS_LIMIT must be positive")

    @property
    def container_memory_bytes(self) -> int:
        return self.container_memory_gib * 1024**3

    @classmethod
    def from_env(cls) -> "R2EConfig":
        def flag(name: str, default: bool) -> bool:
            raw = os.environ.get(name)
            if raw is None:
                return default
            return raw.strip().lower() not in {"0", "false", "no", "off"}

        return cls(
            adapter_bind_host=os.environ.get("R2E_ADAPTER_BIND_HOST", "127.0.0.1"),
            adapter_port=int(os.environ.get("R2E_ADAPTER_PORT", "0")),
            case_concurrency=int(os.environ.get("R2E_CASE_CONCURRENCY", "184")),
            container_start_concurrency=int(
                os.environ.get("R2E_CONTAINER_START_CONCURRENCY", "12")
            ),
            # Default off at the Python layer so a Ray worker restarted inside
            # a job launched before this feature existed cannot silently change
            # that in-progress run. New launches explicitly enable it in train.sh.
            container_limits_enabled=flag("R2E_CONTAINER_LIMITS_ENABLED", False),
            container_memory_gib=int(
                os.environ.get("R2E_CONTAINER_MEMORY_GIB", "8")
            ),
            container_pids_limit=int(
                os.environ.get("R2E_CONTAINER_PIDS_LIMIT", "512")
            ),
            max_context_tokens=int(os.environ.get("R2E_MAX_CONTEXT_TOKENS", "65536")),
            context_safety_margin=int(
                os.environ.get("R2E_CONTEXT_SAFETY_MARGIN", "128")
            ),
            min_generation_tokens=int(
                os.environ.get("R2E_MIN_GENERATION_TOKENS", "256")
            ),
            max_iterations=int(os.environ.get("R2E_MAX_ITERATIONS", "60")),
            verified_max_iterations=int(
                os.environ.get("R2E_VERIFIED_MAX_ITERATIONS", "150")
            ),
            max_consecutive_format_errors=int(
                os.environ.get("R2E_MAX_CONSECUTIVE_FORMAT_ERRORS", "3")
            ),
            step_timeout=int(os.environ.get("R2E_STEP_TIMEOUT", "45")),
            hard_step_timeout=int(os.environ.get("R2E_HARD_STEP_TIMEOUT", "60")),
            setup_timeout=int(os.environ.get("R2E_SETUP_TIMEOUT", "120")),
            patch_timeout=int(os.environ.get("R2E_PATCH_TIMEOUT", "90")),
            grade_timeout=int(os.environ.get("R2E_GRADE_TIMEOUT", "90")),
            agent_wall_time=int(os.environ.get("R2E_AGENT_WALL_TIME", "1400")),
            case_timeout=int(os.environ.get("R2E_CASE_TIMEOUT", "2420")),
            verified_grade_timeout=int(
                os.environ.get("R2E_VERIFIED_GRADE_TIMEOUT", "600")
            ),
            verified_agent_wall_time=int(
                os.environ.get("R2E_VERIFIED_AGENT_WALL_TIME", "1200")
            ),
            verified_case_timeout=int(
                os.environ.get("R2E_VERIFIED_CASE_TIMEOUT", "2220")
            ),
            model_request_timeout=int(
                os.environ.get("R2E_MODEL_REQUEST_TIMEOUT", "180")
            ),
            drop_session_timeout=int(os.environ.get("R2E_DROP_SESSION_TIMEOUT", "30")),
            require_submission=flag("R2E_REQUIRE_SUBMISSION", True),
            add_turn_reminder=flag("R2E_ADD_TURN_REMINDER", True),
            grade_unsubmitted_patches=flag(
                "R2E_GRADE_UNSUBMITTED_PATCHES", False
            ),
            save_trajectory_records=flag("R2E_SAVE_TRAJECTORIES", False),
            trajectory_dir=os.environ.get("R2E_TRAJECTORY_DIR", ""),
        )


CONFIG = R2EConfig.from_env()
_CONTAINER_START_SEMAPHORE = threading.BoundedSemaphore(
    CONFIG.container_start_concurrency
)


@dataclass
class CaseResult:
    reward: float = 0.0
    reward_known: bool = False
    submitted: bool = False
    patch_available: bool = False
    external_stop: bool = False
    timed_out: bool = False
    exit_status: str = ""
    error: str = ""
    n_calls: int = 0
    elapsed_seconds: float = 0.0
    container_start_wait_seconds: float = 0.0
    context_prompt_tokens: int = 0
    context_remaining_tokens: int = 0
    context_budget_exit: bool = False
    generation_length_exit: bool = False
    model_request_timeout: bool = False
    agent_time_exceeded: bool = False
    limits_exceeded: bool = False
    repeated_format_error: bool = False
    format_error_recovered_turns: int = 0
    bad_json_turns: int = 0
    no_tool_turns: int = 0
    bad_json_tokens: int = 0
    no_tool_tokens: int = 0
    format_error_arguments_normalized_turns: int = 0
    assistant_turns: int = 0
    history_assistant_turns: int = 0
    history_reasoning_turns: int = 0
    format_transition_pairs: int = 0
    previous_error_turns: int = 0
    error_after_error_turns: int = 0
    previous_ok_turns: int = 0
    error_after_ok_turns: int = 0
    format_error_recovery_turns: int = 0
    format_error_streak_count: int = 0
    format_error_max_streak: int = 0
    format_error_terminal_streak: int = 0
    sampled_without_eos_turns: int = 0
    verified_fresh_container: bool = False
    training_fresh_container: bool = False
    training_patch_apply_failed: bool = False
    verified_patch_apply_failed: bool = False
    verified_resolution_status: str = ""
    verified_f2p_failure_count: int = 0
    verified_p2p_failure_count: int = 0
    verified_eval_returncode: int = 0
    unsubmitted_worktree_patch_available: bool = False
    unsubmitted_worktree_patch_bytes: int = 0
    unsubmitted_worktree_patch_lines: int = 0
    unsubmitted_patch_grade_attempted: bool = False
    unsubmitted_patch_reward: float = 0.0
    unsubmitted_patch_reward_known: bool = False
    unsubmitted_patch_apply_failed: bool = False
    unsubmitted_patch_grade_timed_out: bool = False
    unsubmitted_patch_grade_elapsed_seconds: float = 0.0
    unsubmitted_patch_resolution_status: str = ""
    unsubmitted_patch_f2p_failure_count: int = 0
    unsubmitted_patch_p2p_failure_count: int = 0
    unsubmitted_patch_eval_returncode: int = 0
    unsubmitted_patch_error: str = ""
    trajectory_record_path: str = ""


def _grade_unsubmitted_r2e_worktree(
    outcome: CaseResult,
    runtime: Any,
    *,
    case_deadline: float,
) -> None:
    """Record a diagnostic grade for an R2E worktree that was not submitted.

    This result is deliberately counterfactual: it never changes the strict
    reward, submission flag, or trainability of the rollout.  It only supplies
    the relaxed score requested for monitoring and later trajectory analysis.
    """

    if (
        not CONFIG.grade_unsubmitted_patches
        or outcome.submitted
        or not outcome.unsubmitted_worktree_patch_available
        or runtime is None
    ):
        return

    started = time.monotonic()
    outcome.unsubmitted_patch_grade_attempted = True
    try:
        raw_reward = _hard_call(
            lambda: runtime._calculate_reward(timeout=CONFIG.grade_timeout),
            _remaining(case_deadline, CONFIG.grade_timeout),
            label="unsubmitted worktree calculate_reward",
            runtime=runtime,
        )
        outcome.unsubmitted_patch_reward = (
            1.0 if float(raw_reward) >= 1.0 else 0.0
        )
        outcome.unsubmitted_patch_reward_known = True
    except TimeoutError as exc:
        outcome.unsubmitted_patch_grade_timed_out = True
        outcome.unsubmitted_patch_error = str(exc)
    except Exception as exc:
        outcome.unsubmitted_patch_error = (
            f"{type(exc).__name__}: {exc}"
        )
    finally:
        outcome.unsubmitted_patch_grade_elapsed_seconds = (
            time.monotonic() - started
        )


def _configure_litellm_logging() -> None:
    """Keep high-volume per-turn LiteLLM INFO logs out of Ray's log relay."""

    for logger_name in (
        "LiteLLM",
        "litellm",
        "litellm.utils",
        "litellm.router",
        "litellm.proxy",
    ):
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    # Keep our low-frequency aggregate latency/lag record visible even when the
    # process-wide default logging level is WARNING.
    logging.getLogger("slime.agent.adapters.openai").setLevel(logging.INFO)
    try:
        import litellm

        litellm.suppress_debug_info = True
        litellm.set_verbose = False
    except (ImportError, AttributeError):
        pass


def _adapter_process_enabled() -> bool:
    """Host the adapter in its own process (see slime.agent.adapter_process)."""
    return os.environ.get("R2E_ADAPTER_PROCESS", "0").strip().lower() in {
        "1",
        "true",
        "yes",
    }


class _AdapterService(metaclass=SingletonMeta):
    def __init__(self, args) -> None:
        _configure_litellm_logging()
        self.tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        max_context = int(getattr(args, "rollout_max_context_len", 0) or 0)
        self.max_context_tokens = max_context or CONFIG.max_context_tokens
        sglang_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
        max_turns_per_sid = max(
            CONFIG.max_iterations, CONFIG.verified_max_iterations
        )
        if _adapter_process_enabled():
            # The adapter's event loop competes with ~184 agent threads for one
            # GIL in this process; hosting it separately is what removes the
            # per-turn resume stalls. Control plane is proxied over HTTP.
            from slime.agent.adapter_process import AdapterProcess
            from .recovery_segments import enabled
            if enabled():
                from .recovery_adapter_process import AdapterProcess

            self.adapter_process = AdapterProcess(
                tokenizer_path=args.hf_checkpoint,
                sglang_url=sglang_url,
                protocol="openai",
                tool_parser=getattr(args, "sglang_tool_call_parser", None) or None,
                reasoning_parser=getattr(args, "sglang_reasoning_parser", None) or None,
                max_turns_per_sid=max_turns_per_sid,
                fork_threshold_tokens=self.max_context_tokens,
                host=CONFIG.adapter_bind_host,
                port=CONFIG.adapter_port,
            )
            self.adapter = self.adapter_process
            self.app_handle = None
            self.adapter_url = self.adapter_process.url
            logger.warning("[r2e_adapter] running in a dedicated process")
        else:
            self.adapter_process = None
            self._start_in_process_adapter(
                sglang_url, args, max_turns_per_sid=max_turns_per_sid
            )
        self.executor = ThreadPoolExecutor(
            max_workers=CONFIG.case_concurrency,
            thread_name_prefix="r2e-case",
        )
        logger.info(
            "[slime-r2e] adapter=%s sglang=%s context=%d "
            "case_concurrency=%d container_start_concurrency=%d",
            self.adapter_url,
            sglang_url,
            self.max_context_tokens,
            CONFIG.case_concurrency,
            CONFIG.container_start_concurrency,
        )

    def _start_in_process_adapter(self, sglang_url, args, *, max_turns_per_sid) -> None:
        self.adapter = OpenAIAdapter(
            tokenizer=self.tokenizer,
            sglang_url=sglang_url,
            tool_parser=getattr(args, "sglang_tool_call_parser", None) or None,
            reasoning_parser=getattr(args, "sglang_reasoning_parser", None) or None,
            # The adapter is shared by training and evaluation sessions. Keep
            # its safety ceiling high enough for the Verified-only turn budget;
            # each Mini-SWE agent still enforces its own scenario-specific
            # step_limit below.
            max_turns_per_sid=max_turns_per_sid,
            # Legacy mode rejects residual multi-sample results. Opt-in recovery
            # mode retains exact segments and centers rewards per rollout id.
            fork_threshold_tokens=self.max_context_tokens,
        )
        from .recovery_segments import enabled
        if enabled():
            from .recovery_trajectory import RecoveryTrajectoryManager
            self.adapter.manager = RecoveryTrajectoryManager(fork_threshold_tokens=self.max_context_tokens)
        self.app_handle = run_app_in_thread(
            self.adapter.app,
            host=CONFIG.adapter_bind_host,
            port=CONFIG.adapter_port,
            thread_name="r2e-openai-adapter",
            runner_kwargs={
                "handler_cancellation": True,
                "access_log_class": FilteredAccessLogger,
            },
        )
        self.adapter_url = (
            f"http://{CONFIG.adapter_bind_host}:{self.app_handle.port}"
        )


def _force_kill_container(runtime: Any) -> None:
    """Best-effort removal used to unblock a wedged Docker exec."""
    if runtime is None:
        return
    try:
        container = getattr(runtime, "container", None)
        if container is not None:
            container.remove(force=True)
            return
    except Exception:
        pass

    # Mini-SWE's DockerEnvironment exposes only a raw container id and starts
    # cleanup asynchronously. Kill it synchronously on hard deadline so a
    # timed-out validation command cannot survive into weight synchronization.
    container_id = getattr(runtime, "container_id", None)
    executable = str(
        getattr(getattr(runtime, "config", None), "executable", "docker")
    )
    if container_id:
        try:
            subprocess.run(
                [executable, "rm", "-f", str(container_id)],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=15,
            )
            runtime.container_id = None
            return
        except Exception:
            pass

    # R2E normally exposes a docker-py object.  Keeping cleanup inside that API
    # avoids requiring a docker CLI in the derived Slime image.
    try:
        import docker

        name = getattr(runtime, "container_name", None)
        if name:
            docker.from_env().containers.get(name).remove(force=True)
    except Exception:
        pass


def _hard_call(
    fn: Callable[[], Any],
    timeout: float,
    *,
    label: str,
    runtime: Any = None,
    cleanup_late_result: Callable[[Any], None] | None = None,
    before_timeout_cleanup: Callable[[], None] | None = None,
) -> Any:
    """Run a blocking Docker/agent call under a true host-side deadline."""
    result: dict[str, Any] = {}
    expired = threading.Event()

    def target() -> None:
        try:
            value = fn()
            if expired.is_set() and cleanup_late_result is not None:
                try:
                    cleanup_late_result(value)
                except Exception:
                    pass
            else:
                result["value"] = value
        except BaseException as exc:
            result["error"] = exc

    thread = threading.Thread(target=target, name=f"deadline-{label}", daemon=True)
    thread.start()
    thread.join(max(float(timeout), 0.001))
    if thread.is_alive():
        expired.set()
        if before_timeout_cleanup is not None:
            try:
                before_timeout_cleanup()
            except Exception:
                logger.debug(
                    "%s pre-cleanup callback failed", label, exc_info=True
                )
        _force_kill_container(runtime)
        raise TimeoutError(f"{label} exceeded hard deadline of {timeout:.1f}s")
    if "error" in result:
        raise result["error"]
    return result.get("value")


def _remaining(deadline: float, requested: int) -> float:
    left = deadline - time.monotonic()
    if left <= 0:
        raise TimeoutError("case exceeded total wall-clock deadline")
    return max(0.001, min(float(requested), left))


@contextmanager
def _container_start_slot() -> Iterator[float]:
    """Limit Docker create/start pressure without limiting active agents.

    Waiting for admission is orchestration queue time, not part of the case's
    execution budget. Callers add the yielded wait duration to their deadline.
    The permit is released as soon as the environment constructor returns.
    """

    wait_started = time.monotonic()
    _CONTAINER_START_SEMAPHORE.acquire()
    waited = time.monotonic() - wait_started
    try:
        yield waited
    finally:
        _CONTAINER_START_SEMAPHORE.release()


def _agent_class(*, add_turn_reminder: bool | None = None) -> type:
    from harnessadapt.openhands.bridge import OfficialAgent, enabled

    if enabled():
        return OfficialAgent
    from minisweagent.agents.default import DefaultAgent

    if not (CONFIG.add_turn_reminder if add_turn_reminder is None else add_turn_reminder):
        return DefaultAgent

    class DefaultAgentWithReminder(DefaultAgent):
        def execute_actions(self, message: dict) -> list[dict]:
            added = super().execute_actions(message)
            step_limit = int(getattr(self.config, "step_limit", 0) or 0)
            if step_limit <= 0 or not added:
                return added
            remaining = max(0, step_limit - int(self.n_calls))
            last = self.messages[-1]
            content = last.get("content")
            if isinstance(content, str):
                replacement = dict(last)
                replacement["content"] = (
                    content + f"\nREMINDER: You have {remaining} turns left"
                )
                self.messages[-1] = replacement
            return added

    return DefaultAgentWithReminder


def _bash_tool_schema() -> dict[str, Any]:
    """Return the exact tool schema Mini-SWE sends to the adapter."""
    from minisweagent.models.utils.actions_toolcall import BASH_TOOL

    return BASH_TOOL


def _normalize_tool_arguments_for_template(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Decode OpenAI JSON-string tool arguments for Qwen's chat template."""
    normalized: list[dict[str, Any]] = []
    for message in messages:
        normalized_message = dict(message)
        tool_calls = message.get("tool_calls")
        if tool_calls:
            normalized_calls = []
            for tool_call in tool_calls:
                normalized_call = dict(tool_call)
                function = dict(tool_call.get("function") or {})
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    try:
                        decoded = json.loads(arguments)
                    except json.JSONDecodeError:
                        decoded = {"arguments": arguments}
                    if not isinstance(decoded, dict):
                        decoded = {"value": decoded}
                    function["arguments"] = decoded
                normalized_call["function"] = function
                normalized_calls.append(normalized_call)
            normalized_message["tool_calls"] = normalized_calls
        normalized.append(normalized_message)
    return normalized


def _classify_format_error(messages: tuple[Any, ...]) -> str:
    feedback = "\n".join(
        str(message.get("content") or "")
        for message in messages
        if isinstance(message, dict) and message.get("role") != "assistant"
    )
    return "no_tool" if _NO_TOOL_FEEDBACK_MARKER in feedback else "bad_json"


def _json_compatible_response(response: Any) -> Any:
    model_dump = getattr(response, "model_dump", None)
    if model_dump is not None:
        try:
            return model_dump(mode="json")
        except TypeError:
            return model_dump()
        except Exception:
            return repr(response)
    if isinstance(response, dict):
        return response
    return repr(response)


def _context_safe_format_error_assistant(response: Any) -> dict[str, Any]:
    """Retain a malformed sampled turn without poisoning the next template."""
    response_dict = _json_compatible_response(response)
    choice: dict[str, Any] = {}
    if isinstance(response_dict, dict):
        choices = response_dict.get("choices") or []
        if choices and isinstance(choices[0], dict):
            choice = choices[0]
    message_data = choice.get("message") if isinstance(choice, dict) else None
    if not isinstance(message_data, dict):
        response_choices = getattr(response, "choices", None) or []
        response_message = (
            getattr(response_choices[0], "message", None)
            if response_choices
            else None
        )
        model_dump = getattr(response_message, "model_dump", None)
        if model_dump is not None:
            try:
                message_data = model_dump(mode="json")
            except TypeError:
                message_data = model_dump()
        elif isinstance(response_message, dict):
            message_data = response_message
        else:
            message_data = {}

    assistant: dict[str, Any] = dict(message_data)
    assistant["role"] = "assistant"
    normalized_calls = []
    arguments_normalized = False
    for tool_call in assistant.get("tool_calls") or []:
        if not isinstance(tool_call, dict):
            continue
        normalized_call = dict(tool_call)
        function = dict(normalized_call.get("function") or {})
        arguments = function.get("arguments")
        requires_wrapper = False
        if isinstance(arguments, str):
            try:
                decoded = json.loads(arguments)
                requires_wrapper = not isinstance(decoded, dict)
            except (json.JSONDecodeError, TypeError):
                requires_wrapper = True
        elif isinstance(arguments, dict):
            function["arguments"] = json.dumps(arguments, ensure_ascii=False)
        else:
            requires_wrapper = True
        if requires_wrapper:
            function["arguments"] = json.dumps(
                {_FORMAT_ERROR_ARGUMENTS_KEY: arguments},
                ensure_ascii=False,
                default=str,
            )
            arguments_normalized = True
        normalized_call["function"] = function
        normalized_calls.append(normalized_call)
    if assistant.get("tool_calls") is not None:
        assistant["tool_calls"] = normalized_calls
    assistant["extra"] = {
        "response": response_dict,
        "format_error_recovered": True,
        "format_error_arguments_normalized": arguments_normalized,
    }
    return assistant


def _response_completion_tokens(response: Any) -> int:
    response_dict = _json_compatible_response(response)
    if not isinstance(response_dict, dict):
        return 0
    usage = response_dict.get("usage") or {}
    try:
        return max(0, int(usage.get("completion_tokens") or 0))
    except (TypeError, ValueError):
        return 0


def _response_prompt_tokens(response: Any) -> int:
    """Read exact adapter prompt usage from a response or echoed message."""

    response_dict = _json_compatible_response(response)
    if not isinstance(response_dict, dict):
        return 0
    usage = response_dict.get("usage") or {}
    if not usage:
        usage = (
            ((response_dict.get("extra") or {}).get("response") or {}).get(
                "usage"
            )
            or {}
        )
    try:
        return max(0, int(usage.get("prompt_tokens") or 0))
    except (TypeError, ValueError):
        return 0


def _response_finish_reason(response: Any) -> str:
    response_dict = _json_compatible_response(response)
    if not isinstance(response_dict, dict):
        return ""
    choices = response_dict.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return ""
    return str(choices[0].get("finish_reason") or "")


def _install_format_error_history_recovery(model: Any) -> Any:
    """Preserve malformed assistant turns and collect disjoint error metrics."""
    if getattr(model, "_slime_format_error_history_recovery", False):
        return model
    parse_actions = getattr(model, "_parse_actions", None)
    if parse_actions is None:
        raise RuntimeError("Mini-SWE model has no _parse_actions hook")

    from minisweagent.exceptions import FormatError

    stats = {
        "format_error_recovered_turns": 0,
        "bad_json_turns": 0,
        "no_tool_turns": 0,
        "bad_json_tokens": 0,
        "no_tool_tokens": 0,
        "format_error_arguments_normalized_turns": 0,
        "assistant_turns": 0,
        "format_transition_pairs": 0,
        "previous_error_turns": 0,
        "error_after_error_turns": 0,
        "previous_ok_turns": 0,
        "error_after_ok_turns": 0,
        "format_error_recovery_turns": 0,
        "format_error_streak_count": 0,
        "format_error_max_streak": 0,
        "sampled_without_eos_turns": 0,
        "_current_error_streak": 0,
        "_previous_turn_was_error": None,
    }

    def record_turn(*, format_error: bool) -> None:
        previous = stats["_previous_turn_was_error"]
        stats["assistant_turns"] += 1
        if format_error:
            stats["_current_error_streak"] += 1
            stats["format_error_max_streak"] = max(
                stats["format_error_max_streak"],
                stats["_current_error_streak"],
            )
            if previous is not True:
                stats["format_error_streak_count"] += 1
        else:
            if previous is True:
                stats["format_error_recovery_turns"] += 1
            stats["_current_error_streak"] = 0
        if previous is not None:
            stats["format_transition_pairs"] += 1
            if previous:
                stats["previous_error_turns"] += 1
                stats["error_after_error_turns"] += int(format_error)
            else:
                stats["previous_ok_turns"] += 1
                stats["error_after_ok_turns"] += int(format_error)
        stats["_previous_turn_was_error"] = format_error

    def parse_actions_with_history(response: Any) -> list[dict[str, Any]]:
        if _response_finish_reason(response).lower() == "length":
            # LitellmModel normally parses tool calls before returning its
            # response. Without intercepting here, a context-length truncation
            # with no complete tool call is raised as FormatError/no-tool and
            # receives misleading corrective feedback.
            stats["assistant_turns"] += 1
            stats["sampled_without_eos_turns"] += 1
            stats["_previous_turn_was_error"] = None
            raise _GenerationLengthStop(response)
        try:
            actions = parse_actions(response)
        except FormatError as exc:
            record_turn(format_error=True)
            messages = list(getattr(exc, "messages", ()) or ())
            error_kind = _classify_format_error(tuple(messages))
            completion_tokens = _response_completion_tokens(response)
            stats["format_error_recovered_turns"] += 1
            stats[f"{error_kind}_turns"] += 1
            stats[f"{error_kind}_tokens"] += completion_tokens

            assistant_index = next(
                (
                    index
                    for index, message in enumerate(messages)
                    if isinstance(message, dict)
                    and message.get("role") == "assistant"
                ),
                None,
            )
            if assistant_index is None:
                messages.insert(0, _context_safe_format_error_assistant(response))
                assistant_index = 0
            assistant = dict(messages[assistant_index])
            extra = dict(assistant.get("extra") or {})
            stats["format_error_arguments_normalized_turns"] += int(
                bool(extra.get("format_error_arguments_normalized"))
            )
            extra["format_error_recovered"] = True
            extra[_FORMAT_ERROR_KIND_KEY] = error_kind
            assistant["extra"] = extra
            messages[assistant_index] = assistant
            # ``extra`` is deliberately stripped by Mini-SWE before an OpenAI
            # request.  Carry one standards-compliant message ``name`` through
            # the wire instead, so the adapter can distinguish this deliberate
            # history rewrite from a genuine branch.  The adapter removes the
            # name before chat-template rendering, so model-visible context is
            # unchanged.
            for index, message in enumerate(messages):
                if not isinstance(message, dict) or message.get("role") != "user":
                    continue
                tagged = dict(message)
                tagged["name"] = _FORMAT_ERROR_MESSAGE_NAME
                messages[index] = tagged
            exc.messages = tuple(messages)
            raise
        record_turn(format_error=False)
        return actions

    model._parse_actions = parse_actions_with_history
    model._slime_format_error_stats = stats
    model._slime_format_error_history_recovery = True
    return model


def _finish_reason(message: dict[str, Any]) -> str:
    """Read LiteLLM's finish reason from a Mini-SWE assistant message."""
    if not isinstance(message, dict):
        return ""
    response = (message.get("extra") or {}).get("response") or {}
    choices = response.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return ""
    return str(choices[0].get("finish_reason") or "")


_MODEL_PERF_LOCK = threading.Lock()
_MODEL_PERF_EVERY = max(
    1,
    int(os.environ.get("SLIME_ADAPTER_METRICS_EVERY", "100")),
)
_MODEL_PERF_TOTALS = {
    "turns": 0.0,
    "errors": 0.0,
    "prepare_ms": 0.0,
    "query_ms": 0.0,
    "postprocess_ms": 0.0,
    "total_ms": 0.0,
    "messages": 0.0,
    "history_assistant": 0.0,
    "prompt_tokens": 0.0,
}
_MODEL_PERF_WINDOW = dict(_MODEL_PERF_TOTALS)
_MODEL_PERF_WINDOW_STARTED_AT: float | None = None


def _record_model_query_performance(
    *,
    prepare_ms: float,
    query_ms: float,
    postprocess_ms: float,
    total_ms: float,
    messages: int,
    history_assistant: int,
    prompt_tokens: int,
    error: bool,
) -> None:
    """Aggregate the synchronous Mini-SWE/LiteLLM side of each model call."""

    global _MODEL_PERF_WINDOW, _MODEL_PERF_WINDOW_STARTED_AT
    now = time.monotonic()
    values = {
        "turns": 1.0,
        "errors": float(error),
        "prepare_ms": float(prepare_ms),
        "query_ms": float(query_ms),
        "postprocess_ms": float(postprocess_ms),
        "total_ms": float(total_ms),
        "messages": float(messages),
        "history_assistant": float(history_assistant),
        "prompt_tokens": float(prompt_tokens),
    }
    with _MODEL_PERF_LOCK:
        if _MODEL_PERF_WINDOW_STARTED_AT is None:
            _MODEL_PERF_WINDOW_STARTED_AT = now
        for counters in (_MODEL_PERF_TOTALS, _MODEL_PERF_WINDOW):
            for key, value in values.items():
                counters[key] += value
        total_turns = int(_MODEL_PERF_TOTALS["turns"])
        if total_turns % _MODEL_PERF_EVERY != 0:
            return

        window = _MODEL_PERF_WINDOW
        window_turns = max(1, int(window["turns"]))
        elapsed = max(
            1e-9,
            now - (_MODEL_PERF_WINDOW_STARTED_AT or now),
        )

        def mean(key: str) -> float:
            return window[key] / window_turns

        known_ms = mean("prepare_ms") + mean("query_ms") + mean(
            "postprocess_ms"
        )
        logger.warning(
            "[r2e_client_perf] turns=%d window=%d rate=%.2f "
            "prepare_ms=%.2f model_query_ms=%.2f postprocess_ms=%.2f "
            "total_ms=%.2f residual_ms=%.2f errors=%d messages=%.1f "
            "history_assistant=%.1f prompt_tokens=%.1f",
            total_turns,
            window_turns,
            window_turns / elapsed,
            mean("prepare_ms"),
            mean("query_ms"),
            mean("postprocess_ms"),
            mean("total_ms"),
            max(0.0, mean("total_ms") - known_ms),
            int(window["errors"]),
            mean("messages"),
            mean("history_assistant"),
            mean("prompt_tokens"),
        )
        _MODEL_PERF_WINDOW = {
            key: 0.0 for key in _MODEL_PERF_TOTALS
        }
        _MODEL_PERF_WINDOW_STARTED_AT = now


class _RemainingContextModel:
    """Delegate exact context capping to the token-capturing adapter.

    The adapter already renders the authoritative prompt IDs and caps each
    request against the session context budget.  Re-tokenizing the same full
    history here doubled CPU work and, under high concurrency, distorted the
    wall-clock policy budget.  Exact prompt usage is copied back from the
    adapter response for diagnostics.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        model_context_length: int,
        safety_margin: int,
        min_generation_tokens: int,
        request_timeout: int,
        case_deadline: float,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self.model_context_length = int(model_context_length)
        self.safety_margin = int(safety_margin)
        self.min_generation_tokens = int(min_generation_tokens)
        self.request_timeout = int(request_timeout)
        self.case_deadline = float(case_deadline)
        self.last_prompt_tokens = 0
        self.last_remaining_tokens = 0
        self.last_history_assistant_turns = 0
        self.last_history_reasoning_turns = 0
        self.context_budget_exit = False
        self.generation_length_exit = False
        self.model_request_timeout = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)

    def _update_prompt_usage(self, response: Any) -> None:
        prompt_tokens = _response_prompt_tokens(response)
        if prompt_tokens <= 0:
            return
        self.last_prompt_tokens = prompt_tokens
        self.last_remaining_tokens = (
            self.model_context_length
            - self.last_prompt_tokens
            - self.safety_margin
        )

    def query(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        from minisweagent.exceptions import LimitsExceeded

        total_started = time.monotonic()
        started = time.monotonic()
        prepare = getattr(self._model, "_prepare_messages_for_api", None)
        prepared = prepare(messages) if prepare is not None else messages
        prepared = _normalize_tool_arguments_for_template(prepared)
        history_assistant = [
            message
            for message in prepared
            if isinstance(message, dict) and message.get("role") == "assistant"
        ]
        self.last_history_assistant_turns = len(history_assistant)
        self.last_history_reasoning_turns = sum(
            isinstance(message.get("reasoning_content"), str)
            and bool(message["reasoning_content"].strip())
            for message in history_assistant
        )
        prepare_ms = (time.monotonic() - started) * 1000.0
        # The adapter replaces this generous client request with the exact
        # remaining context computed from its authoritative prompt token IDs.
        kwargs["max_tokens"] = self.model_context_length
        kwargs["timeout"] = max(
            1,
            min(
                self.request_timeout,
                int(
                    math.ceil(
                        _remaining(self.case_deadline, self.request_timeout)
                    )
                ),
            ),
        )
        started = time.monotonic()
        try:
            result = self._model.query(messages, **kwargs)
        except _GenerationLengthStop as exc:
            query_ms = (time.monotonic() - started) * 1000.0
            if exc.prompt_tokens > 0:
                self.last_prompt_tokens = exc.prompt_tokens
                self.last_remaining_tokens = (
                    self.model_context_length
                    - self.last_prompt_tokens
                    - self.safety_margin
                )
            self.context_budget_exit = True
            self.generation_length_exit = True
            _record_model_query_performance(
                prepare_ms=prepare_ms,
                query_ms=query_ms,
                postprocess_ms=0.0,
                total_ms=(time.monotonic() - total_started) * 1000.0,
                messages=len(prepared),
                history_assistant=len(history_assistant),
                prompt_tokens=self.last_prompt_tokens,
                error=True,
            )
            raise LimitsExceeded(
                {
                    "role": "exit",
                    "content": "GenerationLengthExceeded",
                    "extra": {
                        "exit_status": "GenerationLengthExceeded",
                        "submission": "",
                        "prompt_tokens": self.last_prompt_tokens,
                        "remaining_tokens": self.last_remaining_tokens,
                    },
                }
            ) from None
        except Exception as exc:
            query_ms = (time.monotonic() - started) * 1000.0
            error_text = f"{type(exc).__name__}: {exc}".lower()
            if "timeout" in error_text or "timed out" in error_text:
                self.model_request_timeout = True
            _record_model_query_performance(
                prepare_ms=prepare_ms,
                query_ms=query_ms,
                postprocess_ms=0.0,
                total_ms=(time.monotonic() - total_started) * 1000.0,
                messages=len(prepared),
                history_assistant=len(history_assistant),
                prompt_tokens=0,
                error=True,
            )
            raise
        query_ms = (time.monotonic() - started) * 1000.0

        started = time.monotonic()
        self._update_prompt_usage(result)
        finish_reason = _finish_reason(result).lower()
        postprocess_ms = (time.monotonic() - started) * 1000.0
        _record_model_query_performance(
            prepare_ms=prepare_ms,
            query_ms=query_ms,
            postprocess_ms=postprocess_ms,
            total_ms=(time.monotonic() - total_started) * 1000.0,
            messages=len(prepared),
            history_assistant=len(history_assistant),
            prompt_tokens=self.last_prompt_tokens,
            error=False,
        )
        if finish_reason == "length":
            # max_tokens already equals the context remainder. Retrying another
            # turn cannot succeed, so end the trajectory and let the caller mask
            # this non-submission instead of training a truncated response.
            self.context_budget_exit = True
            self.generation_length_exit = True
            raise LimitsExceeded(
                {
                    "role": "exit",
                    "content": "GenerationLengthExceeded",
                    "extra": {
                        "exit_status": "GenerationLengthExceeded",
                        "submission": "",
                        "prompt_tokens": self.last_prompt_tokens,
                        "remaining_tokens": self.last_remaining_tokens,
                    },
                }
            )
        return result


def _case_model(
    adapter_url: str,
    session_id: str,
    sampling_params: dict[str, Any],
    tokenizer: Any,
    *,
    model_context_length: int,
    case_deadline: float,
):
    from harnessadapt.openhands.bridge import ModelConnection, enabled

    if enabled():
        return ModelConnection(
            base_url=adapter_url.rstrip("/") + "/v1", api_key=session_id,
            temperature=float(sampling_params.get("temperature", 1.0)),
            top_p=float(sampling_params.get("top_p", 1.0)),
            context_tokens=model_context_length,
            max_output_tokens=int(sampling_params.get("max_new_tokens") or model_context_length),
            request_timeout=CONFIG.model_request_timeout, deadline=case_deadline,
        )
    from harnessadapt.training.runtime.fast_model import get_actor_model

    model = get_actor_model(
        "openai/slime-actor",
        {
            "model_kwargs": {
                "api_base": adapter_url.rstrip("/") + "/v1",
                "api_key": session_id,
                "temperature": float(sampling_params.get("temperature", 1.0)),
                "top_p": float(sampling_params.get("top_p", 1.0)),
                "timeout": CONFIG.model_request_timeout,
                "num_retries": 0,
                "drop_params": True,
            },
            "cost_tracking": "ignore_errors",
            "observation_template": HEAD_TAIL_OBSERVATION_TEMPLATE,
        },
    )
    return _RemainingContextModel(
        _install_format_error_history_recovery(model),
        tokenizer,
        model_context_length=model_context_length,
        safety_margin=CONFIG.context_safety_margin,
        min_generation_tokens=CONFIG.min_generation_tokens,
        request_timeout=CONFIG.model_request_timeout,
        case_deadline=case_deadline,
    )


def _close_repo_env(repo_env: Any) -> None:
    if repo_env is None:
        return
    runtime = getattr(repo_env, "runtime", None)
    try:
        _hard_call(
            repo_env.close,
            15,
            label="close",
            runtime=runtime,
        )
    except Exception:
        _force_kill_container(runtime)


def _install_r2e_import_compat() -> None:
    """Supply the one legacy HF Hub symbol imported by this R2E checkout.

    ``HfFolder`` was removed in huggingface_hub 1.x. R2E imports it in a broad
    utility module but the Docker runtime path never calls it. Downgrading
    huggingface_hub would disturb Slime/Transformers, so expose the equivalent
    token methods only for the duration of this process.
    """
    import huggingface_hub

    if hasattr(huggingface_hub, "HfFolder"):
        return

    class HfFolder:
        @staticmethod
        def get_token() -> str | None:
            return huggingface_hub.get_token()

        @staticmethod
        def save_token(token: str) -> None:
            from huggingface_hub.constants import HF_TOKEN_PATH

            os.makedirs(os.path.dirname(HF_TOKEN_PATH), exist_ok=True)
            with open(HF_TOKEN_PATH, "w", encoding="utf-8") as handle:
                handle.write(token)

        @staticmethod
        def delete_token() -> None:
            from huggingface_hub.constants import HF_TOKEN_PATH

            try:
                os.remove(HF_TOKEN_PATH)
            except FileNotFoundError:
                pass

    huggingface_hub.HfFolder = HfFolder


def _verified_image_name(instance: dict[str, Any]) -> str:
    image_name = str(instance.get("image_name") or "").strip()
    if image_name:
        return image_name
    instance_id = str(instance.get("instance_id") or "").strip()
    if not instance_id:
        raise ValueError("Verified instance has no instance_id")
    compatible = instance_id.replace("__", "_1776_")
    return (
        "docker.io/swebench/"
        f"sweb.eval.x86_64.{compatible}:latest"
    ).lower()


def _docker_resource_kwargs() -> dict[str, Any]:
    """Resource limits passed to R2E-Gym's Docker SDK container creation."""

    from harnessadapt.openhands.bridge import docker_kwargs, enabled

    backend_options = docker_kwargs() if enabled() else {}
    if not CONFIG.container_limits_enabled:
        return backend_options
    memory_bytes = CONFIG.container_memory_bytes
    return {
        **backend_options,
        "mem_limit": memory_bytes,
        # Docker interprets memory-swap as RAM + swap. Setting it equal to
        # memory disables per-container swap and keeps a runaway case inside
        # the same hard cgroup budget.
        "memswap_limit": memory_bytes,
        "pids_limit": CONFIG.container_pids_limit,
        "labels": {
            "self_evolve.r2e": "1",
            "self_evolve.resource_limits": "1",
            **({"self_evolve.coevolve_root": os.environ["R2E_COEVOLVE_ROOT"]}
               if os.environ.get("R2E_COEVOLVE_ROOT") else {}),
        },
    }


def _verified_docker_run_args() -> list[str]:
    """Equivalent limits for Mini-SWE's Docker CLI environment."""

    from harnessadapt.openhands.bridge import docker_run_args, enabled

    backend_options = docker_run_args() if enabled() else []
    if not CONFIG.container_limits_enabled:
        return ["--rm", *backend_options]
    memory_bytes = str(CONFIG.container_memory_bytes)
    return [
        "--rm",
        *backend_options,
        "--memory",
        memory_bytes,
        "--memory-swap",
        memory_bytes,
        "--pids-limit",
        str(CONFIG.container_pids_limit),
        "--label",
        "self_evolve.r2e=1",
        "--label",
        "self_evolve.resource_limits=1",
        *(["--label", "self_evolve.coevolve_root=" + os.environ["R2E_CONTAINER_OWNER_ROOT"]]
          if os.environ.get("R2E_CONTAINER_OWNER_ROOT") else []),
    ]


def _new_verified_environment(instance: dict[str, Any]) -> Any:
    from minisweagent.environments import get_environment

    return get_environment(
        {
            "environment_class": "docker",
            "image": _verified_image_name(instance),
            "cwd": "/testbed",
            "timeout": CONFIG.step_timeout,
            "pull_timeout": CONFIG.setup_timeout,
            "executable": "docker",
            "run_args": _verified_docker_run_args(),
            "env": {
                "PAGER": "cat",
                "MANPAGER": "cat",
                "LESS": "-R",
                "PIP_PROGRESS_BAR": "off",
                "TQDM_DISABLE": "1",
            },
        }
    )


def _close_verified_environment(environment: Any) -> None:
    if environment is None:
        return
    try:
        cleanup = getattr(environment, "cleanup", None)
        if cleanup is not None:
            cleanup()
    finally:
        # DockerEnvironment.cleanup() is deliberately asynchronous. A worker
        # may exit before its subprocess runs, so synchronously remove the
        # exact container as the final ownership handoff.
        _force_kill_container(environment)


def _score_swebench_verified_output(
    instance: dict[str, Any], test_output: str
) -> tuple[float, dict[str, Any]]:
    """Use SWE-bench's selected FAIL_TO_PASS/PASS_TO_PASS resolution rules."""

    from r2egym.agenthub.trajectory.swebench_utils import swebench_report
    from swebench.harness.constants import ResolvedStatus
    from swebench.harness.grading import get_resolution_status

    report = swebench_report(instance, test_output)
    resolution_status = get_resolution_status(report)
    summary: dict[str, Any] = {"resolution_status": resolution_status}
    for category in ("FAIL_TO_PASS", "PASS_TO_PASS"):
        category_report = dict(report.get(category) or {})
        success = list(category_report.get("success") or [])
        failure = list(category_report.get("failure") or [])
        summary[category] = {
            "success_count": len(success),
            "failure_count": len(failure),
            "failure": failure[:20],
            "failure_truncated": len(failure) > 20,
        }
    reward = 1.0 if resolution_status == ResolvedStatus.FULL.value else 0.0
    return reward, summary


def _raw_verified_environment(environment: Any) -> Any:
    """Unwrap the model-facing git policy for internal read-only diagnostics."""

    current = environment
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        wrapped = getattr(current, "_environment", None)
        if wrapped is None:
            break
        current = wrapped
    return current


def _validated_worktree_patch(value: Any, *, label: str) -> str:
    """Return a real unified diff and reject Docker/runtime error text.

    Some R2E runtime failures are returned as strings instead of raised as
    exceptions. Treating any non-empty string as a patch made a post-timeout
    ``Error: NotFound(...)`` response look like a valid worktree diff.
    """

    patch = str(value or "")
    if not patch.strip():
        return ""

    # ``runtime.get_patch`` and the fresh-container extractor both normally
    # produce a git diff. Be tolerant of harmless leading diagnostics while
    # ensuring an error message cannot become a positive patch signal.
    git_diff = re.search(r"(?m)^diff --git ", patch)
    if git_diff is not None:
        return patch[git_diff.start() :]

    stripped = patch.lstrip()
    if stripped.startswith("--- ") and re.search(r"(?m)^\+\+\+ ", stripped):
        return stripped

    excerpt = " ".join(patch.strip().split())[:300]
    raise RuntimeError(f"{label} returned non-patch output: {excerpt}")


def _extract_verified_worktree_patch(
    environment: Any, *, timeout: float
) -> str:
    """Return tracked and untracked worktree changes without accepting them."""

    raw_environment = _raw_verified_environment(environment)
    if raw_environment is None:
        return ""
    # Intent-to-add makes untracked files visible to git diff. The rollout
    # container is disposable, and this internal command bypasses the
    # model-facing policy that intentionally blocks `git add`.
    action = {
        "command": (
            "git add -N . >/dev/null 2>&1 || true\n"
            "git -c core.fileMode=false diff --binary -- . "
            "':(exclude)PROBLEM_STATEMENT.md' ':(exclude).harness/**'"
        )
    }
    command_timeout = max(1, int(math.ceil(timeout)))
    result = _hard_call(
        lambda: raw_environment.execute(action, timeout=command_timeout),
        timeout + 5,
        label="verified worktree diff",
        runtime=None,
    )
    if int(result.get("returncode", -1)) != 0:
        raise RuntimeError(
            "git diff failed: " + str(result.get("output") or "")[-1000:]
        )
    return _validated_worktree_patch(
        result.get("output"), label="verified worktree diff"
    )


@dataclass
class _VerifiedPatchGrade:
    attempted: bool = False
    reward: float = 0.0
    reward_known: bool = False
    apply_failed: bool = False
    timed_out: bool = False
    elapsed_seconds: float = 0.0
    container_start_wait_seconds: float = 0.0
    resolution_status: str = ""
    f2p_failure_count: int = 0
    p2p_failure_count: int = 0
    eval_returncode: int = 0
    error: str = ""


def _new_r2e_environment(instance):
    _install_r2e_import_compat()
    from r2egym.agenthub.environment.env import EnvArgs, RepoEnv
    return RepoEnv(EnvArgs(ds=instance), backend='docker',
                   step_timeout=CONFIG.step_timeout, reward_timeout=CONFIG.grade_timeout,
                   verbose=False, runtime_kwargs=_docker_resource_kwargs())


def _grade_r2e_patch_fresh(instance, patch, *, case_deadline):
    """Grade only submitted bytes in a fresh instance of the same R2E task."""
    grade = _VerifiedPatchGrade(attempted=True)
    repo_env = None
    started = time.monotonic()
    try:
        if not patch.strip():
            grade.apply_failed = True
            grade.reward_known = True
            return grade
        with _container_start_slot() as wait_seconds:
            grade.container_start_wait_seconds += wait_seconds
            case_deadline += wait_seconds
            repo_env = _hard_call(lambda: _new_r2e_environment(instance),
                _remaining(case_deadline, CONFIG.setup_timeout), label='training grading setup',
                cleanup_late_result=_close_repo_env)
        runtime = repo_env.runtime
        # Copy data as a file: never interpolate model-produced patch text into
        # shell commands, and preserve the exact submitted bytes/newlines.
        remote_patch = f'/tmp/submitted-{secrets.token_hex(16)}.patch'
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', suffix='.patch') as stream:
            stream.write(patch)
            stream.flush()
            _hard_call(lambda: runtime.copy_to_container(stream.name, remote_patch),
                _remaining(case_deadline, CONFIG.patch_timeout), label='training patch copy', runtime=runtime)
        output, status = _hard_call(
            lambda: runtime.run(f'git apply {remote_patch}', timeout=CONFIG.patch_timeout),
            _remaining(case_deadline, CONFIG.patch_timeout), label='training patch apply', runtime=runtime)
        if str(status) == '-1':
            raise RuntimeError(f'Patch application infrastructure failure: {output}')
        if str(status) != '0':
            grade.apply_failed = True
            grade.reward_known = True
            return grade
        reward = _hard_call(lambda: runtime._calculate_reward(timeout=CONFIG.grade_timeout),
            _remaining(case_deadline, CONFIG.grade_timeout), label='training submitted patch tests', runtime=runtime)
        grade.reward = 1.0 if float(reward) >= 1.0 else 0.0
        grade.reward_known = True
    except TimeoutError as exc:
        grade.timed_out = True
        grade.error = str(exc)
    except Exception as exc:
        grade.error = f'{type(exc).__name__}: {exc}'
    finally:
        _close_repo_env(repo_env)
        grade.elapsed_seconds = time.monotonic() - started
    return grade


def _grade_verified_patch_fresh(
    instance: dict[str, Any],
    patch: str,
    *,
    case_deadline: float,
    label_prefix: str,
) -> _VerifiedPatchGrade:
    """Apply and grade a patch in a new SWE-bench Verified container."""

    grade = _VerifiedPatchGrade(attempted=True)
    environment = None
    started = time.monotonic()
    try:
        with _container_start_slot() as wait_seconds:
            grade.container_start_wait_seconds += wait_seconds
            case_deadline += wait_seconds
            environment = _hard_call(
                lambda: _new_verified_environment(instance),
                _remaining(case_deadline, CONFIG.setup_timeout),
                label=f"{label_prefix} setup",
                cleanup_late_result=_close_verified_environment,
            )
        patch_delimiter = f"PATCH_{secrets.token_hex(16)}"
        apply_action = {
            "command": (
                f"git apply <<'{patch_delimiter}'\n"
                f"{patch}\n"
                f"{patch_delimiter}"
            )
        }
        apply_timeout = max(
            1,
            int(math.ceil(_remaining(case_deadline, CONFIG.step_timeout))),
        )
        apply_result = _hard_call(
            lambda: environment.execute(apply_action, timeout=apply_timeout),
            _remaining(case_deadline, apply_timeout + 5),
            label=f"{label_prefix} patch apply",
            runtime=environment,
        )
        if int(apply_result.get("returncode", -1)) != 0:
            grade.apply_failed = True
            grade.reward_known = True
            grade.error = (
                "git apply: " + str(apply_result.get("output") or "")[-1000:]
            )
            return grade

        eval_script = str(instance.get("eval_script") or "")
        if not eval_script.strip():
            raise ValueError("Verified instance has no eval_script")
        eval_delimiter = f"EVAL_{secrets.token_hex(16)}"
        eval_action = {
            "command": (
                f"bash <<'{eval_delimiter}'\n"
                f"{eval_script}\n"
                f"{eval_delimiter}"
            )
        }
        eval_timeout = max(
            1,
            int(
                math.ceil(
                    _remaining(case_deadline, CONFIG.verified_grade_timeout)
                )
            ),
        )
        test_result = _hard_call(
            lambda: environment.execute(eval_action, timeout=eval_timeout),
            _remaining(case_deadline, eval_timeout + 5),
            label=f"{label_prefix} grading",
            runtime=environment,
        )
        grade.eval_returncode = int(test_result.get("returncode", -1))
        grade.reward, grading = _score_swebench_verified_output(
            instance, str(test_result.get("output") or "")
        )
        grade.reward_known = True
        grade.resolution_status = str(grading["resolution_status"])
        grade.f2p_failure_count = int(
            grading["FAIL_TO_PASS"]["failure_count"]
        )
        grade.p2p_failure_count = int(
            grading["PASS_TO_PASS"]["failure_count"]
        )
        if not grade.reward:
            grade.error = (
                f"resolution={grade.resolution_status} "
                f"f2p_failed={grade.f2p_failure_count} "
                f"p2p_failed={grade.p2p_failure_count}; "
                + str(test_result.get("output") or "")[-1000:]
            )
    except TimeoutError as exc:
        grade.timed_out = True
        grade.error = str(exc)
    except Exception as exc:
        grade.error = f"{type(exc).__name__}: {exc}"
    finally:
        _close_verified_environment(environment)
        grade.elapsed_seconds = time.monotonic() - started
    return grade


def _write_verified_trajectory_record(
    *,
    instance: dict[str, Any],
    session_id: str,
    agent: Any,
    exit_info: dict[str, Any],
    submission: str,
    worktree_patch: str,
    outcome: CaseResult,
) -> None:
    """Atomically preserve a full per-case record outside Slime's PT dump."""

    if not CONFIG.save_trajectory_records or not CONFIG.trajectory_dir:
        return
    target_dir = Path(CONFIG.trajectory_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    instance_id = str(instance.get("instance_id") or "unknown")
    safe_id = "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in instance_id
    )
    suffix = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:12]
    target = target_dir / f"{safe_id}.{suffix}.json.gz"
    temporary = target.with_suffix(target.suffix + f".{secrets.token_hex(4)}.tmp")
    payload = {
        "format_version": 1,
        "instance_id": instance_id,
        "session_id": session_id,
        "official_reward": outcome.reward,
        "official_submitted": outcome.submitted,
        "exit_info": exit_info,
        "explicit_submission_patch": submission,
        "unsubmitted_worktree_patch": worktree_patch,
        "outcome": asdict(outcome),
        "messages": list(getattr(agent, "messages", []) or []),
        "backend_provenance": getattr(agent, "backend_provenance", None),
        "openhands_events": list(getattr(agent, "events", []) or []),
        "budgets": {
            "max_iterations": CONFIG.verified_max_iterations,
            "agent_wall_time": CONFIG.verified_agent_wall_time,
            "grade_timeout": CONFIG.verified_grade_timeout,
            "case_timeout": CONFIG.verified_case_timeout,
        },
    }
    try:
        with gzip.open(
            temporary,
            "wt",
            encoding="utf-8",
            compresslevel=3,
        ) as handle:
            json.dump(payload, handle, ensure_ascii=False, default=str)
        os.replace(temporary, target)
        outcome.trajectory_record_path = str(target)
    except Exception as exc:
        logger.warning(
            "[slime-r2e] failed to save trajectory %s: %s",
            instance_id,
            exc,
        )
        try:
            temporary.unlink(missing_ok=True)
        except Exception:
            pass


def _write_training_trajectory_record(
    *,
    instance: dict[str, Any],
    session_id: str,
    hint: str,
    agent: Any,
    exit_info: dict[str, Any],
    worktree_patch: str,
    outcome: CaseResult,
) -> None:
    """Atomically save one full training conversation and its outcome."""

    if not CONFIG.save_trajectory_records or not CONFIG.trajectory_dir:
        return
    target_dir = Path(CONFIG.trajectory_dir) / "train"
    target_dir.mkdir(parents=True, exist_ok=True)
    instance_id = str(instance.get("instance_id") or "unknown")
    safe_id = "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in instance_id
    )
    suffix = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:12]
    target = target_dir / f"{safe_id}.{suffix}.json.gz"
    temporary = target.with_suffix(target.suffix + f".{secrets.token_hex(4)}.tmp")
    submission = str(exit_info.get("submission") or "")
    payload = {
        "format_version": 1,
        "trajectory_kind": "training",
        "instance_id": instance_id,
        "session_id": session_id,
        "hint_used": bool(hint.strip()),
        "hint": hint,
        "official_reward": outcome.reward,
        "official_submitted": outcome.submitted,
        "exit_info": exit_info,
        "explicit_submission_patch": submission,
        "worktree_patch": worktree_patch,
        "outcome": asdict(outcome),
        "messages": list(getattr(agent, "messages", []) or []),
        "backend_provenance": getattr(agent, "backend_provenance", None),
        "openhands_events": list(getattr(agent, "events", []) or []),
        "budgets": {
            "max_iterations": CONFIG.max_iterations,
            "agent_wall_time": CONFIG.agent_wall_time,
            "grade_timeout": CONFIG.grade_timeout,
            "case_timeout": CONFIG.case_timeout,
        },
    }
    try:
        with gzip.open(
            temporary,
            "wt",
            encoding="utf-8",
            compresslevel=3,
        ) as handle:
            json.dump(payload, handle, ensure_ascii=False, default=str)
        os.replace(temporary, target)
        from .trajectory_markdown import write_trajectory_markdown

        write_trajectory_markdown(payload, target_dir.parent / "train_markdown")
        outcome.trajectory_record_path = str(target)
    except Exception as exc:
        logger.warning(
            "[slime-r2e] failed to save training trajectory %s: %s",
            instance_id,
            exc,
        )
        try:
            temporary.unlink(missing_ok=True)
        except Exception:
            pass


def _populate_outcome_from_agent(
    outcome: CaseResult, agent: Any, model: Any, exit_info: dict[str, Any]
) -> None:
    outcome.exit_status = str(exit_info.get("exit_status") or "")
    submission = str(exit_info.get("submission") or "")
    outcome.submitted = (
        outcome.exit_status.lower() == "submitted"
        and bool(submission.strip())
    )
    outcome.n_calls = int(getattr(agent, "n_calls", 0) or 0)
    outcome.context_prompt_tokens = model.last_prompt_tokens
    outcome.context_remaining_tokens = model.last_remaining_tokens
    outcome.history_assistant_turns = model.last_history_assistant_turns
    outcome.history_reasoning_turns = model.last_history_reasoning_turns
    outcome.context_budget_exit = bool(model.context_budget_exit)
    outcome.generation_length_exit = bool(model.generation_length_exit)
    outcome.model_request_timeout = bool(model.model_request_timeout)
    normalized_status = outcome.exit_status.strip().lower()
    outcome.agent_time_exceeded = normalized_status == "timeexceeded"
    outcome.limits_exceeded = normalized_status == "limitsexceeded"
    outcome.repeated_format_error = normalized_status == "repeatedformaterror"
    format_stats = getattr(model, "_slime_format_error_stats", {}) or {}
    for field in (
        "format_error_recovered_turns",
        "bad_json_turns",
        "no_tool_turns",
        "bad_json_tokens",
        "no_tool_tokens",
        "format_error_arguments_normalized_turns",
        "assistant_turns",
        "format_transition_pairs",
        "previous_error_turns",
        "error_after_error_turns",
        "previous_ok_turns",
        "error_after_ok_turns",
        "format_error_recovery_turns",
        "format_error_streak_count",
        "format_error_max_streak",
        "sampled_without_eos_turns",
    ):
        setattr(outcome, field, int(format_stats.get(field) or 0))
    outcome.format_error_terminal_streak = int(
        format_stats.get("_current_error_streak") or 0
    )
    if outcome.exit_status.lower() in _EXTERNAL_STOP_STATUSES:
        outcome.external_stop = True


def _agent_deadline_exit(exc: TimeoutError) -> dict[str, Any]:
    """Translate the agent-owned hard deadline into a policy-budget exit.

    The host-side ``_hard_call`` and Mini-SWE's internal wall-time check guard
    the same agent budget.  At the boundary the host thread can win the race,
    so returning an empty ``exit_info`` would misclassify a policy timeout as
    an infrastructure timeout.  Keep ``timed_out`` for diagnostics, while the
    explicit status lets downstream code retain the partial trajectory as
    ``INCOMPLETE`` process evidence.  Its Stage-2 solution tokens remain
    masked and never train as a reward-zero completion.
    """

    return {
        "exit_status": "TimeExceeded",
        "submission": "",
        "agent_deadline_error": str(exc),
    }


def _run_case(
    *,
    instance: dict[str, Any],
    hint: str,
    opsd_note_override: str | None = None,
    opsd_priority_block_override: str = "",
    adapter_url: str,
    session_id: str,
    sampling_params: dict[str, Any],
    tokenizer: Any,
    model_context_length: int,
) -> CaseResult:
    """Blocking selected-agent/R2E execution in the bounded case pool."""
    from minisweagent.agents.default import DefaultAgent

    _install_r2e_import_compat()
    from r2egym.agenthub.environment.env import EnvArgs, RepoEnv

    # These helpers contain the established R2E command policy and exact
    # submission protocol used by the current implementation.
    from harnessadapt.policy import (
        INSTANCE_TEMPLATE,
        SYSTEM_TEMPLATE,
        _build_opsd_note,
        _make_environment_cls,
    )

    started = time.monotonic()
    case_deadline = started + CONFIG.case_timeout
    repo_env = None
    runtime = None
    agent: DefaultAgent | None = None
    outcome = CaseResult()
    exit_info: dict[str, Any] = {}
    patch = ""
    timeout_snapshot: dict[str, str] = {}

    try:
        with _container_start_slot() as wait_seconds:
            outcome.container_start_wait_seconds += wait_seconds
            case_deadline += wait_seconds
            repo_env = _hard_call(
                lambda: RepoEnv(
                    EnvArgs(ds=instance),
                    backend="docker",
                    step_timeout=CONFIG.step_timeout,
                    reward_timeout=CONFIG.grade_timeout,
                    verbose=False,
                    runtime_kwargs=_docker_resource_kwargs(),
                ),
                _remaining(case_deadline, CONFIG.setup_timeout),
                label="setup",
                cleanup_late_result=_close_repo_env,
            )
        runtime = repo_env.runtime
        try:
            problem = runtime.get_task_instruction()
        except Exception:
            problem = str(instance.get("problem_statement") or "")
        if not problem:
            raise ValueError("R2E instance has no problem statement")

        # Keep the curriculum's established carrier as the default.  Callers
        # that produce a different kind of guidance (for example, an on-policy
        # scaffold that may be wrong) can reuse the exact same system-prompt
        # slot while supplying appropriately calibrated framing.
        opsd_note = (
            _build_opsd_note(hint)
            if opsd_note_override is None
            else opsd_note_override
        )
        environment = _make_environment_cls()(
            runtime,
            cwd="/testbed",
            timeout=CONFIG.step_timeout,
            hard_step_timeout=CONFIG.hard_step_timeout,
            opsd_note=opsd_note,
            opsd_priority_block=opsd_priority_block_override,
        )
        model = _case_model(
            adapter_url,
            session_id,
            sampling_params,
            tokenizer,
            model_context_length=model_context_length,
            case_deadline=case_deadline,
        )
        agent = _agent_class()(
            model,
            environment,
            system_template=SYSTEM_TEMPLATE,
            instance_template=INSTANCE_TEMPLATE,
            step_limit=CONFIG.max_iterations,
            cost_limit=0.0,
            wall_time_limit_seconds=min(
                CONFIG.agent_wall_time,
                max(1, int(math.floor(case_deadline - time.monotonic()))),
            ),
            max_consecutive_format_errors=CONFIG.max_consecutive_format_errors,
        )

        def snapshot_before_forced_cleanup() -> None:
            """Capture the live diff while the timed-out container still exists."""

            if runtime is None:
                return
            try:
                raw_patch = _hard_call(
                    runtime.get_patch,
                    min(float(CONFIG.patch_timeout), 30.0),
                    label="timeout worktree diff",
                    # The outer agent deadline owns container cleanup. An inner
                    # patch timeout must not remove the container before that
                    # cleanup path gets a chance to run.
                    runtime=None,
                )
                timeout_snapshot["patch"] = _validated_worktree_patch(
                    raw_patch, label="timeout worktree diff"
                )
            except Exception as exc:
                timeout_snapshot["error"] = f"{type(exc).__name__}: {exc}"

        agent_exception = ""
        try:
            exit_info = (
                _hard_call(
                    lambda: agent.run(problem),
                    _remaining(case_deadline, CONFIG.agent_wall_time),
                    label="agent",
                    runtime=runtime,
                    before_timeout_cleanup=snapshot_before_forced_cleanup,
                )
                or {}
            )
        except TimeoutError as exc:
            outcome.timed_out = True
            outcome.external_stop = True
            outcome.error = str(exc)
            exit_info = _agent_deadline_exit(exc)
        except Exception as exc:
            agent_exception = f"{type(exc).__name__}: {exc}"
            outcome.error = agent_exception
            if agent.messages:
                exit_info = dict(agent.messages[-1].get("extra") or {})

        # Mini-SWE uses the submission marker. Official OpenHands requires an
        # SDK FINISHED state and a nonempty patch collected against the initial
        # commit. Both feed the same fresh-container benchmark grader below.
        _populate_outcome_from_agent(outcome, agent, model, exit_info)

        if outcome.timed_out:
            patch = str(timeout_snapshot.get("patch") or "")
            if timeout_snapshot.get("error"):
                outcome.unsubmitted_patch_error = (
                    "timeout worktree diff: " + timeout_snapshot["error"]
                )
        else:
            try:
                raw_patch = _hard_call(
                    runtime.get_patch,
                    _remaining(case_deadline, CONFIG.patch_timeout),
                    label="get_patch",
                    runtime=runtime,
                )
                patch = _validated_worktree_patch(
                    raw_patch, label="get_patch"
                )
            except Exception as exc:
                if not outcome.error:
                    outcome.error = f"get_patch: {type(exc).__name__}: {exc}"

        outcome.patch_available = bool(patch.strip())
        if not outcome.submitted:
            outcome.unsubmitted_worktree_patch_available = (
                outcome.patch_available
            )
            outcome.unsubmitted_worktree_patch_bytes = len(
                patch.encode("utf-8")
            )
            outcome.unsubmitted_worktree_patch_lines = len(patch.splitlines())

        if outcome.timed_out or outcome.external_stop:
            # Reward zero still participates in the GRPO group baseline, while
            # ``generate`` masks all policy tokens for this execution.
            outcome.reward = 0.0
            outcome.reward_known = True
        elif outcome.repeated_format_error:
            # This is a model-behaviour terminal state, not an infrastructure
            # ambiguity. It is a known failed attempt with zero reward.
            outcome.reward = 0.0
            outcome.reward_known = True
        elif agent_exception:
            # An uncaught model/adapter exception is infrastructure-ambiguous.
            outcome.reward_known = False
        elif not outcome.submitted:
            outcome.reward = 0.0
            outcome.reward_known = True
        else:
            _close_repo_env(repo_env)
            repo_env = None
            runtime = None
            outcome.training_fresh_container = True
            grade = _grade_r2e_patch_fresh(instance, str(exit_info.get('submission') or ''),
                                         case_deadline=case_deadline)
            outcome.container_start_wait_seconds += grade.container_start_wait_seconds
            outcome.training_patch_apply_failed = grade.apply_failed
            outcome.reward = grade.reward
            outcome.reward_known = grade.reward_known
            if grade.timed_out:
                outcome.timed_out = True
                outcome.external_stop = True
                outcome.reward = 0.0
                outcome.reward_known = True
            if grade.error:
                outcome.error = 'submitted patch grade: ' + grade.error

        _grade_unsubmitted_r2e_worktree(
            outcome,
            runtime,
            case_deadline=case_deadline,
        )

    except TimeoutError as exc:
        outcome.timed_out = True
        outcome.external_stop = True
        outcome.reward = 0.0
        outcome.reward_known = True
        outcome.error = str(exc)
    except Exception as exc:
        outcome.error = f"{type(exc).__name__}: {exc}"
        logger.warning(
            "[slime-r2e] case setup/execution failed: %s\n%s",
            outcome.error,
            traceback.format_exc(),
        )
    finally:
        _close_repo_env(repo_env)
        outcome.elapsed_seconds = time.monotonic() - started
        _write_training_trajectory_record(
            instance=instance,
            session_id=session_id,
            hint=hint,
            agent=agent,
            exit_info=exit_info,
            worktree_patch=patch,
            outcome=outcome,
        )

    return outcome


def _run_verified_case(
    *,
    instance: dict[str, Any],
    hint: str,
    adapter_url: str,
    session_id: str,
    sampling_params: dict[str, Any],
    tokenizer: Any,
    model_context_length: int,
) -> CaseResult:
    """Roll out and grade one SWE-bench Verified case in separate containers."""

    del hint  # Verified evaluation is always hint-free.
    _install_r2e_import_compat()
    from harnessadapt.policy import (
        INSTANCE_TEMPLATE,
        SYSTEM_TEMPLATE,
        _install_git_command_policy,
    )

    started = time.monotonic()
    case_deadline = started + CONFIG.verified_case_timeout
    agent_environment = None
    agent = None
    model = None
    outcome = CaseResult(verified_fresh_container=True)
    exit_info: dict[str, Any] = {}
    agent_exception = ""
    submission = ""
    worktree_patch = ""
    timeout_snapshot: dict[str, str] = {}

    try:
        with _container_start_slot() as wait_seconds:
            outcome.container_start_wait_seconds += wait_seconds
            case_deadline += wait_seconds
            agent_environment = _hard_call(
                lambda: _new_verified_environment(instance),
                _remaining(case_deadline, CONFIG.setup_timeout),
                label="verified rollout setup",
                cleanup_late_result=_close_verified_environment,
            )
        agent_environment = _install_git_command_policy(agent_environment)
        problem = str(instance.get("problem_statement") or "").strip()
        if not problem:
            raise ValueError("Verified instance has no problem_statement")

        model = _case_model(
            adapter_url,
            session_id,
            sampling_params,
            tokenizer,
            model_context_length=model_context_length,
            case_deadline=case_deadline,
        )
        agent = _agent_class()(
            model,
            agent_environment,
            system_template=SYSTEM_TEMPLATE,
            instance_template=INSTANCE_TEMPLATE,
            step_limit=CONFIG.verified_max_iterations,
            cost_limit=0.0,
            wall_time_limit_seconds=min(
                CONFIG.verified_agent_wall_time,
                max(1, int(math.floor(case_deadline - time.monotonic()))),
            ),
            max_consecutive_format_errors=CONFIG.max_consecutive_format_errors,
        )

        def snapshot_before_forced_cleanup() -> None:
            if outcome.submitted or agent_environment is None:
                return
            try:
                timeout_snapshot["patch"] = _extract_verified_worktree_patch(
                    agent_environment,
                    timeout=min(float(CONFIG.patch_timeout), 30.0),
                )
            except Exception as exc:
                timeout_snapshot["error"] = (
                    f"{type(exc).__name__}: {exc}"
                )

        try:
            exit_info = (
                _hard_call(
                    lambda: agent.run(problem),
                    _remaining(
                        case_deadline, CONFIG.verified_agent_wall_time
                    ),
                    label="verified agent",
                    runtime=agent_environment,
                    before_timeout_cleanup=snapshot_before_forced_cleanup,
                )
                or {}
            )
        except TimeoutError as exc:
            outcome.timed_out = True
            outcome.external_stop = True
            outcome.error = str(exc)
            exit_info = _agent_deadline_exit(exc)
        except Exception as exc:
            agent_exception = f"{type(exc).__name__}: {exc}"
            outcome.error = agent_exception
            if getattr(agent, "messages", None):
                exit_info = dict(agent.messages[-1].get("extra") or {})

        submission = str(exit_info.get("submission") or "")
        _populate_outcome_from_agent(outcome, agent, model, exit_info)
        outcome.patch_available = bool(submission.strip())

        if (
            not outcome.submitted
            and (
                CONFIG.grade_unsubmitted_patches
                or CONFIG.save_trajectory_records
            )
        ):
            worktree_patch = str(timeout_snapshot.get("patch") or "")
            if not worktree_patch and agent_environment is not None:
                try:
                    worktree_patch = _extract_verified_worktree_patch(
                        agent_environment,
                        timeout=_remaining(
                            case_deadline, CONFIG.patch_timeout
                        ),
                    )
                except Exception as exc:
                    outcome.unsubmitted_patch_error = (
                        "worktree diff: "
                        f"{type(exc).__name__}: {exc}"
                    )
            if not worktree_patch and timeout_snapshot.get("error"):
                outcome.unsubmitted_patch_error = (
                    "timeout worktree diff: " + timeout_snapshot["error"]
                )
            outcome.unsubmitted_worktree_patch_available = bool(
                worktree_patch.strip()
            )
            outcome.unsubmitted_worktree_patch_bytes = len(
                worktree_patch.encode("utf-8")
            )
            outcome.unsubmitted_worktree_patch_lines = len(
                worktree_patch.splitlines()
            )

        if outcome.timed_out or outcome.external_stop:
            outcome.reward = 0.0
            outcome.reward_known = True
        elif outcome.repeated_format_error:
            outcome.reward = 0.0
            outcome.reward_known = True
        elif agent_exception:
            outcome.reward_known = False
        elif CONFIG.require_submission and not outcome.submitted:
            outcome.reward = 0.0
            outcome.reward_known = True
        else:
            # Release the rollout sandbox before fresh-container grading so
            # concurrent cases do not keep two containers alive apiece.
            _close_verified_environment(agent_environment)
            agent_environment = None
            grade = _grade_verified_patch_fresh(
                instance,
                submission,
                case_deadline=case_deadline,
                label_prefix="verified",
            )
            outcome.container_start_wait_seconds += (
                grade.container_start_wait_seconds
            )
            outcome.verified_patch_apply_failed = grade.apply_failed
            outcome.verified_eval_returncode = grade.eval_returncode
            outcome.verified_resolution_status = grade.resolution_status
            outcome.verified_f2p_failure_count = grade.f2p_failure_count
            outcome.verified_p2p_failure_count = grade.p2p_failure_count
            if grade.timed_out:
                outcome.timed_out = True
                outcome.external_stop = True
                outcome.reward = 0.0
                outcome.reward_known = True
            else:
                outcome.reward = grade.reward
                outcome.reward_known = grade.reward_known
            if grade.error:
                outcome.error = "verified eval: " + grade.error

        # Counterfactual diagnostic only. A passing worktree patch never
        # changes official reward/submission or trainability.
        if (
            CONFIG.grade_unsubmitted_patches
            and not outcome.submitted
            and worktree_patch.strip()
        ):
            _close_verified_environment(agent_environment)
            agent_environment = None
            diagnostic_grade = _grade_verified_patch_fresh(
                instance,
                worktree_patch,
                case_deadline=case_deadline,
                label_prefix="unsubmitted worktree",
            )
            outcome.container_start_wait_seconds += (
                diagnostic_grade.container_start_wait_seconds
            )
            outcome.unsubmitted_patch_grade_attempted = (
                diagnostic_grade.attempted
            )
            outcome.unsubmitted_patch_reward = diagnostic_grade.reward
            outcome.unsubmitted_patch_reward_known = (
                diagnostic_grade.reward_known
            )
            outcome.unsubmitted_patch_apply_failed = (
                diagnostic_grade.apply_failed
            )
            outcome.unsubmitted_patch_grade_timed_out = (
                diagnostic_grade.timed_out
            )
            outcome.unsubmitted_patch_grade_elapsed_seconds = (
                diagnostic_grade.elapsed_seconds
            )
            outcome.unsubmitted_patch_resolution_status = (
                diagnostic_grade.resolution_status
            )
            outcome.unsubmitted_patch_f2p_failure_count = (
                diagnostic_grade.f2p_failure_count
            )
            outcome.unsubmitted_patch_p2p_failure_count = (
                diagnostic_grade.p2p_failure_count
            )
            outcome.unsubmitted_patch_eval_returncode = (
                diagnostic_grade.eval_returncode
            )
            if diagnostic_grade.error:
                outcome.unsubmitted_patch_error = diagnostic_grade.error
    except TimeoutError as exc:
        outcome.timed_out = True
        outcome.external_stop = True
        outcome.reward = 0.0
        outcome.reward_known = True
        outcome.error = str(exc)
    except Exception as exc:
        outcome.error = f"verified: {type(exc).__name__}: {exc}"
        logger.warning(
            "[slime-r2e] Verified case failed: %s\n%s",
            outcome.error,
            traceback.format_exc(),
        )
    finally:
        _close_verified_environment(agent_environment)
        outcome.elapsed_seconds = time.monotonic() - started
        _write_verified_trajectory_record(
            instance=instance,
            session_id=session_id,
            agent=agent,
            exit_info=exit_info,
            submission=submission,
            worktree_patch=worktree_patch,
            outcome=outcome,
        )

    return outcome


def _session_id(sample: Sample) -> str:
    if sample.session_id:
        return sample.session_id
    label = str(sample.label or "unknown").replace("/", "_")
    if sample.index is not None and sample.group_index is not None:
        return f"r2e-{label}-{sample.group_index}-{sample.index}"
    return f"r2e-{label}-{secrets.token_hex(8)}"


def _masked_result(
    sample: Sample,
    tokenizer: Any,
    *,
    reason: str,
    stage: int,
    hint_used: bool,
    reward: float = 0.0,
    extra: dict[str, Any] | None = None,
) -> Sample:
    """Return a shape-valid, zero-loss sample without marking the group aborted."""
    token_id = (
        tokenizer.eos_token_id
        if tokenizer.eos_token_id is not None
        else tokenizer.pad_token_id
    )
    if token_id is None:
        token_id = 0
    sample.tokens = [int(token_id), int(token_id)]
    sample.response = ""
    sample.response_length = 1
    sample.loss_mask = [0]
    sample.rollout_log_probs = [0.0]
    sample.reward = float(reward)
    sample.remove_sample = True
    sample.status = Sample.Status.COMPLETED
    sample.metadata = {
        "instance_id": str(sample.label or ""),
        "curriculum_stage": stage,
        "hint_used": hint_used,
        "masked_reason": reason,
        **(extra or {}),
    }
    return sample


def _validate_trainable_sample(sample: Sample) -> str | None:
    if sample.response_length <= 0:
        return "empty_response"
    if not sample.tokens:
        return "empty_tokens"
    if sample.loss_mask is None or len(sample.loss_mask) != sample.response_length:
        return "invalid_loss_mask"
    if not any(sample.loss_mask):
        return "no_trainable_tokens"
    if (
        sample.rollout_log_probs is None
        or len(sample.rollout_log_probs) != sample.response_length
    ):
        return "invalid_rollout_logprobs"
    return None


def _count_non_target_language_tokens(
    samples: list[Sample],
    tokenizer: Any,
) -> tuple[int, int]:
    """Count non-ASCII-letter sampled tokens with a bounded decode cache."""
    non_target = 0
    sampled = 0
    for sample in samples:
        response_length = int(sample.response_length or 0)
        if response_length <= 0 or not sample.tokens:
            continue
        response_ids = sample.tokens[-response_length:]
        mask = sample.loss_mask or [1] * response_length
        for token_id, trainable in zip(response_ids, mask, strict=False):
            if not trainable:
                continue
            sampled += 1
            token_id = int(token_id)
            is_non_target = _NON_TARGET_TOKEN_CACHE.get(token_id)
            if is_non_target is None:
                try:
                    text = tokenizer.decode(
                        [token_id],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                except TypeError:
                    text = tokenizer.decode([token_id], skip_special_tokens=True)
                is_non_target = any(
                    not character.isascii() and character.isalpha()
                    for character in text
                )
                _NON_TARGET_TOKEN_CACHE[token_id] = is_non_target
            non_target += int(is_non_target)
    return non_target, sampled


def _trajectory_token_diagnostics(
    samples: list[Sample],
    tokenizer: Any,
) -> dict[str, Any]:
    response_tokens = sum(int(sample.response_length or 0) for sample in samples)
    trainable_tokens = sum(
        sum(int(value) for value in (sample.loss_mask or []))
        for sample in samples
    )
    non_target_tokens, legal_tool_tokens = _count_non_target_language_tokens(
        samples,
        tokenizer,
    )
    observed_weight_versions = sorted(
        {
            str(version)
            for sample in samples
            for version in (sample.weight_versions or [])
            if str(version)
        }
    )
    numeric_weight_versions: list[int] = []
    for version in observed_weight_versions:
        try:
            numeric_weight_versions.append(int(version))
        except ValueError:
            pass
    repeated_16grams = 0
    sampled_16grams = 0
    sampled_unique_tokens: set[int] = set()
    sampled_tokens = 0
    ngram_size = 16
    hash_mask = (1 << 64) - 1
    hash_base = 1_000_003
    high_power = pow(hash_base, ngram_size - 1, 1 << 64)
    for sample in samples:
        response_length = int(sample.response_length or 0)
        if response_length <= 0 or not sample.tokens:
            continue
        response_ids = sample.tokens[-response_length:]
        mask = sample.loss_mask or [1] * response_length
        sampled_ids = [
            int(token_id)
            for token_id, trainable in zip(response_ids, mask, strict=False)
            if trainable
        ]
        sampled_tokens += len(sampled_ids)
        sampled_unique_tokens.update(sampled_ids)
        if len(sampled_ids) < ngram_size:
            continue
        rolling_hash = 0
        for token_id in sampled_ids[:ngram_size]:
            rolling_hash = (
                rolling_hash * hash_base + token_id + 1
            ) & hash_mask
        seen_hashes = {rolling_hash}
        sampled_16grams += 1
        for offset in range(ngram_size, len(sampled_ids)):
            rolling_hash = (
                (rolling_hash - (sampled_ids[offset - ngram_size] + 1) * high_power)
                * hash_base
                + sampled_ids[offset]
                + 1
            ) & hash_mask
            repeated_16grams += int(rolling_hash in seen_hashes)
            seen_hashes.add(rolling_hash)
            sampled_16grams += 1
    return {
        "adapter_sample_count": len(samples),
        "adapter_fork": len(samples) > 1,
        "token_ids_missing": any(
            int(sample.response_length or 0) > 0 and not sample.tokens
            for sample in samples
        ),
        "logprobs_missing": any(
            sample.rollout_log_probs is None
            or len(sample.rollout_log_probs) != int(sample.response_length or 0)
            for sample in samples
        ),
        # Slime's adapter constructs trajectories from SGLang token IDs rather
        # than retokenized text. A fork is the explicit signal that replayed
        # history drifted too far to remain one exact trajectory.
        "token_retokenized": len(samples) > 1,
        "trajectory_response_tokens": response_tokens,
        "raw_trainable_tokens": trainable_tokens,
        "legal_tool_tokens": legal_tool_tokens,
        "non_target_language_tokens": non_target_tokens,
        "weight_version_missing": bool(response_tokens)
        and not observed_weight_versions,
        "weight_version_mixed": len(observed_weight_versions) > 1,
        "weight_version_count": len(observed_weight_versions),
        "weight_version_min": (
            min(numeric_weight_versions) if numeric_weight_versions else -1
        ),
        "weight_version_max": (
            max(numeric_weight_versions) if numeric_weight_versions else -1
        ),
        # Language-independent degeneration signals over model-generated
        # tokens only. A rolling 64-bit hash avoids retaining large n-gram
        # tuples for 64K trajectories.
        "sampled_token_count": sampled_tokens,
        "sampled_unique_token_count": len(sampled_unique_tokens),
        "sampled_16gram_count": sampled_16grams,
        "repeated_16gram_count": repeated_16grams,
        "sampled_16gram_repetition_rate": (
            repeated_16grams / sampled_16grams if sampled_16grams else 0.0
        ),
    }


async def generate(
    args,
    base_sample: Sample,
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> Sample:
    """Slime ``--custom-generate-function-path`` entry point."""
    state = _AdapterService(args)
    metadata = base_sample.metadata if isinstance(base_sample.metadata, dict) else {}
    instance = metadata.get("instance")
    curriculum = metadata.get("curriculum")
    if not isinstance(instance, dict) or not isinstance(curriculum, dict):
        return _masked_result(
            base_sample,
            state.tokenizer,
            reason="missing_instance_or_curriculum",
            stage=-1,
            hint_used=False,
        )
    # Slime assigns group_index only to training samples. Evaluation prompts
    # are independent pass@1 samples and hint_for_sample() ignores the index
    # when evaluation=True, so do not reject the Verified split here.
    if base_sample.group_index is None and not evaluation:
        return _masked_result(
            base_sample,
            state.tokenizer,
            reason="missing_group_index",
            stage=-1,
            hint_used=False,
        )

    stage, hint_used, hint = hint_for_sample(
        base_sample.group_index if base_sample.group_index is not None else 0,
        curriculum,
        evaluation=evaluation,
        hint_mode=os.environ.get("R2E_HINT_MODE", "curriculum"),
    )
    session_id = _session_id(base_sample)
    base_sample.session_id = session_id
    state.adapter.open_session(
        session_id,
        sampling_defaults=sampling_params,
        # The adapter owns authoritative tokenization, so enforce both the
        # safety margin and minimum useful generation budget there.
        max_context_tokens=max(
            1,
            state.max_context_tokens - CONFIG.context_safety_margin,
        ),
        min_generation_tokens=CONFIG.min_generation_tokens,
    )

    try:
        loop = asyncio.get_running_loop()
        data_source = str(metadata.get("data_source") or "")
        case_runner = (
            _run_verified_case
            if evaluation and data_source.lower() == "swe-bench_verified"
            else _run_case
        )
        outcome = await loop.run_in_executor(
            state.executor,
            functools.partial(
                case_runner,
                instance=instance,
                hint=hint,
                adapter_url=state.adapter_url,
                session_id=session_id,
                sampling_params=sampling_params,
                tokenizer=state.tokenizer,
                model_context_length=state.max_context_tokens,
            ),
        )
        diagnostic = {
            "training_fresh_container": outcome.training_fresh_container,
            "training_patch_apply_failed": outcome.training_patch_apply_failed,
            "reward_known": outcome.reward_known,
            "submitted": outcome.submitted,
            "patch_available": outcome.patch_available,
            "external_stop": outcome.external_stop,
            "timed_out": outcome.timed_out,
            "exit_status": outcome.exit_status,
            "rollout_error": outcome.error[:500],
            "agent_calls": outcome.n_calls,
            "case_elapsed_seconds": outcome.elapsed_seconds,
            "container_start_wait_seconds": (
                outcome.container_start_wait_seconds
            ),
            "context_prompt_tokens": outcome.context_prompt_tokens,
            "context_remaining_tokens": outcome.context_remaining_tokens,
            "context_budget_exit": outcome.context_budget_exit,
            "generation_length_exit": outcome.generation_length_exit,
            "model_request_timeout": outcome.model_request_timeout,
            "agent_time_exceeded": outcome.agent_time_exceeded,
            "limits_exceeded": outcome.limits_exceeded,
            "repeated_format_error": outcome.repeated_format_error,
            # Legacy name retained for dashboard compatibility. It means the
            # malformed turn was preserved in history, not that the following
            # turn recovered.
            "format_error_recovered": outcome.format_error_recovered_turns > 0,
            "format_error_recovered_turns": outcome.format_error_recovered_turns,
            "format_error_history_preserved": (
                outcome.format_error_recovered_turns > 0
            ),
            "format_error_history_preserved_turns": (
                outcome.format_error_recovered_turns
            ),
            "format_error_recovery": outcome.format_error_recovery_turns > 0,
            "format_error_recovery_turns": outcome.format_error_recovery_turns,
            "format_error_streak_count": outcome.format_error_streak_count,
            "format_error_max_streak": outcome.format_error_max_streak,
            "format_error_terminal_streak": outcome.format_error_terminal_streak,
            "bad_json_turns": outcome.bad_json_turns,
            "no_tool_turns": outcome.no_tool_turns,
            "bad_json_tokens": outcome.bad_json_tokens,
            "no_tool_tokens": outcome.no_tool_tokens,
            "format_error_arguments_normalized": (
                outcome.format_error_arguments_normalized_turns > 0
            ),
            "format_error_arguments_normalized_turns": (
                outcome.format_error_arguments_normalized_turns
            ),
            "assistant_turns": outcome.assistant_turns,
            "history_assistant_turns": outcome.history_assistant_turns,
            "history_reasoning_turns": outcome.history_reasoning_turns,
            "history_reasoning_missing_turns": max(
                0,
                outcome.history_assistant_turns - outcome.history_reasoning_turns,
            ),
            "format_transition_pairs": outcome.format_transition_pairs,
            "previous_error_turns": outcome.previous_error_turns,
            "error_after_error_turns": outcome.error_after_error_turns,
            "previous_ok_turns": outcome.previous_ok_turns,
            "error_after_ok_turns": outcome.error_after_ok_turns,
            "sampled_without_eos": outcome.sampled_without_eos_turns > 0,
            "sampled_without_eos_turns": outcome.sampled_without_eos_turns,
            "verified_fresh_container": outcome.verified_fresh_container,
            "verified_patch_apply_failed": outcome.verified_patch_apply_failed,
            "verified_resolution_status": outcome.verified_resolution_status,
            "verified_f2p_failure_count": (
                outcome.verified_f2p_failure_count
            ),
            "verified_p2p_failure_count": (
                outcome.verified_p2p_failure_count
            ),
            "verified_eval_returncode": outcome.verified_eval_returncode,
            "unsubmitted_worktree_patch_available": (
                outcome.unsubmitted_worktree_patch_available
            ),
            "unsubmitted_worktree_patch_bytes": (
                outcome.unsubmitted_worktree_patch_bytes
            ),
            "unsubmitted_worktree_patch_lines": (
                outcome.unsubmitted_worktree_patch_lines
            ),
            "unsubmitted_patch_grade_attempted": (
                outcome.unsubmitted_patch_grade_attempted
            ),
            "unsubmitted_patch_reward": outcome.unsubmitted_patch_reward,
            "unsubmitted_patch_reward_known": (
                outcome.unsubmitted_patch_reward_known
            ),
            "unsubmitted_patch_apply_failed": (
                outcome.unsubmitted_patch_apply_failed
            ),
            "unsubmitted_patch_grade_timed_out": (
                outcome.unsubmitted_patch_grade_timed_out
            ),
            "unsubmitted_patch_grade_elapsed_seconds": (
                outcome.unsubmitted_patch_grade_elapsed_seconds
            ),
            "unsubmitted_patch_resolution_status": (
                outcome.unsubmitted_patch_resolution_status
            ),
            "unsubmitted_patch_f2p_failure_count": (
                outcome.unsubmitted_patch_f2p_failure_count
            ),
            "unsubmitted_patch_p2p_failure_count": (
                outcome.unsubmitted_patch_p2p_failure_count
            ),
            "unsubmitted_patch_eval_returncode": (
                outcome.unsubmitted_patch_eval_returncode
            ),
            "unsubmitted_patch_error": outcome.unsubmitted_patch_error[:500],
            "trajectory_record_path": outcome.trajectory_record_path,
        }
        samples = await state.adapter.finish_session(
            session_id,
            base_sample=base_sample,
            reward=outcome.reward,
            extra_metadata={
                "instance_id": str(base_sample.label or ""),
                "data_source": str(metadata.get("data_source") or ""),
                "curriculum_stage": stage,
                "hint_used": hint_used,
                **diagnostic,
            },
            wait_timeout=CONFIG.drop_session_timeout,
        )
        diagnostic.update(
            _trajectory_token_diagnostics(samples, state.tokenizer)
        )
        for sample in samples:
            sample.metadata = {
                **(sample.metadata or {}),
                **diagnostic,
            }
        if not outcome.submitted:
            # This is intentionally unconditional: reward availability,
            # repeated format errors, timeouts, and partial patches do not make
            # a trajectory trainable unless Mini-SWE completed the explicit
            # echo-marker submission protocol.
            return _masked_result(
                base_sample,
                state.tokenizer,
                reason="not_submitted",
                stage=stage,
                hint_used=hint_used,
                reward=outcome.reward if outcome.reward_known else 0.0,
                extra=diagnostic,
            )
        if not outcome.reward_known:
            return _masked_result(
                base_sample,
                state.tokenizer,
                reason="reward_unknown",
                stage=stage,
                hint_used=hint_used,
                reward=0.0,
                extra=diagnostic,
            )
        if outcome.external_stop or outcome.timed_out:
            return _masked_result(
                base_sample,
                state.tokenizer,
                reason="external_stop",
                stage=stage,
                hint_used=hint_used,
                reward=outcome.reward,
                extra=diagnostic,
            )

        from .recovery_segments import enabled as segments_enabled, mark_segments
        if len(samples) != 1 and not (segments_enabled() and samples):
            return _masked_result(
                base_sample,
                state.tokenizer,
                reason=f"adapter_sample_count:{len(samples)}",
                stage=stage,
                hint_used=hint_used,
                reward=outcome.reward,
                extra=diagnostic,
            )
        if segments_enabled():
            for segment in samples:
                if invalid := _validate_trainable_sample(segment):
                    raise ValueError(f'Invalid recovery segment: {invalid}')
            return mark_segments(samples, base_sample, session_id, outcome.reward)
        result = samples[0]
        if invalid := _validate_trainable_sample(result):
            return _masked_result(
                base_sample,
                state.tokenizer,
                reason=invalid,
                stage=stage,
                hint_used=hint_used,
                reward=outcome.reward,
                extra=diagnostic,
            )
        result.reward = outcome.reward
        logger.info(
            "[slime-r2e] instance=%s stage=%d hint=%s reward=%.0f "
            "submitted=%s calls=%d tokens=%d elapsed=%.1fs",
            base_sample.label,
            stage,
            hint_used,
            outcome.reward,
            outcome.submitted,
            outcome.n_calls,
            result.response_length,
            outcome.elapsed_seconds,
        )
        return result
    except Exception as exc:
        logger.warning(
            "[slime-r2e] instance=%s rollout failed: %s\n%s",
            base_sample.label,
            exc,
            traceback.format_exc(),
        )
        return _masked_result(
            base_sample,
            state.tokenizer,
            reason=f"exception:{type(exc).__name__}",
            stage=stage,
            hint_used=hint_used,
        )
    finally:
        await state.adapter.drop_session(
            session_id,
            wait_timeout=CONFIG.drop_session_timeout,
        )
