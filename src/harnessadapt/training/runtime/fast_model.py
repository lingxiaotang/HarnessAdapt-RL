"""Low-overhead OpenAI-compatible client for the slime actor adapter.

Mini-SWE's ``LitellmModel`` routes every turn through litellm into the OpenAI
SDK, whose ``maybe_transform`` walks the entire request body against its
TypedDict annotations.  At R2E conversation sizes (~86 messages / ~26K prompt
tokens) that is ~3,300 recursive calls and ~13.5 ms of CPU per turn, against
~0.9 ms for a plain POST.  Because the walk is pure Python it holds the GIL,
and the adapter's asyncio loop shares the RolloutManager process with ~184
agent threads doing exactly this work, so it lands directly on rollout latency.

Only the transport changes here.  ``_query`` builds and posts the body itself
and rebuilds a litellm ``ModelResponse`` from the reply, so ``query``,
``_parse_actions``, ``_calculate_cost``, ``format_observation_messages`` and
the format-error recovery hook are all inherited unchanged and the assistant
message handed back to Mini-SWE is identical to the litellm path.

Set ``R2E_FAST_MODEL_CLIENT=0`` to fall back to the stock ``LitellmModel``.
"""

from __future__ import annotations

import copy
import json
import os
import threading
from typing import Any

# Keys that configure the transport rather than the completion request; they
# must never reach the wire body.
_TRANSPORT_KEYS = frozenset(
    {
        "api_base",
        "base_url",
        "api_key",
        "api_version",
        "timeout",
        "num_retries",
        "max_retries",
        "drop_params",
        "custom_llm_provider",
        "mock_response",
        "cost_tracking",
        "litellm_model_registry",
    }
)

_local = threading.local()


def _session():
    """One persistent connection per agent thread, reused across cases."""
    session = getattr(_local, "session", None)
    if session is None:
        import requests

        session = requests.Session()
        # each case thread issues one request at a time; a single pooled
        # connection per thread keeps the adapter's accept queue predictable
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=1,
            pool_maxsize=4,
            max_retries=0,
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        _local.session = session
    return session


def _wire_model_name(model_name: str) -> str:
    """``openai/slime-actor`` -> ``slime-actor`` (litellm strips the provider)."""
    return model_name.split("/", 1)[1] if "/" in model_name else model_name


def _http_response(status: int, text: str, url: str):
    """Some litellm exceptions unwrap ``response.request``, so build a real one."""
    import httpx

    return httpx.Response(
        status_code=status, text=text, request=httpx.Request("POST", url)
    )


def _raise_for_status(status: int, text: str, model: str, url: str) -> None:
    """Reproduce litellm's OpenAI status -> exception mapping.

    Mini-SWE retries on everything except ``LitellmModel.abort_exceptions``, so
    which class is raised decides whether a turn is retried up to 10 times or
    fails the case immediately.
    """
    from litellm import exceptions as E

    provider = "openai"
    detail = text[:500]
    response = _http_response(status, detail, url)
    if status == 400:
        lowered = detail.lower()
        if "context" in lowered and ("length" in lowered or "window" in lowered):
            raise E.ContextWindowExceededError(
                message=detail, model=model, llm_provider=provider, response=response
            )
        raise E.BadRequestError(
            message=detail, model=model, llm_provider=provider, response=response
        )
    if status == 401:
        raise E.AuthenticationError(
            message=detail, model=model, llm_provider=provider, response=response
        )
    if status == 403:
        raise E.PermissionDeniedError(
            message=detail, model=model, llm_provider=provider, response=response
        )
    if status == 404:
        raise E.NotFoundError(
            message=detail, model=model, llm_provider=provider, response=response
        )
    if status == 429:
        raise E.RateLimitError(
            message=detail, model=model, llm_provider=provider, response=response
        )
    if status == 503:
        raise E.ServiceUnavailableError(
            message=detail, model=model, llm_provider=provider, response=response
        )
    if status >= 500:
        raise E.InternalServerError(
            message=detail, model=model, llm_provider=provider, response=response
        )
    if status >= 400:
        raise E.APIError(
            status_code=status,
            message=detail,
            model=model,
            llm_provider=provider,
            request=response.request,
        )


def _to_model_response(payload: dict):
    """Build the exact object litellm hands back for an OpenAI chat completion.

    litellm runs the decoded body through ``convert_to_model_response_object``,
    which fills in fields a bare ``ModelResponse(**payload)`` leaves out
    (``provider_specific_fields``, ``service_tier``, ``moderation``).  Those
    are echoed back to the adapter inside the assistant turn, so reproducing
    them keeps the wire history byte-identical.
    """
    from litellm.types.utils import ModelResponse

    try:
        from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (  # noqa: E501
            convert_to_model_response_object,
        )
    except ImportError:  # pragma: no cover - litellm layout changed
        return ModelResponse(**payload)
    return convert_to_model_response_object(
        response_object=payload,
        model_response_object=ModelResponse(),
        stream=False,
    )


def build_fast_model_class():
    """Build the subclass lazily so importing this module stays cheap."""
    from minisweagent.models.litellm_model import LitellmModel
    from minisweagent.models.utils.actions_toolcall import BASH_TOOL

    class FastOpenAIModel(LitellmModel):
        """``LitellmModel`` with the request path replaced by a plain POST."""

        def _query(self, messages: list[dict[str, Any]], **kwargs):
            import requests
            from litellm import exceptions as E

            params = self.config.model_kwargs | kwargs
            api_base = str(params.get("api_base") or "").rstrip("/")
            if not api_base:
                raise E.BadRequestError(
                    message="fast model client requires api_base",
                    model=self.config.model_name,
                    llm_provider="openai",
                )

            body: dict[str, Any] = {
                key: value
                for key, value in params.items()
                if key not in _TRANSPORT_KEYS and value is not None
            }
            body["model"] = _wire_model_name(self.config.model_name)
            body["messages"] = messages
            body["tools"] = [BASH_TOOL]

            headers = {"Content-Type": "application/json"}
            api_key = params.get("api_key")
            if api_key:
                # the adapter resolves the session id from this bearer token
                headers["Authorization"] = f"Bearer {api_key}"

            url = f"{api_base}/chat/completions"
            try:
                response = _session().post(
                    url,
                    data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                    headers=headers,
                    timeout=params.get("timeout"),
                )
            except requests.exceptions.Timeout as exc:
                raise E.Timeout(
                    message=f"request timed out: {exc}",
                    model=self.config.model_name,
                    llm_provider="openai",
                ) from exc
            except requests.exceptions.RequestException as exc:
                raise E.APIConnectionError(
                    message=f"{type(exc).__name__}: {exc}",
                    model=self.config.model_name,
                    llm_provider="openai",
                ) from exc

            if response.status_code >= 400:
                _raise_for_status(
                    response.status_code,
                    response.text,
                    self.config.model_name,
                    url,
                )

            try:
                payload = response.json()
            except ValueError as exc:
                raise E.APIError(
                    status_code=response.status_code,
                    message=f"non-JSON response: {response.text[:500]}",
                    model=self.config.model_name,
                    llm_provider="openai",
                    request=_http_response(
                        response.status_code, response.text[:500], url
                    ).request,
                ) from exc

            # Same object litellm hands back, so every downstream consumer
            # (_parse_actions, model_dump, usage, finish_reason) is unchanged.
            return _to_model_response(payload)

    return FastOpenAIModel


def fast_client_enabled() -> bool:
    return os.environ.get("R2E_FAST_MODEL_CLIENT", "1").strip().lower() not in {
        "0",
        "false",
        "no",
    }


_announced = False


def _announce(client: str) -> None:
    """Say once per process which client is live, so a run log can prove it."""
    global _announced
    if _announced:
        return
    _announced = True
    import logging

    logging.getLogger(__name__).warning("[r2e_actor_client] using %s", client)


def get_actor_model(model_name: str, config: dict[str, Any]):
    """Return the actor model, preferring the low-overhead client.

    Falls back to Mini-SWE's ``get_model`` when disabled or when the fast
    subclass cannot be built, so a litellm/Mini-SWE upgrade can never take the
    rollout down.
    """
    if fast_client_enabled():
        try:
            # get_model() deep-copies the config before constructing; match it
            # so callers can keep reusing their config dict.
            model = build_fast_model_class()(
                model_name=model_name, **copy.deepcopy(config)
            )
            _announce("FastOpenAIModel")
            return model
        except Exception:  # pragma: no cover - defensive fallback
            import logging

            logging.getLogger(__name__).exception(
                "fast model client unavailable; falling back to LitellmModel"
            )
    from minisweagent.models import get_model

    _announce("LitellmModel")
    return get_model(model_name, config)

