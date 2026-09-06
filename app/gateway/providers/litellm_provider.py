"""LiteLLM adapter. `litellm` is imported lazily: it is slow to import and tests never need it."""

from __future__ import annotations

import json
import os
from typing import Any

from app.gateway.errors import ProviderNonRetryable, ProviderRetryable
from app.gateway.types import LLMRequest, ProviderResponse
from app.tools.schemas import ToolCallProposal


def _parse_tool_calls(message: Any) -> list[ToolCallProposal]:
    proposals: list[ToolCallProposal] = []
    for idx, call in enumerate(getattr(message, "tool_calls", None) or []):
        fn = getattr(call, "function", None)
        name = getattr(fn, "name", None) or ""
        raw = getattr(fn, "arguments", None)
        raw_str = raw if isinstance(raw, str) else json.dumps(raw or {})
        args: dict | None = None
        err: str | None = None
        try:
            parsed = json.loads(raw_str) if raw_str.strip() else {}
            if isinstance(parsed, dict):
                args = parsed
            else:
                err = f"arguments must be a JSON object, got {type(parsed).__name__}"
        except json.JSONDecodeError as exc:
            err = f"arguments are not valid JSON: {exc.msg}"
        proposals.append(
            ToolCallProposal(
                id=getattr(call, "id", None) or f"call_{idx}",
                name=name,
                raw_arguments=raw_str,
                arguments=args,
                parse_error=err,
            )
        )
    return proposals


class LiteLLMProvider:
    def __init__(self, api_keys: dict[str, str | None] | None = None):
        os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
        import litellm  # noqa: PLC0415 – deliberate lazy import

        litellm.suppress_debug_info = True
        litellm.drop_params = True  # silently drop params a provider doesn't support
        litellm.num_retries = 0  # the gateway owns retries
        self._litellm = litellm
        for provider, key in (api_keys or {}).items():
            if key:
                os.environ.setdefault(
                    {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "gemini": "GEMINI_API_KEY"}.get(
                        provider, f"{provider.upper()}_API_KEY"
                    ),
                    key,
                )

    def supports_response_schema(self, model: str) -> bool:
        try:
            return bool(self._litellm.supports_response_schema(model=model))
        except Exception:  # unknown model in cost map → be conservative
            return False

    def complete(self, request: LLMRequest) -> ProviderResponse:
        lt = self._litellm
        kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": request.messages,
            "temperature": request.temperature,
            "timeout": request.timeout_s,
        }
        if request.max_tokens:
            kwargs["max_tokens"] = request.max_tokens
        if request.tools:
            kwargs["tools"] = request.tools
            kwargs["tool_choice"] = request.tool_choice or "auto"
        if request.response_format is not None:
            kwargs["response_format"] = request.response_format

        try:
            resp = lt.completion(**kwargs)
        except (
            lt.RateLimitError,
            lt.ServiceUnavailableError,
            lt.InternalServerError,
            lt.APIConnectionError,
            lt.Timeout,
        ) as exc:
            raise ProviderRetryable(
                f"{type(exc).__name__}: {exc}", http_status=getattr(exc, "status_code", None)
            ) from exc
        except (
            lt.AuthenticationError,
            lt.PermissionDeniedError,
            lt.BadRequestError,
            lt.NotFoundError,
            lt.ContentPolicyViolationError,
            lt.ContextWindowExceededError,
            lt.UnprocessableEntityError,
        ) as exc:
            raise ProviderNonRetryable(
                f"{type(exc).__name__}: {exc}", http_status=getattr(exc, "status_code", None)
            ) from exc
        except lt.APIError as exc:  # generic: classify by status code if present
            status = getattr(exc, "status_code", None)
            if status and (status == 429 or status >= 500):
                raise ProviderRetryable(f"APIError {status}: {exc}", http_status=status) from exc
            raise ProviderNonRetryable(f"APIError {status}: {exc}", http_status=status) from exc

        choice = resp.choices[0]
        message = choice.message
        usage = getattr(resp, "usage", None)
        cost: float | None
        try:
            cost = float(lt.completion_cost(completion_response=resp))
        except Exception:
            cost = None
        return ProviderResponse(
            text=getattr(message, "content", None),
            tool_calls=_parse_tool_calls(message),
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            total_tokens=int(getattr(usage, "total_tokens", 0) or 0),
            cost_usd=cost,
            model=getattr(resp, "model", request.model) or request.model,
            finish_reason=getattr(choice, "finish_reason", None),
        )
