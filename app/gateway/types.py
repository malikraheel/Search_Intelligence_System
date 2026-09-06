from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from app.domain.models import LLMUsage
from app.tools.schemas import ToolCallProposal


class Role(StrEnum):
    PLANNER = "planner"
    RETRIEVAL = "retrieval"
    ANALYST = "analyst"
    REPORTER = "reporter"


@dataclass
class LLMRequest:
    """Provider-agnostic request (OpenAI wire shape, which LiteLLM normalises for every provider)."""

    model: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    response_format: Any | None = None  # Pydantic class, {"type": "json_object"}, or None
    temperature: float = 0.0
    max_tokens: int | None = None
    timeout_s: float = 30.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderResponse:
    """What a provider adapter returns. Token counts are 0 when the provider omits them."""

    text: str | None
    tool_calls: list[ToolCallProposal] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float | None = None
    model: str = ""
    finish_reason: str | None = None


class LLMResponse(BaseModel):
    text: str | None = None
    tool_calls: list[ToolCallProposal] = Field(default_factory=list)
    parsed: Any | None = None
    usage: LLMUsage
    model: str
    attempts: int = 1
    fallback_used: bool = False
    cached: bool = False
    latency_ms: float = 0.0
