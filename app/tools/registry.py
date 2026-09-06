"""Tool registry: maps tool names → argument schema + DataForSEO endpoint, and validates
LLM-proposed arguments without ever raising on malformed input."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError

from app.tools.schemas import KeywordVolumeInput, LlmResponsesInput, SerpOrganicInput


@dataclass(frozen=True)
class ToolSpec:
    name: str
    input_model: type[BaseModel]
    endpoint_path: str
    description: str


TOOLS: dict[str, ToolSpec] = {
    "serp_organic": ToolSpec(
        name="serp_organic",
        input_model=SerpOrganicInput,
        endpoint_path="/v3/serp/google/organic/live/advanced",
        description=(SerpOrganicInput.__doc__ or "").strip(),
    ),
    "llm_responses": ToolSpec(
        name="llm_responses",
        input_model=LlmResponsesInput,
        endpoint_path="/v3/ai_optimization/chat_gpt/llm_responses/live",
        description=(LlmResponsesInput.__doc__ or "").strip(),
    ),
    "keyword_volume": ToolSpec(
        name="keyword_volume",
        input_model=KeywordVolumeInput,
        endpoint_path="/v3/keywords_data/google_ads/search_volume/live",
        description=(KeywordVolumeInput.__doc__ or "").strip(),
    ),
}


def get_tool(name: str) -> ToolSpec:
    try:
        return TOOLS[name]
    except KeyError as exc:
        raise KeyError(f"unknown tool '{name}'; known: {sorted(TOOLS)}") from exc


def to_openai_tools(names: list[str] | None = None) -> list[dict[str, Any]]:
    """OpenAI-format tool definitions (LiteLLM normalises this shape for every provider)."""
    specs = [TOOLS[n] for n in names] if names else list(TOOLS.values())
    return [
        {
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.input_model.model_json_schema(),
            },
        }
        for spec in specs
    ]


@dataclass
class ValidationOutcome:
    tool_name: str
    ok: bool
    input: BaseModel | None = None
    errors: list[str] = field(default_factory=list)

    @property
    def spec(self) -> ToolSpec | None:
        return TOOLS.get(self.tool_name)


def _coerce_arguments(raw: Any) -> tuple[dict[str, Any] | None, str | None]:
    if raw is None:
        return None, "arguments missing"
    if isinstance(raw, dict):
        return raw, None
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None, "arguments empty"
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            return None, f"arguments are not valid JSON: {exc.msg} at pos {exc.pos}"
        if not isinstance(parsed, dict):
            return None, f"arguments must be a JSON object, got {type(parsed).__name__}"
        return parsed, None
    return None, f"arguments have unsupported type {type(raw).__name__}"


def validate_tool_call(
    tool_name: str, raw_arguments: Any, *, defaults: dict[str, Any] | None = None
) -> ValidationOutcome:
    """Validate a proposed tool call. Never raises.

    `defaults` are applied for *missing* keys only (e.g. location_code / language_code from the
    profile), so a partially-specified call from the LLM can still be executed safely.
    """
    spec = TOOLS.get(tool_name)
    if spec is None:
        return ValidationOutcome(tool_name=tool_name, ok=False, errors=[f"unknown tool '{tool_name}'"])

    args, err = _coerce_arguments(raw_arguments)
    if err:
        return ValidationOutcome(tool_name=tool_name, ok=False, errors=[err])
    assert args is not None

    merged = dict(args)
    for key, value in (defaults or {}).items():
        if key in spec.input_model.model_fields and key not in merged:
            merged[key] = value

    try:
        model = spec.input_model.model_validate(merged)
    except ValidationError as exc:
        messages = [f"{'.'.join(str(p) for p in e['loc']) or '<root>'}: {e['msg']}" for e in exc.errors()]
        return ValidationOutcome(tool_name=tool_name, ok=False, errors=messages)
    return ValidationOutcome(tool_name=tool_name, ok=True, input=model)
