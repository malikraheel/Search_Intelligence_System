"""Tool-argument schemas: one Pydantic model per DataForSEO endpoint.

These are the JSON schemas the LLM sees (via `model_json_schema()`) AND the validators the
code runs on the LLM's proposed arguments before any real API call is made.
`extra="forbid"` catches hallucinated fields; bounds catch out-of-range values.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SerpOrganicInput(BaseModel):
    """Fetch Google organic search results (live, advanced) for ONE keyword, including any
    AI Overview block and its cited sources. Use this to find where a domain ranks."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    keyword: str = Field(min_length=1, max_length=200, description="The exact search query to look up.")
    location_code: int = Field(default=2840, ge=1, description="DataForSEO location code. 2840 = United States.")
    language_code: str = Field(default="en", pattern=r"^[a-z]{2}$", description="ISO 639-1 language code.")
    device: Literal["desktop", "mobile"] = Field(default="desktop", description="Device type for the SERP.")
    depth: int = Field(default=10, ge=10, le=100, multiple_of=10, description="Number of results (10–100, step 10).")


class LlmResponsesInput(BaseModel):
    """Ask an LLM (via DataForSEO AI Optimization) a prompt and capture the answer plus the
    web sources it cites. Use this to check whether a domain is mentioned/cited in AI answers."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    user_prompt: str = Field(min_length=1, max_length=500, description="Natural-language prompt to send to the LLM.")
    model_name: str = Field(default="gpt-4o-mini", min_length=1, description="LLM model to query.")
    web_search: bool = Field(default=True, description="Allow the LLM to use web search (needed for citations).")
    max_output_tokens: int = Field(default=1024, ge=16, le=4096, description="Max answer length in tokens.")


class KeywordVolumeInput(BaseModel):
    """Fetch Google Ads monthly search volume and competition index for a BATCH of keywords.
    Call this once with all keywords rather than once per keyword."""

    model_config = ConfigDict(extra="forbid")

    keywords: list[str] = Field(min_length=1, max_length=1000, description="Keywords to look up (1–1000).")
    location_code: int = Field(default=2840, ge=1, description="DataForSEO location code. 2840 = United States.")
    language_code: str = Field(default="en", pattern=r"^[a-z]{2}$", description="ISO 639-1 language code.")

    @field_validator("keywords")
    @classmethod
    def _clean_keywords(cls, value: list[str]) -> list[str]:
        seen: set[str] = set()
        cleaned: list[str] = []
        for raw in value:
            if not isinstance(raw, str):
                raise ValueError("keywords must be strings")
            kw = " ".join(raw.split())
            if not kw:
                raise ValueError("keywords must not be empty")
            if len(kw) > 80:
                raise ValueError(f"keyword too long (>80 chars): {kw[:30]}…")
            key = kw.lower()
            if key not in seen:
                seen.add(key)
                cleaned.append(kw)
        return cleaned


class ToolCallProposal(BaseModel):
    """A tool call as proposed by the LLM — NOT yet validated. `arguments` may be None when the
    model emitted non-JSON; `parse_error` then explains why."""

    id: str
    name: str
    raw_arguments: str
    arguments: dict | None = None
    parse_error: str | None = None
