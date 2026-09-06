"""Request/response models for the REST API."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.dataforseo.domains import normalize_domain

_DOMAIN_RE = re.compile(r"^(?=.{1,253}$)([a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}$")


def _validate_domain(value: str) -> str:
    dom = normalize_domain(value)
    if not _DOMAIN_RE.match(dom):
        raise ValueError(f"'{value}' is not a valid domain name")
    return dom


# ── profiles ──────────────────────────────────────────────────────────────────
class ProfileCreate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    name: str = Field(min_length=1, max_length=200, examples=["Surfer SEO"])
    domain: str = Field(min_length=3, max_length=253, examples=["surferseo.com"])
    industry: str = Field(min_length=1, max_length=200, examples=["SEO Software"])
    description: str = Field(default="", max_length=2000)
    competitors: list[str] = Field(default_factory=list, max_length=20)

    @field_validator("domain")
    @classmethod
    def _domain(cls, v: str) -> str:
        return _validate_domain(v)

    @field_validator("competitors")
    @classmethod
    def _competitors(cls, v: list[str]) -> list[str]:
        out: list[str] = []
        for c in v:
            dom = _validate_domain(c)
            if dom not in out:
                out.append(dom)
        return out


class ProfileOut(BaseModel):
    profile_uuid: str
    name: str
    domain: str
    industry: str
    description: str
    competitors: list[str]
    status: str
    created_at: datetime


class ProfileStats(BaseModel):
    total_runs: int
    last_run_uuid: str | None
    last_run_status: str | None
    last_run_at: datetime | None
    average_opportunity_score: float | None
    tracked_queries: int
    visible_queries: int


class ProfileDetailOut(ProfileOut):
    stats: ProfileStats


# ── runs ──────────────────────────────────────────────────────────────────────
class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str | None = Field(
        default=None, max_length=1000, description="Optional natural-language research question."
    )


class ReportOut(BaseModel):
    json_: dict[str, Any] = Field(alias="json")
    summary: str
    model_config = ConfigDict(populate_by_name=True)


class RunOut(BaseModel):
    run_uuid: str
    profile_uuid: str
    mode: Literal["full", "recheck"]
    status: Literal["running", "completed", "partial", "failed"]
    question: str
    planned_calls_count: int
    records_extracted_count: int
    api_calls_count: int
    llm_calls_count: int
    total_tokens: int
    total_cost_usd: float
    top_insights: list[dict[str, Any]]
    recommendations_count: int
    report: ReportOut | None
    degradation: list[str]
    error: str | None
    started_at: datetime
    finished_at: datetime | None
    duration_ms: int | None


class NodeExecutionOut(BaseModel):
    seq: int
    node_name: str
    branch_key: str | None
    status: str
    duration_ms: float
    retry_count: int
    api_calls: int
    tokens: int
    input_summary: dict[str, Any]
    output_summary: dict[str, Any]
    error: str | None
    started_at: datetime


class RunTraceOut(BaseModel):
    run_uuid: str
    status: str
    nodes: list[NodeExecutionOut]


# ── queries ───────────────────────────────────────────────────────────────────
class QueryOut(BaseModel):
    query_uuid: str
    query_text: str
    intent: str | None
    estimated_search_volume: int | None
    competitive_difficulty: int
    opportunity_score: float
    score_components: dict[str, float] | None
    domain_visible: bool | None
    visibility_status: Literal["visible", "not_visible", "unknown"]
    visibility_position: int | None
    ai_overview_present: bool | None
    ai_cited: bool | None
    competitor_positions: dict[str, int]
    competitors_cited: list[str]
    error_flag: bool
    error_detail: str | None
    discovered_at: datetime
    last_checked_at: datetime
    last_run_uuid: str | None


class Page[T](BaseModel):
    items: list[T]
    page: int
    per_page: int
    total: int


# ── recommendations ───────────────────────────────────────────────────────────
class RecommendationOut(BaseModel):
    recommendation_uuid: str
    target_query_uuid: str | None
    target_query_text: str | None
    content_type: str
    title: str
    rationale: str
    target_keywords: list[str]
    priority: Literal["high", "medium", "low"]
    run_uuid: str
    created_at: datetime


class RecheckOut(BaseModel):
    run: RunOut
    query: QueryOut
    recommendations: list[RecommendationOut]


class ErrorOut(BaseModel):
    error: str
    detail: str | None = None
