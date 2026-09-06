"""Pipeline data contracts shared between graph nodes (Pydantic v2)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


def now_utc() -> datetime:
    return datetime.now(UTC)


def new_id() -> str:
    return str(uuid4())


# ── profile context (input) ─────────────────────────────────────────────────────
class ProfileCtx(BaseModel):
    profile_uuid: str
    name: str
    domain: str
    industry: str
    description: str = ""
    competitors: list[str] = Field(default_factory=list)


# ── planning ───────────────────────────────────────────────────────────────────
Intent = Literal["brand", "category", "comparison", "informational", "transactional"]


class SubQuery(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    query_text: str = Field(min_length=3, max_length=200, description="Search query a real user would type.")
    intent: Intent = Field(description="Why a user searches this.")
    priority: int = Field(default=3, ge=1, le=5, description="1 = highest business priority.")
    wants_ai_check: bool = Field(default=True, description="Also check AI-answer visibility for this query.")


class RetrievalPlan(BaseModel):
    """Output of the Query Planner: which searches are needed to answer the question."""

    sub_queries: list[SubQuery] = Field(min_length=1, max_length=8)
    rationale: str = Field(default="", max_length=1000)


ToolName = Literal["serp_organic", "llm_responses", "keyword_volume"]


class PlannedCall(BaseModel):
    """One retrieval unit dispatched to a retrieval branch."""

    call_id: str = Field(default_factory=new_id)
    tool: ToolName
    query_text: str  # for keyword_volume this is a label; keywords carries the batch
    intent: Intent | None = None
    keywords: list[str] = Field(default_factory=list)


# ── retrieval results ──────────────────────────────────────────────────────────
class ErrorInfo(BaseModel):
    kind: str
    message: str
    http_status: int | None = None
    api_status_code: int | None = None
    retryable: bool = False


class RawResult(BaseModel):
    call_id: str
    tool: ToolName
    query_text: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    status: Literal["ok", "degraded"]
    payload: dict[str, Any] | None = None
    error: ErrorInfo | None = None
    latency_ms: float = 0.0
    retry_count: int = 0
    fallback_reason: str | None = None


class ApiCallRecord(BaseModel):
    tool: str
    outcome: str  # ok | retryable_error | non_retryable_error | breaker_open
    latency_ms: float
    attempts: int


# ── observability records carried in state ─────────────────────────────────────
class LLMUsage(BaseModel):
    role: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    cost_unknown: bool = False
    cached: bool = False


class NodeTrace(BaseModel):
    node: str
    branch_key: str | None = None
    status: Literal["success", "failure", "degraded"]
    duration_ms: float
    retry_count: int = 0
    api_calls: int = 0
    tokens: int = 0
    input_summary: dict[str, Any] = Field(default_factory=dict)
    output_summary: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    started_at: datetime = Field(default_factory=now_utc)


class RunError(BaseModel):
    node: str
    kind: str
    message: str
    query_text: str | None = None


# ── normalized data ────────────────────────────────────────────────────────────
VisibilityStatus = Literal["visible", "not_visible", "unknown"]


class ScoreComponents(BaseModel):
    vol_norm: float
    difficulty_norm: float
    visibility_gap: float
    ai_gap: float


class QueryRecord(BaseModel):
    """Clean, per-query record produced by the Extraction/Normalization agent."""

    query_text: str
    intent: Intent | None = None
    estimated_search_volume: int | None = None
    competition_index: int | None = None
    competitive_difficulty: int = Field(ge=0, le=100)
    opportunity_score: float = Field(ge=0.0, le=1.0)
    score_components: ScoreComponents | None = None
    domain_visible: bool | None = None
    visibility_status: VisibilityStatus = "unknown"
    visibility_position: int | None = None
    ai_overview_present: bool | None = None
    ai_cited: bool | None = None
    competitor_positions: dict[str, int] = Field(default_factory=dict)
    competitors_cited: list[str] = Field(default_factory=list)
    top_domains: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)  # tools that contributed
    error_flags: list[str] = Field(default_factory=list)
    checked_at: datetime = Field(default_factory=now_utc)


# ── analysis ───────────────────────────────────────────────────────────────────
class Insight(BaseModel):
    query_text: str = Field(description="The query this insight is about (must match a record).")
    headline: str = Field(max_length=200)
    evidence: str = Field(max_length=800, description="Concrete data points backing the insight.")
    relevance_score: float = Field(ge=0.0, le=1.0)
    opportunity_score: float = Field(default=0.0, ge=0.0, le=1.0)


ContentType = Literal["blog_post", "landing_page", "comparison", "faq", "guide", "video"]
Priority = Literal["high", "medium", "low"]


class RecommendationDraft(BaseModel):
    target_query_text: str = Field(description="The query this content targets (must match a record).")
    content_type: ContentType
    title: str = Field(max_length=200)
    rationale: str = Field(max_length=800)
    target_keywords: list[str] = Field(min_length=1, max_length=8)
    priority: Priority


class AnalysisResult(BaseModel):
    executive_summary: str = Field(max_length=1200)
    insights: list[Insight] = Field(min_length=1, max_length=12)
    recommendations: list[RecommendationDraft] = Field(default_factory=list, max_length=12)


class Report(BaseModel):
    report_json: dict[str, Any]
    summary: str
