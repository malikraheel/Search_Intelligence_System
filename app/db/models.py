"""Persistence model: profiles → runs → (queries, recommendations, node executions)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.domain.models import new_id, now_utc


class Profile(Base):
    __tablename__ = "profiles"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    domain: Mapped[str] = mapped_column(String(253), nullable=False, index=True)
    industry: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    competitors: Mapped[list[str]] = mapped_column(JSON, default=list)
    status: Mapped[str] = mapped_column(String(20), default="created")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc, onupdate=now_utc)

    runs: Mapped[list[PipelineRun]] = relationship(back_populates="profile", cascade="all, delete-orphan")
    queries: Mapped[list[Query]] = relationship(back_populates="profile", cascade="all, delete-orphan")


class PipelineRun(Base):
    __tablename__ = "pipeline_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    profile_id: Mapped[str] = mapped_column(ForeignKey("profiles.id", ondelete="CASCADE"), index=True)
    mode: Mapped[str] = mapped_column(String(10), default="full")  # full | recheck
    status: Mapped[str] = mapped_column(String(12), default="running")  # running|completed|partial|failed
    question: Mapped[str] = mapped_column(Text, default="")
    recheck_query_id: Mapped[str | None] = mapped_column(String(36), nullable=True)

    planned_calls_count: Mapped[int] = mapped_column(Integer, default=0)
    records_extracted_count: Mapped[int] = mapped_column(Integer, default=0)
    api_calls_count: Mapped[int] = mapped_column(Integer, default=0)
    llm_calls_count: Mapped[int] = mapped_column(Integer, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_cost_usd: Mapped[float] = mapped_column(Float, default=0.0)

    report_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    degradation: Mapped[list[str]] = mapped_column(JSON, default=list)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

    profile: Mapped[Profile] = relationship(back_populates="runs")
    node_executions: Mapped[list[NodeExecution]] = relationship(back_populates="run", cascade="all, delete-orphan")
    recommendations: Mapped[list[Recommendation]] = relationship(back_populates="run", cascade="all, delete-orphan")

    __table_args__ = (Index("ix_runs_profile_started", "profile_id", "started_at"),)


class Query(Base):
    __tablename__ = "queries"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    profile_id: Mapped[str] = mapped_column(ForeignKey("profiles.id", ondelete="CASCADE"), index=True)
    last_run_id: Mapped[str] = mapped_column(ForeignKey("pipeline_runs.id", ondelete="SET NULL"), index=True)

    query_text: Mapped[str] = mapped_column(String(200), nullable=False)
    intent: Mapped[str | None] = mapped_column(String(20), nullable=True)
    estimated_search_volume: Mapped[int | None] = mapped_column(Integer, nullable=True)
    competition_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    competitive_difficulty: Mapped[int] = mapped_column(Integer, default=50)
    opportunity_score: Mapped[float] = mapped_column(Float, default=0.0, index=True)
    score_components: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    domain_visible: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    visibility_status: Mapped[str] = mapped_column(String(12), default="unknown", index=True)
    visibility_position: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ai_overview_present: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    ai_cited: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    competitor_positions: Mapped[dict[str, int]] = mapped_column(JSON, default=dict)
    competitors_cited: Mapped[list[str]] = mapped_column(JSON, default=list)
    top_domains: Mapped[list[str]] = mapped_column(JSON, default=list)
    sources: Mapped[list[str]] = mapped_column(JSON, default=list)

    error_flag: Mapped[bool] = mapped_column(Boolean, default=False)
    error_detail: Mapped[str | None] = mapped_column(Text, nullable=True)

    discovered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)
    last_checked_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)

    profile: Mapped[Profile] = relationship(back_populates="queries")
    recommendations: Mapped[list[Recommendation]] = relationship(back_populates="target_query")

    __table_args__ = (UniqueConstraint("profile_id", "query_text", name="uq_query_per_profile"),)


class Recommendation(Base):
    __tablename__ = "recommendations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    profile_id: Mapped[str] = mapped_column(ForeignKey("profiles.id", ondelete="CASCADE"), index=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("pipeline_runs.id", ondelete="CASCADE"), index=True)
    target_query_id: Mapped[str | None] = mapped_column(ForeignKey("queries.id", ondelete="SET NULL"), nullable=True)

    content_type: Mapped[str] = mapped_column(String(30), nullable=False)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    rationale: Mapped[str] = mapped_column(Text, default="")
    target_keywords: Mapped[list[str]] = mapped_column(JSON, default=list)
    priority: Mapped[str] = mapped_column(String(10), default="medium")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)

    run: Mapped[PipelineRun] = relationship(back_populates="recommendations")
    target_query: Mapped[Query | None] = relationship(back_populates="recommendations")


class NodeExecution(Base):
    """One row per node execution → the persisted trace of a run."""

    __tablename__ = "node_executions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    run_id: Mapped[str] = mapped_column(ForeignKey("pipeline_runs.id", ondelete="CASCADE"), index=True)
    seq: Mapped[int] = mapped_column(Integer, default=0)
    node_name: Mapped[str] = mapped_column(String(60), nullable=False)
    branch_key: Mapped[str | None] = mapped_column(String(16), nullable=True)
    status: Mapped[str] = mapped_column(String(12), nullable=False)
    duration_ms: Mapped[float] = mapped_column(Float, default=0.0)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    api_calls: Mapped[int] = mapped_column(Integer, default=0)
    tokens: Mapped[int] = mapped_column(Integer, default=0)
    input_summary: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    output_summary: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)

    run: Mapped[PipelineRun] = relationship(back_populates="node_executions")
