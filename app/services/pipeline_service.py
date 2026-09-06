"""Runs the DAG for a profile (or a single query) and persists everything afterwards.

Graph nodes never touch the database: the service builds the initial state, invokes the compiled
graph synchronously, then writes run / queries / recommendations / node traces in one transaction.
"""

from __future__ import annotations

import time
from typing import Any

from langgraph.graph.state import CompiledStateGraph
from sqlalchemy.orm import Session

from app.db import repositories as repo
from app.db.models import PipelineRun, Profile, Query
from app.domain.models import ProfileCtx
from app.graph.builder import default_question, initial_state, invoke_graph
from app.graph.deps import Deps
from app.graph.state import RunState
from app.observability.logging import bind_context, get_logger

log = get_logger("pipeline")


def profile_ctx(profile: Profile) -> ProfileCtx:
    return ProfileCtx(
        profile_uuid=profile.id,
        name=profile.name,
        domain=profile.domain,
        industry=profile.industry,
        description=profile.description or "",
        competitors=list(profile.competitors or []),
    )


class PipelineService:
    def __init__(self, deps: Deps, graph: CompiledStateGraph):
        self.deps = deps
        self.graph = graph

    # ── public ────────────────────────────────────────────────────────────────
    def run_profile(self, session: Session, profile: Profile, question: str | None = None) -> PipelineRun:
        ctx = profile_ctx(profile)
        q = (question or "").strip() or default_question(ctx)
        run = repo.create_run(session, profile.id, mode="full", question=q)
        session.commit()
        state = initial_state(ctx, mode="full", question=q, run_id=run.id)
        return self._execute(session, run, state)

    def recheck_query(self, session: Session, profile: Profile, query: Query) -> PipelineRun:
        ctx = profile_ctx(profile)
        q = f"Re-check visibility of {ctx.name} for '{query.query_text}'."
        run = repo.create_run(session, profile.id, mode="recheck", question=q, recheck_query_id=query.id)
        session.commit()
        state = initial_state(ctx, mode="recheck", question=q, recheck_query=query.query_text, run_id=run.id)
        return self._execute(session, run, state, only_query_texts={query.query_text})

    # ── internals ─────────────────────────────────────────────────────────────
    def _execute(
        self, session: Session, run: PipelineRun, state: RunState, *, only_query_texts: set[str] | None = None
    ) -> PipelineRun:
        settings = self.deps.settings
        started = time.perf_counter()
        with bind_context(run_id=run.id, profile_uuid=run.profile_id, mode=run.mode):
            log.info("run.start", question=state["question"])
            try:
                final = invoke_graph(
                    self.graph,
                    state,
                    recursion_limit=settings.pipeline_recursion_limit,
                    max_concurrency=settings.pipeline_max_concurrency,
                )
            except Exception as exc:  # unexpected bug/infra failure → persist as failed, never leave 'running'
                log.error("run.crashed", error=repr(exc), duration_ms=round((time.perf_counter() - started) * 1000, 1))
                repo.finish_run(
                    session,
                    run,
                    status="failed",
                    planned_calls=0,
                    records=0,
                    api_calls=0,
                    llm_calls=0,
                    total_tokens=0,
                    total_cost_usd=0.0,
                    report_json=None,
                    summary=None,
                    degradation=["unexpected_exception"],
                    error=f"{type(exc).__name__}: {exc}",
                )
                session.commit()
                self.deps.metrics.record_run("failed")
                return run

            self._persist(session, run, final, only_query_texts)
            session.commit()
            self.deps.metrics.record_run(run.status)
            log.info(
                "run.summary",
                status=run.status,
                planned_calls=run.planned_calls_count,
                records=run.records_extracted_count,
                api_calls=run.api_calls_count,
                llm_calls=run.llm_calls_count,
                total_tokens=run.total_tokens,
                cost_usd=run.total_cost_usd,
                degradation=run.degradation,
                duration_ms=run.duration_ms,
                metrics=self.deps.metrics.snapshot(),
            )
            return run

    def _persist(self, session: Session, run: PipelineRun, final: RunState, only_query_texts: set[str] | None) -> None:
        report = final.get("report")
        usage = final.get("llm_usage", [])
        records = final.get("records", [])
        repo.finish_run(
            session,
            run,
            status=final.get("status", "failed"),
            planned_calls=len(final.get("planned_calls", [])),
            records=len(records),
            api_calls=len(final.get("api_calls", [])),
            llm_calls=len(usage),
            total_tokens=sum(u.total_tokens for u in usage),
            total_cost_usd=round(sum(u.cost_usd for u in usage), 6),
            report_json=report.report_json if report else None,
            summary=report.summary if report else None,
            degradation=list(
                dict.fromkeys(
                    final.get("degradation", []) + (report.report_json.get("degradation", []) if report else [])
                )
            ),
        )
        query_map = repo.upsert_queries(session, run.profile_id, run.id, records)
        analysis = final.get("analysis")
        if analysis is not None:
            repo.replace_recommendations(
                session, run.profile_id, run.id, analysis.recommendations, query_map, only_query_texts=only_query_texts
            )
        repo.save_node_executions(session, run.id, final.get("node_traces", []))


def run_to_dict(run: PipelineRun) -> dict[str, Any]:
    """Shape shared by /run, /recheck and GET /runs/{id}."""
    report = run.report_json or {}
    analysis = report.get("analysis") or {}
    insights = sorted(
        analysis.get("insights", []),
        key=lambda i: i.get("relevance_score", 0) + i.get("opportunity_score", 0),
        reverse=True,
    )
    return {
        "run_uuid": run.id,
        "profile_uuid": run.profile_id,
        "mode": run.mode,
        "status": run.status,
        "question": run.question,
        "planned_calls_count": run.planned_calls_count,
        "records_extracted_count": run.records_extracted_count,
        "api_calls_count": run.api_calls_count,
        "llm_calls_count": run.llm_calls_count,
        "total_tokens": run.total_tokens,
        "total_cost_usd": run.total_cost_usd,
        "top_insights": insights[:5],
        "recommendations_count": len(analysis.get("recommendations", [])),
        "report": {"json": report, "summary": run.summary or ""} if run.report_json else None,
        "degradation": run.degradation or [],
        "error": run.error,
        "started_at": repo.as_utc(run.started_at),
        "finished_at": repo.as_utc(run.finished_at),
        "duration_ms": run.duration_ms,
    }
