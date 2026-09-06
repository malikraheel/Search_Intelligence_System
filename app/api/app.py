"""FastAPI application factory."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.engine import Engine
from starlette.middleware.base import BaseHTTPMiddleware

from app.api.routers import profiles, queries, runs
from app.config import Settings, get_settings
from app.dataforseo.client import build_client
from app.db.base import init_db, make_engine, make_session_factory
from app.gateway import build_gateway
from app.graph.builder import build_graph
from app.graph.deps import Deps
from app.observability.logging import bind_permanent, clear_context, configure_logging, get_logger
from app.observability.metrics import MetricsRegistry
from app.resilience.errors import AppError
from app.services.pipeline_service import PipelineService

log = get_logger("api")


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Bind a correlation id to every log line emitted while handling the request."""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        clear_context()
        bind_permanent(request_id=request_id, path=request.url.path, method=request.method)
        try:
            response = await call_next(request)
        finally:
            clear_context()
        response.headers["X-Request-ID"] = request_id
        return response


def build_deps(settings: Settings, metrics: MetricsRegistry | None = None) -> Deps:
    metrics = metrics or MetricsRegistry()
    return Deps(
        settings=settings,
        gateway=build_gateway(settings, metrics),
        dfs_client=build_client(settings, metrics),
        metrics=metrics,
    )


def create_app(settings: Settings | None = None, *, deps: Deps | None = None, engine: Engine | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_format)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        eng = engine or make_engine(settings.database_url)
        init_db(eng)
        app.state.engine = eng
        app.state.session_factory = make_session_factory(eng)
        app.state.deps = deps or build_deps(settings)
        app.state.graph = build_graph(app.state.deps)
        app.state.pipeline = PipelineService(app.state.deps, app.state.graph)
        log.info(
            "app.start",
            dataforseo_mode=settings.dataforseo_mode,
            database=settings.database_url.split("@")[-1],
            planner_model=settings.gateway_planner_model,
            analyst_model=settings.gateway_analyst_model,
        )
        yield
        log.info("app.stop")

    app = FastAPI(
        title="Agentic Search Intelligence API",
        version="0.1.0",
        description="LangGraph DAG of single-responsibility agents answering brand AI/search visibility questions.",
        lifespan=lifespan,
    )
    app.add_middleware(RequestContextMiddleware)
    app.include_router(profiles.router)
    app.include_router(queries.router)
    app.include_router(runs.router)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422, content={"error": "validation_error", "detail": jsonable_encoder(exc.errors())}
        )

    @app.exception_handler(AppError)
    async def _app_error(_: Request, exc: AppError) -> JSONResponse:
        status = 503 if exc.retryable else 502
        log.error("api.upstream_error", error=exc.to_dict())
        return JSONResponse(status_code=status, content={"error": exc.kind, "detail": exc.message})

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
        log.error("api.unhandled", error=repr(exc))
        return JSONResponse(status_code=500, content={"error": "internal_error", "detail": type(exc).__name__})

    return app


def app_factory() -> FastAPI:  # for `uvicorn app.api.app:app_factory --factory`
    return create_app()
