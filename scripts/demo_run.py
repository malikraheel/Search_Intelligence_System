"""End-to-end demo without the HTTP server: create the sample profile, run the DAG, print the report.

Examples
    uv run scripts/demo_run.py                       # mock DataForSEO + real OpenAI (needs OPENAI_API_KEY)
    uv run scripts/demo_run.py --fake-llm            # fully offline (scripted LLM)
    uv run scripts/demo_run.py --chaos               # inject failures from DATAFORSEO_CHAOS_SCRIPT
    uv run scripts/demo_run.py --chaos "serp_organic:429,429,ok;llm_responses:401" --fake-llm
    uv run scripts/demo_run.py --recheck --json      # also recheck the top query; dump the full report JSON
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from app.api.app import build_deps  # noqa: E402
from app.config import Settings  # noqa: E402
from app.dataforseo.client import build_client  # noqa: E402
from app.db import repositories as repo  # noqa: E402
from app.db.base import init_db, make_engine, make_session_factory  # noqa: E402
from app.graph.builder import build_graph  # noqa: E402
from app.graph.deps import Deps  # noqa: E402
from app.observability.logging import configure_logging, get_logger  # noqa: E402
from app.observability.metrics import MetricsRegistry  # noqa: E402
from app.services.pipeline_service import PipelineService, run_to_dict  # noqa: E402

SAMPLE_PROFILE = {
    "name": "Surfer SEO",
    "domain": "surferseo.com",
    "industry": "SEO Software",
    "description": "AI-powered SEO content optimization tool",
    "competitors": ["clearscope.io", "marketmuse.com", "frase.io"],
}


def make_deps(settings: Settings, fake_llm: bool) -> Deps:
    metrics = MetricsRegistry()
    if not fake_llm:
        return build_deps(settings, metrics)
    from tests.fakes import make_fake_gateway  # scripted, role-aware fake LLM shipped with the test-suite

    gateway, _ = make_fake_gateway(metrics)
    return Deps(settings=settings, gateway=gateway, dfs_client=build_client(settings, metrics), metrics=metrics)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chaos", nargs="?", const="__default__", help="chaos mode; optional script overrides env")
    parser.add_argument("--fake-llm", action="store_true", help="use the scripted fake LLM (no API key needed)")
    parser.add_argument("--recheck", action="store_true", help="recheck the top-opportunity query afterwards")
    parser.add_argument("--json", action="store_true", help="print the full report JSON")
    parser.add_argument("--question", default=None)
    parser.add_argument("--db", default="sqlite:///./demo.db")
    parser.add_argument("--log-format", default=None, choices=["json", "console"])
    args = parser.parse_args()

    overrides: dict = {"database_url": args.db}
    if args.chaos:
        overrides["dataforseo_mode"] = "chaos"
        if args.chaos != "__default__":
            overrides["dataforseo_chaos_script"] = args.chaos
    if args.log_format:
        overrides["log_format"] = args.log_format
    settings = Settings().model_copy(update=overrides)
    configure_logging(settings.log_level, settings.log_format)
    log = get_logger("demo")

    fake_llm = args.fake_llm or not settings.openai_api_key
    if fake_llm and not args.fake_llm:
        log.warning("demo.no_openai_key", detail="OPENAI_API_KEY not set → using the scripted fake LLM")

    deps = make_deps(settings, fake_llm)
    engine = make_engine(settings.database_url)
    init_db(engine)
    session = make_session_factory(engine)()
    service = PipelineService(deps, build_graph(deps))

    profile = repo.create_profile(session, **SAMPLE_PROFILE)
    session.commit()
    log.info(
        "demo.profile_created", profile_uuid=profile.id, dataforseo_mode=settings.dataforseo_mode, fake_llm=fake_llm
    )

    run = service.run_profile(session, profile, question=args.question)
    out = run_to_dict(run)
    report = out.pop("report") or {}

    print("\n=== RUN ===")
    print(json.dumps({k: v for k, v in out.items() if k != "top_insights"}, indent=2, default=str))
    print("\n=== TOP INSIGHTS ===")
    for i in out["top_insights"]:
        print(f"- [{i.get('opportunity_score', 0):.2f}] {i['headline']} — {i['evidence'][:120]}")
    print("\n=== QUERIES (by opportunity) ===")
    rows, _ = repo.list_queries(session, profile.id, per_page=50)
    for q in rows:
        pos = q.visibility_position if q.visibility_position is not None else "-"
        flag = f"  ⚠ {q.error_detail}" if q.error_flag else ""
        print(
            f"- {q.opportunity_score:.2f}  {q.query_text:<40} vol={q.estimated_search_volume!s:>6} "
            f"diff={q.competitive_difficulty:>3} {q.visibility_status:<11} pos={pos!s:<3} ai_cited={q.ai_cited}{flag}"
        )
    print("\n=== RECOMMENDATIONS ===")
    for r in repo.list_recommendations(session, profile.id):
        print(f"- [{r.priority}] {r.content_type}: {r.title}  (keywords: {', '.join(r.target_keywords[:4])})")
    print("\n=== SUMMARY ===")
    print(report.get("summary", ""))

    if args.recheck and rows:
        top = rows[0]
        print(f"\n=== RECHECK '{top.query_text}' ===")
        rerun = service.recheck_query(session, profile, top)
        session.refresh(top)
        print(
            json.dumps(
                {
                    k: run_to_dict(rerun)[k]
                    for k in ("run_uuid", "status", "planned_calls_count", "records_extracted_count", "degradation")
                },
                indent=2,
            )
        )
        print(
            f"updated: score={top.opportunity_score:.2f} status={top.visibility_status} pos={top.visibility_position}"
        )

    if args.json:
        print("\n=== REPORT JSON ===")
        print(json.dumps(report.get("json", {}), indent=2, default=str))

    print("\n=== METRICS ===")
    print(json.dumps(deps.metrics.snapshot(), indent=2))
    session.close()
    return 0 if run.status != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
