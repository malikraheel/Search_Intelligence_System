"""Assembles the LangGraph DAG.

START ─┬─(full)──→ query_planner → plan_validator ─┬─(valid)──────────→ dispatch_retrieval
       │                               ↑           ├─(invalid, 1st)──→ plan_repair ──┘
       │                               └───────────┤
       │                                           └─(invalid, final)→ fallback_plan → dispatch_retrieval
       └─(recheck)→ recheck_plan ──────────────────────────────────────→ dispatch_retrieval
dispatch_retrieval ══Send()×N══→ [retrieval_branch subgraph] → normalize → normalize_validator
    normalize_validator ─┬─(records)→ analyze ─┬─(ok)→ report → END
                         │                     └─(llm failed)→ analyze_fallback → report
                         └─(no records)→ degraded_report → END
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from app.domain.models import ProfileCtx, new_id
from app.graph.deps import Deps
from app.graph.nodes import analyze as an
from app.graph.nodes import normalize as nz
from app.graph.nodes import planner as pl
from app.graph.nodes import report as rp
from app.graph.nodes import retrieval as rt
from app.graph.state import BranchInput, BranchOutput, BranchState, RunState
from app.graph.tracing import traced_node

RETRIEVAL_BRANCH = "retrieval_branch"


def _node(deps: Deps, name: str, fn: Callable[..., dict[str, Any]]) -> Callable[[dict[str, Any]], dict[str, Any]]:
    return traced_node(name, deps.metrics)(partial(fn, deps=deps))


def build_retrieval_subgraph(deps: Deps) -> CompiledStateGraph:
    g: StateGraph = StateGraph(BranchState, input_schema=BranchInput, output_schema=BranchOutput)
    g.add_node("retrieval_agent", _node(deps, "retrieval_agent", rt.retrieval_agent))
    g.add_node("tool_arg_validator", _node(deps, "tool_arg_validator", rt.tool_arg_validator))
    g.add_node("execute_tool", _node(deps, "execute_tool", rt.execute_tool))
    g.add_node("retrieval_fallback", _node(deps, "retrieval_fallback", rt.retrieval_fallback))

    g.add_edge(START, "retrieval_agent")
    g.add_edge("retrieval_agent", "tool_arg_validator")
    g.add_conditional_edges(
        "tool_arg_validator",
        rt.route_after_validation,
        {
            "execute_tool": "execute_tool",
            "retrieval_agent": "retrieval_agent",
            "retrieval_fallback": "retrieval_fallback",
        },
    )
    g.add_conditional_edges(
        "execute_tool", rt.route_after_execute, {"done": END, "retrieval_fallback": "retrieval_fallback"}
    )
    g.add_edge("retrieval_fallback", END)
    return g.compile(name=RETRIEVAL_BRANCH)


def route_entry(state: RunState) -> str:
    return "recheck" if state.get("mode") == "recheck" else "full"


def fan_out_retrieval(state: RunState) -> list[Send]:
    """Map step: one retrieval branch per planned call, executed in parallel."""
    return [
        Send(RETRIEVAL_BRANCH, {"run_id": state["run_id"], "profile": state["profile"], "planned_call": pc})
        for pc in state["planned_calls"]
    ]


def build_graph(deps: Deps) -> CompiledStateGraph:
    g: StateGraph = StateGraph(RunState)

    g.add_node("query_planner", _node(deps, "query_planner", pl.query_planner))
    g.add_node("plan_validator", _node(deps, "plan_validator", pl.plan_validator))
    g.add_node("plan_repair", _node(deps, "plan_repair", pl.plan_repair))
    g.add_node("fallback_plan", _node(deps, "fallback_plan", pl.fallback_plan))
    g.add_node("recheck_plan", _node(deps, "recheck_plan", pl.recheck_plan))
    g.add_node("dispatch_retrieval", _node(deps, "dispatch_retrieval", pl.dispatch_retrieval))
    g.add_node(RETRIEVAL_BRANCH, build_retrieval_subgraph(deps))
    g.add_node("normalize", _node(deps, "normalize", nz.normalize))
    g.add_node("normalize_validator", _node(deps, "normalize_validator", nz.normalize_validator))
    g.add_node("analyze", _node(deps, "analyze", an.analyze))
    g.add_node("analyze_fallback", _node(deps, "analyze_fallback", an.analyze_fallback))
    g.add_node("report", _node(deps, "report", rp.report))
    g.add_node("degraded_report", _node(deps, "degraded_report", rp.degraded_report))

    g.add_conditional_edges(START, route_entry, {"full": "query_planner", "recheck": "recheck_plan"})
    g.add_edge("query_planner", "plan_validator")
    g.add_conditional_edges(
        "plan_validator",
        pl.route_after_plan_validation,
        {"dispatch_retrieval": "dispatch_retrieval", "plan_repair": "plan_repair", "fallback_plan": "fallback_plan"},
    )
    g.add_edge("plan_repair", "plan_validator")
    g.add_edge("fallback_plan", "dispatch_retrieval")
    g.add_edge("recheck_plan", "dispatch_retrieval")
    g.add_conditional_edges("dispatch_retrieval", fan_out_retrieval, [RETRIEVAL_BRANCH])
    g.add_edge(RETRIEVAL_BRANCH, "normalize")
    g.add_edge("normalize", "normalize_validator")
    g.add_conditional_edges(
        "normalize_validator",
        nz.route_after_normalize_validation,
        {"analyze": "analyze", "degraded_report": "degraded_report"},
    )
    g.add_conditional_edges(
        "analyze", an.route_after_analyze, {"report": "report", "analyze_fallback": "analyze_fallback"}
    )
    g.add_edge("analyze_fallback", "report")
    g.add_edge("report", END)
    g.add_edge("degraded_report", END)
    return g.compile(name="visibility_graph")


def default_question(profile: ProfileCtx) -> str:
    return (
        f"How does {profile.name} ({profile.domain}) show up in AI answers and Google search results "
        f"for '{profile.industry}' queries, and where are the biggest content opportunities?"
    )


def initial_state(
    profile: ProfileCtx,
    *,
    mode: str = "full",
    question: str | None = None,
    recheck_query: str | None = None,
    run_id: str | None = None,
) -> RunState:
    return {
        "mode": mode,  # type: ignore[typeddict-item]
        "run_id": run_id or new_id(),
        "profile": profile,
        "question": question or default_question(profile),
        "recheck_query": recheck_query,
        "plan": None,
        "plan_errors": [],
        "plan_repair_attempts": 0,
        "planned_calls": [],
        "records": [],
        "dropped_records": 0,
        "analysis": None,
        "report": None,
    }


def invoke_graph(
    graph: CompiledStateGraph,
    state: RunState,
    *,
    recursion_limit: int = 60,
    max_concurrency: int = 4,
) -> RunState:
    return graph.invoke(state, config={"recursion_limit": recursion_limit, "max_concurrency": max_concurrency})  # type: ignore[return-value]


def mermaid(graph: CompiledStateGraph) -> str:
    return graph.get_graph(xray=1).draw_mermaid()
