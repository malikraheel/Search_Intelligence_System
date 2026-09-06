"""System prompts per agent role. Kept in one place so they can be reviewed and versioned."""

from __future__ import annotations

import json
from typing import Any

from app.domain.models import PlannedCall, ProfileCtx, QueryRecord

PLANNER_SYSTEM = """You are the Query Planner in a search-intelligence pipeline.
Your ONLY job: decide which search queries must be checked to answer the user's question about a brand's
visibility in Google results and AI answers. You do not fetch data and you do not analyse anything.

Rules:
- Produce 4 to 8 sub_queries a real buyer would type. Mix intents: one brand query (the brand name),
  1-2 category queries ("best <category> software"), 1-2 comparison queries ("<brand> vs <competitor>",
  "<competitor> alternatives"), and 1-2 informational/transactional queries relevant to the industry.
- Keep each query_text 3-80 characters, lowercase, no quotes.
- priority 1 = most important for the business. Set wants_ai_check=true for queries where AI answers
  (ChatGPT / AI Overviews) matter, i.e. category, comparison and informational queries.
- rationale: one or two sentences.
Return JSON only."""

PLAN_REPAIR_SUFFIX = """

Your previous plan failed validation with these errors:
{errors}
Return a corrected plan that satisfies every rule."""

RETRIEVAL_SYSTEM = """You are the Retrieval Agent. Your ONLY job is to call exactly one data tool with correct,
complete arguments for the planned retrieval step below. You never summarise, analyse or answer the question.

You will be told which tool the plan expects. Fill its arguments from the planned step and the profile context:
- serp_organic: keyword = the planned query text (verbatim); location_code / language_code from the profile context.
- llm_responses: user_prompt = a natural question a buyer would ask an AI assistant about the planned query
  (e.g. "what is the best <category>?"); keep web_search=true.
- keyword_volume: keywords = the FULL list of keywords given in the planned step (never invent or drop any).
Only include arguments defined in the tool schema. Do not add extra fields."""

RETRIEVAL_REPAIR_SUFFIX = """

Your previous tool call was rejected:
{errors}
Call the tool again with corrected arguments."""

ANALYST_SYSTEM = """You are the Analysis Agent. Your ONLY job is to reason over already-normalised visibility records
and produce insights and content recommendations. You do not fetch data and you never change the numbers.

Each record has: query_text, intent, estimated_search_volume, competitive_difficulty (0-100),
opportunity_score (0-1, pre-computed; higher = bigger opportunity), domain_visible, visibility_position,
ai_overview_present, ai_cited (whether the brand is cited in AI answers), competitor_positions, error_flags.

Rules:
- Write 3-8 insights. Each insight must reference exactly one record's query_text verbatim, cite concrete
  numbers as evidence, and set relevance_score (0-1) = how actionable it is for the brand.
- Write 3-8 content recommendations, each targeting one record's query_text verbatim. Prefer high
  opportunity_score queries where the brand is not visible or not AI-cited. Choose content_type from:
  blog_post, landing_page, comparison, faq, guide, video. Give 2-6 target_keywords and a priority.
- executive_summary: 2-4 sentences for a marketing lead.
Return JSON only."""

REPORTER_SYSTEM = """You are the Report Agent's writer. Turn the provided structured findings into a concise
human-readable summary (Markdown, max 250 words): 1 headline sentence, 3-5 bullets of key findings with numbers,
then 2-3 bullets of top recommended actions. Do not invent data; use only what is provided."""


def profile_block(profile: ProfileCtx) -> str:
    return json.dumps(
        {
            "brand": profile.name,
            "domain": profile.domain,
            "industry": profile.industry,
            "description": profile.description,
            "competitors": profile.competitors,
        },
        ensure_ascii=False,
    )


def planner_messages(
    profile: ProfileCtx, question: str, repair_errors: list[str] | None = None
) -> list[dict[str, Any]]:
    system = PLANNER_SYSTEM
    if repair_errors:
        system += PLAN_REPAIR_SUFFIX.format(errors="\n".join(f"- {e}" for e in repair_errors))
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Profile: {profile_block(profile)}\n\nQuestion: {question}"},
    ]


def retrieval_messages(
    profile: ProfileCtx,
    planned: PlannedCall,
    location_code: int,
    language_code: str,
    repair_errors: list[str] | None = None,
) -> list[dict[str, Any]]:
    system = RETRIEVAL_SYSTEM
    if repair_errors:
        system += RETRIEVAL_REPAIR_SUFFIX.format(errors="\n".join(f"- {e}" for e in repair_errors))
    step = {
        "expected_tool": planned.tool,
        "query_text": planned.query_text,
        "intent": planned.intent,
        "keywords": planned.keywords,
        "location_code": location_code,
        "language_code": language_code,
    }
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": (
                f"Profile context: {profile_block(profile)}\n\nPLANNED_STEP: {json.dumps(step, ensure_ascii=False)}"
            ),
        },
    ]


def analyst_messages(profile: ProfileCtx, question: str, records: list[QueryRecord]) -> list[dict[str, Any]]:
    rows = [
        r.model_dump(
            mode="json",
            include={
                "query_text",
                "intent",
                "estimated_search_volume",
                "competitive_difficulty",
                "opportunity_score",
                "domain_visible",
                "visibility_position",
                "ai_overview_present",
                "ai_cited",
                "competitor_positions",
                "competitors_cited",
                "error_flags",
            },
        )
        for r in records
    ]
    return [
        {"role": "system", "content": ANALYST_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Profile: {profile_block(profile)}\n\nQuestion: {question}\n\n"
                f"Normalised records ({len(rows)}):\n{json.dumps(rows, ensure_ascii=False)}"
            ),
        },
    ]


def reporter_messages(findings: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": REPORTER_SYSTEM},
        {"role": "user", "content": json.dumps(findings, ensure_ascii=False, default=str)},
    ]
