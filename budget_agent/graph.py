"""The agent graph.

Five working nodes and a rejection node. The reason this is a graph and not a
function call is the branch: a question that does not parse, or parses into
something the schema refuses, has to stop before the engine runs and come back
with a reason. Keeping that as an explicit edge makes the failure path as
visible as the happy one.

    parse -> validate -> baseline -> simulate -> explain
       |          |
       +----------+--> reject

`parse` has two implementations behind it. With no API key it is the rules
parser and nothing else, which is the default and the only path the tests take.
With a key set it asks a model first (see `llm.py`); if the model's proposal
does not survive the schema the rules parser runs instead and the question is
answered exactly as it would have been. Either way `validate` is unchanged and
no number in the answer comes from a model.

State is a plain dict and every node appends to `trace`, so the UI can show
what the agent did without any extra instrumentation.
"""

from __future__ import annotations

import operator
from functools import lru_cache
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from . import fiscal, llm
from .engine import Baseline, EngineConfig, ScenarioResult, build_baseline, simulate
from .models import Scenario
from .parser import parse_question


class AgentState(TypedDict, total=False):
    question: str
    config: dict
    trace: Annotated[list[str], operator.add]
    errors: list[str]
    draft: dict | None
    scenario: dict | None
    parse_coverage: float
    ignored_words: list[str]
    parser: str
    status: str
    baseline_summary: dict
    result_summary: dict
    narrative: str
    # The engine objects ride along in memory for the UI. They are not
    # serializable, which is fine as long as this graph runs without a
    # checkpointer; swap them for ids if you ever add one.
    baseline: Any
    scenario_obj: Any
    result: Any


@lru_cache(maxsize=8)
def _cached_baseline(config_json: str) -> Baseline:
    return build_baseline(EngineConfig.model_validate_json(config_json))


def get_baseline(config: EngineConfig) -> Baseline:
    return _cached_baseline(config.model_dump_json())


def parse_node(state: AgentState) -> dict:
    """Model first if there is a key, rules otherwise, rules again if the
    model's proposal does not hold up. The fallback is silent by design: a
    question the model fumbles is answered by the parser that has always
    answered it, not by an error message about a feature the user did not
    ask for."""
    proposed = llm.parse_question(state["question"])
    if proposed is not None and proposed.draft is not None:
        return {
            "draft": proposed.draft,
            "errors": [],
            "trace": [f"[parse] {line}" for line in proposed.trace],
            "status": "parsed",
            "parser": "model",
        }

    parsed = parse_question(state["question"])
    trace = [f"[parse] {line}" for line in (proposed.trace if proposed else [])]
    trace += [f"[parse] {line}" for line in parsed.trace]
    trace.append(
        f"[parse] matched {parsed.tokens_matched}/{parsed.tokens_total} "
        f"significant tokens"
    )
    return {
        "draft": parsed.draft,
        "errors": list(parsed.errors),
        "parse_coverage": parsed.coverage,
        "ignored_words": parsed.ignored_words,
        "trace": trace,
        "status": "parsed" if parsed.ok else "rejected",
        "parser": "rules",
    }


def validate_node(state: AgentState) -> dict:
    try:
        scenario = Scenario.model_validate(state["draft"])
    except ValidationError as exc:
        messages = [
            f"{'.'.join(str(p) for p in err['loc']) or 'scenario'}: {err['msg']}"
            for err in exc.errors()
        ]
        return {
            "errors": list(state.get("errors", [])) + messages,
            "status": "rejected",
            "trace": [f"[validate] rejected: {m}" for m in messages],
        }
    return {
        "scenario": scenario.model_dump(mode="json"),
        "trace": [f"[validate] accepted: {scenario.description}"],
        "status": "validated",
        # the parsed object rides along so the engine does not re-validate
        "scenario_obj": scenario,
    }


def baseline_node(state: AgentState) -> dict:
    config = EngineConfig.model_validate(state.get("config") or {})
    baseline = get_baseline(config)
    summary = {
        "plan": f"FY{fiscal.PLAN_FY}",
        "months": len(baseline.months),
        "lines": len(baseline.lines),
        "total_usd": round(baseline.total, 2),
        "run_rate_usd_per_month": round(float(baseline.run_rate.sum()), 2),
    }
    return {
        "baseline": baseline,
        "baseline_summary": summary,
        "trace": [
            f"[baseline] projected {summary['lines']} lines over "
            f"{summary['months']} months: ${summary['total_usd']:,.0f}"
        ],
    }


def simulate_node(state: AgentState) -> dict:
    config = EngineConfig.model_validate(state.get("config") or {})
    scenario = state.get("scenario_obj") or Scenario.model_validate(state["scenario"])
    result = simulate(scenario, state["baseline"], config)
    return {
        "result": result,
        "result_summary": result.summary(),
        "trace": [
            f"[simulate] net {result.net_savings:+,.0f} USD against a "
            f"${result.baseline_total:,.0f} baseline"
        ]
        + [f"[simulate] warning: {w}" for w in result.warnings],
    }


def explain_node(state: AgentState) -> dict:
    result: ScenarioResult = state["result"]
    return {
        "narrative": narrate(result),
        "status": "done",
        "trace": ["[explain] built the narrative and the assumption ledger"],
    }


def reject_node(state: AgentState) -> dict:
    return {
        "status": "rejected",
        "narrative": "",
        "trace": ["[reject] nothing was calculated"],
    }


def _sentence(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


def narrate(result: ScenarioResult) -> str:
    e = result.effects
    scenario = result.scenario
    parts = [
        f"{_sentence(scenario.description)} moves the FY{fiscal.PLAN_FY} plan "
        f"from ${result.baseline_total:,.0f} to ${result.scenario_total:,.0f}, "
        f"a net {'saving' if result.net_savings >= 0 else 'increase'} of "
        f"${abs(result.net_savings):,.0f} "
        f"({abs(result.net_savings) / result.baseline_total * 100:.2f}% of plan)."
    ]
    if e["realized_reduction_usd"] > 0:
        detail = [f"${e['realized_reduction_usd']:,.0f} comes off the targeted lines"]
        if e["blocked_by_minimum_commitment_usd"] > 0:
            detail.append(
                f"${e['blocked_by_minimum_commitment_usd']:,.0f} could not be cut "
                f"because of contractual minimums"
            )
        if e["deferred_by_notice_usd"] > 0:
            detail.append(
                f"${e['deferred_by_notice_usd']:,.0f} falls inside a notice period "
                f"and does not land in the window"
            )
        parts.append("; ".join(detail) + ".")
    give_back = (
        e["contractor_backfill_usd"] + e["shift_reinvestment_usd"] + e["one_time_exit_fees_usd"]
    )
    if give_back > 0:
        pieces = []
        if e["contractor_backfill_usd"]:
            pieces.append(f"${e['contractor_backfill_usd']:,.0f} of backfill cover")
        if e["shift_reinvestment_usd"]:
            pieces.append(f"${e['shift_reinvestment_usd']:,.0f} spent in the destination category")
        if e["one_time_exit_fees_usd"]:
            pieces.append(f"${e['one_time_exit_fees_usd']:,.0f} of one-time exit fees")
        parts.append(
            "Against that, " + " and ".join(pieces)
            + f", so {give_back / max(e['realized_reduction_usd'], 1e-9) * 100:.0f}% "
            "of the gross reduction is given back."
        )
    if e["increase_usd"] > 0:
        parts.append(f"The scenario also adds ${e['increase_usd']:,.0f} of new spend.")

    by_department = (
        result.by_line.groupby("department", observed=True)["delta_usd"].sum().sort_values()
    )
    if len(by_department):
        department = by_department.index[0]
        parts.append(
            f"{department} carries most of the change "
            f"(${by_department.iloc[0]:,.0f})."
        )
    worst = result.monthly.loc[result.monthly["delta_usd"].abs().idxmax()]
    parts.append(
        f"The biggest single month is {fiscal.label(worst['month'])} at "
        f"${worst['delta_usd']:,.0f}."
    )
    return " ".join(parts)


def _route_after_parse(state: AgentState) -> str:
    return "reject" if state.get("status") == "rejected" or not state.get("draft") else "validate"


def _route_after_validate(state: AgentState) -> str:
    return "reject" if state.get("status") == "rejected" else "baseline"


def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("parse", parse_node)
    builder.add_node("validate", validate_node)
    builder.add_node("baseline", baseline_node)
    builder.add_node("simulate", simulate_node)
    builder.add_node("explain", explain_node)
    builder.add_node("reject", reject_node)

    builder.add_edge(START, "parse")
    builder.add_conditional_edges("parse", _route_after_parse, ["validate", "reject"])
    builder.add_conditional_edges("validate", _route_after_validate, ["baseline", "reject"])
    builder.add_edge("baseline", "simulate")
    builder.add_edge("simulate", "explain")
    builder.add_edge("explain", END)
    builder.add_edge("reject", END)
    return builder.compile()


AGENT = build_graph()


def ask(question: str, config: EngineConfig | None = None) -> AgentState:
    config = config or EngineConfig()
    return AGENT.invoke(
        {"question": question, "config": config.model_dump(), "trace": [], "errors": []}
    )
