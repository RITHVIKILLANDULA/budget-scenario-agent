"""Budget Scenario Agent -- Streamlit front end.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from budget_agent import fiscal, store, vocab
from budget_agent.engine import EngineConfig, compare
from budget_agent.graph import AGENT, ask, get_baseline

st.set_page_config(page_title="Budget Scenario Agent", layout="wide")

EXAMPLES = [
    "what happens if we cut contractor spend 15% in Q3",
    "freeze travel for the year",
    "move 30% of contractor spend into software",
    "cut $40k a month from marketing programs in Q2 and increase cloud by 8%",
    "reduce facilities by 20%",
    "cut the Atlantis team by 10%",
]

SOURCE_BADGE = {
    "derived": ":blue[derived from the ledger]",
    "default": ":gray[model default]",
    "override": ":orange[you changed this]",
    "structural": ":gray[how the model is built]",
}


def sidebar_config() -> EngineConfig:
    st.sidebar.header("Assumptions")
    st.sidebar.caption(
        "These are the knobs behind every number on the right. Change one and "
        "the whole answer moves, which is the point."
    )
    defaults = EngineConfig()
    use_seasonality = st.sidebar.toggle(
        "Apply seasonality", value=defaults.use_seasonality,
        help="Off means every month of the plan is the same size.",
    )
    honor_notice = st.sidebar.toggle(
        "Honour notice periods", value=defaults.honor_notice,
        help="A cut cannot land before the vendor's contractual notice has run.",
    )
    respect_floors = st.sidebar.toggle(
        "Respect minimum commitments", value=defaults.respect_floors,
        help="Spend cannot fall below a vendor's contracted floor.",
    )
    backfill = st.sidebar.slider(
        "Contractor backfill", 0.0, 1.0, defaults.contractor_backfill_ratio, 0.05,
        help="Share of a contractor cut that comes back as agency or overtime cost.",
    )
    exit_fee = st.sidebar.slider(
        "Exit fee (months)", 0.0, 3.0, defaults.exit_fee_months, 0.25,
        help="One-time charge on vendors that have a notice period.",
    )
    shift_efficiency = st.sidebar.slider(
        "Shift efficiency", 0.0, 1.5, defaults.shift_efficiency, 0.05,
        help="Dollars spent in the destination per dollar cut from the source.",
    )
    growth_clip_high = st.sidebar.slider(
        "Growth cap (per year)", 0.0, 1.0, defaults.growth_clip_high, 0.05,
        help="Ceiling on the growth fitted from history.",
    )

    st.sidebar.divider()
    st.sidebar.subheader("Data")
    db = store.status()
    if db["enabled"]:
        st.sidebar.success(
            f"Postgres connected: {db['rows']:,} expense rows, {db['runs']} runs logged"
        )
    else:
        st.sidebar.info(f"In-process ledger ({db['detail']})")
    st.sidebar.caption(
        f"Synthetic ledger, seed {defaults.seed}. "
        f"{fiscal.label(fiscal.HISTORY_START)}..{fiscal.label(fiscal.HISTORY_END)} "
        f"of actuals, planning FY{fiscal.PLAN_FY}."
    )
    return EngineConfig(
        use_seasonality=use_seasonality,
        honor_notice=honor_notice,
        respect_floors=respect_floors,
        contractor_backfill_ratio=backfill,
        exit_fee_months=exit_fee,
        shift_efficiency=shift_efficiency,
        growth_clip_high=growth_clip_high,
    )


def monthly_chart(monthly: pd.DataFrame) -> alt.Chart:
    long = monthly.melt(
        id_vars="month", value_vars=["baseline_usd", "scenario_usd"],
        var_name="series", value_name="usd",
    )
    long["series"] = long["series"].map(
        {"baseline_usd": "baseline", "scenario_usd": "scenario"}
    )
    return (
        alt.Chart(long)
        .mark_line(point=True)
        .encode(
            x=alt.X("yearmonth(month):T", title=None),
            y=alt.Y("usd:Q", title="USD per month", scale=alt.Scale(zero=False)),
            color=alt.Color(
                "series:N", title=None,
                scale=alt.Scale(domain=["baseline", "scenario"], range=["#8a8a8a", "#1f77b4"]),
            ),
            tooltip=["yearmonth(month):T", "series:N", alt.Tooltip("usd:Q", format=",.0f")],
        )
        .properties(height=300)
    )


def delta_chart(monthly: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(monthly)
        .mark_bar()
        .encode(
            x=alt.X("yearmonth(month):T", title=None),
            y=alt.Y("delta_usd:Q", title="change vs baseline"),
            color=alt.condition(
                alt.datum.delta_usd < 0, alt.value("#2e7d32"), alt.value("#c62828")
            ),
            tooltip=["yearmonth(month):T", alt.Tooltip("delta_usd:Q", format=",.0f")],
        )
        .properties(height=190)
    )


def category_chart(by_category: pd.DataFrame) -> alt.Chart:
    frame = by_category[by_category["delta_usd"].abs() > 0.5].copy()
    frame["label"] = frame["category"].map(vocab.category_label)
    return (
        alt.Chart(frame)
        .mark_bar()
        .encode(
            y=alt.Y("label:N", sort="x", title=None),
            x=alt.X("delta_usd:Q", title="change vs baseline (USD)"),
            color=alt.condition(
                alt.datum.delta_usd < 0, alt.value("#2e7d32"), alt.value("#c62828")
            ),
            tooltip=["label:N", alt.Tooltip("delta_usd:Q", format=",.0f")],
        )
        .properties(height=max(140, 30 * len(frame)))
    )


def render_assumptions(result) -> None:
    st.subheader("Assumptions behind this number")
    st.caption(
        "Nothing here is hidden in the code. Anything marked orange is a value "
        "you changed in the sidebar."
    )
    for assumption in result.assumptions:
        st.markdown(
            f"**{assumption.label}** — {assumption.value}  \n"
            f"{SOURCE_BADGE.get(assumption.source, assumption.source)}"
        )
        st.caption(assumption.detail)


def render_result(state, config: EngineConfig) -> None:
    result = state["result"]
    summary = state["result_summary"]

    top = st.columns(4)
    top[0].metric("FY baseline", f"${result.baseline_total/1e6:,.2f}M")
    top[1].metric("Scenario", f"${result.scenario_total/1e6:,.2f}M")
    top[2].metric(
        "Net change", f"${result.net_delta:,.0f}",
        delta=f"{result.net_delta / result.baseline_total * 100:.2f}% of plan",
        delta_color="inverse",
    )
    top[3].metric(
        "Gross reduction", f"${result.effects['realized_reduction_usd']:,.0f}",
        help="Before backfill, reinvestment and one-time fees.",
    )

    st.write(state["narrative"])
    for warning in result.warnings:
        st.warning(warning)

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.altair_chart(monthly_chart(result.monthly), width="stretch")
        st.altair_chart(delta_chart(result.monthly), width="stretch")
        st.altair_chart(category_chart(result.by_category), width="stretch")
    with right:
        render_assumptions(result)

    effects = pd.DataFrame(
        [{"effect": k.replace("_usd", "").replace("_", " "), "usd": v}
         for k, v in result.effects.items() if abs(v) > 0.005]
    )
    with st.expander("Where the money went", expanded=False):
        st.dataframe(
            effects.style.format({"usd": "{:,.0f}"}), hide_index=True,
            width="stretch",
        )
        st.caption(
            "These reconcile to the net change exactly: "
            "-(realized reduction) + increases + backfill + reinvestment + fees."
        )
    with st.expander("Structured scenario and agent trace"):
        st.json(state["scenario"], expanded=False)
        st.code("\n".join(state["trace"]), language="text")
    with st.expander("Line-level detail"):
        frame = result.by_line.copy()
        frame["category"] = frame["category"].map(vocab.category_label)
        st.dataframe(
            frame.style.format(
                {"baseline_usd": "{:,.0f}", "scenario_usd": "{:,.0f}", "delta_usd": "{:,.0f}"}
            ),
            hide_index=True, width="stretch",
        )


def render_rejection(state) -> None:
    st.error("That question was rejected before any numbers were calculated.")
    for error in state.get("errors", []):
        st.markdown(f"- {error}")
    with st.expander("What the parser did see", expanded=True):
        st.code("\n".join(state.get("trace", [])) or "nothing matched", language="text")
    st.caption(
        "The language layer is rule-based, so this is usually a vocabulary miss "
        "rather than a misunderstanding. Known categories: "
        + ", ".join(vocab.category_label(c) for c in vocab.CATEGORY_KEYS)
        + ". Known departments: " + ", ".join(vocab.DEPARTMENT_KEYS) + "."
    )


def main() -> None:
    config = sidebar_config()
    baseline = get_baseline(config)

    st.title("Budget Scenario Agent")
    st.caption(
        f"Ask a budget question in English. It is parsed into a validated "
        f"scenario, run against a ${baseline.total/1e6:,.1f}M FY{fiscal.PLAN_FY} "
        f"plan built from {len(baseline.lines)} ledger lines, and every "
        f"assumption is listed back to you."
    )

    if "question" not in st.session_state:
        # ?q=... makes a scenario shareable, and is how the screenshot script
        # puts the app into a given state.
        st.session_state.question = st.query_params.get("q", EXAMPLES[0])

    chips = st.columns(3)
    for i, example in enumerate(EXAMPLES):
        if chips[i % 3].button(example, width="stretch", key=f"ex{i}"):
            st.session_state.question = example

    question = st.text_input("Question", key="question")
    st.button("Run scenario", type="primary")

    if question:
        state = ask(question, config)
        store.log_run(
            question, state.get("status", "unknown"),
            state.get("scenario"), state.get("result_summary"),
        )
        if state.get("status") == "done":
            render_result(state, config)
        else:
            render_rejection(state)

    st.divider()
    st.subheader("Compare two scenarios")
    st.caption("Both run against the same baseline and the same assumptions.")
    cols = st.columns(2)
    question_a = cols[0].text_input(
        "Scenario A", value=st.query_params.get("a", EXAMPLES[0]), key="cmp_a"
    )
    question_b = cols[1].text_input(
        "Scenario B", value=st.query_params.get("b", EXAMPLES[2]), key="cmp_b"
    )
    if question_a and question_b:
        states = [ask(question_a, config), ask(question_b, config)]
        if any(s.get("status") != "done" for s in states):
            for label, s in zip("AB", states):
                if s.get("status") != "done":
                    st.error(f"Scenario {label} was rejected: {'; '.join(s['errors'])}")
        else:
            results = [s["result"] for s in states]
            table = compare(results)
            st.dataframe(
                table.style.format(
                    {
                        c: ("{:.2f}" if c.endswith("%") else "{:,.0f}")
                        for c in table.columns
                        if c != "scenario"
                    }
                ),
                hide_index=True, width="stretch",
            )
            merged = pd.concat(
                [
                    r.monthly.assign(scenario=f"{label}: {r.scenario.name}")
                    for label, r in zip("AB", results)
                ]
            )
            chart = (
                alt.Chart(merged)
                .mark_line(point=True)
                .encode(
                    x=alt.X("yearmonth(month):T", title=None),
                    y=alt.Y("delta_usd:Q", title="change vs baseline (USD)"),
                    color=alt.Color("scenario:N", title=None),
                    tooltip=["yearmonth(month):T", "scenario:N",
                             alt.Tooltip("delta_usd:Q", format=",.0f")],
                )
                .properties(height=240)
            )
            st.altair_chart(chart, width="stretch")
            best, other = sorted(results, key=lambda r: -r.net_savings)
            st.info(
                f"**{best.scenario.name}** saves "
                f"${best.net_savings - other.net_savings:,.0f} more than "
                f"**{other.scenario.name}**, but gives back "
                f"${best.effects['contractor_backfill_usd'] + best.effects['shift_reinvestment_usd'] + best.effects['one_time_exit_fees_usd']:,.0f} "
                f"against ${other.effects['contractor_backfill_usd'] + other.effects['shift_reinvestment_usd'] + other.effects['one_time_exit_fees_usd']:,.0f} "
                f"in second-order cost."
            )

    runs = store.recent_runs()
    if runs is not None and len(runs):
        with st.expander(f"Run history from Postgres ({len(runs)} most recent)"):
            st.dataframe(runs, hide_index=True, width="stretch")


if __name__ == "__main__":
    main()
