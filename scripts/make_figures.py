"""Build the figures the README points at.

Everything here is drawn from a real run of the real code. The waterfall in
particular is the actual effect decomposition for the scenario named below, so
if the engine changes the picture changes with it.

    .venv/bin/python scripts/make_figures.py
"""

from __future__ import annotations

import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from budget_agent import fiscal, vocab  # noqa: E402
from budget_agent.data import generate_ledger  # noqa: E402
from budget_agent.graph import AGENT, ask  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[1] / "docs" / "img"
INK = "#1f2933"
MUTED = "#7b8794"
GRID = "#e4e7eb"
GREEN = "#2e7d32"
RED = "#c62828"
BLUE = "#1f77b4"

plt.rcParams.update(
    {
        "font.size": 10,
        "font.family": "DejaVu Sans",
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def architecture() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 4.8), dpi=150)
    ax.set_xlim(0, 110)
    ax.set_ylim(0, 46)
    ax.axis("off")

    def box(x, y, w, h, title, body, fill="#f5f7fa", edge=MUTED):
        ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.2",
                linewidth=1.2, edgecolor=edge, facecolor=fill,
            )
        )
        ax.text(x + w / 2, y + h - 3.4, title, ha="center", va="top",
                fontsize=10, fontweight="bold", color=INK)
        ax.text(x + w / 2, y + h - 8.2, body, ha="center", va="top",
                fontsize=8.4, color=MUTED, linespacing=1.5)

    def arrow(x1, y1, x2, y2, style="-|>"):
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=12,
                linewidth=1.1, color=MUTED, shrinkA=0, shrinkB=0,
            )
        )

    box(1, 25, 22, 17, "Synthetic ledger",
        "24 months of monthly\ncost lines, seeded\n\n57 lines, 19 vendors")
    box(1, 4, 22, 17, "Question",
        "plain English,\nparsed by rules,\nno model call")
    box(28, 25, 22, 17, "Baseline",
        "trailing run rate\n+ fitted growth\n+ seasonality index")
    box(28, 4, 22, 17, "Validated scenario",
        "Pydantic levers,\ntargets and windows\n(rejected if invalid)")
    box(55, 15, 23, 19, "Engine",
        "applies the levers,\nthen the constraints,\nthen the second-order\neffects",
        fill="#eef4fb", edge=BLUE)
    box(53, 1, 27, 12, "Vendor terms",
        "minimum commitments\nand notice periods")
    box(85, 25, 24, 17, "Result",
        "monthly path, category\nand line deltas,\na decomposition that\nreconciles")
    box(85, 4, 24, 17, "Assumption ledger",
        "every value the answer\ndepends on, with its\nprovenance",
        fill="#fdf4ec", edge="#c47d3a")

    arrow(23.8, 33.5, 27.2, 33.5)
    arrow(23.8, 12.5, 27.2, 12.5)
    arrow(50.8, 33.5, 54.2, 29.5)
    arrow(50.8, 12.5, 54.2, 20.0)
    arrow(66.5, 13.8, 66.5, 15.4)
    arrow(78.8, 29.5, 84.2, 33.5)
    arrow(78.8, 20.0, 84.2, 12.5)
    ax.set_xlim(0, 112)
    ax.set_ylim(0, 46)
    ax.text(56, 44.5, "Budget Scenario Agent", fontsize=13, fontweight="bold",
            ha="center", color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "architecture.png", bbox_inches="tight")
    plt.close(fig)
    print("architecture.png")


def agent_graph() -> None:
    """LangGraph's own render of the compiled topology."""
    try:
        png = AGENT.get_graph().draw_mermaid_png()
    except Exception as exc:
        print(f"skipped agent_graph.png ({type(exc).__name__}: {exc})")
        return
    (OUT / "agent_graph.png").write_bytes(png)
    (OUT / "agent_graph.mmd").write_text(AGENT.get_graph().draw_mermaid())
    print("agent_graph.png")


def ledger_history() -> None:
    ledger = generate_ledger()
    pivot = (
        ledger.pivot_table(index="month", columns="category", values="amount_usd", aggfunc="sum")
        .sort_index()
    )
    order = pivot.sum().sort_values(ascending=False).index
    pivot = pivot[order]

    fig, ax = plt.subplots(figsize=(11, 4.4), dpi=150)
    colors = plt.cm.tab20(range(len(order)))
    ax.stackplot(
        pivot.index, [pivot[c] / 1e6 for c in order],
        labels=[vocab.category_label(c) for c in order], colors=colors, alpha=0.92,
    )
    ax.set_ylabel("USD per month (millions)")
    ax.set_title("Synthetic ledger: 24 months of actuals, seed 20260401")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(loc="upper left", ncol=4, frameon=False, fontsize=8)
    ax.set_ylim(0, pivot.sum(axis=1).max() / 1e6 * 1.35)

    for month, note in [
        ("2025-04", "cloud region\nadded"),
        ("2025-10", "contractor\nproject ramps"),
        ("2026-02", "media channel\ncut"),
    ]:
        x = pivot.index[pivot.index.strftime("%Y-%m") == month][0]
        ax.axvline(x, color=INK, linewidth=0.8, linestyle=":", alpha=0.6)
        ax.annotate(
            note, xy=(x, pivot.sum(axis=1).max() / 1e6 * 1.02), fontsize=7.5,
            color=INK, ha="center", va="bottom",
        )
    fig.tight_layout()
    fig.savefig(OUT / "ledger_history.png", bbox_inches="tight")
    plt.close(fig)
    print("ledger_history.png")


def scenario_output(question: str) -> None:
    state = ask(question)
    assert state["status"] == "done", state.get("errors")
    result = state["result"]
    effects = result.effects

    fig, (left, right) = plt.subplots(
        1, 2, figsize=(12, 4.6), dpi=150, gridspec_kw={"width_ratios": [1.15, 1]}
    )

    monthly = result.monthly
    left.plot(monthly["month"], monthly["baseline_usd"] / 1e6, color=MUTED,
              linewidth=2, label="baseline", marker="o", markersize=3.5)
    left.plot(monthly["month"], monthly["scenario_usd"] / 1e6, color=BLUE,
              linewidth=2, label="scenario", marker="o", markersize=3.5)
    left.fill_between(
        monthly["month"], monthly["baseline_usd"] / 1e6, monthly["scenario_usd"] / 1e6,
        color=GREEN, alpha=0.18,
    )
    left.set_ylabel("USD per month (millions)")
    left.set_title(f'"{question}"', fontsize=10.5)
    left.grid(axis="y", color=GRID, linewidth=0.8)
    left.set_axisbelow(True)
    for spine in ("top", "right"):
        left.spines[spine].set_visible(False)
    left.legend(frameon=False, fontsize=9)
    left.tick_params(axis="x", rotation=30, labelsize=8)

    steps = [
        ("gross\nreduction", -effects["realized_reduction_usd"], GREEN),
        ("backfill\ncover", effects["contractor_backfill_usd"], RED),
        ("one-time\nexit fees", effects["one_time_exit_fees_usd"], RED),
    ]
    if effects["shift_reinvestment_usd"]:
        steps.insert(1, ("reinvested in\ndestination", effects["shift_reinvestment_usd"], RED))
    if effects["increase_usd"]:
        steps.insert(1, ("new\nspend", effects["increase_usd"], RED))

    running = 0.0
    tops = []
    for i, (label, value, color) in enumerate(steps):
        right.bar(i, value / 1e3, bottom=running / 1e3, color=color, alpha=0.85, width=0.62)
        end = running + value
        offset = 6 if value > 0 else -6
        right.text(
            i, end / 1e3 + offset, f"{value/1e3:+,.0f}k", ha="center",
            va="bottom" if value > 0 else "top", fontsize=8.5, color=INK,
        )
        tops.append(end / 1e3)
        running = end
    for i, top in enumerate(tops[:-1]):
        right.plot([i + 0.31, i + 1 - 0.31], [top, top], color=MUTED,
                   linewidth=0.9, linestyle="--")
    right.plot([len(steps) - 1 + 0.31, len(steps) - 0.31], [tops[-1], tops[-1]],
               color=MUTED, linewidth=0.9, linestyle="--")
    right.bar(len(steps), running / 1e3, color=INK, alpha=0.9, width=0.62)
    right.text(len(steps), running / 1e3 - 6, f"{running/1e3:+,.0f}k", ha="center",
               va="top", fontsize=9.5, fontweight="bold", color=INK)
    right.axhline(0, color=MUTED, linewidth=1)
    right.set_xticks(range(len(steps) + 1))
    right.set_xticklabels([s[0] for s in steps] + ["net change\nto the plan"], fontsize=8.5)
    right.set_ylabel("USD thousands")
    right.set_title("What the cut actually delivers", fontsize=10.5)
    right.grid(axis="y", color=GRID, linewidth=0.8)
    right.set_axisbelow(True)
    low = min(min(tops), running / 1e3, 0)
    high = max(max(tops), 0)
    right.set_ylim(low - abs(low) * 0.22 - 8, high + abs(high) * 0.15 + 10)
    for spine in ("top", "right"):
        right.spines[spine].set_visible(False)

    blocked = effects["blocked_by_minimum_commitment_usd"]
    deferred = effects["deferred_by_notice_usd"]
    if blocked or deferred:
        right.text(
            0.02, 0.03,
            f"blocked by minimums: ${blocked:,.0f}\ndeferred by notice: ${deferred:,.0f}",
            transform=right.transAxes, fontsize=8, color=MUTED, va="bottom",
        )

    fig.tight_layout()
    fig.savefig(OUT / "scenario_output.png", bbox_inches="tight")
    plt.close(fig)
    print("scenario_output.png")
    print(f"  {state['narrative']}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    architecture()
    agent_graph()
    ledger_history()
    scenario_output("what happens if we cut contractor spend 15% in Q3")


if __name__ == "__main__":
    main()
