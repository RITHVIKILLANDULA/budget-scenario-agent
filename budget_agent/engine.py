"""Baseline projection and scenario arithmetic.

All of it is plain Python and numpy. Nothing here calls a model, and every
number the UI shows can be traced back to a line in this file. The engine also
emits an assumption ledger: if a figure depends on a choice I made, that choice
is listed back to the analyst with its value and where it came from.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from . import fiscal, vocab
from .data import DEFAULT_SEED, generate_ledger
from .models import Lever, LeverKind, Scenario

LINE_KEYS = ["department", "category", "vendor"]


class EngineConfig(BaseModel):
    """Every knob the analyst is allowed to turn, with its default."""

    seed: int = DEFAULT_SEED
    use_seasonality: bool = True
    growth_clip_low: float = Field(default=-0.25, le=0)
    growth_clip_high: float = Field(default=0.35, ge=0)
    honor_notice: bool = True
    respect_floors: bool = True
    contractor_backfill_ratio: float = Field(default=0.30, ge=0, le=1)
    exit_fee_months: float = Field(default=0.5, ge=0, le=6)
    shift_efficiency: float = Field(default=0.60, ge=0, le=2)


@dataclass(frozen=True)
class Assumption:
    key: str
    label: str
    value: str
    source: str  # "derived" | "default" | "override" | "structural"
    detail: str


@dataclass
class Baseline:
    """The FY plan before anyone touches it."""

    lines: pd.DataFrame  # one row per ledger line, LINE_KEYS
    matrix: np.ndarray  # lines x 12 plan months
    months: list[dt.date]
    run_rate: np.ndarray  # trailing-12-month monthly average per line
    growth: np.ndarray  # fitted annual growth per line
    seasonality: pd.DataFrame  # category x calendar month
    config: EngineConfig

    @property
    def total(self) -> float:
        return float(self.matrix.sum())

    def frame(self) -> pd.DataFrame:
        out = self.lines.copy()
        for i, month in enumerate(self.months):
            out[fiscal.label(month)] = self.matrix[:, i]
        return out


@dataclass
class ScenarioResult:
    scenario: Scenario
    baseline: Baseline
    monthly: pd.DataFrame  # month, baseline_usd, scenario_usd, delta_usd
    by_category: pd.DataFrame
    by_line: pd.DataFrame
    effects: dict[str, float]
    assumptions: list[Assumption] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def baseline_total(self) -> float:
        return float(self.monthly["baseline_usd"].sum())

    @property
    def scenario_total(self) -> float:
        return float(self.monthly["scenario_usd"].sum())

    @property
    def net_delta(self) -> float:
        """Negative means the scenario spends less than the baseline."""
        return self.scenario_total - self.baseline_total

    @property
    def net_savings(self) -> float:
        return -self.net_delta

    def summary(self) -> dict:
        return {
            "scenario": self.scenario.name,
            "baseline_usd": round(self.baseline_total, 2),
            "scenario_usd": round(self.scenario_total, 2),
            "net_delta_usd": round(self.net_delta, 2),
            "net_savings_usd": round(self.net_savings, 2),
            "net_savings_pct": round(100 * self.net_savings / self.baseline_total, 3),
            **{k: round(v, 2) for k, v in self.effects.items()},
        }


# --- baseline ---------------------------------------------------------------


def _seasonality_table(ledger: pd.DataFrame) -> pd.DataFrame:
    monthly = (
        ledger.assign(cal_month=ledger["month"].dt.month)
        .groupby(["category", "cal_month"], observed=True)["amount_usd"]
        .mean()
        .unstack("cal_month")
    )
    return monthly.div(monthly.mean(axis=1), axis=0)


def build_baseline(config: EngineConfig | None = None) -> Baseline:
    config = config or EngineConfig()
    ledger = generate_ledger(config.seed)
    plan_months = fiscal.PLAN_MONTHS

    wide = (
        ledger.pivot_table(
            index=LINE_KEYS, columns="month", values="amount_usd", aggfunc="sum", observed=True
        )
        .fillna(0.0)
        .sort_index(axis=1)
    )
    history = wide.to_numpy(dtype=float)

    ttm = history[:, -12:]
    run_rate = ttm.sum(axis=1) / 12.0

    first_half = history[:, -12:-6].sum(axis=1)
    second_half = history[:, -6:].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(first_half > 0, second_half / first_half, 1.0)
    growth = np.clip(ratio**2 - 1.0, config.growth_clip_low, config.growth_clip_high)

    season = _seasonality_table(ledger)
    lines = wide.index.to_frame(index=False)

    matrix = np.empty((len(lines), 12), dtype=float)
    for i, month in enumerate(plan_months):
        trend = (1.0 + growth) ** ((i + 0.5) / 12.0)
        if config.use_seasonality:
            factor = season.loc[lines["category"].to_numpy(), month.month].to_numpy()
        else:
            factor = np.ones(len(lines))
        matrix[:, i] = run_rate * trend * factor

    return Baseline(
        lines=lines,
        matrix=matrix,
        months=list(plan_months),
        run_rate=run_rate,
        growth=growth,
        seasonality=season,
        config=config,
    )


# --- scenario ---------------------------------------------------------------


def _line_mask(lines: pd.DataFrame, lever: Lever) -> np.ndarray:
    mask = np.ones(len(lines), dtype=bool)
    target = lever.target
    if target.departments:
        mask &= lines["department"].isin(target.departments).to_numpy()
    if target.categories:
        mask &= lines["category"].isin(target.categories).to_numpy()
    if target.vendors:
        mask &= lines["vendor"].isin(target.vendors).to_numpy()
    return mask


def _notice_by_line(lines: pd.DataFrame) -> np.ndarray:
    return lines["vendor"].map(
        {v.name: v.notice_months for v in vocab.VENDORS}
    ).fillna(0).to_numpy(dtype=int)


def _floor_ratio_by_line(lines: pd.DataFrame) -> np.ndarray:
    return lines["vendor"].map(
        {v.name: v.min_commit_ratio for v in vocab.VENDORS}
    ).fillna(0.0).to_numpy(dtype=float)


class _Additions:
    """Costs the scenario creates rather than removes."""

    def __init__(self, months: list[dt.date]):
        self.months = months
        self.rows: dict[tuple[str, str, str], np.ndarray] = {}

    def add(self, department: str, category: str, vendor: str, col: int, amount: float) -> None:
        if amount == 0:
            return
        key = (department, category, vendor)
        vec = self.rows.setdefault(key, np.zeros(len(self.months)))
        vec[col] += amount

    def frame(self) -> pd.DataFrame:
        if not self.rows:
            return pd.DataFrame(columns=LINE_KEYS + [fiscal.label(m) for m in self.months])
        data = [
            {"department": d, "category": c, "vendor": v,
             **{fiscal.label(m): vec[i] for i, m in enumerate(self.months)}}
            for (d, c, v), vec in self.rows.items()
        ]
        return pd.DataFrame(data)

    def total(self) -> float:
        return float(sum(vec.sum() for vec in self.rows.values()))


def _spread(target_amount: float, weights: np.ndarray) -> np.ndarray:
    total = weights.sum()
    if total <= 0:
        return np.zeros_like(weights)
    return target_amount * weights / total


def simulate(scenario: Scenario, baseline: Baseline, config: EngineConfig | None = None) -> ScenarioResult:
    config = config or baseline.config
    lines = baseline.lines
    months = baseline.months
    month_index = {m: i for i, m in enumerate(months)}

    scen = baseline.matrix.copy()
    additions = _Additions(months)
    notice = _notice_by_line(lines)
    floor_ratio = _floor_ratio_by_line(lines)
    departments = lines["department"].to_numpy()
    categories = lines["category"].to_numpy()
    vendors = lines["vendor"].to_numpy()

    effects = {
        "intended_reduction_usd": 0.0,
        "blocked_by_minimum_commitment_usd": 0.0,
        "deferred_by_notice_usd": 0.0,
        "realized_reduction_usd": 0.0,
        "increase_usd": 0.0,
        "contractor_backfill_usd": 0.0,
        "shift_reinvestment_usd": 0.0,
        "one_time_exit_fees_usd": 0.0,
    }
    warnings: list[str] = []
    binding_floors: set[str] = set()
    deferred_vendors: set[str] = set()
    touched_categories: set[str] = set()

    for lever in scenario.levers:
        mask = _line_mask(lines, lever)
        if not mask.any():
            warnings.append(
                f"'{lever.description}' matched no ledger lines, so it changed nothing"
            )
            continue
        cols = [month_index[m] for m in lever.window.resolve()]
        rows = np.flatnonzero(mask)
        touched_categories.update(categories[rows].tolist())

        # per-line reduction realized by this lever, used for second-order effects
        realized_by_row = np.zeros(len(lines))
        first_effective_col: dict[int, int] = {}

        for col in cols:
            current = scen[:, col]
            desired = current.copy()

            if lever.kind == LeverKind.PERCENT_CHANGE:
                desired[rows] = current[rows] * (1.0 + lever.pct / 100.0)
            elif lever.kind == LeverKind.ABSOLUTE_MONTHLY:
                delta = _spread(lever.amount_usd_per_month, current[rows])
                desired[rows] = current[rows] + delta
            elif lever.kind == LeverKind.FREEZE:
                desired[rows] = np.minimum(current[rows], baseline.run_rate[rows])
            elif lever.kind == LeverKind.SHIFT:
                desired[rows] = current[rows] * (1.0 - lever.pct / 100.0)

            wanted = current[rows] - desired[rows]  # positive = a cut
            cut_rows = wanted > 0

            # notice periods: a reduction cannot land before the vendor's notice
            # has run from the start of the plan year.
            if config.honor_notice:
                blocked_by_notice = cut_rows & (col < notice[rows])
                if blocked_by_notice.any():
                    effects["deferred_by_notice_usd"] += float(wanted[blocked_by_notice].sum())
                    deferred_vendors.update(np.unique(vendors[rows][blocked_by_notice]).tolist())
                    wanted = np.where(blocked_by_notice, 0.0, wanted)

            # contractual minimum commitments
            if config.respect_floors:
                floors = floor_ratio[rows] * baseline.matrix[rows, col]
                allowed = np.maximum(current[rows] - floors, 0.0)
                over = (wanted > allowed) & (wanted > 0)
                if over.any():
                    effects["blocked_by_minimum_commitment_usd"] += float(
                        (wanted[over] - allowed[over]).sum()
                    )
                    binding_floors.update(np.unique(vendors[rows][over]).tolist())
                    wanted = np.where(wanted > allowed, allowed, wanted)

            increases = np.where(wanted < 0, -wanted, 0.0)
            reductions = np.where(wanted > 0, wanted, 0.0)
            effects["intended_reduction_usd"] += float(
                np.maximum(current[rows] - desired[rows], 0.0).sum()
            )
            effects["realized_reduction_usd"] += float(reductions.sum())
            effects["increase_usd"] += float(increases.sum())

            scen[rows, col] = current[rows] - wanted
            realized_by_row[rows] += reductions
            for local, row in enumerate(rows):
                if reductions[local] > 0 and row not in first_effective_col:
                    first_effective_col[row] = col

            # Second-order: contractor cuts partly come back as agency premium.
            # A shift lever already books an explicit replacement cost in the
            # destination category, so backfill would double-count it there.
            if config.contractor_backfill_ratio > 0 and lever.kind != LeverKind.SHIFT:
                contractor_rows = categories[rows] == "contractors"
                backfill = reductions * contractor_rows * config.contractor_backfill_ratio
                for local, row in enumerate(rows):
                    if backfill[local] > 0:
                        additions.add(
                            departments[row], "professional_services",
                            "Backfill cover", col, float(backfill[local]),
                        )
                        effects["contractor_backfill_usd"] += float(backfill[local])

            # shift: part of what is cut is spent again in the destination
            if lever.kind == LeverKind.SHIFT and reductions.sum() > 0:
                for local, row in enumerate(rows):
                    if reductions[local] <= 0:
                        continue
                    amount = reductions[local] * config.shift_efficiency
                    destination = _shift_destination_line(
                        lines, departments[row], lever.to_category
                    )
                    additions.add(destination[0], lever.to_category, destination[1], col, float(amount))
                    effects["shift_reinvestment_usd"] += float(amount)

        # one-time exit fees on contracted vendors that were actually reduced
        if config.exit_fee_months > 0:
            for row, col in first_effective_col.items():
                if notice[row] <= 0 or realized_by_row[row] <= 0:
                    continue
                monthly_average = realized_by_row[row] / max(len(cols), 1)
                fee = monthly_average * config.exit_fee_months
                additions.add(
                    departments[row], categories[row],
                    f"{vendors[row]} (exit fee)", col, float(fee),
                )
                effects["one_time_exit_fees_usd"] += float(fee)

    month_labels = [fiscal.label(m) for m in months]
    base_frame = lines.copy()
    scen_frame = lines.copy()
    for i, label in enumerate(month_labels):
        base_frame[label] = baseline.matrix[:, i]
        scen_frame[label] = scen[:, i]

    additions_frame = additions.frame()
    if len(additions_frame):
        zeros = {label: 0.0 for label in month_labels}
        padded = additions_frame.assign(**{k: additions_frame.get(k, 0.0) for k in zeros})
        base_pad = padded[LINE_KEYS].assign(**zeros)
        base_frame = pd.concat([base_frame, base_pad], ignore_index=True)
        scen_frame = pd.concat([scen_frame, padded[LINE_KEYS + month_labels]], ignore_index=True)

    monthly = pd.DataFrame(
        {
            "month": months,
            "baseline_usd": base_frame[month_labels].sum().to_numpy(),
            "scenario_usd": scen_frame[month_labels].sum().to_numpy(),
        }
    )
    monthly["delta_usd"] = monthly["scenario_usd"] - monthly["baseline_usd"]

    by_category = (
        pd.DataFrame(
            {
                "category": base_frame["category"],
                "baseline_usd": base_frame[month_labels].sum(axis=1),
                "scenario_usd": scen_frame[month_labels].sum(axis=1),
            }
        )
        .groupby("category", as_index=False, observed=True)
        .sum()
    )
    by_category["delta_usd"] = by_category["scenario_usd"] - by_category["baseline_usd"]
    by_category = by_category.sort_values("delta_usd").reset_index(drop=True)

    by_line = base_frame[LINE_KEYS].copy()
    by_line["baseline_usd"] = base_frame[month_labels].sum(axis=1)
    by_line["scenario_usd"] = scen_frame[month_labels].sum(axis=1)
    by_line["delta_usd"] = by_line["scenario_usd"] - by_line["baseline_usd"]
    by_line = by_line[by_line["delta_usd"].abs() > 0.005].sort_values("delta_usd")

    result = ScenarioResult(
        scenario=scenario,
        baseline=baseline,
        monthly=monthly,
        by_category=by_category,
        by_line=by_line.reset_index(drop=True),
        effects=effects,
        warnings=warnings,
    )
    result.assumptions = build_assumptions(
        scenario, baseline, config, effects, binding_floors, deferred_vendors, touched_categories
    )
    if effects["blocked_by_minimum_commitment_usd"] > 0:
        share = 100 * effects["blocked_by_minimum_commitment_usd"] / max(
            effects["intended_reduction_usd"], 1e-9
        )
        warnings.append(
            f"{share:.1f}% of the intended reduction is blocked by minimum "
            f"commitments on {', '.join(sorted(binding_floors))}"
        )
    if effects["deferred_by_notice_usd"] > 0:
        warnings.append(
            f"${effects['deferred_by_notice_usd']:,.0f} of the cut cannot start "
            f"inside the window because of notice periods on "
            f"{', '.join(sorted(deferred_vendors))}"
        )
    return result


def _shift_destination_line(lines: pd.DataFrame, department: str, category: str) -> tuple[str, str]:
    """Where reinvested spend lands: the biggest existing line in that
    department and category, or a new unallocated line if there is none."""
    candidates = lines[(lines["department"] == department) & (lines["category"] == category)]
    if len(candidates):
        return department, str(candidates.iloc[0]["vendor"])
    return department, "(unallocated)"


# --- assumptions ------------------------------------------------------------


def build_assumptions(
    scenario: Scenario,
    baseline: Baseline,
    config: EngineConfig,
    effects: dict[str, float],
    binding_floors: set[str],
    deferred_vendors: set[str],
    touched_categories: set[str],
) -> list[Assumption]:
    defaults = EngineConfig()
    out: list[Assumption] = []

    def source_for(field_name: str) -> str:
        return "override" if getattr(config, field_name) != getattr(defaults, field_name) else "default"

    out.append(
        Assumption(
            "baseline_run_rate",
            "Baseline run rate",
            f"${baseline.run_rate.sum():,.0f}/month over {len(baseline.lines)} ledger lines",
            "derived",
            "Each line starts from its own trailing-12-month average "
            f"({fiscal.label(fiscal.add_months(fiscal.HISTORY_END, -11))}.."
            f"{fiscal.label(fiscal.HISTORY_END)}).",
        )
    )
    weighted_growth = float(
        np.average(baseline.growth, weights=np.maximum(baseline.run_rate, 1e-9))
    )
    out.append(
        Assumption(
            "growth",
            "Growth carried into the plan",
            f"{weighted_growth * 100:+.1f}%/yr weighted average",
            "derived",
            "Fitted per line as (last 6 months / prior 6 months) squared, clipped to "
            f"[{config.growth_clip_low:+.0%}, {config.growth_clip_high:+.0%}]. Two years "
            "of history is not enough to separate trend from noise, so the clip is "
            "doing real work here.",
        )
    )
    out.append(
        Assumption(
            "seasonality",
            "Seasonality",
            "on (per-category index)" if config.use_seasonality else "off (flat months)",
            source_for("use_seasonality"),
            "Index estimated from 24 months of actuals, so each calendar month has "
            "only two observations behind it.",
        )
    )
    out.append(
        Assumption(
            "fiscal_calendar",
            "Fiscal calendar",
            f"FY starts in {dt.date(2000, fiscal.FY_START_MONTH, 1):%B}",
            "structural",
            "Quarters in the question resolve against the fiscal year, not the "
            f"calendar year: {fiscal.quarter_label(3)}.",
        )
    )
    if binding_floors or config.respect_floors:
        out.append(
            Assumption(
                "minimum_commitments",
                "Contractual minimums",
                "enforced" if config.respect_floors else "ignored",
                source_for("respect_floors"),
                (
                    "Binding on " + ", ".join(sorted(binding_floors))
                    if binding_floors
                    else "No vendor in this scenario hits its contracted floor."
                ),
            )
        )
    if deferred_vendors or config.honor_notice:
        out.append(
            Assumption(
                "notice_periods",
                "Notice periods",
                "honoured" if config.honor_notice else "ignored",
                source_for("honor_notice"),
                (
                    "Reductions start after notice runs from the first month of the "
                    "plan year. Deferred: " + ", ".join(sorted(deferred_vendors))
                    if deferred_vendors
                    else "No reduction in this scenario falls inside a notice period."
                ),
            )
        )
    if "contractors" in touched_categories and any(
        l.kind != LeverKind.SHIFT for l in scenario.levers
    ):
        out.append(
            Assumption(
                "contractor_backfill",
                "Contractor backfill",
                f"{config.contractor_backfill_ratio:.0%} of any contractor cut returns as cost",
                source_for("contractor_backfill_ratio"),
                "Booked to professional services in the same department. This is a "
                "planning rule of thumb, not something measured from the ledger.",
            )
        )
    if effects["one_time_exit_fees_usd"] > 0 or config.exit_fee_months != defaults.exit_fee_months:
        out.append(
            Assumption(
                "exit_fees",
                "Exit fees",
                f"{config.exit_fee_months:g} months of the monthly reduction, once",
                source_for("exit_fee_months"),
                "Charged only on vendors that have a notice period, in the first "
                "month the reduction actually lands.",
            )
        )
    if any(l.kind == LeverKind.SHIFT for l in scenario.levers):
        out.append(
            Assumption(
                "shift_efficiency",
                "Shift efficiency",
                f"{config.shift_efficiency:.0%} of what is cut is spent in the destination",
                source_for("shift_efficiency"),
                "Replacing people with tooling is assumed to cost less than the people, "
                "which is the entire bet and the number most worth arguing about.",
            )
        )
    if len(scenario.levers) > 1:
        out.append(
            Assumption(
                "lever_order",
                "Lever order",
                f"{len(scenario.levers)} levers applied in the order written",
                "structural",
                "Overlapping levers compound: a second percentage applies to the "
                "value the first one left behind.",
            )
        )
    out.append(
        Assumption(
            "no_new_lines",
            "Plan composition",
            "no new vendors or departments",
            "structural",
            "The plan year contains the same 57 ledger lines as the trailing year. "
            "New spend the business has already committed to is not in here.",
        )
    )
    out.append(
        Assumption(
            "nominal_usd",
            "Currency",
            "nominal USD, no inflation adjustment",
            "structural",
            "Price rises only enter through the fitted per-line growth.",
        )
    )
    return out


def reconcile(result: ScenarioResult) -> float:
    """The decomposition must add up to the headline delta. Used by the tests."""
    e = result.effects
    rebuilt = (
        -e["realized_reduction_usd"]
        + e["increase_usd"]
        + e["contractor_backfill_usd"]
        + e["shift_reinvestment_usd"]
        + e["one_time_exit_fees_usd"]
    )
    return rebuilt - result.net_delta


def compare(results: list[ScenarioResult]) -> pd.DataFrame:
    """Side-by-side table for two or more runs against the same baseline."""
    rows = []
    for result in results:
        summary = result.summary()
        rows.append(
            {
                "scenario": summary["scenario"],
                "net saving $": summary["net_savings_usd"],
                "net saving %": summary["net_savings_pct"],
                "gross cut $": summary["realized_reduction_usd"],
                "blocked $": summary["blocked_by_minimum_commitment_usd"],
                "deferred $": summary["deferred_by_notice_usd"],
                "backfill $": summary["contractor_backfill_usd"],
                "reinvested $": summary["shift_reinvestment_usd"],
                "one-time $": summary["one_time_exit_fees_usd"],
            }
        )
    return pd.DataFrame(rows)
