import numpy as np
import pytest

from budget_agent import fiscal, vocab
from budget_agent.engine import (
    EngineConfig, build_baseline, compare, reconcile, simulate,
)
from budget_agent.models import Lever, Scenario, Target, Window


def scenario(*levers, name="test"):
    return Scenario(name=name, levers=list(levers))


def cut(category, pct=-10, **window):
    return Lever(
        kind="percent_change",
        target=Target(categories=[category]),
        window=Window(**window) if window else Window(),
        pct=pct,
    )


# --- baseline ---------------------------------------------------------------


def test_baseline_covers_the_plan_year(baseline):
    assert baseline.matrix.shape == (57, 12)
    assert baseline.months == fiscal.PLAN_MONTHS
    assert (baseline.matrix > 0).all()


def test_with_growth_and_seasonality_off_the_plan_is_the_run_rate(flat_baseline):
    expected = np.repeat(flat_baseline.run_rate[:, None], 12, axis=1)
    np.testing.assert_allclose(flat_baseline.matrix, expected, rtol=1e-12)


def test_seasonality_indices_average_one(baseline):
    np.testing.assert_allclose(baseline.seasonality.mean(axis=1).to_numpy(), 1.0, rtol=1e-9)


def test_growth_is_clipped(baseline):
    assert baseline.growth.min() >= baseline.config.growth_clip_low - 1e-12
    assert baseline.growth.max() <= baseline.config.growth_clip_high + 1e-12


def test_baseline_is_deterministic():
    assert build_baseline().total == build_baseline().total


# --- percent changes --------------------------------------------------------


def test_an_unconstrained_cut_lands_in_full(flat_baseline):
    """Voyager Travel Desk has no floor and no notice, so 10% off travel is
    exactly 10% off travel."""
    result = simulate(scenario(cut("travel", -10)), flat_baseline)
    travel = result.by_category.set_index("category").loc["travel"]
    assert travel["delta_usd"] == pytest.approx(-0.10 * travel["baseline_usd"], rel=1e-9)


def test_an_increase_adds_spend(flat_baseline):
    result = simulate(scenario(cut("travel", 25)), flat_baseline)
    assert result.net_delta > 0
    assert result.effects["increase_usd"] == pytest.approx(result.net_delta, rel=1e-9)


def test_a_quarter_window_only_touches_three_months(flat_baseline):
    result = simulate(scenario(cut("travel", -20, kind="quarter", quarter=3)), flat_baseline)
    changed = result.monthly[result.monthly["delta_usd"].abs() > 0.005]
    assert len(changed) == 3
    assert [m.isoformat() for m in changed["month"]] == [
        "2027-01-01", "2027-02-01", "2027-03-01"
    ]


# --- constraints ------------------------------------------------------------


def test_a_contractual_floor_blocks_part_of_the_cut(flat_baseline):
    """Harborview Properties is a lease with a 95% floor."""
    result = simulate(
        scenario(Lever(kind="percent_change",
                       target=Target(vendors=["Harborview Properties"]), pct=-50)),
        flat_baseline,
    )
    assert result.effects["blocked_by_minimum_commitment_usd"] > 0
    achieved = -result.by_line["delta_usd"].sum()
    baseline_amount = result.by_line["baseline_usd"].sum()
    assert achieved <= 0.05 * baseline_amount + 1e-6


def test_floors_can_be_switched_off(flat_baseline):
    config = EngineConfig(
        use_seasonality=False, growth_clip_low=0.0, growth_clip_high=0.0,
        respect_floors=False, honor_notice=False, exit_fee_months=0,
    )
    result = simulate(
        scenario(Lever(kind="percent_change",
                       target=Target(vendors=["Harborview Properties"]), pct=-50)),
        flat_baseline, config,
    )
    assert result.effects["blocked_by_minimum_commitment_usd"] == 0
    assert result.effects["realized_reduction_usd"] == pytest.approx(
        0.5 * result.by_line["baseline_usd"].sum(), rel=1e-9
    )


def test_a_notice_period_defers_the_whole_cut(flat_baseline):
    """Kestrel CRM needs six months' notice, so nothing lands in Q1."""
    result = simulate(
        scenario(Lever(kind="percent_change", target=Target(vendors=["Kestrel CRM"]),
                       window=Window(kind="quarter", quarter=1), pct=-30)),
        flat_baseline,
    )
    assert result.effects["realized_reduction_usd"] == 0
    assert result.effects["deferred_by_notice_usd"] > 0
    assert result.net_delta == pytest.approx(0.0, abs=1e-6)


def test_the_same_cut_lands_in_q3_when_notice_has_run(flat_baseline):
    result = simulate(
        scenario(Lever(kind="percent_change", target=Target(vendors=["Kestrel CRM"]),
                       window=Window(kind="quarter", quarter=3), pct=-30)),
        flat_baseline,
    )
    assert result.effects["realized_reduction_usd"] > 0


def test_notice_can_be_switched_off(flat_baseline):
    config = EngineConfig(
        use_seasonality=False, growth_clip_low=0.0, growth_clip_high=0.0,
        honor_notice=False,
    )
    result = simulate(
        scenario(Lever(kind="percent_change", target=Target(vendors=["Kestrel CRM"]),
                       window=Window(kind="quarter", quarter=1), pct=-30)),
        flat_baseline, config,
    )
    assert result.effects["deferred_by_notice_usd"] == 0
    assert result.effects["realized_reduction_usd"] > 0


# --- second-order effects ---------------------------------------------------


def test_contractor_backfill_is_a_fixed_share_of_what_is_cut(flat_baseline):
    config = EngineConfig(
        use_seasonality=False, growth_clip_low=0.0, growth_clip_high=0.0,
        contractor_backfill_ratio=0.5, exit_fee_months=0,
    )
    result = simulate(scenario(cut("contractors", -20)), flat_baseline, config)
    assert result.effects["contractor_backfill_usd"] == pytest.approx(
        0.5 * result.effects["realized_reduction_usd"], rel=1e-9
    )


def test_backfill_lands_in_professional_services(flat_baseline):
    result = simulate(scenario(cut("contractors", -20)), flat_baseline)
    by_category = result.by_category.set_index("category")
    assert by_category.loc["contractors", "delta_usd"] < 0
    assert by_category.loc["professional_services", "delta_usd"] > 0


def test_cutting_a_category_without_contractors_creates_no_backfill(flat_baseline):
    result = simulate(scenario(cut("travel", -20)), flat_baseline)
    assert result.effects["contractor_backfill_usd"] == 0


def test_exit_fees_are_charged_once_per_line(flat_baseline):
    config = EngineConfig(
        use_seasonality=False, growth_clip_low=0.0, growth_clip_high=0.0,
        exit_fee_months=1.0, contractor_backfill_ratio=0,
    )
    result = simulate(
        scenario(Lever(kind="percent_change", target=Target(vendors=["Halyard QA Services"]),
                       window=Window(kind="half", half=2), pct=-10)),
        flat_baseline, config,
    )
    monthly_reduction = result.effects["realized_reduction_usd"] / 6
    assert result.effects["one_time_exit_fees_usd"] == pytest.approx(
        monthly_reduction, rel=1e-9
    )


def test_a_vendor_without_notice_pays_no_exit_fee(flat_baseline):
    result = simulate(
        scenario(Lever(kind="percent_change",
                       target=Target(vendors=["Voyager Travel Desk"]), pct=-10)),
        flat_baseline,
    )
    assert result.effects["one_time_exit_fees_usd"] == 0


# --- other lever kinds ------------------------------------------------------


def test_a_freeze_caps_spend_at_the_run_rate(baseline):
    result = simulate(
        scenario(Lever(kind="freeze", target=Target(categories=["travel"]))), baseline
    )
    assert result.net_delta < 0
    merged = result.by_line[result.by_line["category"] == "travel"]
    assert (merged["scenario_usd"] <= merged["baseline_usd"] + 1e-6).all()


def test_a_freeze_never_raises_a_month_above_baseline(baseline):
    result = simulate(
        scenario(Lever(kind="freeze", target=Target(categories=["hardware"]))), baseline
    )
    assert (result.monthly["delta_usd"] <= 1e-6).all()


def test_an_absolute_cut_is_exactly_the_amount_asked_for(flat_baseline):
    result = simulate(
        scenario(Lever(kind="absolute_monthly", target=Target(categories=["travel"]),
                       window=Window(kind="quarter", quarter=2),
                       amount_usd_per_month=-40_000)),
        flat_baseline,
    )
    assert result.net_delta == pytest.approx(-120_000, rel=1e-9)


def test_a_shift_reinvests_at_the_configured_efficiency(flat_baseline):
    config = EngineConfig(
        use_seasonality=False, growth_clip_low=0.0, growth_clip_high=0.0,
        shift_efficiency=0.6, exit_fee_months=0,
    )
    result = simulate(
        scenario(Lever(kind="shift", target=Target(categories=["contractors"]),
                       pct=30, to_category="software")),
        flat_baseline, config,
    )
    assert result.effects["shift_reinvestment_usd"] == pytest.approx(
        0.6 * result.effects["realized_reduction_usd"], rel=1e-9
    )
    by_category = result.by_category.set_index("category")
    assert by_category.loc["software", "delta_usd"] > 0
    assert result.effects["contractor_backfill_usd"] == 0


def test_a_shift_at_efficiency_one_is_cost_neutral_before_fees(flat_baseline):
    config = EngineConfig(
        use_seasonality=False, growth_clip_low=0.0, growth_clip_high=0.0,
        shift_efficiency=1.0, exit_fee_months=0, honor_notice=False,
    )
    result = simulate(
        scenario(Lever(kind="shift", target=Target(categories=["contractors"]),
                       pct=30, to_category="software")),
        flat_baseline, config,
    )
    assert result.net_delta == pytest.approx(0.0, abs=1e-6)


def test_levers_compound_in_order(flat_baseline):
    once = simulate(scenario(cut("travel", -10)), flat_baseline)
    twice = simulate(scenario(cut("travel", -10), cut("travel", -10)), flat_baseline)
    assert twice.net_delta < once.net_delta
    assert twice.net_delta == pytest.approx(once.net_delta * 1.9, rel=1e-9)


# --- decomposition ----------------------------------------------------------


@pytest.mark.parametrize(
    "levers",
    [
        (cut("contractors", -15, kind="quarter", quarter=3),),
        (cut("facilities", -20),),
        (cut("cloud", 20, kind="half", half=2), cut("marketing_programs", -10)),
        (Lever(kind="freeze", target=Target(categories=["travel"])),),
        (Lever(kind="shift", target=Target(categories=["contractors"]),
               pct=30, to_category="software"),),
        (Lever(kind="absolute_monthly", target=Target(categories=["travel"]),
               amount_usd_per_month=-25_000),),
    ],
)
def test_the_decomposition_reconciles_to_the_headline(baseline, levers):
    result = simulate(scenario(*levers), baseline)
    assert reconcile(result) == pytest.approx(0.0, abs=0.01)


def test_a_no_op_target_warns_instead_of_crashing(baseline):
    result = simulate(
        scenario(Lever(kind="percent_change",
                       target=Target(departments=["People"], categories=["cloud"]),
                       pct=-10)),
        baseline,
    )
    assert result.net_delta == 0
    assert any("matched no ledger lines" in w for w in result.warnings)


# --- assumptions ------------------------------------------------------------


def test_every_result_reports_its_assumptions(baseline):
    result = simulate(scenario(cut("contractors", -15)), baseline)
    keys = {a.key for a in result.assumptions}
    assert {"baseline_run_rate", "growth", "seasonality", "fiscal_calendar"} <= keys
    assert all(a.detail for a in result.assumptions)


def test_a_changed_knob_is_marked_as_an_override(baseline):
    config = EngineConfig(contractor_backfill_ratio=0.75)
    result = simulate(scenario(cut("contractors", -15)), baseline, config)
    backfill = next(a for a in result.assumptions if a.key == "contractor_backfill")
    assert backfill.source == "override"
    assert "75%" in backfill.value


def test_untouched_knobs_stay_marked_as_defaults(baseline):
    result = simulate(scenario(cut("contractors", -15)), baseline)
    backfill = next(a for a in result.assumptions if a.key == "contractor_backfill")
    assert backfill.source == "default"


def test_shift_efficiency_is_only_listed_for_shifts(baseline):
    plain = simulate(scenario(cut("contractors", -15)), baseline)
    assert not any(a.key == "shift_efficiency" for a in plain.assumptions)
    shifted = simulate(
        scenario(Lever(kind="shift", target=Target(categories=["contractors"]),
                       pct=30, to_category="software")),
        baseline,
    )
    assert any(a.key == "shift_efficiency" for a in shifted.assumptions)


def test_binding_floors_are_named_in_the_assumption(baseline):
    result = simulate(
        scenario(Lever(kind="percent_change",
                       target=Target(vendors=["Harborview Properties"]), pct=-50)),
        baseline,
    )
    minimums = next(a for a in result.assumptions if a.key == "minimum_commitments")
    assert "Harborview Properties" in minimums.detail


# --- comparison -------------------------------------------------------------


def test_compare_puts_two_runs_side_by_side(baseline):
    results = [
        simulate(scenario(cut("contractors", -15, kind="quarter", quarter=3), name="A"), baseline),
        simulate(scenario(Lever(kind="freeze", target=Target(categories=["travel"])), name="B"), baseline),
    ]
    table = compare(results)
    assert list(table["scenario"]) == ["A", "B"]
    assert "net saving $" in table.columns
    assert len(table) == 2


def test_summary_is_json_friendly(baseline):
    summary = simulate(scenario(cut("contractors", -15)), baseline).summary()
    import json

    assert json.loads(json.dumps(summary))["scenario"] == "test"
    assert summary["net_savings_usd"] > 0
