import pytest
from pydantic import ValidationError

from budget_agent import fiscal
from budget_agent.models import Lever, LeverKind, Scenario, Target, Window


def build(**overrides):
    payload = {
        "name": "test",
        "levers": [
            {
                "kind": "percent_change",
                "target": {"categories": ["contractors"]},
                "pct": -15,
            }
        ],
    }
    payload.update(overrides)
    return Scenario.model_validate(payload)


def test_a_valid_scenario_round_trips():
    scenario = build()
    assert scenario.levers[0].pct == -15
    assert scenario.model_dump(mode="json")["levers"][0]["kind"] == "percent_change"


def test_department_names_are_canonicalised():
    target = Target(departments=["engineering", "FINANCE"])
    assert target.departments == ("Engineering", "Finance")


def test_unknown_department_is_rejected_with_a_suggestion():
    with pytest.raises(ValidationError) as caught:
        Target(departments=["Engneering"])
    message = str(caught.value)
    assert "unknown department" in message
    assert "Did you mean Engineering" in message


def test_unknown_category_is_rejected():
    with pytest.raises(ValidationError, match="unknown category"):
        Target(categories=["crypto"])


def test_vendor_outside_the_named_category_is_rejected():
    with pytest.raises(ValidationError, match="selects nothing"):
        Target(vendors=["Voyager Travel Desk"], categories=["cloud"])


def test_a_cut_deeper_than_everything_is_rejected():
    with pytest.raises(ValidationError):
        Lever(kind="percent_change", target=Target(categories=["travel"]), pct=-150)


def test_a_zero_percent_change_is_rejected():
    with pytest.raises(ValidationError, match="not a scenario"):
        Lever(kind="percent_change", target=Target(categories=["travel"]), pct=0)


def test_a_scenario_needs_at_least_one_lever():
    with pytest.raises(ValidationError):
        build(levers=[])


def test_a_freeze_takes_no_magnitude():
    with pytest.raises(ValidationError, match="holds the run rate flat"):
        Lever(kind="freeze", target=Target(categories=["travel"]), pct=-10)


def test_a_shift_needs_a_destination():
    with pytest.raises(ValidationError, match="destination category"):
        Lever(kind="shift", target=Target(categories=["contractors"]), pct=30)


def test_a_shift_cannot_target_its_own_category():
    with pytest.raises(ValidationError, match="into itself"):
        Lever(
            kind="shift", target=Target(categories=["contractors"]),
            pct=30, to_category="contractors",
        )


def test_a_shift_needs_an_explicit_source():
    with pytest.raises(ValidationError, match="explicit source"):
        Lever(kind="shift", target=Target(departments=["Engineering"]),
              pct=30, to_category="software")


def test_an_absolute_lever_needs_a_dollar_amount():
    with pytest.raises(ValidationError, match="non-zero dollar"):
        Lever(kind="absolute_monthly", target=Target(categories=["travel"]))


def test_a_year_outside_the_horizon_is_rejected():
    with pytest.raises(ValidationError, match="outside the planning horizon"):
        Window(kind="quarter", quarter=3, fiscal_year=2031)


def test_a_bad_quarter_is_rejected():
    with pytest.raises(ValidationError, match="quarter must be 1-4"):
        Window(kind="quarter", quarter=7)


def test_a_month_outside_the_plan_year_is_rejected():
    import datetime as dt

    with pytest.raises(ValidationError, match="outside FY2027"):
        Window(kind="months", months=[dt.date(2025, 1, 1)])


@pytest.mark.parametrize(
    "window,expected",
    [
        (Window(), 12),
        (Window(kind="quarter", quarter=2), 3),
        (Window(kind="half", half=1), 6),
    ],
)
def test_window_resolution(window, expected):
    assert len(window.resolve()) == expected


def test_q3_label_names_the_calendar_months():
    assert Window(kind="quarter", quarter=3).label == "Q3 FY2027 (2027-01..2027-03)"


def test_lever_descriptions_read_like_english():
    assert Lever(
        kind="percent_change", target=Target(categories=["contractors"]),
        window=Window(kind="quarter", quarter=3), pct=-15,
    ).description == "cut Contractors by 15% over Q3 FY2027 (2027-01..2027-03)"


def test_an_empty_target_means_everything():
    assert Target().is_everything
    assert Target().label == "all discretionary spend"


def test_models_are_frozen():
    scenario = build()
    with pytest.raises(ValidationError):
        scenario.levers[0].pct = -20
