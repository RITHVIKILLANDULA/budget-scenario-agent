import pytest

from budget_agent import fiscal
from budget_agent.parser import parse_question


def lever(question):
    result = parse_question(question)
    assert result.ok, result.errors
    return result.draft["levers"][0]


def test_the_headline_question():
    result = parse_question("what happens if we cut contractor spend 15% in Q3")
    assert result.ok
    (one,) = result.draft["levers"]
    assert one["kind"] == "percent_change"
    assert one["pct"] == -15.0
    assert one["target"]["categories"] == ["contractors"]
    assert one["window"] == {"kind": "quarter", "quarter": 3}
    assert result.coverage == 1.0


def test_quarter_resolves_on_the_fiscal_calendar():
    result = parse_question("cut travel 10% in Q3")
    assert any("2027-01" in line for line in result.trace)


@pytest.mark.parametrize(
    "question,expected",
    [
        ("cut travel by 10%", -10.0),
        ("reduce travel by 10 percent", -10.0),
        ("increase travel by 10%", 10.0),
        ("grow travel 10%", 10.0),
        ("trim travel by half", -50.0),
        ("cut travel by a quarter", -25.0),
        ("travel 12%", -12.0),
    ],
)
def test_magnitude_and_direction(question, expected):
    assert lever(question)["pct"] == pytest.approx(expected, rel=1e-3)


@pytest.mark.parametrize(
    "question,kind",
    [
        ("freeze travel for the year", "freeze"),
        ("hold travel flat", "freeze"),
        ("cut travel by 10%", "percent_change"),
        ("cut $40k a month from travel", "absolute_monthly"),
        ("move 30% of contractor spend into software", "shift"),
        ("reallocate 20% of contractors to cloud", "shift"),
    ],
)
def test_lever_kind(question, kind):
    assert lever(question)["kind"] == kind


def test_dollar_amount_per_month_is_taken_literally():
    assert lever("cut $40k a month from travel in Q2")["amount_usd_per_month"] == -40_000


def test_dollar_amount_without_a_period_is_spread_over_the_window():
    result = parse_question("take $480k out of travel")
    one = result.draft["levers"][0]
    assert one["amount_usd_per_month"] == pytest.approx(-40_000)
    assert any("12-month window" in line for line in result.trace)


@pytest.mark.parametrize(
    "question,kind,size",
    [("cut travel 10% in Q1", "quarter", 3), ("cut travel 10% in h2", "half", 6),
     ("cut travel 10%", "year", 12), ("cut travel 10% in march", "months", 1)],
)
def test_window_kinds(question, kind, size):
    window = lever(question)["window"]
    assert window["kind"] == kind
    months = {
        "quarter": lambda w: fiscal.quarter_months(w["quarter"]),
        "half": lambda w: fiscal.half_months(w["half"]),
        "year": lambda w: fiscal.PLAN_MONTHS,
        "months": lambda w: w["months"],
    }[kind](window)
    assert len(months) == size


def test_bare_marketing_reads_as_the_category():
    target = lever("cut marketing by 10%")["target"]
    assert target["categories"] == ["marketing_programs"]
    assert target["departments"] == []


def test_in_marketing_reads_as_the_department():
    target = lever("cut travel in marketing by 10%")["target"]
    assert target["departments"] == ["Marketing"]
    assert target["categories"] == ["travel"]


def test_the_ambiguity_is_written_into_the_trace():
    result = parse_question("cut marketing by 10%")
    assert any("ambiguous" in line or "matches both" in line for line in result.trace)


def test_vendor_by_alias():
    assert lever("cut brightpath by 20%")["target"]["vendors"] == ["Brightpath Staffing"]


def test_two_clauses_become_two_levers():
    result = parse_question("increase cloud by 20% in H2 and cut marketing programs by 10%")
    assert len(result.draft["levers"]) == 2
    assert result.draft["levers"][0]["pct"] == 20.0
    assert result.draft["levers"][1]["pct"] == -10.0


def test_and_inside_a_target_does_not_split():
    result = parse_question("cut travel and hardware by 10%")
    assert len(result.draft["levers"]) == 1
    assert set(result.draft["levers"][0]["target"]["categories"]) == {"travel", "hardware"}


def test_shift_destination_is_not_also_the_source():
    one = lever("move 30% of contractor spend into software")
    assert one["target"]["categories"] == ["contractors"]
    assert one["to_category"] == "software"


def test_unknown_target_is_rejected():
    result = parse_question("what if we cut the Atlantis team by 10%")
    assert not result.ok
    assert "Atlantis" in result.errors[0] or "atlantis" in result.errors[0]


def test_missing_magnitude_is_rejected():
    result = parse_question("cut contractors")
    assert not result.ok
    assert "no size" in result.errors[0]


def test_empty_question_is_rejected():
    assert not parse_question("   ").ok


def test_nonsense_is_rejected_without_a_traceback():
    result = parse_question("how tall is the empire state building")
    assert not result.ok
    assert result.draft is None


def test_across_the_board_needs_no_named_target():
    one = lever("cut everything by 5%")
    assert one["target"] == {"departments": [], "categories": [], "vendors": []}


def test_ignored_words_are_reported():
    result = parse_question("cut contractor spend 15% in Q3 please by friday")
    assert result.ok
    assert "friday" in result.ignored_words
    assert result.coverage < 1.0


def test_scenario_name_is_readable():
    assert parse_question("cut contractor spend 15% in Q3").draft["name"] == (
        "Contractors -15% Q3"
    )
    assert "->" in parse_question("move 30% of contractors to software").draft["name"]
