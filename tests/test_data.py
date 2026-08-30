import pandas as pd
import pytest

from budget_agent import fiscal, vocab
from budget_agent.data import DEFAULT_SEED, generate_ledger, ledger_summary, vendor_terms


def test_ledger_shape():
    ledger = generate_ledger()
    assert list(ledger.columns) == [
        "month", "fiscal_year", "fiscal_quarter", "department",
        "category", "vendor", "amount_usd",
    ]
    assert ledger["month"].nunique() == 24
    assert ledger["month"].min() == pd.Timestamp(fiscal.HISTORY_START)
    assert ledger["month"].max() == pd.Timestamp(fiscal.HISTORY_END)
    assert (ledger["amount_usd"] > 0).all()


def test_every_vendor_and_department_appears():
    ledger = generate_ledger()
    assert set(ledger["vendor"]) == set(vocab.VENDOR_KEYS)
    assert set(ledger["department"]) == set(vocab.DEPARTMENT_KEYS)
    assert set(ledger["category"]) == set(vocab.CATEGORY_KEYS)


def test_allocations_sum_to_one():
    for spec in vocab.VENDORS:
        assert abs(sum(spec.allocation.values()) - 1.0) < 1e-9, spec.name


def test_same_seed_same_numbers():
    a, b = generate_ledger(DEFAULT_SEED), generate_ledger(DEFAULT_SEED)
    pd.testing.assert_frame_equal(a, b)


def test_different_seed_different_numbers():
    a, b = generate_ledger(DEFAULT_SEED), generate_ledger(DEFAULT_SEED + 1)
    assert a["amount_usd"].sum() != b["amount_usd"].sum()
    assert len(a) == len(b)


def test_cloud_step_change_is_visible():
    """Northwind Cloud steps up 22% from 2025-04; it should survive the noise."""
    ledger = generate_ledger()
    northwind = ledger[ledger["vendor"] == "Northwind Cloud"]
    before = northwind[
        (northwind["month"] >= "2024-10-01") & (northwind["month"] < "2025-04-01")
    ]["amount_usd"].sum()
    after = northwind[
        (northwind["month"] >= "2025-04-01") & (northwind["month"] < "2025-10-01")
    ]["amount_usd"].sum()
    assert 1.15 < after / before < 1.40


def test_travel_is_seasonal():
    ledger = generate_ledger()
    travel = ledger[ledger["category"] == "travel"].copy()
    travel["cal_month"] = travel["month"].dt.month
    by_month = travel.groupby("cal_month")["amount_usd"].mean()
    assert by_month.idxmin() == 12  # nobody travels in December
    assert by_month.max() / by_month.min() > 1.5


def test_vendor_terms_cover_every_vendor():
    terms = vendor_terms()
    assert set(terms["vendor"]) == set(vocab.VENDOR_KEYS)
    assert terms["min_commit_ratio"].between(0, 1).all()
    assert (terms["notice_months"] >= 0).all()


def test_summary_reports_the_trailing_year():
    summary = ledger_summary(generate_ledger())
    assert summary["months"] == 24
    assert summary["lines"] == 57
    assert summary["ttm_usd"] < summary["total_usd"]
    assert summary["first_month"] == "2024-07"
    assert summary["last_month"] == "2026-06"


@pytest.mark.parametrize(
    "date_str,expected_fy,expected_q",
    [("2026-07-01", 2027, 1), ("2026-12-01", 2027, 2), ("2027-01-01", 2027, 3),
     ("2027-06-01", 2027, 4), ("2026-06-01", 2026, 4)],
)
def test_fiscal_calendar(date_str, expected_fy, expected_q):
    import datetime as dt

    date = dt.date.fromisoformat(date_str)
    assert fiscal.fiscal_year(date) == expected_fy
    assert fiscal.fiscal_quarter(date) == expected_q


def test_q3_is_the_calendar_new_year():
    months = fiscal.quarter_months(3, 2027)
    assert [m.isoformat() for m in months] == ["2027-01-01", "2027-02-01", "2027-03-01"]
