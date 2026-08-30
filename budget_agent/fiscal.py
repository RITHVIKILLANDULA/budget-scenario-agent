"""Fiscal calendar helpers.

The company in this dataset runs a July-June fiscal year, which is the whole
reason quarter parsing is interesting: "Q3" in a budget question means
January-March, not July-September. Getting that wrong silently is exactly the
kind of thing this tool is supposed to make visible.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

FY_START_MONTH = 7  # July

# 24 months of actuals, then a 12-month planning horizon.
HISTORY_START = dt.date(2024, 7, 1)
HISTORY_END = dt.date(2026, 6, 1)  # inclusive, last closed month
PLAN_FY = 2027  # FY27 = 2026-07 .. 2027-06


def month_start(d: dt.date) -> dt.date:
    return dt.date(d.year, d.month, 1)


def add_months(d: dt.date, n: int) -> dt.date:
    total = (d.year * 12 + d.month - 1) + n
    return dt.date(total // 12, total % 12 + 1, 1)


def month_range(start: dt.date, end: dt.date) -> list[dt.date]:
    """Inclusive list of month-start dates."""
    if end < start:
        return []
    out, cur = [], month_start(start)
    end = month_start(end)
    while cur <= end:
        out.append(cur)
        cur = add_months(cur, 1)
    return out


def fiscal_year(d: dt.date) -> int:
    """FY label for a date. 2026-07-01 -> 2027."""
    return d.year + 1 if d.month >= FY_START_MONTH else d.year


def fiscal_quarter(d: dt.date) -> int:
    offset = (d.month - FY_START_MONTH) % 12
    return offset // 3 + 1


def fy_months(fy: int = PLAN_FY) -> list[dt.date]:
    start = dt.date(fy - 1, FY_START_MONTH, 1)
    return [add_months(start, i) for i in range(12)]


def quarter_months(quarter: int, fy: int = PLAN_FY) -> list[dt.date]:
    if quarter not in (1, 2, 3, 4):
        raise ValueError(f"quarter must be 1-4, got {quarter}")
    start = dt.date(fy - 1, FY_START_MONTH, 1)
    first = add_months(start, (quarter - 1) * 3)
    return [add_months(first, i) for i in range(3)]


def half_months(half: int, fy: int = PLAN_FY) -> list[dt.date]:
    if half not in (1, 2):
        raise ValueError(f"half must be 1 or 2, got {half}")
    return quarter_months(1 if half == 1 else 3, fy) + quarter_months(
        2 if half == 1 else 4, fy
    )


HISTORY_MONTHS = month_range(HISTORY_START, HISTORY_END)
PLAN_MONTHS = fy_months(PLAN_FY)


def label(d: dt.date) -> str:
    return d.strftime("%Y-%m")


def quarter_label(quarter: int, fy: int = PLAN_FY) -> str:
    months = quarter_months(quarter, fy)
    return f"Q{quarter} FY{fy} ({label(months[0])}..{label(months[-1])})"
