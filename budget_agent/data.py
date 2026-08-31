"""Synthetic expense ledger.

Every number in this project is generated here from a seed. Nothing is real
company data. Each (vendor, department) line gets its own RNG stream derived
from the seed and the line name, so adding a vendor to the catalog does not
shuffle the numbers of every other line.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from functools import lru_cache

import numpy as np
import pandas as pd

from . import fiscal, vocab

DEFAULT_SEED = 20260401

LEDGER_COLUMNS = [
    "month",
    "fiscal_year",
    "fiscal_quarter",
    "department",
    "category",
    "vendor",
    "amount_usd",
]


def _line_rng(seed: int, key: str) -> np.random.Generator:
    # Python's str hash is salted per process, so derive the stream from a
    # stable digest instead. Two runs of this file must produce byte-identical
    # ledgers or none of the numbers in the README mean anything.
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=4).digest()
    return np.random.default_rng([seed, int.from_bytes(digest, "big")])


def _step_multiplier(spec: vocab.VendorSpec, month: dt.date) -> float:
    mult = 1.0
    for step_date, factor in spec.steps:
        if month >= step_date:
            mult *= factor
    return mult


@lru_cache(maxsize=4)
def generate_ledger(seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """24 months of monthly cost lines across departments, vendors, categories."""
    months = fiscal.HISTORY_MONTHS
    rows: list[dict] = []

    for spec in vocab.VENDORS:
        season = vocab.SEASONALITY[spec.category]
        for department, share in spec.allocation.items():
            if share <= 0:
                continue
            rng = _line_rng(seed, f"{spec.name}|{department}")
            base = spec.base_monthly * share
            noise = rng.lognormal(mean=0.0, sigma=spec.volatility, size=len(months))
            for i, month in enumerate(months):
                trend = (1.0 + spec.drift) ** (i / 12.0)
                amount = (
                    base
                    * trend
                    * season[month.month - 1]
                    * _step_multiplier(spec, month)
                    * noise[i]
                )
                rows.append(
                    {
                        "month": pd.Timestamp(month),
                        "fiscal_year": fiscal.fiscal_year(month),
                        "fiscal_quarter": fiscal.fiscal_quarter(month),
                        "department": department,
                        "category": spec.category,
                        "vendor": spec.name,
                        "amount_usd": round(float(amount), 2),
                    }
                )

    frame = pd.DataFrame(rows, columns=LEDGER_COLUMNS)
    return frame.sort_values(["month", "department", "category", "vendor"]).reset_index(
        drop=True
    )


def vendor_terms() -> pd.DataFrame:
    """Contract terms that constrain what a cut can actually achieve."""
    return pd.DataFrame(
        [
            {
                "vendor": spec.name,
                "category": spec.category,
                "min_commit_ratio": spec.min_commit_ratio,
                "notice_months": spec.notice_months,
            }
            for spec in vocab.VENDORS
        ]
    )


def ledger_summary(frame: pd.DataFrame) -> dict:
    ttm = frame[frame["month"] >= pd.Timestamp(fiscal.add_months(fiscal.HISTORY_END, -11))]
    return {
        "rows": int(len(frame)),
        "months": int(frame["month"].nunique()),
        "first_month": fiscal.label(frame["month"].min().date()),
        "last_month": fiscal.label(frame["month"].max().date()),
        "lines": int(frame.groupby(["department", "vendor"], observed=True).ngroups),
        "total_usd": float(frame["amount_usd"].sum()),
        "ttm_usd": float(ttm["amount_usd"].sum()),
    }
