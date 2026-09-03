"""Scenario schema.

The point of these models is the rejection path. A question that names a
department we do not have, or asks for a 400% cut, or targets a fiscal year
outside the planning horizon, must fail here -- before the engine touches a
single number. The UI shows the failure instead of a plausible-looking chart.
"""

from __future__ import annotations

import datetime as dt
import difflib
from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from . import fiscal, vocab


class LeverKind(str, Enum):
    PERCENT_CHANGE = "percent_change"
    ABSOLUTE_MONTHLY = "absolute_monthly"
    FREEZE = "freeze"
    SHIFT = "shift"


def _resolve(value: str, options: tuple[str, ...], kind: str) -> str:
    lowered = {o.lower(): o for o in options}
    key = value.strip().lower()
    if key in lowered:
        return lowered[key]
    close = difflib.get_close_matches(key, list(lowered), n=2, cutoff=0.6)
    hint = f" Did you mean {' or '.join(lowered[c] for c in close)}?" if close else ""
    raise ValueError(
        f"unknown {kind} {value!r}; the ledger only has "
        f"{', '.join(options)}.{hint}"
    )


class Window(BaseModel):
    """When in the plan year a lever applies."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["year", "quarter", "half", "months"] = "year"
    quarter: int | None = None
    half: int | None = None
    months: list[dt.date] | None = None
    fiscal_year: int = fiscal.PLAN_FY

    @field_validator("fiscal_year")
    @classmethod
    def _fy_in_scope(cls, v: int) -> int:
        if v != fiscal.PLAN_FY:
            raise ValueError(
                f"FY{v} is outside the planning horizon; this model only plans "
                f"FY{fiscal.PLAN_FY} ({fiscal.label(fiscal.PLAN_MONTHS[0])}.."
                f"{fiscal.label(fiscal.PLAN_MONTHS[-1])})"
            )
        return v

    @model_validator(mode="after")
    def _consistent(self) -> "Window":
        if self.kind == "quarter" and self.quarter not in (1, 2, 3, 4):
            raise ValueError(f"quarter must be 1-4, got {self.quarter!r}")
        if self.kind == "half" and self.half not in (1, 2):
            raise ValueError(f"half must be 1 or 2, got {self.half!r}")
        if self.kind == "months":
            if not self.months:
                raise ValueError("months window needs at least one month")
            allowed = set(fiscal.fy_months(self.fiscal_year))
            stray = sorted(m for m in self.months if m not in allowed)
            if stray:
                raise ValueError(
                    f"{', '.join(fiscal.label(m) for m in stray)} "
                    f"is outside FY{self.fiscal_year}"
                )
        return self

    def resolve(self) -> list[dt.date]:
        if self.kind == "year":
            return fiscal.fy_months(self.fiscal_year)
        if self.kind == "quarter":
            return fiscal.quarter_months(int(self.quarter), self.fiscal_year)
        if self.kind == "half":
            return fiscal.half_months(int(self.half), self.fiscal_year)
        return sorted(self.months or [])

    @property
    def label(self) -> str:
        months = self.resolve()
        span = f"{fiscal.label(months[0])}..{fiscal.label(months[-1])}"
        if self.kind == "year":
            return f"FY{self.fiscal_year} ({span})"
        if self.kind == "quarter":
            return f"Q{self.quarter} FY{self.fiscal_year} ({span})"
        if self.kind == "half":
            return f"H{self.half} FY{self.fiscal_year} ({span})"
        return span


class Target(BaseModel):
    """Which ledger lines a lever touches. Empty means every line."""

    model_config = ConfigDict(frozen=True)

    departments: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()
    vendors: tuple[str, ...] = ()

    @field_validator("departments", mode="before")
    @classmethod
    def _departments(cls, v):
        return tuple(_resolve(x, vocab.DEPARTMENT_KEYS, "department") for x in v or ())

    @field_validator("categories", mode="before")
    @classmethod
    def _categories(cls, v):
        return tuple(_resolve(x, vocab.CATEGORY_KEYS, "category") for x in v or ())

    @field_validator("vendors", mode="before")
    @classmethod
    def _vendors(cls, v):
        return tuple(_resolve(x, vocab.VENDOR_KEYS, "vendor") for x in v or ())

    @model_validator(mode="after")
    def _vendor_matches_category(self) -> "Target":
        if self.vendors and self.categories:
            bad = [
                v
                for v in self.vendors
                if vocab.VENDORS_BY_NAME[v].category not in self.categories
            ]
            if bad:
                raise ValueError(
                    f"{', '.join(bad)} is not in "
                    f"{', '.join(self.categories)}; that combination selects nothing"
                )
        return self

    @property
    def is_everything(self) -> bool:
        return not (self.departments or self.categories or self.vendors)

    @property
    def label(self) -> str:
        if self.is_everything:
            return "all discretionary spend"
        parts = []
        if self.vendors:
            parts.append(" + ".join(self.vendors))
        if self.categories:
            parts.append(" + ".join(vocab.category_label(c) for c in self.categories))
        if self.departments:
            parts.append("in " + " + ".join(self.departments))
        return " ".join(parts)


class Lever(BaseModel):
    """One change to the baseline."""

    model_config = ConfigDict(frozen=True)

    kind: LeverKind
    target: Target = Target()
    window: Window = Window()
    pct: Annotated[float, Field(ge=-100, le=200)] | None = None
    amount_usd_per_month: float | None = None
    to_category: str | None = None

    @field_validator("to_category", mode="before")
    @classmethod
    def _to_category(cls, v):
        return _resolve(v, vocab.CATEGORY_KEYS, "category") if v else None

    @model_validator(mode="after")
    def _fields_match_kind(self) -> "Lever":
        if self.kind in (LeverKind.PERCENT_CHANGE, LeverKind.SHIFT):
            if self.pct is None:
                raise ValueError(f"{self.kind.value} needs a percentage")
            if self.pct == 0:
                raise ValueError("a 0% change is not a scenario")
        if self.kind == LeverKind.ABSOLUTE_MONTHLY:
            if not self.amount_usd_per_month:
                raise ValueError("absolute_monthly needs a non-zero dollar amount")
        if self.kind == LeverKind.SHIFT:
            if self.pct <= 0:
                raise ValueError("a shift moves a positive share of spend")
            if not self.to_category:
                raise ValueError("a shift needs a destination category")
            if self.to_category in self.target.categories:
                raise ValueError(
                    f"cannot shift {vocab.category_label(self.to_category)} into itself"
                )
            if not self.target.categories and not self.target.vendors:
                raise ValueError("a shift needs an explicit source category or vendor")
        if self.kind == LeverKind.FREEZE and (
            self.pct is not None or self.amount_usd_per_month is not None
        ):
            raise ValueError("a freeze takes no magnitude; it holds the run rate flat")
        return self

    @property
    def description(self) -> str:
        scope = self.target.label
        when = self.window.label
        if self.kind == LeverKind.PERCENT_CHANGE:
            verb = "cut" if self.pct < 0 else "increase"
            return f"{verb} {scope} by {abs(self.pct):.4g}% over {when}"
        if self.kind == LeverKind.ABSOLUTE_MONTHLY:
            verb = "cut" if self.amount_usd_per_month < 0 else "add"
            return (
                f"{verb} ${abs(self.amount_usd_per_month):,.0f}/month of {scope} "
                f"over {when}"
            )
        if self.kind == LeverKind.FREEZE:
            return f"freeze {scope} at the run rate entering {when}"
        return (
            f"shift {self.pct:.4g}% of {scope} into "
            f"{vocab.category_label(self.to_category)} over {when}"
        )


class Scenario(BaseModel):
    """A validated, runnable plan change."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1, max_length=80)
    levers: tuple[Lever, ...] = Field(min_length=1)
    question: str | None = None
    fiscal_year: int = fiscal.PLAN_FY

    @model_validator(mode="after")
    def _windows_match_year(self) -> "Scenario":
        for lever in self.levers:
            if lever.window.fiscal_year != self.fiscal_year:
                raise ValueError(
                    f"lever window FY{lever.window.fiscal_year} does not match "
                    f"scenario FY{self.fiscal_year}"
                )
        return self

    @property
    def description(self) -> str:
        return "; ".join(lever.description for lever in self.levers)
