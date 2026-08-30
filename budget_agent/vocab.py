"""The chart of accounts.

Everything downstream keys off this file: the data generator builds the ledger
from it, the Pydantic models validate against it, and the parser matches user
words against the alias lists. If a department or category is not in here, a
question mentioning it gets rejected rather than quietly ignored.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field


def _norm(vec: tuple[float, ...]) -> tuple[float, ...]:
    m = sum(vec) / len(vec)
    return tuple(round(v / m, 6) for v in vec)


# Calendar-month seasonality multipliers, index 0 == January.
SEASONALITY: dict[str, tuple[float, ...]] = {
    # audit + planning cycle lands in the first fiscal quarter (Jul-Sep)
    "professional_services": _norm(
        (0.9, 0.9, 1.0, 0.95, 0.9, 0.85, 1.25, 1.3, 1.2, 1.0, 0.95, 0.9)
    ),
    # conference season in spring and early autumn, nothing in December
    "travel": _norm((0.7, 0.9, 1.3, 1.15, 1.0, 0.85, 0.8, 0.9, 1.35, 1.2, 0.95, 0.55)),
    # campaign pushes before the summer and again in the autumn
    "marketing_programs": _norm(
        (0.8, 0.95, 1.1, 1.2, 1.25, 1.05, 0.8, 0.9, 1.2, 1.15, 1.0, 0.7)
    ),
    # year-end project crunch, quiet over the holidays
    "contractors": _norm((1.1, 1.15, 1.2, 1.05, 1.0, 0.95, 0.9, 0.95, 1.0, 1.0, 0.95, 0.8)),
    # onboarding waves after the fiscal year starts and again in January
    "hardware": _norm((1.2, 0.9, 0.85, 0.8, 0.8, 0.9, 1.35, 1.4, 1.1, 0.95, 0.9, 0.85)),
    "cloud": _norm((0.98, 0.96, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.02, 1.04, 1.06, 1.02)),
    "software": _norm((1.0,) * 12),
    "facilities": _norm((1.0,) * 12),
}

CATEGORIES: dict[str, dict] = {
    "contractors": {
        "label": "Contractors",
        "aliases": [
            "contractor", "contractors", "contractor spend", "contract labor",
            "contract labour", "contingent labor", "staff aug", "staffing", "temps",
        ],
    },
    "cloud": {
        "label": "Cloud",
        "aliases": ["cloud", "cloud infrastructure", "infra", "infrastructure", "hosting", "compute"],
    },
    "software": {
        "label": "Software",
        "aliases": ["software", "saas", "licenses", "licences", "tooling", "tools", "subscriptions"],
    },
    "travel": {
        "label": "Travel",
        "aliases": ["travel", "t&e", "travel and entertainment", "flights", "trips"],
    },
    "marketing_programs": {
        "label": "Marketing programs",
        "aliases": ["marketing programs", "marketing program", "marketing", "campaigns", "advertising", "ad spend", "media spend", "events"],
    },
    "professional_services": {
        "label": "Professional services",
        "aliases": ["professional services", "consulting", "consultants", "advisory", "legal", "audit fees"],
    },
    "facilities": {
        "label": "Facilities",
        "aliases": ["facilities", "office", "offices", "rent", "real estate", "workplace"],
    },
    "hardware": {
        "label": "Hardware",
        "aliases": ["hardware", "laptops", "devices", "equipment"],
    },
}

DEPARTMENTS: dict[str, dict] = {
    "Engineering": {"aliases": ["engineering", "eng", "r&d", "product engineering"], "fte": 142},
    "Sales": {"aliases": ["sales", "sales org", "account executives"], "fte": 88},
    "Marketing": {"aliases": ["marketing team", "marketing org", "demand gen"], "fte": 31},
    "Customer Success": {"aliases": ["customer success", "cs", "support", "customer experience"], "fte": 64},
    "Finance": {"aliases": ["finance", "accounting", "fp&a"], "fte": 22},
    "People": {"aliases": ["people", "people ops", "hr", "talent", "recruiting"], "fte": 17},
}


@dataclass(frozen=True)
class VendorSpec:
    """One supplier, its commercial terms, and how its spend is shaped."""

    name: str
    category: str
    # department -> share of this vendor's monthly spend
    allocation: dict[str, float]
    base_monthly: float
    drift: float  # annual trend, compounded monthly
    volatility: float  # lognormal sigma on the monthly draw
    min_commit_ratio: float  # contracted floor as a share of current run rate
    notice_months: int  # contractual notice before a reduction can take effect
    steps: tuple[tuple[dt.date, float], ...] = ()
    aliases: tuple[str, ...] = ()


VENDORS: tuple[VendorSpec, ...] = (
    VendorSpec(
        "Brightpath Staffing", "contractors",
        {"Engineering": 0.62, "Customer Success": 0.23, "Finance": 0.15},
        118_000, 0.06, 0.09, 0.25, 2,
        steps=((dt.date(2025, 10, 1), 1.45), (dt.date(2026, 4, 1), 0.82)),
        aliases=("brightpath",),
    ),
    VendorSpec(
        "Meridian Contract Dev", "contractors",
        {"Engineering": 0.85, "Marketing": 0.15},
        74_500, 0.11, 0.12, 0.0, 1,
        aliases=("meridian",),
    ),
    VendorSpec(
        "Halyard QA Services", "contractors",
        {"Engineering": 1.0},
        31_200, -0.04, 0.14, 0.4, 3,
        aliases=("halyard", "halyard qa"),
    ),
    VendorSpec(
        "Northwind Cloud", "cloud",
        {"Engineering": 0.78, "Customer Success": 0.12, "Marketing": 0.10},
        196_000, 0.14, 0.05, 0.55, 0,
        steps=((dt.date(2025, 4, 1), 1.22),),
        aliases=("northwind",),
    ),
    VendorSpec(
        "Cirrus Object Store", "cloud",
        {"Engineering": 0.9, "Finance": 0.1},
        42_800, 0.19, 0.07, 0.3, 0,
        aliases=("cirrus",),
    ),
    VendorSpec(
        "Beacon CDN", "cloud",
        {"Engineering": 0.6, "Marketing": 0.4},
        23_400, 0.08, 0.1, 0.5, 1,
        aliases=("beacon", "cdn"),
    ),
    VendorSpec(
        "Arclight Analytics", "software",
        {"Engineering": 0.35, "Sales": 0.25, "Marketing": 0.2, "Finance": 0.2},
        61_000, 0.05, 0.02, 0.85, 3,
        steps=((dt.date(2026, 1, 1), 1.15),),
        aliases=("arclight",),
    ),
    VendorSpec(
        "Kestrel CRM", "software",
        {"Sales": 0.7, "Marketing": 0.2, "Customer Success": 0.1},
        88_000, 0.09, 0.02, 0.9, 6,
        aliases=("kestrel", "crm"),
    ),
    VendorSpec(
        "Foundry Design Suite", "software",
        {"Engineering": 0.5, "Marketing": 0.5},
        14_600, 0.03, 0.03, 0.6, 1,
        aliases=("foundry",),
    ),
    VendorSpec(
        "Lumen Ticketing", "software",
        {"Customer Success": 0.75, "Engineering": 0.25},
        27_300, 0.07, 0.03, 0.8, 2,
        aliases=("lumen",),
    ),
    VendorSpec(
        "Voyager Travel Desk", "travel",
        {"Sales": 0.48, "Engineering": 0.16, "Marketing": 0.14, "Customer Success": 0.14, "Finance": 0.05, "People": 0.03},
        97_000, 0.1, 0.16, 0.0, 0,
        aliases=("voyager",),
    ),
    VendorSpec(
        "Signal Events", "marketing_programs",
        {"Marketing": 0.82, "Sales": 0.18},
        134_000, 0.12, 0.18, 0.2, 2,
        aliases=("signal", "signal events"),
    ),
    VendorSpec(
        "Tradewind Media", "marketing_programs",
        {"Marketing": 1.0},
        118_500, 0.16, 0.13, 0.0, 1,
        steps=((dt.date(2026, 2, 1), 0.65),),
        aliases=("tradewind",),
    ),
    VendorSpec(
        "Orchard Content Studio", "marketing_programs",
        {"Marketing": 0.85, "Customer Success": 0.15},
        38_900, 0.04, 0.11, 0.3, 1,
        aliases=("orchard",),
    ),
    VendorSpec(
        "Ashfield Advisory", "professional_services",
        {"Finance": 0.68, "People": 0.18, "Engineering": 0.14},
        52_400, 0.03, 0.15, 0.0, 1,
        aliases=("ashfield",),
    ),
    VendorSpec(
        "Coldbrook Legal", "professional_services",
        {"Finance": 0.55, "People": 0.25, "Sales": 0.2},
        44_100, 0.07, 0.19, 0.0, 0,
        aliases=("coldbrook",),
    ),
    VendorSpec(
        "Harborview Properties", "facilities",
        {"Engineering": 0.34, "Sales": 0.24, "Customer Success": 0.18, "Marketing": 0.1, "Finance": 0.08, "People": 0.06},
        163_000, 0.02, 0.01, 0.95, 6,
        steps=((dt.date(2025, 7, 1), 1.08),),
        aliases=("harborview",),
    ),
    VendorSpec(
        "Ridgeline Facilities", "facilities",
        {"Engineering": 0.4, "Sales": 0.3, "Customer Success": 0.2, "People": 0.1},
        29_700, 0.05, 0.06, 0.5, 2,
        aliases=("ridgeline",),
    ),
    VendorSpec(
        "Tessellate Devices", "hardware",
        {"Engineering": 0.55, "Sales": 0.2, "Customer Success": 0.13, "Marketing": 0.06, "Finance": 0.04, "People": 0.02},
        58_200, 0.08, 0.2, 0.0, 0,
        aliases=("tessellate", "laptops"),
    ),
)

VENDORS_BY_NAME: dict[str, VendorSpec] = {v.name: v for v in VENDORS}
CATEGORY_KEYS: tuple[str, ...] = tuple(CATEGORIES)
DEPARTMENT_KEYS: tuple[str, ...] = tuple(DEPARTMENTS)
VENDOR_KEYS: tuple[str, ...] = tuple(VENDORS_BY_NAME)


def category_label(key: str) -> str:
    return CATEGORIES[key]["label"]


def vendors_in_category(category: str) -> list[str]:
    return [v.name for v in VENDORS if v.category == category]
