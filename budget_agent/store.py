"""Optional Postgres backing.

The app runs perfectly well without this. Set BUDGET_DB_URL and it will read
the ledger out of SQL instead of generating it in-process, and append every
scenario run to a table so you can see what was asked over time. That is the
only difference.
"""

from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd

from .data import DEFAULT_SEED, generate_ledger, vendor_terms

SCHEMA = """
CREATE TABLE IF NOT EXISTS expense_lines (
    id              bigserial PRIMARY KEY,
    month           date        NOT NULL,
    fiscal_year     int         NOT NULL,
    fiscal_quarter  int         NOT NULL,
    department      text        NOT NULL,
    category        text        NOT NULL,
    vendor          text        NOT NULL,
    amount_usd      numeric(14, 2) NOT NULL
);
CREATE INDEX IF NOT EXISTS expense_lines_month_idx ON expense_lines (month);
CREATE INDEX IF NOT EXISTS expense_lines_cat_idx ON expense_lines (category);

CREATE TABLE IF NOT EXISTS vendor_terms (
    vendor           text PRIMARY KEY,
    category         text    NOT NULL,
    min_commit_ratio numeric(4, 3) NOT NULL,
    notice_months    int     NOT NULL
);

CREATE TABLE IF NOT EXISTS scenario_runs (
    id          bigserial PRIMARY KEY,
    created_at  timestamptz NOT NULL DEFAULT now(),
    question    text        NOT NULL,
    status      text        NOT NULL,
    scenario    jsonb,
    summary     jsonb
);
"""


def database_url() -> str | None:
    return os.environ.get("BUDGET_DB_URL") or None


def connect(url: str | None = None):
    import psycopg  # imported lazily so the no-Docker path never needs it

    return psycopg.connect(url or database_url(), autocommit=True)


def status() -> dict[str, Any]:
    url = database_url()
    if not url:
        return {"enabled": False, "detail": "BUDGET_DB_URL is not set"}
    try:
        with connect(url) as conn, conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM expense_lines")
            rows = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM scenario_runs")
            runs = cur.fetchone()[0]
        return {"enabled": True, "rows": rows, "runs": runs, "detail": "connected"}
    except Exception as exc:  # surfaced in the sidebar rather than crashing the app
        return {"enabled": False, "detail": f"{type(exc).__name__}: {exc}"}


def init_schema(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(SCHEMA)


def seed(conn, seed_value: int = DEFAULT_SEED, replace: bool = False) -> int:
    """Load the generated ledger and vendor terms into Postgres."""
    ledger = generate_ledger(seed_value)
    terms = vendor_terms()
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM expense_lines")
        existing = cur.fetchone()[0]
        if existing and not replace:
            return 0
        cur.execute("TRUNCATE expense_lines RESTART IDENTITY")
        cur.execute("TRUNCATE vendor_terms")
        with cur.copy(
            "COPY expense_lines (month, fiscal_year, fiscal_quarter, department, "
            "category, vendor, amount_usd) FROM STDIN"
        ) as copy:
            for row in ledger.itertuples(index=False):
                copy.write_row(
                    (
                        row.month.date(),
                        int(row.fiscal_year),
                        int(row.fiscal_quarter),
                        row.department,
                        row.category,
                        row.vendor,
                        float(row.amount_usd),
                    )
                )
        with cur.copy(
            "COPY vendor_terms (vendor, category, min_commit_ratio, notice_months) FROM STDIN"
        ) as copy:
            for row in terms.itertuples(index=False):
                copy.write_row(
                    (row.vendor, row.category, float(row.min_commit_ratio), int(row.notice_months))
                )
    return len(ledger)


def fetch_ledger(url: str | None = None) -> pd.DataFrame | None:
    """The ledger as stored in Postgres, or None if there is no usable database."""
    url = url or database_url()
    if not url:
        return None
    try:
        with connect(url) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT month, fiscal_year, fiscal_quarter, department, category, "
                "vendor, amount_usd FROM expense_lines ORDER BY month, department, "
                "category, vendor"
            )
            rows = cur.fetchall()
    except Exception:
        return None
    if not rows:
        return None
    frame = pd.DataFrame(
        rows,
        columns=[
            "month", "fiscal_year", "fiscal_quarter", "department",
            "category", "vendor", "amount_usd",
        ],
    )
    frame["month"] = pd.to_datetime(frame["month"])
    frame["amount_usd"] = frame["amount_usd"].astype(float)
    return frame


def category_history(url: str | None = None) -> pd.DataFrame | None:
    """Monthly rollup done in SQL rather than pandas, when a database is there."""
    url = url or database_url()
    if not url:
        return None
    try:
        with connect(url) as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT month, category, sum(amount_usd) AS amount_usd
                FROM expense_lines
                GROUP BY month, category
                ORDER BY month, category
                """
            )
            rows = cur.fetchall()
    except Exception:
        return None
    frame = pd.DataFrame(rows, columns=["month", "category", "amount_usd"])
    frame["month"] = pd.to_datetime(frame["month"])
    frame["amount_usd"] = frame["amount_usd"].astype(float)
    return frame


def log_run(question: str, status_value: str, scenario: dict | None, summary: dict | None) -> bool:
    url = database_url()
    if not url:
        return False
    try:
        with connect(url) as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO scenario_runs (question, status, scenario, summary) "
                "VALUES (%s, %s, %s, %s)",
                (
                    question,
                    status_value,
                    json.dumps(scenario) if scenario else None,
                    json.dumps(summary) if summary else None,
                ),
            )
        return True
    except Exception:
        return False


def recent_runs(limit: int = 10) -> pd.DataFrame | None:
    url = database_url()
    if not url:
        return None
    try:
        with connect(url) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT created_at, question, status, "
                "(summary->>'net_savings_usd')::numeric AS net_savings_usd "
                "FROM scenario_runs ORDER BY created_at DESC LIMIT %s",
                (limit,),
            )
            rows = cur.fetchall()
    except Exception:
        return None
    return pd.DataFrame(rows, columns=["created_at", "question", "status", "net_savings_usd"])
