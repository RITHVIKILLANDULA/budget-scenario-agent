"""Postgres round trip.

Skipped unless BUDGET_DB_URL points at a reachable database:

    docker compose up -d db
    BUDGET_DB_URL=postgresql://budget:budget@localhost:55432/budget \
        .venv/bin/python -m pytest tests/test_store.py -q
"""

import os

import pandas as pd
import pytest

from budget_agent import store
from budget_agent.data import generate_ledger
from budget_agent.engine import build_baseline
from budget_agent.graph import ask

pytestmark = pytest.mark.skipif(
    not os.environ.get("BUDGET_DB_URL"), reason="BUDGET_DB_URL is not set"
)


@pytest.fixture(scope="module")
def seeded():
    with store.connect() as connection:
        store.init_schema(connection)
        store.seed(connection, replace=True)
    return store.database_url()


def test_status_reports_a_live_connection(seeded):
    status = store.status()
    assert status["enabled"]
    assert status["rows"] == len(generate_ledger())


def test_the_stored_ledger_matches_the_generated_one(seeded):
    stored = store.fetch_ledger()
    generated = generate_ledger()
    assert len(stored) == len(generated)
    assert stored["amount_usd"].sum() == pytest.approx(
        generated["amount_usd"].sum(), abs=0.01
    )
    pd.testing.assert_series_equal(
        stored.groupby("category")["amount_usd"].sum().sort_index(),
        generated.groupby("category")["amount_usd"].sum().sort_index(),
        rtol=1e-9,
    )


def test_the_sql_rollup_agrees_with_pandas(seeded):
    sql = store.category_history()
    pandas_side = (
        generate_ledger()
        .groupby(["month", "category"], as_index=False)["amount_usd"].sum()
    )
    assert len(sql) == len(pandas_side)
    assert sql["amount_usd"].sum() == pytest.approx(pandas_side["amount_usd"].sum(), abs=0.01)


def test_the_answer_is_the_same_whichever_side_the_ledger_came_from(seeded):
    from_db = build_baseline()
    assert from_db.source == "postgres"
    generated_total = round(from_db.total, 2)

    saved = os.environ.pop("BUDGET_DB_URL")
    try:
        in_process = build_baseline()
        assert in_process.source == "in-process"
        assert round(in_process.total, 2) == generated_total
    finally:
        os.environ["BUDGET_DB_URL"] = saved


def test_runs_are_logged(seeded):
    before = store.status()["runs"]
    state = ask("cut contractor spend 15% in Q3")
    assert store.log_run(
        state["question"], state["status"], state["scenario"], state["result_summary"]
    )
    after = store.recent_runs(limit=5)
    assert store.status()["runs"] == before + 1
    assert after.iloc[0]["question"] == "cut contractor spend 15% in Q3"
    assert float(after.iloc[0]["net_savings_usd"]) == pytest.approx(
        state["result_summary"]["net_savings_usd"], abs=0.01
    )


def test_a_rejected_question_is_logged_without_a_scenario(seeded):
    state = ask("cut the Atlantis team by 10%")
    assert store.log_run(state["question"], state["status"], None, None)
    latest = store.recent_runs(limit=1).iloc[0]
    assert latest["status"] == "rejected"
    assert latest["net_savings_usd"] is None
