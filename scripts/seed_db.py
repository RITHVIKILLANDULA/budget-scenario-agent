"""Load the generated ledger into Postgres.

    BUDGET_DB_URL=postgresql://budget:budget@localhost:55432/budget \
        .venv/bin/python scripts/seed_db.py

Idempotent: it does nothing if the table already has rows, unless you pass
--replace.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from budget_agent import store  # noqa: E402
from budget_agent.data import DEFAULT_SEED  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--replace", action="store_true", help="reload even if rows exist")
    parser.add_argument("--wait", type=int, default=0, help="seconds to wait for the database")
    args = parser.parse_args()

    url = store.database_url()
    if not url:
        print("BUDGET_DB_URL is not set", file=sys.stderr)
        return 2

    deadline = time.time() + args.wait
    while True:
        try:
            connection = store.connect(url)
            break
        except Exception as exc:
            if time.time() >= deadline:
                print(f"cannot reach the database: {exc}", file=sys.stderr)
                return 1
            time.sleep(1)

    with connection:
        store.init_schema(connection)
        written = store.seed(connection, args.seed, replace=args.replace)
    if written:
        print(f"wrote {written:,} expense rows with seed {args.seed}")
    else:
        print("expense_lines already populated; pass --replace to reload")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
