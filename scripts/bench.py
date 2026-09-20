"""Timings for the numbers quoted in the README.

    .venv/bin/python scripts/bench.py
"""

from __future__ import annotations

import pathlib
import statistics
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

QUESTIONS = [
    "what happens if we cut contractor spend 15% in Q3",
    "freeze travel for the year",
    "move 30% of contractor spend into software",
    "cut $40k a month from marketing programs in Q2 and increase cloud by 8%",
    "reduce facilities by 20%",
    "cut the Atlantis team by 10%",
]


def cold_start() -> float:
    code = (
        "import time; t=time.perf_counter(); import sys; sys.path.insert(0,'.');"
        "from budget_agent.graph import ask;"
        "ask('cut contractor spend 15% in Q3');"
        "print(time.perf_counter()-t)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        cwd=pathlib.Path(__file__).resolve().parents[1],
    )
    return float(out.stdout.strip())


def main() -> None:
    print(f"python            {sys.version.split()[0]}")

    started = time.perf_counter()
    from budget_agent.data import generate_ledger, ledger_summary
    from budget_agent.engine import EngineConfig, build_baseline
    from budget_agent.graph import ask
    print(f"import            {1000 * (time.perf_counter() - started):.0f} ms")

    generate_ledger.cache_clear()
    started = time.perf_counter()
    ledger = generate_ledger()
    print(f"generate ledger   {1000 * (time.perf_counter() - started):.0f} ms "
          f"({len(ledger):,} rows)")
    print(f"ledger summary    {ledger_summary(ledger)}")

    started = time.perf_counter()
    baseline = build_baseline(EngineConfig())
    print(f"build baseline    {1000 * (time.perf_counter() - started):.0f} ms "
          f"(${baseline.total:,.0f} over {len(baseline.lines)} lines)")

    for question in QUESTIONS:
        ask(question)  # warm the cached baseline
    timings: list[float] = []
    for _ in range(40):
        for question in QUESTIONS:
            started = time.perf_counter()
            ask(question)
            timings.append(time.perf_counter() - started)
    timings.sort()
    print(
        f"question -> answer median {1000 * statistics.median(timings):.1f} ms, "
        f"p95 {1000 * timings[int(0.95 * len(timings))]:.1f} ms, "
        f"max {1000 * timings[-1]:.1f} ms  (n={len(timings)})"
    )
    print(f"cold process      {cold_start():.2f} s (import to first answer)")


if __name__ == "__main__":
    main()
