"""Timings for the numbers quoted in the README.

    .venv/bin/python scripts/bench.py
    .venv/bin/python scripts/bench.py --model   # also times the model parser

The main figures are the rules path, which is the one that runs with no key
and the one the README quotes, so the model is switched off here even if a
key is sitting in the environment. Leaving it on would put an HTTP call in
the middle of a microbenchmark.
"""

from __future__ import annotations

import os
import pathlib
import statistics
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

WITH_MODEL = "--model" in sys.argv
os.environ["BUDGET_LLM"] = "off"

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


def model_path() -> None:
    """Six live calls, spaced out. The endpoint this talks to allows 8,000
    tokens a minute and one question costs about 1,100, so a tighter loop
    measures the rate limiter rather than the model."""
    from budget_agent import llm

    os.environ.pop("BUDGET_LLM", None)  # on for this section only
    if not llm.status()["enabled"]:
        print("model path        off (no GROQ_API_KEY)")
        return
    asked = [
        "cut travel by fifteen percent",
        "cut contractors 10% in Q1 but only 5% in Q2",
        "trim the dev team's cloud bill by a tenth",
        "hold marketing flat and take an eighth out of consultants",
        "drop the Harborview lease by a fifth in the back half",
        "pull a tenth out of every software licence next year",
    ]
    timings, accepted = [], 0
    for i, question in enumerate(asked):
        if i:
            time.sleep(10)
        started = time.perf_counter()
        parsed = llm.parse_question(question)
        timings.append(time.perf_counter() - started)
        accepted += parsed.draft is not None
    timings.sort()
    print(
        f"model parse       median {timings[len(timings) // 2]:.2f} s, "
        f"min {timings[0]:.2f} s, max {timings[-1]:.2f} s, "
        f"{accepted}/{len(asked)} accepted by the schema"
    )


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
    if WITH_MODEL:
        model_path()


if __name__ == "__main__":
    main()
