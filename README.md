# Budget Scenario Agent

Type a budget question in English, get a costed scenario back with every assumption it used printed
next to the answer.

"What happens if we cut contractor spend 15% in Q3" becomes a validated structured object, runs
against a synthetic 24-month expense ledger, and comes back as a month-by-month plan delta. The
part I actually cared about is the last bit: the headline number is nearly always less impressive
than the gross cut, because vendors have notice periods and minimum commitments and because cutting
contractors quietly creates agency cost somewhere else. The tool exists to make that gap visible
instead of letting a spreadsheet hide it.

![The app after running a scenario](docs/img/ui_scenario.png)

## Running it

No API key, no Docker, no model download. On a clean clone:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app.py
```

That opens on <http://localhost:8501> with the ledger generated in-process. Nothing is fetched and
nothing is written to disk.

To run it against real Postgres instead:

```bash
docker compose up --build
```

Same app on port 8501, but the ledger is loaded into Postgres on first boot and every question you
ask is logged to a `scenario_runs` table. Both paths give identical numbers, because the database holds
the same generated rows.

## What you can ask

The parser is rules, not a model (more on that below). It handles four kinds of lever:

| Question | Becomes |
| --- | --- |
| `cut contractor spend 15% in Q3` | `percent_change`, Contractors, −15%, 2027-01..2027-03 |
| `freeze travel for the year` | `freeze`, Travel, capped at the trailing run rate |
| `cut $40k a month from marketing programs in Q2` | `absolute_monthly`, −$40,000/mo |
| `move 30% of contractor spend into software` | `shift`, Contractors → Software at 60% efficiency |

Clauses combine: *"increase cloud by 20% in H2 and cut marketing programs by 10%"* produces two
levers applied in order. `Q3` means fiscal Q3. This company's year starts in July, so that is
January to March, and the trace says so rather than leaving you to find out later.

Questions it cannot answer are rejected before any arithmetic happens:

![A rejected question](docs/img/ui_rejected.png)

## How it works

Five working nodes and a rejection node, compiled with LangGraph. The diagram below is LangGraph's
own export of the compiled graph, not a drawing of it:

<img src="docs/img/agent_graph.png" alt="The compiled LangGraph topology" width="260">

`parse` turns text into a draft dict, `validate` turns the draft into a Pydantic `Scenario` or a
list of reasons, `baseline` projects the plan year, `simulate` applies the levers, `explain` writes
the narrative. Both `parse` and `validate` can route to `reject`, and that edge is the reason this
is a graph rather than a function — the failure path has to be as visible as the happy one. Every
node appends to `trace`, which the UI shows verbatim.

![Data flow](docs/img/architecture.png)

### The language layer is rules, deliberately

`budget_agent/parser.py` is regexes plus the alias lists in `vocab.py`. No LLM call, which is why
this repo has no API key and why the parser has 35 tests that assert exact output.

That is a real trade-off and I would not pretend otherwise. An LLM would handle *"can we find a
million dollars without touching headcount"*; the rules parser cannot, and tells you so. What the
rules buy is that the language layer is inspectable: when it gets something wrong you can see
which pattern fired, and the fix is a line of code rather than a prompt change you cannot test.
It also reports which words it ignored, so *"cut contractor spend 15% in Q3 please by friday"*
runs but flags `friday` as unused.

If I swapped it, the LLM would slot in as a replacement for `parse_node` only: emit the same draft
dict, keep `validate` exactly as it is. The schema is the safety net either way, and I would rather
have the schema than the model.

### The schema is the gate

`models.py` is where out-of-scope requests die. Unknown department, a cut deeper than 100%, a
fiscal year outside the planning horizon, a vendor that is not in the category you named, a freeze
with a percentage attached — all rejected by Pydantic validators before the engine is touched.
Unknown names come back with a suggestion from `difflib`, so `Engneering` tells you about
`Engineering`.

### The arithmetic

The baseline is a per-line trailing-12-month run rate, times a growth factor fitted as
(last 6 months / prior 6 months) squared and clipped to [−25%, +35%], times a per-category
seasonality index estimated from the 24 months of history. For FY27 that is $21,493,291 across 57
ledger lines.

Then each lever is applied in three passes, in this order:

1. **The intent.** −15% on the lines the target selects, over the months the window resolves to.
2. **The constraints.** A reduction cannot land before the vendor's notice period has run from the
   start of the plan year, and cannot push a line below its contracted minimum. What is lost to
   each is tracked separately rather than silently dropped.
3. **The second-order effects.** 30% of any contractor reduction comes back as agency cover booked
   to professional services in the same department; vendors with a notice period charge a one-time
   exit fee of half a month's reduction; a `shift` lever spends 60% of what it cut in the
   destination category.

Those three numbers — 30%, half a month, 60% — are planning rules of thumb, not measurements. They
are sliders in the sidebar, and when you move one the assumption ledger marks it as yours.

The decomposition reconciles to the cent, and there is a test that asserts it for six different
scenario shapes:

```
-(realized reduction) + increases + backfill + reinvestment + one-time fees == net change
```

![Baseline against scenario, and the effect decomposition](docs/img/scenario_output.png)

The left panel is the FY27 monthly path; the right is where a 15% Q3 contractor cut actually goes.
$174k comes off, $81k comes back, $93k lands. That 47% give-back is the entire point of the tool.

### The assumption ledger

Every result carries a list of the values it depended on, each tagged `derived`, `default`,
`override` or `structural`, with a sentence saying where it came from. It is context-sensitive:
shift efficiency only appears for shift levers, and the minimum-commitment entry names the vendors
that actually bound. The right-hand column of the screenshot at the top is the whole of it.

## The data

Synthetic, generated from seed `20260401`: 24 months of monthly cost lines, 19 vendors across 8
categories and 6 departments, 57 (department, vendor) lines, 1,368 rows, $38.3M of history. Each
line gets its own RNG stream derived from a BLAKE2 digest of its name, so adding a vendor to the
catalog does not reshuffle everything else, and there is a test that runs the generator in
subprocesses under three different `PYTHONHASHSEED` values and asserts the totals match.

Seasonality and a handful of step changes are declared explicitly in `vocab.py` rather than being
emergent, so the fixtures can assert them:

![Two years of synthetic actuals](docs/img/ledger_history.png)

Each vendor also carries commercial terms: a minimum commitment as a share of run rate, and a
notice period in months. Harborview Properties is a lease: 95% floor, six months' notice, so a
"cut facilities 20%" scenario mostly bounces off it. Voyager Travel Desk has neither, so travel
cuts land in full. That asymmetry is what makes comparing two scenarios interesting.

![Comparing two scenarios](docs/img/ui_compare.png)

## Numbers

Measured on an M-series Mac, Python 3.14.6, with `scripts/bench.py`:

| | |
| --- | --- |
| Generate the ledger (1,368 rows) | 7 ms |
| Build the FY27 baseline (57 lines × 12 months) | 30 ms |
| Question to answer, median over 240 runs | 8.0 ms |
| Same, p95 | 18.5 ms |
| Cold process, import to first answer | 0.47 s |
| Test suite | 128 tests in 1.3 s |

Nothing is cached between questions except the baseline, which is keyed on the engine config and
rebuilt whenever you move a slider. The app is fast because there is no network call anywhere in
the path.

## Postgres

`BUDGET_DB_URL` is the only switch. Unset, the ledger is generated in-process. Set and reachable,
`budget_agent/store.py` reads `expense_lines` over SQL, does the category rollup with a `GROUP BY`
instead of pandas, and appends each question to `scenario_runs` with the scenario and summary as
`jsonb`. If the database is unreachable the app says so in the sidebar and carries on with the
in-process ledger rather than failing.

To seed a database you are running yourself:

```bash
docker compose up -d db
BUDGET_DB_URL=postgresql://budget:budget@localhost:55432/budget \
  .venv/bin/python scripts/seed_db.py
```

## Tests

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
```

128 tests, no database needed. They assert behaviour, not coverage: that a floor blocks the right
number of dollars, that a six-month notice period makes a Q1 cut a no-op and a Q3 cut land, that
two 10% cuts compound to 19%, that a shift at 100% efficiency is cost-neutral, that the compiled
graph has exactly the six nodes this README claims.

Six more tests in `tests/test_store.py` run only when `BUDGET_DB_URL` is set. They check that the
Postgres round trip preserves the ledger to the cent and that both paths produce the same baseline:

```bash
docker compose up -d db
BUDGET_DB_URL=postgresql://budget:budget@localhost:55432/budget \
  .venv/bin/python -m pytest tests/test_store.py -q
```

## What it does not do

- **The data is invented.** Vendor names, amounts, contract terms, all of it. The point is the
  mechanism, not the numbers.
- **The parser is narrow.** It understands the four lever shapes above and the vocabulary in
  `vocab.py`. Anything else is rejected rather than guessed at.
- **Seasonality is thin.** Two years of history means two observations per calendar month. The
  index is honest about being an estimate, but I would not plan a real Q4 on it.
- **No headcount model.** These are vendor and category lines only. "Cut contractor spend" moves
  invoices, not people, and the backfill ratio is the only nod to the difference.
- **No optimiser.** You propose a scenario and it costs it. It will not search for the cheapest way
  to save a million dollars.
- **Single user, no persistence of edits.** Slider positions live in the Streamlit session. The
  Postgres table is a log, not state.

## Layout

```
app.py                  Streamlit UI
budget_agent/
  vocab.py              chart of accounts: departments, categories, vendors, terms
  fiscal.py             July-June fiscal calendar
  data.py               seeded ledger generator
  parser.py             English -> draft scenario, rules only
  models.py             Pydantic scenario schema and the rejection rules
  engine.py             baseline, levers, constraints, assumption ledger
  graph.py              the LangGraph graph
  store.py              optional Postgres
scripts/
  seed_db.py            load the ledger into Postgres
  bench.py              the timings in this README
  make_figures.py       the figures in this README
  screenshot.py         the app screenshots in this README
tests/                  134 tests (128 + 6 that need a database)
```

MIT licensed. See `LICENSE`.
