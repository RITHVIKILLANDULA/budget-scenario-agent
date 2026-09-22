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

To put a model in front of the parser, give it a key:

```bash
cp .env.example .env        # then paste a key into it
# or just: export GROQ_API_KEY=...
```

That is the only difference the key makes, and the section below is the argument for it. With no
key the app is the app above: same UI, same answers, no warning that anything is missing.

To run it against real Postgres instead:

```bash
docker compose up --build
```

Same app on port 8501, but the ledger is loaded into Postgres on first boot and every question you
ask is logged to a `scenario_runs` table. Both paths give identical numbers, because the database holds
the same generated rows.

## What you can ask

The parser is rules by default and a model when you give it a key (more on both below). Either
way it produces one of four kinds of lever:

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

`parse` turns text into a draft dict — with the model if there is a key and with the regexes
otherwise, which is a choice inside the node rather than another node — `validate` turns the draft
into a Pydantic `Scenario` or a list of reasons, `baseline` projects the plan year, `simulate` applies the levers, `explain` writes
the narrative. Both `parse` and `validate` can route to `reject`, and that edge is the reason this
is a graph rather than a function — the failure path has to be as visible as the happy one. Every
node appends to `trace`, which the UI shows verbatim.

![Data flow](docs/img/architecture.png)

### The language layer is rules first

`budget_agent/parser.py` is regexes plus the alias lists in `vocab.py`. No model call, no key, 35
tests that assert exact output. That is what runs on a clean clone, what runs on the public deploy,
and what produces every number in this README.

What the rules buy is that the language layer is inspectable: when it gets something wrong you can
see which pattern fired, and the fix is a line of code rather than a prompt change you cannot test.
It also reports which words it ignored, so *"cut contractor spend 15% in Q3 please by friday"* runs
but flags `friday` as unused.

### A model in front of them

Set `GROQ_API_KEY` and the question goes to a model before it goes to the regexes. The model does
the job the previous version of this section said a model would do, and nothing more: emit the same
draft dict the rules parser emits, using only the vocabulary that is actually in the ledger.

It is sent the real department, category and vendor names out of `vocab.py`, the fiscal calendar,
and the months the data covers, and it must reply with one JSON object. Then three things happen
before anything reaches the screen:

1. `budget_agent/llm.py` checks every name in the reply against `vocab.py`. A department that does
   not exist kills the reply; it is not fuzzy-matched to the nearest one.
2. The draft goes through `Scenario.model_validate` — the same call, the same validators, the same
   rejection rules the rules parser's output goes through.
3. If either step fails, or the call errors, times out or gets rate-limited, the question goes to
   the rules parser and is answered exactly as it would have been. The reason is written to the
   trace and nowhere else, because a fallback that worked is not an error the reader needs.

No number on the screen comes from the model. It is never asked for one. It returns a request, the
request has to survive the schema, and Python costs it. That is the same arrangement as before —
the model just replaced the regexes at the front of it.

#### Where it earns its place

I ran both parsers over the same questions to find these. They are not written to flatter the
feature, and the numbers are from `budget_agent.graph.ask` with the default assumptions.

| Question | Rules parser | With the model |
| --- | --- | --- |
| *cut travel by fifteen percent* | rejected: "has no size" | −15% Travel over FY27, saves $202,924 |
| *cut contractors 10% in Q1 but only 5% in Q2* | one lever, −10% in Q1, $16,764 | two levers, $40,140 |
| *trim the dev team's cloud bill by a tenth* | −10% Cloud in all six departments, $487,334 | −10% Cloud in Engineering, $385,337 |
| *cut everything except engineering by 5%* | −5% on Engineering alone, $352,486 | −5% on the other five departments, $498,050 |

The first one is the harmless kind of failure: it says no. The other three are the reason I
bothered, because none of them is rejected — the rules parser answers a slightly different
question and hands back a number that looks perfectly reasonable. "dev team" is not an alias in
`vocab.py` and "but only 5% in Q2" is not a clause the splitter knows, so both drop out quietly,
and the only trace of it is the ignored-words line. On the cloud question that silent drop is
$102,000 in a $21.5M plan.

The last row is the one that bothers me. "except" is not a word the rules parser has any concept
of, so it matches the one department it can see and cuts precisely the department the question
asked it to leave alone. That is not a near miss, it is the inverse, and nothing in the output
says so.

It is not uniformly better, and its failure mode is worse than a rejection. *"halve what we spend
on consultants after the new year"* comes back with the window `["2027-01-01", "2027-06-01"]` —
the two ends of the range I meant rather than the six months inside it. That is a legal window
over two real months of the plan year, so `Scenario.model_validate` passes it, and the answer is
$149,917 where the reading I intended is $677,285. The gate catches malformed replies and invented
vocabulary. It cannot catch this, because a two-month cut is an ordinary thing to ask for and the
schema has no way to know it is not what I meant. I have not found a fix, and it reproduces.

It also declines a lot, which I would rather it did. *"can we find a million dollars without
touching headcount"* comes back as an empty lever list, falls through to the rules parser, and gets
the rejection it always got. There is still no optimiser behind any of this.

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

Measured on an M-series Mac, Python 3.14.6, with `scripts/bench.py` — except the last
row, which is `pytest -q`:

| | |
| --- | --- |
| Generate the ledger (1,368 rows) | 7 ms |
| Build the FY27 baseline (57 lines × 12 months) | 30 ms |
| Question to answer, median over 240 runs | 8.0 ms |
| Same, p95 | 10 ms |
| Cold process, import to first answer | 0.47 s |
| Test suite | 168 tests in 1.4 s |

The median is stable to a few tenths of a millisecond across runs. The p95 is not:
on a loaded machine I have seen it near 18 ms, so treat the tail as a property of
whatever else is running rather than of this code.

With a key set the parse stops being free. `scripts/bench.py --model` times six live questions:
median 0.47 s on one run and 1.59 s on the next, fastest 0.38 s, slowest 1.88 s, six of six
accepted by the schema. That spread between runs is the endpoint's and not this code's, which is
the main thing a network call costs you. Everything after the parse is the same 8 ms of
arithmetic.

Nothing is cached between questions except the baseline, which is keyed on the engine config and
rebuilt whenever you move a slider — and the model's reply, which is cached per question because
Streamlit re-runs the whole script on every widget change and the compare panel asks two more
questions on each of those. Without a key the app is fast because there is no network call
anywhere in the path. With one, the parse is the only network call there is.

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

168 tests, no database needed. They assert behaviour, not coverage: that a floor blocks the right
number of dollars, that a six-month notice period makes a Q1 cut a no-op and a Q3 cut land, that
two 10% cuts compound to 19%, that a shift at 100% efficiency is cost-neutral, that the compiled
graph has exactly the six nodes this README claims.

Forty of them are the model path. They hand `llm.parse_question` a stub that returns whatever a
model might have said — good JSON, fenced JSON, prose, an invented department, a 400% cut, a
timeout — and assert what comes out the other side. The suite makes no network calls and takes no
key; `tests/conftest.py` switches the model off for every test so that a key sitting in your shell
cannot quietly put an HTTP request in front of all 168 of them.

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
  `vocab.py`. Anything else is rejected rather than guessed at. A key widens the phrasing that
  gets through, not the vocabulary and not the lever shapes: a name the ledger does not hold is
  thrown out in `llm.py` before Pydantic is even asked.
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
  llm.py                English -> draft scenario, model, optional
  models.py             Pydantic scenario schema and the rejection rules
  engine.py             baseline, levers, constraints, assumption ledger
  graph.py              the LangGraph graph
  store.py              optional Postgres
scripts/
  seed_db.py            load the ledger into Postgres
  bench.py              the timings in this README
  make_figures.py       the figures in this README
  screenshot.py         the app screenshots in this README
tests/                  174 tests (168 + 6 that need a database)
.env.example            the optional key, and the switch to ignore it
```

MIT licensed. See `LICENSE`.
