"""The model-backed parser, with no network anywhere.

Every test here hands `llm.parse_question` a stub client, which is just a
callable that takes the prompt and returns whatever a model might have said.
That is the whole seam: the HTTP call is one function, and nothing above it
knows or cares where the string came from.

What these tests are actually asserting is the rule the feature lives by --
the model proposes, the schema disposes, and a proposal that does not survive
the schema costs the user nothing.
"""

from __future__ import annotations

import json

import pytest

from budget_agent import fiscal, llm, vocab
from budget_agent.engine import EngineConfig
from budget_agent.graph import ask
from budget_agent.parser import parse_question as rules_parse

CONTRACTOR_CUT = {
    "levers": [
        {
            "kind": "percent_change",
            "pct": -15,
            "target": {"departments": [], "categories": ["contractors"], "vendors": []},
            "window": {"kind": "quarter", "quarter": 3},
        }
    ]
}


def replies(payload) -> callable:
    """A stub client that always says the same thing."""
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return lambda prompt: text


def parse(payload, question="cut contractor spend 15% in Q3"):
    return llm.parse_question(question, client=replies(payload))


# --- off by default ----------------------------------------------------------

def test_no_key_means_no_model_and_no_call(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("BUDGET_LLM", raising=False)
    monkeypatch.setattr(llm, "_load_env_file", lambda *a, **k: None)
    assert llm.config() is None
    assert llm.parse_question("cut travel by 10%") is None
    assert llm.status() == {
        "enabled": False,
        "model": None,
        "detail": "GROQ_API_KEY is not set",
    }


def test_the_switch_beats_the_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "not-a-real-key")
    monkeypatch.setenv("BUDGET_LLM", "off")
    assert llm.config() is None


def test_a_key_turns_it_on(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "not-a-real-key")
    monkeypatch.delenv("BUDGET_LLM", raising=False)
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    config = llm.config()
    assert config is not None
    assert config.model == llm.DEFAULT_MODEL
    assert config.base_url.startswith("https://")
    assert "key" not in llm.status()["detail"]


def test_the_status_the_ui_reads_never_carries_the_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "sk-secret-value")
    monkeypatch.delenv("BUDGET_LLM", raising=False)
    assert "sk-secret-value" not in json.dumps(llm.status())


# --- the prompt --------------------------------------------------------------

def test_the_prompt_carries_the_real_vocabulary():
    block = llm.vocabulary_block()
    for key in vocab.CATEGORY_KEYS:
        assert key in block
    for department in vocab.DEPARTMENT_KEYS:
        assert department in block
    for spec in vocab.VENDORS:
        assert spec.name in block


def test_the_prompt_carries_the_real_dates():
    block = llm.vocabulary_block()
    assert fiscal.label(fiscal.HISTORY_START) in block
    assert fiscal.label(fiscal.HISTORY_END) in block
    assert f"FY{fiscal.PLAN_FY}" in block
    # the trap the whole fiscal module exists for
    assert "2027-01..2027-03" in block


def test_the_question_is_in_the_prompt():
    assert "cut travel by 10%" in llm.build_prompt("  cut travel by 10%  ")


# --- a reply that works ------------------------------------------------------

def test_a_good_reply_becomes_the_same_draft_the_rules_would_have_built():
    got = parse(CONTRACTOR_CUT)
    assert got.draft is not None
    mine = got.draft["levers"][0]
    theirs = rules_parse("cut contractor spend 15% in Q3").draft["levers"][0]
    assert mine["kind"] == theirs["kind"] == "percent_change"
    assert mine["pct"] == theirs["pct"] == -15.0
    assert mine["target"]["categories"] == theirs["target"]["categories"]
    assert mine["window"] == theirs["window"]


def test_the_name_is_derived_not_taken_from_the_model():
    payload = json.loads(json.dumps(CONTRACTOR_CUT))
    payload["name"] = "A DRAMATIC 40% SAVING"
    got = parse(payload)
    assert got.draft["name"] == "Contractors -15% Q3"


def test_a_fenced_reply_is_read():
    text = "```json\n" + json.dumps(CONTRACTOR_CUT) + "\n```"
    assert parse(text).draft is not None


def test_prose_around_the_object_is_tolerated():
    text = "Sure. " + json.dumps(CONTRACTOR_CUT) + " Hope that helps."
    assert parse(text).draft is not None


def test_a_bare_lever_is_read_as_one_lever():
    assert parse(CONTRACTOR_CUT["levers"][0]).draft is not None


def test_every_lever_kind_survives_the_round_trip():
    payloads = {
        "freeze": {"kind": "freeze", "target": {"categories": ["travel"]}, "window": {"kind": "year"}},
        "absolute_monthly": {
            "kind": "absolute_monthly", "amount_usd_per_month": -40000,
            "target": {"categories": ["marketing_programs"]}, "window": {"kind": "quarter", "quarter": 2},
        },
        "shift": {
            "kind": "shift", "pct": 30, "to_category": "software",
            "target": {"categories": ["contractors"]}, "window": {"kind": "year"},
        },
    }
    for kind, lever in payloads.items():
        got = parse({"levers": [lever]})
        assert got.draft is not None, got.trace
        assert got.draft["levers"][0]["kind"] == kind


def test_a_month_without_a_day_is_normalised():
    got = parse(
        {"levers": [{"kind": "freeze", "target": {"categories": ["travel"]},
                     "window": {"kind": "months", "months": ["2026-10", "2026-11-01"]}}]}
    )
    assert got.draft["levers"][0]["window"]["months"] == ["2026-10-01", "2026-11-01"]


def test_a_synonym_the_rules_miss_lands_on_the_right_department():
    """'the dev team' is not in vocab.py, so the rules parser reads the
    category and quietly drops the scope. A model that maps it to Engineering
    is the entire argument for this feature."""
    question = "trim the dev team's cloud bill by a tenth"
    theirs = rules_parse(question).draft["levers"][0]
    assert theirs["target"]["departments"] == []

    mine = parse(
        {"levers": [{"kind": "percent_change", "pct": -10,
                     "target": {"departments": ["Engineering"], "categories": ["cloud"]},
                     "window": {"kind": "year"}}]},
        question=question,
    ).draft["levers"][0]
    assert mine["target"]["departments"] == ["Engineering"]


# --- replies that do not work ------------------------------------------------

@pytest.mark.parametrize(
    "reply,because",
    [
        ("not json at all", "not JSON"),
        ("[1, 2, 3]", "not a JSON object"),
        ({"levers": []}, "declined"),
        ({"levers": "contractors"}, "no levers"),
        ({"levers": [{"kind": "rewrite_the_ledger"}]}, "unknown lever kind"),
        ({"levers": [{"kind": "percent_change", "pct": "a lot",
                      "target": {"categories": ["travel"]}}]}, "not a number"),
        ({"levers": [{"kind": "percent_change", "pct": -10,
                      "target": {"departments": ["DevOps"]}}]}, "not a department"),
        ({"levers": [{"kind": "percent_change", "pct": -10,
                      "target": {"categories": ["headcount"]}}]}, "not a category"),
        ({"levers": [{"kind": "percent_change", "pct": -10,
                      "target": {"vendors": ["Initech"]}}]}, "not a vendor"),
        ({"levers": [{"kind": "percent_change", "pct": -10,
                      "target": {"categories": ["travel"]},
                      "window": {"kind": "decade"}}]}, "unknown window kind"),
    ],
)
def test_a_reply_that_does_not_hold_up_is_dropped(reply, because):
    got = parse(reply)
    assert got.draft is None
    assert because in got.trace[0]


def test_too_many_levers_is_refused():
    lever = {"kind": "percent_change", "pct": -1, "target": {"categories": ["travel"]}}
    got = parse({"levers": [lever] * (llm.MAX_LEVERS + 1)})
    assert got.draft is None
    assert "more than this tool runs" in got.trace[0]


@pytest.mark.parametrize(
    "lever",
    [
        {"kind": "percent_change", "pct": -400, "target": {"categories": ["travel"]}},
        {"kind": "percent_change", "pct": 0, "target": {"categories": ["travel"]}},
        {"kind": "shift", "pct": 30, "to_category": "contractors",
         "target": {"categories": ["contractors"]}},
        {"kind": "freeze", "pct": -10, "target": {"categories": ["travel"]}},
        {"kind": "percent_change", "pct": -10,
         "target": {"categories": ["cloud"], "vendors": ["Voyager Travel Desk"]}},
        {"kind": "percent_change", "pct": -10, "target": {"categories": ["travel"]},
         "window": {"kind": "months", "months": ["2029-01-01"]}},
        {"kind": "percent_change", "pct": -10, "target": {"categories": ["travel"]},
         "window": {"kind": "quarter", "quarter": 9}},
    ],
)
def test_the_schema_is_still_the_gate(lever):
    """None of these are malformed JSON. They are well-formed requests that
    the Pydantic model refuses, and the model gets no say in that."""
    got = parse({"levers": [lever]})
    assert got.draft is None
    assert got.trace and "model parse" in got.trace[0]


def test_a_client_that_raises_is_not_an_error_the_user_sees():
    def timing_out(prompt):
        raise TimeoutError("read timed out")

    got = llm.parse_question("cut travel by 10%", client=timing_out)
    assert got.draft is None
    assert "TimeoutError" in got.trace[0]


def test_an_http_failure_is_not_an_error_the_user_sees():
    def failing(prompt):
        raise llm.LLMError("HTTP 429")

    got = llm.parse_question("cut travel by 10%", client=failing)
    assert got.draft is None
    assert "HTTP 429" in got.trace[0]


# --- through the graph -------------------------------------------------------

def use_model(monkeypatch, payload):
    """Point the graph's parse node at a stub instead of the network."""
    real = llm.parse_question

    def stub(asked, client=None, cfg=None):
        return real(asked, client=replies(payload))

    monkeypatch.setattr(llm, "parse_question", stub)


def test_the_graph_says_which_parser_ran(monkeypatch):
    assert ask("cut contractor spend 15% in Q3")["parser"] == "rules"
    use_model(monkeypatch, CONTRACTOR_CUT)
    assert ask("cut contractor spend 15% in Q3")["parser"] == "model"


def test_the_model_path_gives_the_identical_answer(monkeypatch):
    """The point of routing both parsers into the same schema: when they agree
    on the request, the arithmetic cannot tell them apart."""
    by_rules = ask("cut contractor spend 15% in Q3")
    use_model(monkeypatch, CONTRACTOR_CUT)
    by_model = ask("cut contractor spend 15% in Q3")
    assert by_model["status"] == by_rules["status"] == "done"
    assert by_model["result_summary"] == by_rules["result_summary"]
    assert by_model["narrative"] == by_rules["narrative"]


def test_a_bad_proposal_falls_back_silently(monkeypatch):
    use_model(monkeypatch, {"levers": [{"kind": "percent_change", "pct": -900,
                                        "target": {"categories": ["contractors"]}}]})
    state = ask("cut contractor spend 15% in Q3")
    assert state["status"] == "done"
    assert state["parser"] == "rules"
    assert not state["errors"]
    assert state["result_summary"]["net_savings_usd"] > 0
    # the reason is in the trace, which is behind an expander, and nowhere else
    assert any("model parse failed the schema" in line for line in state["trace"])


def test_a_question_neither_parser_can_take_is_still_rejected(monkeypatch):
    use_model(monkeypatch, {"levers": []})
    state = ask("cut the Atlantis team by 10%")
    assert state["status"] == "rejected"
    assert state["parser"] == "rules"
    assert state.get("result") is None
    assert any("Atlantis" in e or "does not name" in e for e in state["errors"])


def test_the_model_cannot_reach_the_numbers(monkeypatch):
    """A model that proposes a cut on a vendor with a 95% floor and six months
    of notice does not get to decide what that cut is worth."""
    use_model(
        monkeypatch,
        {"levers": [{"kind": "percent_change", "pct": -20,
                     "target": {"vendors": ["Harborview Properties"]},
                     "window": {"kind": "quarter", "quarter": 1}}]},
    )
    state = ask("take 20% off the Harborview lease in Q1", EngineConfig())
    assert state["parser"] == "model"
    effects = state["result"].effects
    assert effects["blocked_by_minimum_commitment_usd"] + effects["deferred_by_notice_usd"] > 0
    assert state["result_summary"]["net_savings_usd"] == pytest.approx(0, abs=1.0)
