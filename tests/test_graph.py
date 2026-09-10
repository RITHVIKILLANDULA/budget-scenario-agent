import pytest

from budget_agent.engine import EngineConfig
from budget_agent.graph import AGENT, ask, narrate


def test_the_happy_path_runs_every_node():
    state = ask("what happens if we cut contractor spend 15% in Q3")
    assert state["status"] == "done"
    tags = {line.split("]")[0] + "]" for line in state["trace"]}
    assert {"[parse]", "[validate]", "[baseline]", "[simulate]", "[explain]"} <= tags
    assert "[reject]" not in tags


def test_the_happy_path_returns_numbers():
    state = ask("cut contractor spend 15% in Q3")
    summary = state["result_summary"]
    assert summary["baseline_usd"] > 0
    assert summary["net_savings_usd"] > 0
    assert summary["net_savings_usd"] < summary["realized_reduction_usd"]


def test_an_unknown_department_stops_at_parse():
    state = ask("cut the Atlantis team by 10%")
    assert state["status"] == "rejected"
    assert state.get("result") is None
    assert any("[reject]" in line for line in state["trace"])
    assert not any("[simulate]" in line for line in state["trace"])


def test_something_that_parses_but_fails_the_schema_stops_at_validate():
    state = ask("cut travel by 150%")
    assert state["status"] == "rejected"
    assert any("[validate] rejected" in line for line in state["trace"])
    assert not any("[baseline]" in line for line in state["trace"])
    assert state.get("result") is None


def test_rejection_explains_itself():
    state = ask("cut contractors")
    assert state["status"] == "rejected"
    assert state["errors"]
    assert "no size" in state["errors"][0]


def test_the_structured_scenario_is_serialisable():
    import json

    state = ask("freeze travel for the year")
    assert json.loads(json.dumps(state["scenario"]))["levers"][0]["kind"] == "freeze"


def test_config_flows_through_to_the_engine():
    loose = ask("cut facilities by 30%", EngineConfig(respect_floors=False))
    strict = ask("cut facilities by 30%", EngineConfig(respect_floors=True))
    assert loose["result_summary"]["net_savings_usd"] > strict["result_summary"]["net_savings_usd"]


def test_the_same_question_gives_the_same_answer():
    first = ask("cut contractor spend 15% in Q3")["result_summary"]
    second = ask("cut contractor spend 15% in Q3")["result_summary"]
    assert first == second


def test_the_compiled_topology_is_what_the_readme_claims():
    graph = AGENT.get_graph()
    nodes = {n for n in graph.nodes if not n.startswith("__")}
    assert nodes == {"parse", "validate", "baseline", "simulate", "explain", "reject"}
    edges = {(e.source, e.target) for e in graph.edges}
    assert ("parse", "validate") in edges
    assert ("parse", "reject") in edges
    assert ("validate", "baseline") in edges
    assert ("validate", "reject") in edges
    assert ("baseline", "simulate") in edges
    assert ("simulate", "explain") in edges


def test_the_narrative_quotes_the_headline_numbers():
    state = ask("cut contractor spend 15% in Q3")
    text = state["narrative"]
    assert "FY2027" in text
    assert f"{state['result'].net_savings:,.0f}" in text
    assert "backfill" in text


def test_the_narrative_mentions_what_could_not_be_cut():
    state = ask("reduce facilities by 20%")
    assert "contractual minimums" in state["narrative"]


@pytest.mark.parametrize(
    "question",
    [
        "what happens if we cut contractor spend 15% in Q3",
        "freeze travel for the year",
        "move 30% of contractor spend into software",
        "cut $40k a month from marketing programs in Q2 and increase cloud by 8%",
        "reduce facilities by 20%",
    ],
)
def test_every_example_in_the_ui_actually_runs(question):
    state = ask(question)
    assert state["status"] == "done", state.get("errors")
    assert state["narrative"]
    assert len(state["result"].assumptions) >= 5
