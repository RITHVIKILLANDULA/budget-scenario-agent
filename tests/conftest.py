import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from budget_agent.engine import EngineConfig, build_baseline


@pytest.fixture(scope="session")
def baseline():
    return build_baseline(EngineConfig())


@pytest.fixture(scope="session")
def flat_baseline():
    """No growth, no seasonality: every plan month equals the run rate.
    Makes the arithmetic tests exact instead of approximate."""
    return build_baseline(
        EngineConfig(use_seasonality=False, growth_clip_low=0.0, growth_clip_high=0.0)
    )


@pytest.fixture(autouse=True)
def model_path_off(monkeypatch):
    """Keep the whole suite off the network.

    The model-backed parser is exercised with a stub in test_llm.py. Without
    this, running the tests on a machine that happens to have GROQ_API_KEY
    exported would put a live HTTP call in front of every question, which is
    both slow and a thing that can fail for reasons that are not the code's
    fault.
    """
    monkeypatch.setenv("BUDGET_LLM", "off")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
