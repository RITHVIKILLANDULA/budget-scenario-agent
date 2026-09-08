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
