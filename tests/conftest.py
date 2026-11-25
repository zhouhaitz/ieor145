"""Shared fixtures: the baseline scenario and a small fast version of it."""

from pathlib import Path

import pytest

from bakery_sim.config import Scenario, load_scenario

BASELINE_PATH = Path(__file__).resolve().parents[1] / "configs" / "baseline.json"


@pytest.fixture
def baseline_scenario() -> Scenario:
    return load_scenario(BASELINE_PATH)


@pytest.fixture
def small_scenario(baseline_scenario: Scenario) -> Scenario:
    """Baseline with few days, for fast tests."""
    return baseline_scenario.with_overrides(n_history_days=7, n_test_days=3)
