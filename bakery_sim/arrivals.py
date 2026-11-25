"""
Customer arrivals and pre-drawn customer randomness (Subagent A).

Each customer type k arrives by its own Poisson process. The expected number of
type-k arrivals in open hour h on a given day of week is

    lambda_k(h, dow) = daily_customers[dow] * type_share[dow][k] * normalized_shape_k(h)

and every hour's rate is scaled by the day's demand multiplier M_d ~ Gamma(mean 1, cv).
All randomness for a day comes from SeedSequence([scenario.seed, day_index]) so it does
not depend on prices or on which pricing policy is running.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bakery_sim.config import Scenario
from bakery_sim.interfaces import DayCustomers


def day_of_week_for(scenario: Scenario, day_index: int) -> str:
    """Day-of-week label ("Mon".."Sun") for a day index, counted from scenario.start_date."""
    raise NotImplementedError


def hourly_arrival_rates(scenario: Scenario, day_of_week: str) -> dict[str, np.ndarray]:
    """Expected arrivals per open hour for each type at demand multiplier 1."""
    raise NotImplementedError


def expected_arrivals_table(scenario: Scenario) -> pd.DataFrame:
    """Columns: day_of_week, hour, type_id, expected_arrivals (for LP-based policies)."""
    raise NotImplementedError


def draw_demand_multiplier(rng: np.random.Generator, coefficient_of_variation: float) -> float:
    """Gamma draw with mean 1 and the given coefficient of variation (cv=0 returns 1.0)."""
    raise NotImplementedError


def draw_day_customers(scenario: Scenario, day_index: int) -> DayCustomers:
    """Draw every customer of one day with their basket size and pick noise, sorted by arrival time."""
    raise NotImplementedError
