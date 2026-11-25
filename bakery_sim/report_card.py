"""
Plausibility report card (Subagent C).

Computes the outputs a real bakery would recognize and checks each against the sourced
band in scenario.targets. Run after every generation; a failing row means the inputs
(not the outputs) need revisiting.
"""

from __future__ import annotations

import pandas as pd

from bakery_sim.config import Scenario
from bakery_sim.dataset import Dataset

METRICS = [
    "customers_per_day",
    "purchase_rate",
    "items_per_purchase",
    "average_ticket",
    "leftover_share",
    "busiest_hour_customers",
    "max_item_revenue_share",
]


def compute_metrics(dataset: Dataset, scenario: Scenario) -> dict[str, float]:
    """Value of every metric in METRICS for this dataset."""
    raise NotImplementedError


def report_card(dataset: Dataset, scenario: Scenario) -> pd.DataFrame:
    """Columns: metric, value, low, high, passed."""
    raise NotImplementedError
