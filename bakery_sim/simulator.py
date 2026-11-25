"""
Day-by-day bakery simulation (Subagent B).

For each day: day-old stock carries in (if enabled), scheduled batches are baked, each
customer arrives in time order and picks items one at a time from what is on the shelf at
the current hour's prices, and at closing leftovers are carried over (multi-day items, if
enabled) or salvaged. Customers come from arrivals.draw_day_customers, so results depend
on the policy only through prices.
"""

from __future__ import annotations

from bakery_sim.config import Scenario
from bakery_sim.dataset import Dataset
from bakery_sim.interfaces import PricingPolicy


def simulate(
    scenario: Scenario,
    policy: PricingPolicy,
    day_indices: list[int],
    split: str,
    unlimited_inventory: bool = False,
) -> Dataset:
    """Run consecutive days under one policy; carryover state starts empty at the first day."""
    raise NotImplementedError


def simulate_history(scenario: Scenario) -> Dataset:
    """History days (0..n_history_days-1) under HistoricalPricingPolicy, split='history'."""
    raise NotImplementedError


def simulate_test(scenario: Scenario, policy: PricingPolicy) -> Dataset:
    """Test days (n_history_days..n_days-1) under the given policy, split='test'."""
    raise NotImplementedError
