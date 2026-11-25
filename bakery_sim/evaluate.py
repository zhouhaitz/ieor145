"""
Policy evaluation: run pricing policies on the same test days and compare them.

Every policy faces the identical customers (same seeds), so differences in profit, revenue
and waste come only from the prices each policy sets.
"""

from __future__ import annotations

import pandas as pd

from bakery_sim.config import Scenario
from bakery_sim.interfaces import NO_PURCHASE, PricingPolicy
from bakery_sim.simulator import simulate_test


def unit_costs(scenario: Scenario) -> dict[str, float]:
    """Cost to bake one unit of each product."""
    return {product.product_id: product.unit_cost for product in scenario.products}


def salvage_values(scenario: Scenario) -> dict[str, float]:
    """Value recovered per leftover unit of each product (e.g. resale app)."""
    return {product.product_id: scenario.salvage_fraction * product.base_price for product in scenario.products}


def evaluate_policy(scenario: Scenario, policy: PricingPolicy) -> dict[str, float]:
    """Per-day profit, revenue, waste and sell-through of one policy on the scenario's test days."""
    test_run = simulate_test(scenario, policy)
    summary = test_run.daily_summary(unit_costs(scenario), salvage_values(scenario))
    day_count = test_run.days["day_index"].nunique()
    purchases = test_run.choices[test_run.choices["chosen_product_id"] != NO_PURCHASE]
    return {
        "policy": policy.name,
        "profit_per_day": summary["profit"].sum() / day_count,
        "revenue_per_day": summary["revenue"].sum() / day_count,
        "salvage_per_day": summary["salvage_revenue"].sum() / day_count,
        "units_wasted_per_day": summary["wasted_units"].sum() / day_count,
        "sell_through": summary["sold_units"].sum() / summary["baked_units"].sum(),
        "average_price_paid": purchases["price_paid"].mean(),
    }


def compare_policies(scenario: Scenario, policies: list[PricingPolicy]) -> pd.DataFrame:
    """One row per policy, all run on the same test days and customers."""
    return pd.DataFrame([evaluate_policy(scenario, policy) for policy in policies])
