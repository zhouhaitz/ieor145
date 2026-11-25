"""Policy evaluation runs every policy on identical customers and reports comparable per-day numbers."""

import pytest

from bakery_sim.evaluate import compare_policies, evaluate_policy
from bakery_sim.example_policies import MyopicMnlPolicy
from bakery_sim.interfaces import PricingContext
from bakery_sim.pricing import FixedPricePolicy, HistoricalPricingPolicy


def test_compare_policies_has_one_row_per_policy(small_scenario):
    table = compare_policies(small_scenario, [FixedPricePolicy(), HistoricalPricingPolicy(small_scenario)])
    assert table["policy"].tolist() == ["fixed_price", "historical"]
    assert (table["sell_through"].between(0, 1)).all()
    assert (table["units_wasted_per_day"] >= 0).all()


def test_evaluation_is_deterministic(small_scenario):
    assert evaluate_policy(small_scenario, FixedPricePolicy()) == evaluate_policy(small_scenario, FixedPricePolicy())


def test_myopic_policy_prices_every_stocked_option_on_the_ladder(baseline_scenario):
    inventory = {option: 5 for option in baseline_scenario.option_keys()}
    context = PricingContext(baseline_scenario, day_index=60, day_of_week="Mon", hour=12, inventory=inventory)
    decisions = MyopicMnlPolicy().prices(context)
    assert set(decisions) == set(inventory)
    assert all(decision.level in baseline_scenario.price_ladder for decision in decisions.values())


def test_myopic_policy_skips_sold_out_items(baseline_scenario):
    inventory = {option: 0 for option in baseline_scenario.option_keys()} | {("croissant", 0): 3}
    context = PricingContext(baseline_scenario, day_index=60, day_of_week="Sat", hour=9, inventory=inventory)
    assert set(MyopicMnlPolicy().prices(context)) == {("croissant", 0)}


@pytest.mark.slow
def test_myopic_policy_runs_on_full_test_days(baseline_scenario):
    row = evaluate_policy(baseline_scenario, MyopicMnlPolicy())
    assert row["profit_per_day"] > 0
