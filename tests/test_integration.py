"""End-to-end runs with the real arrivals and choice modules (slow)."""

import pandas as pd
import pytest

from bakery_sim.pricing import FixedPricePolicy, HistoricalPricingPolicy
from bakery_sim.simulator import simulate, simulate_history

pytestmark = pytest.mark.slow


def test_history_runs_and_inventory_is_conserved(small_scenario):
    dataset = simulate_history(small_scenario)
    assert len(dataset.days) == small_scenario.n_history_days
    inventory = dataset.inventory
    carried_and_wasted = inventory["carried_over_units"] + inventory["wasted_units"]
    assert (inventory["start_units"] + inventory["baked_units"] - inventory["sold_units"] == carried_and_wasted).all()
    assert (
        (inventory[["start_units", "baked_units", "sold_units", "carried_over_units", "wasted_units"]] >= 0).all().all()
    )
    non_none = dataset.choices[dataset.choices["chosen_product_id"] != "none"]
    sold_from_choices = non_none.groupby(["day_index", "chosen_product_id", "chosen_age"]).size().to_dict()
    sold_from_inventory = inventory.set_index(["day_index", "product_id", "age"])["sold_units"]
    assert sold_from_choices == sold_from_inventory[sold_from_inventory > 0].to_dict()


def test_same_seed_gives_identical_datasets(small_scenario):
    first = simulate_history(small_scenario)
    second = simulate_history(small_scenario)
    for table_name in ["days", "prices", "choices", "inventory"]:
        pd.testing.assert_frame_equal(getattr(first, table_name), getattr(second, table_name))


def test_different_policies_see_identical_arrivals(small_scenario):
    day_indices = list(range(small_scenario.n_history_days, small_scenario.n_days))
    fixed = simulate(small_scenario, FixedPricePolicy(), day_indices, split="test")
    historical = simulate(small_scenario, HistoricalPricingPolicy(small_scenario), day_indices, split="test")
    arrival_columns = ["day_index", "visit_id", "arrival_minute", "type_id"]
    first_picks_fixed = fixed.choices[fixed.choices["pick_number"] == 0][arrival_columns].reset_index(drop=True)
    first_picks_historical = historical.choices[historical.choices["pick_number"] == 0][arrival_columns]
    pd.testing.assert_frame_equal(first_picks_fixed, first_picks_historical.reset_index(drop=True))
    pd.testing.assert_frame_equal(fixed.days[["demand_multiplier"]], historical.days[["demand_multiplier"]])


@pytest.mark.slow
def test_unlimited_inventory_offers_no_day_old(small_scenario):
    from bakery_sim.pricing import FixedPricePolicy
    from bakery_sim.simulator import simulate

    dataset = simulate(
        small_scenario, FixedPricePolicy(), day_indices=[0, 1, 2], split="history", unlimited_inventory=True
    )
    assert (dataset.choices["options"].str.contains("@1:")).sum() == 0
