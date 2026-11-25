"""Config loads, is internally consistent, and round-trips through JSON."""

import pytest

from bakery_sim.config import DAYS_OF_WEEK, load_scenario, save_scenario


def test_baseline_loads(baseline_scenario):
    assert baseline_scenario.scenario_id == "baseline"
    assert len(baseline_scenario.open_hours) == 11


def test_shapes_match_open_hours(baseline_scenario):
    for customer_type in baseline_scenario.customer_types:
        assert len(customer_type.hourly_shape) == len(baseline_scenario.open_hours)
        assert len(customer_type.weekend_hourly_shape) == len(baseline_scenario.open_hours)


def test_type_shares_sum_to_one(baseline_scenario):
    for day_of_week in DAYS_OF_WEEK:
        assert sum(baseline_scenario.type_share[day_of_week].values()) == pytest.approx(1.0)


def test_bake_shares_sum_to_one(baseline_scenario):
    for product in baseline_scenario.products:
        shares = [batch.share for batch in baseline_scenario.bake_schedule if batch.product_id == product.product_id]
        assert sum(shares) == pytest.approx(1.0)


def test_every_type_values_every_product(baseline_scenario):
    product_ids = {product.product_id for product in baseline_scenario.products}
    for customer_type in baseline_scenario.customer_types:
        assert set(customer_type.wtp_multiplier) == product_ids


def test_option_keys_include_day_old_only_for_multi_day_items(baseline_scenario):
    keys = baseline_scenario.option_keys()
    assert ("sourdough", 1) in keys
    assert ("croissant", 1) not in keys


def test_round_trip(baseline_scenario, tmp_path):
    path = tmp_path / "scenario.json"
    save_scenario(baseline_scenario, path)
    assert load_scenario(path) == baseline_scenario
