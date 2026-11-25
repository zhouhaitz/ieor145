"""MNL estimation: option parsing, gradient, and recovery of known parameters from the baseline history."""

import numpy as np
import pytest
from scipy.optimize import check_grad

from bakery_sim.estimate import (
    build_pick_observations,
    fit_mnl,
    negative_log_likelihood,
    parse_options,
    recovery_table,
    register_bias_table,
)
from bakery_sim.simulator import simulate_history

MAX_RELATIVE_ERROR_W = 0.15
MAX_RELATIVE_ERROR_SIGMA = 0.15  # identified by deal-of-the-day, promo and end-of-day price variation in the history
MEANINGFUL_PURCHASE_SHARE = 0.15  # a type "buys meaningfully" a product if it is at least this share of its items


def test_parse_options():
    assert parse_options("") == {}
    assert parse_options("croissant@0:5.00;sourdough@1:3.85") == {("croissant", 0): 5.0, ("sourdough", 1): 3.85}


def test_gradient_matches_finite_differences(small_scenario):
    choices = simulate_history(small_scenario).choices
    observations = build_pick_observations(choices, small_scenario)
    parameters = np.array([1.0, 0.5, 2.0, 1.5, 0.8, 0.3])

    def value(point):
        return negative_log_likelihood(point, observations)[0]

    def gradient(point):
        return negative_log_likelihood(point, observations)[1]

    assert check_grad(value, gradient, parameters) < 1e-5


def test_fit_mnl_returns_one_row_per_parameter(small_scenario):
    choices = simulate_history(small_scenario).view("truth").choices
    pooled = fit_mnl(choices, small_scenario)
    assert list(pooled.columns) == ["parameter", "product_id", "estimate"]
    assert pooled["parameter"].tolist() == ["w"] * len(small_scenario.products) + ["sigma", "day_old_value_share"]
    assert pooled.loc[pooled["parameter"] == "sigma", "estimate"].iloc[0] > 0
    one_type = fit_mnl(choices, small_scenario, type_id="regular")
    assert len(one_type) == len(pooled)


def test_per_type_fit_needs_truth_view(small_scenario):
    door_counter_choices = simulate_history(small_scenario).view("door_counter").choices
    with pytest.raises(ValueError, match="type_id"):
        fit_mnl(door_counter_choices, small_scenario, type_id="regular")


@pytest.fixture(scope="module")
def baseline_history(request):
    from pathlib import Path

    from bakery_sim.config import load_scenario

    scenario = load_scenario(Path(__file__).resolve().parents[1] / "configs" / "baseline.json")
    return scenario, simulate_history(scenario)


@pytest.mark.slow
def test_recovery_of_known_parameters(baseline_history):
    scenario, dataset = baseline_history
    truth_choices = dataset.view("truth").choices
    table = recovery_table(truth_choices, scenario)
    purchases = truth_choices[truth_choices["chosen_product_id"] != "none"]
    for customer_type in scenario.customer_types:
        type_rows = table[table["type_id"] == customer_type.type_id]
        item_share = purchases[purchases["type_id"] == customer_type.type_id]["chosen_product_id"].value_counts(
            normalize=True
        )
        meaningful = item_share[item_share >= MEANINGFUL_PURCHASE_SHARE].index
        wtp_rows = type_rows[(type_rows["parameter"] == "w") & type_rows["product_id"].isin(meaningful)]
        assert len(wtp_rows) > 0
        assert (wtp_rows["relative_error"].abs() < MAX_RELATIVE_ERROR_W).all(), wtp_rows
        sigma_row = type_rows[type_rows["parameter"] == "sigma"]
        assert (sigma_row["relative_error"].abs() < MAX_RELATIVE_ERROR_SIGMA).all(), sigma_row


@pytest.mark.slow
def test_register_data_overstates_willingness_to_pay(baseline_history):
    scenario, dataset = baseline_history
    table = register_bias_table(dataset, scenario)
    assert list(table.columns) == ["parameter", "product_id", "door_counter_estimate", "register_estimate"]
    wtp_rows = table[table["parameter"] == "w"]
    assert (wtp_rows["register_estimate"] > wtp_rows["door_counter_estimate"]).all()
