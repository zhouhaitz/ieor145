"""Calibration: type-share fit, bake quantities, which inputs calibration touches, determinism."""

import dataclasses

import numpy as np
import pytest

from bakery_sim.calibrate import (
    CALIBRATED_INPUTS,
    apply_calibration,
    band_score,
    calibrate,
    estimate_full_price_demand,
    fit_type_shares,
    normalized_type_shapes,
    set_bake_quantities,
)
from bakery_sim.config import DAYS_OF_WEEK, validate_scenario


def shape_error(scenario, day_of_week):
    shares = np.array([scenario.type_share[day_of_week][t.type_id] for t in scenario.customer_types])
    fitted_shape = normalized_type_shapes(scenario, day_of_week) @ shares
    return np.abs(fitted_shape - np.array(scenario.observed_hourly_shape[day_of_week])).max()


def uniform_shares(scenario):
    equal_share = 1 / len(scenario.customer_types)
    return {
        day: {customer_type.type_id: equal_share for customer_type in scenario.customer_types} for day in DAYS_OF_WEEK
    }


def test_fit_type_shares_gives_valid_shares_and_lowers_shape_error(baseline_scenario):
    uniform = baseline_scenario.with_overrides(type_share=uniform_shares(baseline_scenario))
    fitted = fit_type_shares(uniform)
    validate_scenario(fitted)
    for day_of_week in DAYS_OF_WEEK:
        shares = np.array(list(fitted.type_share[day_of_week].values()))
        assert (shares >= 0).all() and abs(shares.sum() - 1) < 1e-9
        assert shape_error(fitted, day_of_week) < shape_error(uniform, day_of_week)


def test_fit_type_shares_changes_only_type_share(baseline_scenario):
    fitted = fit_type_shares(baseline_scenario)
    assert dataclasses.replace(fitted, type_share=baseline_scenario.type_share) == baseline_scenario


def test_set_bake_quantities_scales_with_supply_ratio(small_scenario):
    demand = estimate_full_price_demand(small_scenario, n_days=7)
    assert demand.shape == (7, len(small_scenario.products)) and (demand > 0).all().all()
    low = set_bake_quantities(small_scenario, supply_ratio=1.0, n_days=7)
    high = set_bake_quantities(small_scenario, supply_ratio=2.0, n_days=7)
    default = set_bake_quantities(small_scenario, n_days=7)
    validate_scenario(high)
    for day_of_week in DAYS_OF_WEEK:
        for product in small_scenario.products:
            low_units = low.bake_quantities[day_of_week][product.product_id]
            high_units = high.bake_quantities[day_of_week][product.product_id]
            assert abs(high_units - 2 * low_units) <= 1
            assert low_units == round(demand.loc[day_of_week, product.product_id])
    assert default.bake_quantities == set_bake_quantities(small_scenario, supply_ratio=1.2, n_days=7).bake_quantities


def test_apply_calibration_touches_only_calibrated_inputs(baseline_scenario):
    scaled = apply_calibration(baseline_scenario, wtp_level_scale=1.1, wtp_spread_scale=0.8, basket_scale=1.5)
    assert dataclasses.replace(scaled, customer_types=baseline_scenario.customer_types) == baseline_scenario
    for before, after in zip(baseline_scenario.customer_types, scaled.customer_types, strict=True):
        assert (
            dataclasses.replace(
                after,
                wtp_multiplier=before.wtp_multiplier,
                wtp_spread=before.wtp_spread,
                basket_extra_mean=before.basket_extra_mean,
            )
            == before
        )
        assert after.wtp_spread == pytest.approx(before.wtp_spread * 0.8, abs=1e-4)
        assert after.basket_extra_mean == pytest.approx(before.basket_extra_mean * 1.5, abs=1e-4)
        for product_id, multiplier in before.wtp_multiplier.items():
            assert after.wtp_multiplier[product_id] == pytest.approx(multiplier * 1.1, abs=1e-4)


def test_apply_calibration_with_unit_scales_is_identity(baseline_scenario):
    assert apply_calibration(baseline_scenario, 1.0, 1.0, 1.0) == baseline_scenario


def test_calibrated_input_names():
    assert CALIBRATED_INPUTS == ["type_share", "wtp_level_scale", "wtp_spread_scale", "basket_scale", "bake_quantities"]


def test_band_score_is_zero_inside_band_and_grows_with_distance_outside():
    targets = {"metric": [0.0, 1.0]}
    inside_middle, inside_edge = band_score({"metric": 0.5}, targets), band_score({"metric": 0.9}, targets)
    just_outside, far_outside = band_score({"metric": 1.1}, targets), band_score({"metric": 2.0}, targets)
    assert inside_middle == inside_edge == 0.0
    assert 0.0 < just_outside < far_outside
    assert band_score({"other": 5.0}, targets) == 0.0


def test_closeness_penalty_prefers_scales_near_one():
    from bakery_sim.calibrate import closeness_penalty

    assert closeness_penalty((1.0, 1.0, 1.0)) == 0.0
    assert 0.0 < closeness_penalty((1.1, 1.0, 1.0)) < closeness_penalty((1.6, 1.0, 1.0))


def test_calibrate_is_deterministic_on_a_tiny_grid(small_scenario):
    grid = {"wtp_level_scales": (0.9, 1.1), "wtp_spread_scales": (1.0,), "basket_scales": (1.0,)}
    first_scenario, first_log = calibrate(small_scenario, n_days=7, **grid)
    second_scenario, second_log = calibrate(small_scenario, n_days=7, **grid)
    assert first_scenario == second_scenario
    assert first_log.equals(second_log)
    assert len(first_log) == 2 and first_log["chosen"].sum() == 1
    assert {"wtp_level_scale", "wtp_spread_scale", "basket_scale", "score", "seed", "hourly_shape_error"} <= set(
        first_log.columns
    )


@pytest.mark.slow
def test_calibrate_picks_lowest_score_on_a_small_grid(baseline_scenario):
    grid = {"wtp_level_scales": (0.9, 1.1), "wtp_spread_scales": (0.8, 1.2), "basket_scales": (1.0, 1.4)}
    _, log = calibrate(baseline_scenario, **grid)
    assert len(log) == 8
    assert log.loc[log["chosen"], "score"].iloc[0] == log["score"].min()
