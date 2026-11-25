"""Report-card metrics on a tiny hand-built dataset, pass/fail logic, and the full baseline (slow)."""

import numpy as np
import pandas as pd
import pytest

from bakery_sim.dataset import Dataset
from bakery_sim.interfaces import TABLE_COLUMNS
from bakery_sim.report_card import (
    METRICS,
    compute_metrics,
    expected_hourly_shape_error,
    format_report_card,
    report_card,
)
from bakery_sim.simulator import simulate_history


def pick(day_index, visit_id, hour, pick_number, product, price, age=0):
    """One choices row; product 'none' means the customer left."""
    chosen_age = -1 if product == "none" else age
    return (day_index, visit_id, 0.0, hour, "regular", pick_number, "croissant@0:5.00", product, chosen_age, price)


@pytest.fixture
def tiny_dataset() -> Dataset:
    """Two days (Mon, Tue), four visits: 1 item, walk-out, 2 items, 1 item."""
    choice_rows = [
        pick(0, 0, 7, 0, "croissant", 5.0),
        pick(0, 0, 7, 1, "none", 0.0),
        pick(0, 1, 7, 0, "none", 0.0),
        pick(0, 2, 8, 0, "croissant", 5.0),
        pick(0, 2, 8, 1, "cookie", 3.5),
        pick(0, 2, 8, 2, "none", 0.0),
        pick(1, 100_000, 8, 0, "sourdough", 7.0),
        pick(1, 100_000, 8, 1, "none", 0.0),
    ]
    return Dataset(
        scenario_id="tiny",
        days=pd.DataFrame(
            [(0, "2026-09-07", "Mon", "history", 1.0), (1, "2026-09-08", "Tue", "history", 1.0)],
            columns=TABLE_COLUMNS["days"],
        ),
        prices=pd.DataFrame(columns=TABLE_COLUMNS["prices"]),
        choices=pd.DataFrame(choice_rows, columns=TABLE_COLUMNS["choices"]),
        inventory=pd.DataFrame([(0, "croissant", 0, 0, 100, 60, 0, 10, np.nan)], columns=TABLE_COLUMNS["inventory"]),
    )


def test_metric_arithmetic(tiny_dataset, baseline_scenario):
    metrics = compute_metrics(tiny_dataset, baseline_scenario)
    assert set(metrics) == set(METRICS)
    assert metrics["customers_per_day"] == pytest.approx(4 / 2)
    assert metrics["purchase_rate"] == pytest.approx(3 / 4)
    assert metrics["items_per_purchase"] == pytest.approx((1 + 2 + 1) / 3)
    assert metrics["average_ticket"] == pytest.approx((5.0 + 8.5 + 7.0) / 3)
    assert metrics["leftover_share"] == pytest.approx(10 / 100)
    assert metrics["busiest_hour_customers"] == pytest.approx(2.0)  # Monday 7am, one Monday
    assert metrics["max_item_revenue_share"] == pytest.approx(10.0 / 20.5)


def test_realized_hourly_shape_error_is_largest_gap_over_weekday_and_hour(tiny_dataset, baseline_scenario):
    metrics = compute_metrics(tiny_dataset, baseline_scenario)
    tuesday_observed_share_at_8am = baseline_scenario.observed_hourly_shape["Tue"][1]
    assert metrics["realized_hourly_shape_error"] == pytest.approx(1.0 - tuesday_observed_share_at_8am)


def test_graded_hourly_shape_error_is_the_noise_free_model_fit(tiny_dataset, baseline_scenario):
    from bakery_sim.report_card import expected_hourly_shape_error

    metrics = compute_metrics(tiny_dataset, baseline_scenario)
    assert metrics["hourly_shape_error"] == pytest.approx(expected_hourly_shape_error(baseline_scenario))


def test_price_elasticity_is_positive_and_grows_when_customers_are_more_price_sensitive(baseline_scenario):
    from bakery_sim.calibrate import apply_calibration
    from bakery_sim.report_card import price_elasticity

    less_sensitive = price_elasticity(apply_calibration(baseline_scenario, 1.0, 1.5), n_days=7)
    more_sensitive = price_elasticity(apply_calibration(baseline_scenario, 1.0, 0.7), n_days=7)
    assert 0.0 < less_sensitive < more_sensitive


def test_report_card_pass_fail_and_missing_targets(tiny_dataset, baseline_scenario):
    targets = {"customers_per_day": [1.0, 3.0], "purchase_rate": [0.8, 0.9]}
    card = report_card(tiny_dataset, baseline_scenario.with_overrides(targets=targets)).set_index("metric")
    assert list(card.columns) == ["value", "low", "high", "passed"]
    assert card.loc["customers_per_day", "passed"]
    assert not card.loc["purchase_rate", "passed"]
    assert np.isnan(card.loc["average_ticket", "low"]) and card.loc["average_ticket", "passed"]


def test_format_report_card_has_one_line_per_metric(tiny_dataset, baseline_scenario):
    text = format_report_card(report_card(tiny_dataset, baseline_scenario))
    assert len(text.splitlines()) == len(METRICS)
    assert "FAIL" in text and "PASS" in text


def test_expected_shape_of_calibrated_baseline_is_inside_the_band(baseline_scenario):
    low, high = baseline_scenario.targets["hourly_shape_error"]
    assert low <= expected_hourly_shape_error(baseline_scenario) <= high


@pytest.mark.slow
def test_calibrated_baseline_report_card(baseline_scenario):
    card = report_card(simulate_history(baseline_scenario), baseline_scenario)
    assert card["passed"].all(), format_report_card(card)
