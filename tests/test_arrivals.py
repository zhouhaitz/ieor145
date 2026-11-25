"""Tests for arrivals.py: rates, the demand multiplier, and deterministic per-day customer draws."""

import numpy as np
import pytest

from bakery_sim.arrivals import (
    day_of_week_for,
    draw_day_customers,
    draw_demand_multiplier,
    expected_arrivals_table,
    hourly_arrival_rates,
)
from bakery_sim.config import DAYS_OF_WEEK, Scenario


def test_day_of_week_for_counts_from_start_date(baseline_scenario: Scenario) -> None:
    assert baseline_scenario.start_date == "2026-09-07"
    assert day_of_week_for(baseline_scenario, day_index=0) == "Mon"
    assert day_of_week_for(baseline_scenario, day_index=5) == "Sat"
    assert day_of_week_for(baseline_scenario, day_index=7) == "Mon"


def test_rates_sum_to_daily_customers(baseline_scenario: Scenario) -> None:
    for day_of_week in DAYS_OF_WEEK:
        rates_by_type = hourly_arrival_rates(baseline_scenario, day_of_week=day_of_week)
        total_rate = sum(rates.sum() for rates in rates_by_type.values())
        assert total_rate == pytest.approx(baseline_scenario.daily_customers[day_of_week])


def test_type_totals_equal_daily_customers_times_share(baseline_scenario: Scenario) -> None:
    for day_of_week in DAYS_OF_WEEK:
        rates_by_type = hourly_arrival_rates(baseline_scenario, day_of_week=day_of_week)
        for type_id, rates in rates_by_type.items():
            expected_total = (
                baseline_scenario.daily_customers[day_of_week] * baseline_scenario.type_share[day_of_week][type_id]
            )
            assert len(rates) == len(baseline_scenario.open_hours)
            assert rates.sum() == pytest.approx(expected_total)


def test_weekend_uses_weekend_shape(baseline_scenario: Scenario) -> None:
    commuter = baseline_scenario.customer_type("commuter")
    saturday_rates = hourly_arrival_rates(baseline_scenario, day_of_week="Sat")["commuter"]
    monday_rates = hourly_arrival_rates(baseline_scenario, day_of_week="Mon")["commuter"]
    weekend_shape = np.array(commuter.weekend_hourly_shape, dtype=float)
    weekday_shape = np.array(commuter.hourly_shape, dtype=float)
    np.testing.assert_allclose(saturday_rates / saturday_rates.sum(), weekend_shape / weekend_shape.sum())
    np.testing.assert_allclose(monday_rates / monday_rates.sum(), weekday_shape / weekday_shape.sum())


def test_expected_arrivals_table_shape_and_sums(baseline_scenario: Scenario) -> None:
    table = expected_arrivals_table(baseline_scenario)
    assert list(table.columns) == ["day_of_week", "hour", "type_id", "expected_arrivals"]
    assert len(table) == len(DAYS_OF_WEEK) * len(baseline_scenario.open_hours) * len(baseline_scenario.customer_types)
    totals_by_day = table.groupby("day_of_week")["expected_arrivals"].sum()
    for day_of_week in DAYS_OF_WEEK:
        assert totals_by_day[day_of_week] == pytest.approx(baseline_scenario.daily_customers[day_of_week])
    assert set(table["hour"]) == set(baseline_scenario.open_hours)


def test_demand_multiplier_mean_and_cv() -> None:
    target_cv = 0.15
    rng = np.random.default_rng(0)
    draws = np.array([draw_demand_multiplier(rng, target_cv) for _ in range(20000)])
    assert draws.mean() == pytest.approx(1.0, abs=0.01)
    assert draws.std() / draws.mean() == pytest.approx(target_cv, rel=0.05)


def test_zero_cv_gives_multiplier_of_one() -> None:
    assert draw_demand_multiplier(np.random.default_rng(0), coefficient_of_variation=0.0) == 1.0


def test_same_seed_and_day_give_identical_customers(baseline_scenario: Scenario) -> None:
    first_day = draw_day_customers(baseline_scenario, day_index=3)
    second_day = draw_day_customers(baseline_scenario, day_index=3)
    assert first_day.demand_multiplier == second_day.demand_multiplier
    assert len(first_day.customers) == len(second_day.customers)
    for first_customer, second_customer in zip(first_day.customers, second_day.customers, strict=True):
        assert first_customer.arrival_minute == second_customer.arrival_minute
        assert first_customer.type_id == second_customer.type_id
        assert first_customer.planned_basket_size == second_customer.planned_basket_size
        np.testing.assert_array_equal(first_customer.pick_noise, second_customer.pick_noise)


def test_different_days_give_different_customers(baseline_scenario: Scenario) -> None:
    first_day = draw_day_customers(baseline_scenario, day_index=0)
    second_day = draw_day_customers(baseline_scenario, day_index=1)
    first_minutes = [customer.arrival_minute for customer in first_day.customers]
    second_minutes = [customer.arrival_minute for customer in second_day.customers]
    assert first_minutes != second_minutes
    assert first_day.demand_multiplier != second_day.demand_multiplier


def test_customers_are_sorted_and_within_open_hours(baseline_scenario: Scenario) -> None:
    day = draw_day_customers(baseline_scenario, day_index=5)
    minutes = [customer.arrival_minute for customer in day.customers]
    assert minutes == sorted(minutes)
    assert day.day_of_week == "Sat"
    for customer in day.customers:
        assert customer.hour in baseline_scenario.open_hours
        hour_start_minute = (customer.hour - baseline_scenario.open_hour) * 60
        assert hour_start_minute <= customer.arrival_minute < hour_start_minute + 60


def test_pick_noise_shape_and_basket_size(baseline_scenario: Scenario) -> None:
    day = draw_day_customers(baseline_scenario, day_index=2)
    noise_columns = 1 + len(baseline_scenario.option_keys())
    assert len(day.customers) > 0
    for customer in day.customers:
        assert customer.planned_basket_size >= 1
        assert customer.pick_noise.shape == (customer.planned_basket_size, noise_columns)


@pytest.mark.slow
def test_mean_daily_arrivals_match_expectation(baseline_scenario: Scenario) -> None:
    day_count = 7 * 60
    arrivals_by_day_of_week: dict[str, list[int]] = {day_of_week: [] for day_of_week in DAYS_OF_WEEK}
    for day_index in range(day_count):
        day = draw_day_customers(baseline_scenario, day_index=day_index)
        arrivals_by_day_of_week[day.day_of_week].append(len(day.customers))
    for day_of_week, counts in arrivals_by_day_of_week.items():
        expected = baseline_scenario.daily_customers[day_of_week]
        assert np.mean(counts) == pytest.approx(expected, rel=0.05)
