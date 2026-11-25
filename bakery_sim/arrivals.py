"""
Customer arrivals and pre-drawn customer randomness (Subagent A).

Each customer type k arrives by its own Poisson process. The expected number of
type-k arrivals in open hour h on a given day of week is

    lambda_k(h, dow) = daily_customers[dow] * type_share[dow][k] * normalized_shape_k(h)

and every hour's rate is scaled by the day's demand multiplier M_d ~ Gamma(mean 1, cv).
All randomness for a day comes from SeedSequence([scenario.seed, day_index]) so it does
not depend on prices or on which pricing policy is running.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from bakery_sim.config import DAYS_OF_WEEK, Scenario
from bakery_sim.interfaces import Customer, DayCustomers

MINUTES_PER_HOUR = 60
WEEKEND_DAYS = ("Sat", "Sun")
SPAWNED_STREAM_COUNT = 3  # multiplier, arrivals, customer details


def day_of_week_for(scenario: Scenario, day_index: int) -> str:
    """Day-of-week label ("Mon".."Sun") for a day index, counted from scenario.start_date."""
    day_date = date.fromisoformat(scenario.start_date) + timedelta(days=day_index)
    return DAYS_OF_WEEK[day_date.weekday()]


def hourly_arrival_rates(scenario: Scenario, day_of_week: str) -> dict[str, np.ndarray]:
    """Expected arrivals per open hour for each type at demand multiplier 1."""
    rates_by_type = {}
    for customer_type in scenario.customer_types:
        is_weekend = day_of_week in WEEKEND_DAYS
        shape = np.array(customer_type.weekend_hourly_shape if is_weekend else customer_type.hourly_shape, dtype=float)
        expected_daily_arrivals = (
            scenario.daily_customers[day_of_week] * scenario.type_share[day_of_week][customer_type.type_id]
        )
        rates_by_type[customer_type.type_id] = expected_daily_arrivals * shape / shape.sum()
    return rates_by_type


def expected_arrivals_table(scenario: Scenario) -> pd.DataFrame:
    """Columns: day_of_week, hour, type_id, expected_arrivals (for LP-based policies)."""
    rows = []
    for day_of_week in DAYS_OF_WEEK:
        rates_by_type = hourly_arrival_rates(scenario, day_of_week=day_of_week)
        for hour_position, hour in enumerate(scenario.open_hours):
            for customer_type in scenario.customer_types:
                rows.append(
                    {
                        "day_of_week": day_of_week,
                        "hour": hour,
                        "type_id": customer_type.type_id,
                        "expected_arrivals": rates_by_type[customer_type.type_id][hour_position],
                    }
                )
    return pd.DataFrame(rows, columns=["day_of_week", "hour", "type_id", "expected_arrivals"])


def draw_demand_multiplier(rng: np.random.Generator, coefficient_of_variation: float) -> float:
    """Gamma draw with mean 1 and the given coefficient of variation (cv=0 returns 1.0)."""
    if coefficient_of_variation == 0:
        return 1.0
    squared_cv = coefficient_of_variation**2
    return float(rng.gamma(shape=1.0 / squared_cv, scale=squared_cv))


def draw_day_customers(scenario: Scenario, day_index: int) -> DayCustomers:
    """Draw every customer of one day with their basket size and pick noise, sorted by arrival time."""
    seed_sequence = np.random.SeedSequence([scenario.seed, day_index])
    multiplier_rng, arrivals_rng, customer_rng = [
        np.random.default_rng(child) for child in seed_sequence.spawn(SPAWNED_STREAM_COUNT)
    ]
    day_of_week = day_of_week_for(scenario, day_index=day_index)
    demand_multiplier = draw_demand_multiplier(multiplier_rng, scenario.demand_multiplier_cv)
    rates_by_type = hourly_arrival_rates(scenario, day_of_week=day_of_week)

    arrivals = []  # (arrival_minute, hour, type_id)
    for hour_position, hour in enumerate(scenario.open_hours):
        hour_start_minute = hour_position * MINUTES_PER_HOUR
        for customer_type in scenario.customer_types:
            expected_count = demand_multiplier * rates_by_type[customer_type.type_id][hour_position]
            arrival_count = arrivals_rng.poisson(lam=expected_count)
            arrival_minutes = arrivals_rng.uniform(
                low=hour_start_minute, high=hour_start_minute + MINUTES_PER_HOUR, size=arrival_count
            )
            arrivals.extend((float(minute), hour, customer_type.type_id) for minute in arrival_minutes)
    arrivals.sort(key=lambda arrival: arrival[0])

    noise_columns = 1 + len(scenario.option_keys())
    customers = []
    for arrival_minute, hour, type_id in arrivals:
        customer_type = scenario.customer_type(type_id)
        planned_basket_size = 1 + int(customer_rng.poisson(lam=customer_type.basket_extra_mean))
        pick_noise = customer_rng.gumbel(size=(planned_basket_size, noise_columns))
        customers.append(
            Customer(
                arrival_minute=arrival_minute,
                hour=hour,
                type_id=type_id,
                planned_basket_size=planned_basket_size,
                pick_noise=pick_noise,
            )
        )
    return DayCustomers(
        day_index=day_index, day_of_week=day_of_week, demand_multiplier=demand_multiplier, customers=customers
    )
