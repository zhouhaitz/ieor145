"""
Plausibility report card.

Computes the outputs a real bakery would recognize (traffic, purchase rate, basket size,
ticket, leftovers, busiest hour, item mix, hourly traffic shape, price elasticity) and checks
each against the sourced band in scenario.targets. Run after every generation; a failing row
means the inputs (not the outputs) need revisiting.

hourly_shape_error is the model's EXPECTED traffic shape vs. Popular Times (noise-free, the fit
we control); realized_hourly_shape_error is the same gap in one simulated history (informational,
it also contains sampling noise). price_elasticity is measured by re-running the same customers
(common random numbers) with every fresh price cut one ladder step and unlimited stock.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bakery_sim.arrivals import hourly_arrival_rates
from bakery_sim.config import DAYS_OF_WEEK, Scenario
from bakery_sim.dataset import Dataset
from bakery_sim.interfaces import NO_PURCHASE
from bakery_sim.pricing import FixedPricePolicy

METRICS = [
    "customers_per_day",
    "purchase_rate",
    "items_per_purchase",
    "average_ticket",
    "leftover_share",
    "busiest_hour_customers",
    "max_item_revenue_share",
    "hourly_shape_error",
    "realized_hourly_shape_error",
    "price_elasticity",
]

ELASTICITY_DAYS = 28  # days simulated per price level when measuring price elasticity
ELASTICITY_CUT_LEVEL = 0.85  # one ladder step below menu price


def _visits_with_day_of_week(dataset: Dataset) -> pd.DataFrame:
    """One row per visit with its day of week attached."""
    day_of_week_by_day = dataset.days.set_index("day_index")["day_of_week"]
    visits = dataset.visits()
    return visits.assign(day_of_week=visits["day_index"].map(day_of_week_by_day))


def _busiest_hour_customers(visits: pd.DataFrame, days: pd.DataFrame, open_hours: list[int]) -> float:
    """Largest average number of visits arriving in one (day of week, hour) cell."""
    visits_per_day_hour = visits.groupby(["day_index", "hour"]).size().unstack("hour")
    visits_per_day_hour = visits_per_day_hour.reindex(index=days["day_index"], columns=open_hours, fill_value=0)
    visits_per_day_hour = visits_per_day_hour.fillna(0.0)
    day_of_week_per_day = days.set_index("day_index")["day_of_week"]
    average_by_weekday = visits_per_day_hour.groupby(day_of_week_per_day).mean()
    return float(average_by_weekday.to_numpy().max())


def _hourly_shape_error(visits: pd.DataFrame, days: pd.DataFrame, scenario: Scenario) -> float:
    """Max absolute gap between simulated and observed share of a weekday's visits per open hour."""
    visits_per_day_hour = visits.groupby(["day_index", "hour"]).size().unstack("hour")
    visits_per_day_hour = visits_per_day_hour.reindex(index=days["day_index"], columns=scenario.open_hours)
    visits_per_day_hour = visits_per_day_hour.fillna(0.0)
    share_per_day_hour = visits_per_day_hour.div(visits_per_day_hour.sum(axis="columns"), axis="index")
    day_of_week_per_day = days.set_index("day_index")["day_of_week"]
    simulated_share = share_per_day_hour.groupby(day_of_week_per_day).mean()
    largest_gap = 0.0
    for day_of_week in DAYS_OF_WEEK:
        if day_of_week not in simulated_share.index:
            continue
        observed_share = np.array(scenario.observed_hourly_shape[day_of_week], dtype=float)
        gap = np.abs(simulated_share.loc[day_of_week].to_numpy() - observed_share).max()
        largest_gap = max(largest_gap, float(gap))
    return largest_gap


def expected_hourly_shape_error(scenario: Scenario) -> float:
    """Noise-free version of hourly_shape_error: the gap between the model's expected shape and the observed one."""
    largest_gap = 0.0
    for day_of_week in DAYS_OF_WEEK:
        expected_per_hour = sum(hourly_arrival_rates(scenario, day_of_week=day_of_week).values())
        expected_share = expected_per_hour / expected_per_hour.sum()
        observed_share = np.array(scenario.observed_hourly_shape[day_of_week], dtype=float)
        largest_gap = max(largest_gap, float(np.abs(expected_share - observed_share).max()))
    return largest_gap


def price_elasticity(scenario: Scenario, n_days: int = ELASTICITY_DAYS) -> float:
    """Arc elasticity of total fresh units sold when every fresh price drops one ladder step (absolute value)."""
    from bakery_sim.simulator import simulate  # local import: simulator imports nothing from this module

    units_sold = {}
    for level in (1.0, ELASTICITY_CUT_LEVEL):
        dataset = simulate(
            scenario=scenario,
            policy=FixedPricePolicy(fresh_level=level),
            day_indices=list(range(n_days)),
            split="history",
            unlimited_inventory=True,
        )
        units_sold[level] = dataset.inventory.loc[dataset.inventory["age"] == 0, "sold_units"].sum()
    quantity_change = (units_sold[ELASTICITY_CUT_LEVEL] - units_sold[1.0]) / (
        (units_sold[ELASTICITY_CUT_LEVEL] + units_sold[1.0]) / 2
    )
    price_change = (ELASTICITY_CUT_LEVEL - 1.0) / ((ELASTICITY_CUT_LEVEL + 1.0) / 2)
    return float(abs(quantity_change / price_change))


def compute_metrics(dataset: Dataset, scenario: Scenario) -> dict[str, float]:
    """Value of every metric in METRICS for this dataset."""
    visits = dataset.visits()
    bought = visits[visits["items_bought"] > 0]
    inventory_totals = dataset.inventory[["baked_units", "wasted_units"]].sum()
    purchases = dataset.choices[dataset.choices["chosen_product_id"] != NO_PURCHASE]
    revenue_by_product = purchases.groupby("chosen_product_id")["price_paid"].sum()
    return {
        "customers_per_day": len(visits) / len(dataset.days),
        "purchase_rate": len(bought) / len(visits),
        "items_per_purchase": float(bought["items_bought"].mean()),
        "average_ticket": float(bought["spend"].mean()),
        "leftover_share": float(inventory_totals["wasted_units"] / inventory_totals["baked_units"]),
        "busiest_hour_customers": _busiest_hour_customers(visits, dataset.days, scenario.open_hours),
        "max_item_revenue_share": float(revenue_by_product.max() / revenue_by_product.sum()),
        "hourly_shape_error": expected_hourly_shape_error(scenario),
        "realized_hourly_shape_error": _hourly_shape_error(visits, dataset.days, scenario),
        "price_elasticity": price_elasticity(scenario),
    }


def report_card(dataset: Dataset, scenario: Scenario) -> pd.DataFrame:
    """Columns: metric, value, low, high, passed (metrics without a target have NaN bounds and pass)."""
    metric_values = compute_metrics(dataset, scenario)
    rows = []
    for metric, value in metric_values.items():
        low, high = scenario.targets.get(metric, [np.nan, np.nan])
        rows.append({"metric": metric, "value": value, "low": low, "high": high, "passed": low <= value <= high})
    card = pd.DataFrame(rows, columns=["metric", "value", "low", "high", "passed"])
    card["passed"] = card["passed"] | card["low"].isna()
    return card


def format_report_card(card: pd.DataFrame) -> str:
    """The report card as a printable table, one row per metric."""
    lines = []
    for row in card.itertuples():
        status = "PASS" if row.passed else "FAIL"
        band = "no target" if np.isnan(row.low) else f"[{row.low:g}, {row.high:g}]"
        lines.append(f"{status}  {row.metric:<26}{row.value:>10.3f}  {band}")
    return "\n".join(lines)
