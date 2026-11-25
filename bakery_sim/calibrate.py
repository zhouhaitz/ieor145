"""
Calibration: set unobservable inputs so outputs match sourced targets.

Only inputs listed in CALIBRATED_INPUTS may be tuned; observable inputs (prices, hours,
arrival shapes, daily traffic) stay fixed from sources, and outputs are never edited.
Type shares are fitted so total hourly traffic matches Popular Times; willingness-to-pay
level and spread and basket size are grid-searched against the report-card bands, with bake
quantities re-derived from the supply ratio for every candidate. Every candidate is logged.
"""

from __future__ import annotations

import itertools
from dataclasses import replace

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from bakery_sim.config import DAYS_OF_WEEK, Scenario
from bakery_sim.pricing import FixedPricePolicy
from bakery_sim.report_card import compute_metrics
from bakery_sim.simulator import simulate, simulate_history

CALIBRATED_INPUTS = ["type_share", "wtp_level_scale", "wtp_spread_scale", "basket_scale", "bake_quantities"]

WEEKEND_DAYS = ("Sat", "Sun")
SUM_TO_ONE_WEIGHT = 1e3  # weight of the sum-to-one row in the non-negative least squares system
SHARE_DECIMALS = 4
INPUT_DECIMALS = 4  # calibrated inputs are rounded so the saved config stays readable
CLOSENESS_TIE_BREAK_WEIGHT = 0.01  # among in-band candidates, prefer scales closest to 1 (the stated assumptions)
MINIMUM_DAYS_FOR_DEMAND = len(DAYS_OF_WEEK)  # need every day of week at least once

DEFAULT_WTP_LEVEL_SCALES = (0.85, 0.95, 1.05, 1.15, 1.25)
DEFAULT_WTP_SPREAD_SCALES = (0.6, 0.85, 1.1, 1.35, 1.6)
DEFAULT_BASKET_SCALES = (0.6, 0.85, 1.1, 1.35, 1.6)


def normalized_type_shapes(scenario: Scenario, day_of_week: str) -> np.ndarray:
    """Matrix (open hours x types) whose columns are each type's hourly shape summing to 1."""
    is_weekend = day_of_week in WEEKEND_DAYS
    columns = []
    for customer_type in scenario.customer_types:
        shape = np.array(customer_type.weekend_hourly_shape if is_weekend else customer_type.hourly_shape, dtype=float)
        columns.append(shape / shape.sum())
    return np.column_stack(columns)


def round_shares_to_sum_to_one(shares: np.ndarray) -> np.ndarray:
    """Round shares to SHARE_DECIMALS and put the rounding remainder on the largest share."""
    rounded = np.round(shares, SHARE_DECIMALS)
    largest = int(np.argmax(rounded))
    others_total = rounded.sum() - rounded[largest]
    rounded[largest] = round(1.0 - others_total, SHARE_DECIMALS)
    return rounded


def fit_type_shares(scenario: Scenario) -> Scenario:
    """Scenario whose type_share per day of week best reproduces observed_hourly_shape (no simulation)."""
    type_ids = [customer_type.type_id for customer_type in scenario.customer_types]
    fitted_shares = {}
    for day_of_week in DAYS_OF_WEEK:
        shape_matrix = normalized_type_shapes(scenario, day_of_week)
        observed = np.array(scenario.observed_hourly_shape[day_of_week], dtype=float)
        sum_to_one_row = SUM_TO_ONE_WEIGHT * np.ones((1, len(type_ids)))
        augmented_matrix = np.vstack([shape_matrix, sum_to_one_row])
        augmented_target = np.concatenate([observed, [SUM_TO_ONE_WEIGHT]])
        raw_shares, _ = nnls(augmented_matrix, augmented_target)
        rounded_shares = round_shares_to_sum_to_one(raw_shares / raw_shares.sum())
        fitted_shares[day_of_week] = dict(zip(type_ids, rounded_shares.tolist(), strict=True))
    return replace(scenario, type_share=fitted_shares)


def estimate_full_price_demand(scenario: Scenario, n_days: int = 28) -> pd.DataFrame:
    """Mean units demanded per day_of_week x product at menu price with unlimited stock."""
    if n_days < MINIMUM_DAYS_FOR_DEMAND:
        raise ValueError(f"n_days must be at least {MINIMUM_DAYS_FOR_DEMAND} to cover every day of week")
    dataset = simulate(
        scenario=scenario,
        policy=FixedPricePolicy(fresh_level=1.0),
        day_indices=list(range(n_days)),
        split="history",
        unlimited_inventory=True,
    )
    fresh_sales = dataset.inventory[dataset.inventory["age"] == 0]
    day_of_week_by_day = dataset.days.set_index("day_index")["day_of_week"]
    fresh_sales = fresh_sales.assign(day_of_week=fresh_sales["day_index"].map(day_of_week_by_day))
    units_per_day = fresh_sales.groupby(["day_of_week", "product_id", "day_index"])["sold_units"].sum()
    mean_units = units_per_day.groupby(["day_of_week", "product_id"]).mean().unstack("product_id")
    product_ids = [product.product_id for product in scenario.products]
    return mean_units.reindex(index=DAYS_OF_WEEK, columns=product_ids)


def set_bake_quantities(scenario: Scenario, supply_ratio: float | None = None, n_days: int = 28) -> Scenario:
    """Scenario with bake_quantities = round(supply_ratio * full-price demand) per day of week."""
    ratio = scenario.supply_ratio if supply_ratio is None else supply_ratio
    mean_demand = estimate_full_price_demand(scenario, n_days=n_days)
    bake_quantities = {
        day_of_week: {
            product_id: int(round(ratio * mean_demand.loc[day_of_week, product_id]))
            for product_id in mean_demand.columns
        }
        for day_of_week in DAYS_OF_WEEK
    }
    return replace(scenario, bake_quantities=bake_quantities, supply_ratio=ratio)


def apply_calibration(
    scenario: Scenario, wtp_level_scale: float, wtp_spread_scale: float, basket_scale: float = 1.0
) -> Scenario:
    """Scale every type's wtp_multiplier, wtp_spread and basket_extra_mean by the given factors."""
    scaled_types = [
        replace(
            customer_type,
            wtp_multiplier={
                product_id: round(multiplier * wtp_level_scale, INPUT_DECIMALS)
                for product_id, multiplier in customer_type.wtp_multiplier.items()
            },
            wtp_spread=round(customer_type.wtp_spread * wtp_spread_scale, INPUT_DECIMALS),
            basket_extra_mean=round(customer_type.basket_extra_mean * basket_scale, INPUT_DECIMALS),
        )
        for customer_type in scenario.customer_types
    ]
    return replace(scenario, customer_types=scaled_types)


def band_score(metric_values: dict[str, float], targets: dict[str, list[float]]) -> float:
    """Squared distance outside each target band, measured in band widths (0 when every metric is in band)."""
    score = 0.0
    for metric, (low, high) in targets.items():
        if metric not in metric_values:
            continue
        value = metric_values[metric]
        distance_outside = max(low - value, 0.0, value - high)
        score += (distance_outside / (high - low)) ** 2
    return score


def closeness_penalty(scales: tuple[float, ...]) -> float:
    """Small penalty for moving scales away from 1, so ties go to the starting assumptions."""
    return CLOSENESS_TIE_BREAK_WEIGHT * float(sum(np.log(scale) ** 2 for scale in scales))


def evaluate_candidate(
    scenario: Scenario, wtp_level_scale: float, wtp_spread_scale: float, basket_scale: float, n_days: int
) -> tuple[Scenario, dict[str, float]]:
    """Apply one set of scales, re-derive bake quantities, simulate history, and compute metrics."""
    scaled = apply_calibration(scenario, wtp_level_scale, wtp_spread_scale, basket_scale)
    stocked = set_bake_quantities(scaled, n_days=n_days)
    history = simulate_history(stocked.with_overrides(n_history_days=n_days, n_test_days=0))
    return stocked, compute_metrics(history, stocked)


def calibrate(
    scenario: Scenario,
    n_days: int = 28,
    wtp_level_scales: tuple[float, ...] = DEFAULT_WTP_LEVEL_SCALES,
    wtp_spread_scales: tuple[float, ...] = DEFAULT_WTP_SPREAD_SCALES,
    basket_scales: tuple[float, ...] = DEFAULT_BASKET_SCALES,
) -> tuple[Scenario, pd.DataFrame]:
    """Fit type shares, then grid-search the scales against scenario.targets; returns (scenario, log)."""
    shares_fitted = fit_type_shares(scenario)
    log_rows = []
    best_score, best_scenario = np.inf, shares_fitted
    for level_scale, spread_scale, basket_scale in itertools.product(
        wtp_level_scales, wtp_spread_scales, basket_scales
    ):
        candidate, metric_values = evaluate_candidate(shares_fitted, level_scale, spread_scale, basket_scale, n_days)
        score = band_score(metric_values, shares_fitted.targets) + closeness_penalty(
            (level_scale, spread_scale, basket_scale)
        )
        log_rows.append(
            {
                "wtp_level_scale": level_scale,
                "wtp_spread_scale": spread_scale,
                "basket_scale": basket_scale,
                **metric_values,
                "score": score,
                "seed": scenario.seed,
            }
        )
        if score < best_score:
            best_score, best_scenario = score, candidate
    log = pd.DataFrame(log_rows)
    log["chosen"] = log.index == log["score"].idxmin()
    return best_scenario, log
