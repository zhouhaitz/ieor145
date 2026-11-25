"""
MNL estimation by maximum likelihood (Lectures 14-15).

Utility (w_j * r**age - p)/sigma = beta_j * r**age - alpha * p with beta_j = w_j/sigma, alpha = 1/sigma,
and r = share of value a day-old item keeps. The log-likelihood is concave in (beta, alpha) for fixed r;
all three are fitted jointly with an analytic gradient. Fitted values are mapped back to dollars
(w, sigma) and r to compare with the scenario's true inputs (the recovery check).
Each row of choices.csv is one pick: the options on the shelf (with prices) and what was
chosen, where leaving has utility zero. A tiny ridge on beta keeps estimates finite when
the data contain no walk-outs (register view), which is exactly the case that misleads.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from bakery_sim.config import Scenario
from bakery_sim.dataset import Dataset
from bakery_sim.interfaces import NO_PURCHASE, OPTION_SEPARATOR

MIN_PRICE_COEFFICIENT = 1e-3  # alpha = 1/sigma lower bound, so sigma stays finite
BETA_RIDGE_PENALTY = 1e-5  # per-row ridge on beta; negligible when walk-outs are observed
MAX_ITERATIONS = 2000
GRADIENT_TOLERANCE = 1e-9
PARAMETERS = ["w", "sigma", "day_old_value_share"]
DAY_OLD_SHARE_BOUNDS = (0.01, 1.5)  # search range for r; true values lie in (0, 1]


def parse_options(options: str) -> dict[tuple[str, int], float]:
    """'croissant@0:5.00;sourdough@1:3.85' -> {('croissant', 0): 5.0, ('sourdough', 1): 3.85}."""
    if not options:
        return {}
    parsed = {}
    for option_text in options.split(OPTION_SEPARATOR):
        product_and_age, price_text = option_text.split(":")
        product_id, age_text = product_and_age.split("@")
        parsed[(product_id, int(age_text))] = float(price_text)
    return parsed


@dataclass(frozen=True)
class PickObservations:
    """All picks flattened: one entry per (pick, option on the shelf), with the chosen option marked."""

    product_index: np.ndarray  # per option entry: position of its product in the scenario's product list
    age: np.ndarray  # per option entry: 0 fresh, 1 day-old
    price: np.ndarray  # per option entry: dollars
    row_of_option: np.ndarray  # per option entry: which pick it belongs to
    row_starts: np.ndarray  # per pick: index of its first option entry
    chosen_entry: np.ndarray  # per pick: option entry that was chosen, -1 if the customer left
    product_count: int

    @property
    def pick_count(self) -> int:
        return len(self.row_starts)


def build_pick_observations(choices: pd.DataFrame, scenario: Scenario, type_id: str | None = None) -> PickObservations:
    """Flatten the picks that had something on the shelf (optionally only one customer type's)."""
    picks = choices[choices["options"].fillna("") != ""]
    if type_id is not None:
        if "type_id" not in choices.columns:
            raise ValueError("Per-type estimation needs the truth view (the type_id column)")
        picks = picks[picks["type_id"] == type_id]
    product_positions = {product.product_id: position for position, product in enumerate(scenario.products)}
    product_index, age, price, row_of_option, row_starts, chosen_entry = [], [], [], [], [], []
    for pick_position, (options_text, chosen_product, chosen_age) in enumerate(
        zip(picks["options"], picks["chosen_product_id"], picks["chosen_age"], strict=True)
    ):
        row_starts.append(len(product_index))
        chosen_entry.append(-1)
        for (product_id, option_age), option_price in parse_options(options_text).items():
            if chosen_product != NO_PURCHASE and (product_id, option_age) == (chosen_product, chosen_age):
                chosen_entry[-1] = len(product_index)
            product_index.append(product_positions[product_id])
            age.append(option_age)
            price.append(option_price)
            row_of_option.append(pick_position)
    return PickObservations(
        product_index=np.array(product_index),
        age=np.array(age, dtype=float),
        price=np.array(price, dtype=float),
        row_of_option=np.array(row_of_option),
        row_starts=np.array(row_starts),
        chosen_entry=np.array(chosen_entry),
        product_count=len(scenario.products),
    )


def negative_log_likelihood(parameters: np.ndarray, observations: PickObservations) -> tuple[float, np.ndarray]:
    """Average negative log-likelihood per pick and its gradient; parameters = [beta..., alpha, r]."""
    product_count = observations.product_count
    beta = parameters[:product_count]
    alpha, day_old_share = parameters[product_count], parameters[product_count + 1]
    value_multiplier = 1.0 + (day_old_share - 1.0) * observations.age  # 1 for fresh, r for day-old
    option_beta = beta[observations.product_index]
    utility = option_beta * value_multiplier - alpha * observations.price

    row_maximum = np.maximum(np.maximum.reduceat(utility, observations.row_starts), 0.0)  # 0.0 = leaving
    shifted_exponentials = np.exp(utility - row_maximum[observations.row_of_option])
    denominator = np.exp(-row_maximum) + np.bincount(
        observations.row_of_option, weights=shifted_exponentials, minlength=observations.pick_count
    )
    probability = shifted_exponentials / denominator[observations.row_of_option]

    chose_something = observations.chosen_entry >= 0
    chosen = observations.chosen_entry[chose_something]
    log_likelihood = utility[chosen].sum() - (row_maximum + np.log(denominator)).sum()

    beta_gradient = np.bincount(
        observations.product_index, weights=probability * value_multiplier, minlength=product_count
    )
    beta_gradient -= np.bincount(
        observations.product_index[chosen], weights=value_multiplier[chosen], minlength=product_count
    )
    alpha_gradient = observations.price[chosen].sum() - probability @ observations.price
    share_sensitivity = option_beta * observations.age  # d utility / d r
    share_gradient = probability @ share_sensitivity - share_sensitivity[chosen].sum()
    gradient = np.concatenate([beta_gradient, [alpha_gradient, share_gradient]])

    ridge_value = 0.5 * BETA_RIDGE_PENALTY * beta @ beta
    ridge_gradient = np.concatenate([BETA_RIDGE_PENALTY * beta, [0.0, 0.0]])
    return -log_likelihood / observations.pick_count + ridge_value, gradient / observations.pick_count + ridge_gradient


def fit_mnl(choices: pd.DataFrame, scenario: Scenario, type_id: str | None = None) -> pd.DataFrame:
    """Fit one MNL (for one type, or pooled if type_id is None); columns: parameter, product_id, estimate."""
    observations = build_pick_observations(choices, scenario, type_id=type_id)
    product_count = observations.product_count
    start = np.concatenate([np.zeros(product_count), [1.0, 1.0]])
    bounds = [(None, None)] * product_count + [(MIN_PRICE_COEFFICIENT, None), DAY_OLD_SHARE_BOUNDS]
    result = minimize(
        negative_log_likelihood,
        start,
        args=(observations,),
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": MAX_ITERATIONS, "gtol": GRADIENT_TOLERANCE, "ftol": 1e-14},
    )
    beta, alpha, day_old_share = result.x[:product_count], result.x[product_count], result.x[product_count + 1]
    rows = [
        {"parameter": "w", "product_id": product.product_id, "estimate": beta[position] / alpha}
        for position, product in enumerate(scenario.products)
    ]
    rows.append({"parameter": "sigma", "product_id": None, "estimate": 1.0 / alpha})
    rows.append({"parameter": "day_old_value_share", "product_id": None, "estimate": day_old_share})
    return pd.DataFrame(rows, columns=["parameter", "product_id", "estimate"])


def recovery_table(choices: pd.DataFrame, scenario: Scenario) -> pd.DataFrame:
    """True vs. estimated w, sigma, day-old value share per type, with relative error (estimate - true) / true."""
    tables = []
    for customer_type in scenario.customer_types:
        fitted = fit_mnl(choices, scenario, type_id=customer_type.type_id)
        true_values = {
            ("w", product.product_id): customer_type.wtp_multiplier[product.product_id] * product.base_price
            for product in scenario.products
        }
        true_values[("sigma", None)] = customer_type.wtp_spread
        true_values[("day_old_value_share", None)] = customer_type.day_old_value_share
        fitted["true"] = [
            true_values[(parameter, None if pd.isna(product_id) else product_id)]
            for parameter, product_id in zip(fitted["parameter"], fitted["product_id"], strict=True)
        ]
        fitted["relative_error"] = (fitted["estimate"] - fitted["true"]) / fitted["true"]
        fitted.insert(0, "type_id", customer_type.type_id)
        tables.append(fitted)
    return pd.concat(tables, ignore_index=True)[
        ["type_id", "parameter", "product_id", "true", "estimate", "relative_error"]
    ]


def register_bias_table(dataset: Dataset, scenario: Scenario) -> pd.DataFrame:
    """Pooled MNL on door-counter vs register data (columns: parameter, product_id, <view>_estimate)."""
    door_counter_fit = fit_mnl(dataset.view("door_counter").choices, scenario)
    register_fit = fit_mnl(dataset.view("register").choices, scenario)
    table = door_counter_fit.rename(columns={"estimate": "door_counter_estimate"})
    table["register_estimate"] = register_fit["estimate"].to_numpy()
    return table
