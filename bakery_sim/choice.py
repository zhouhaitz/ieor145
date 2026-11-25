"""
Customer choice: multinomial logit (MNL) from willingness to pay (Subagent A).

A type-k customer's utility for option j = (product, age) at price p is

    U_kj = (w_kj * r_k ** age - p) / sigma_k + epsilon_kj,   U_k0 = epsilon_k0 (leave)

with w_kj = wtp_multiplier[k][product] * base_price, r_k = day_old_value_share (a day-old item keeps
that share of its value, for any product sold day-old), and epsilon i.i.d. standard Gumbel,
which gives MNL choice probabilities (Lecture 15). With one item on offer this reduces to
1 - F(p) with a logistic F (Lecture 16). The Gumbel noise is pre-drawn per customer and
pick (Customer.pick_noise), so a choice is the argmax over options currently on the shelf.
"""

from __future__ import annotations

import numpy as np

from bakery_sim.config import CustomerType, Scenario
from bakery_sim.interfaces import Customer, OptionKey


def mean_utilities(
    scenario: Scenario,
    customer_type: CustomerType,
    options: list[OptionKey],
    prices: dict[OptionKey, float],
) -> np.ndarray:
    """Deterministic utility (w * r**age - p) / sigma for each option, in the order given."""
    utilities = []
    for option in options:
        product_id, age = option
        willingness_to_pay = customer_type.wtp_multiplier[product_id] * scenario.product(product_id).base_price
        net_value = willingness_to_pay * customer_type.day_old_value_share**age - prices[option]
        utilities.append(net_value / customer_type.wtp_spread)
    return np.array(utilities, dtype=float)


def choice_probabilities(
    scenario: Scenario,
    customer_type: CustomerType,
    options: list[OptionKey],
    prices: dict[OptionKey, float],
) -> dict[OptionKey | None, float]:
    """MNL probabilities for each option plus None (leave); sums to 1."""
    leave_utility = 0.0
    utilities = np.concatenate([[leave_utility], mean_utilities(scenario, customer_type, options, prices)])
    exponentials = np.exp(utilities - utilities.max())  # subtract the max for numerical stability
    probabilities = exponentials / exponentials.sum()
    return dict(zip([None, *options], probabilities.tolist(), strict=True))


def choose_option(
    scenario: Scenario,
    customer: Customer,
    pick_number: int,
    available_options: list[OptionKey],
    prices: dict[OptionKey, float],
) -> OptionKey | None:
    """The option the customer picks on this pick (None = leave), using their pre-drawn noise."""
    if not available_options:
        return None
    customer_type = scenario.customer_type(customer.type_id)
    noise = customer.pick_noise[pick_number]
    option_keys = scenario.option_keys()
    noise_for_options = np.array([noise[1 + option_keys.index(option)] for option in available_options])
    scores = mean_utilities(scenario, customer_type, available_options, prices) + noise_for_options
    best_position = int(np.argmax(scores))
    if scores[best_position] > noise[0]:
        return available_options[best_position]
    return None
