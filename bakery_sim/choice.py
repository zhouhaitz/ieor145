"""
Customer choice: multinomial logit (MNL) from willingness to pay (Subagent A).

A type-k customer's utility for option j = (product, age) at price p is

    U_kj = (w_kj - p - delta_k * age) / sigma_k + epsilon_kj,   U_k0 = epsilon_k0 (leave)

with w_kj = wtp_multiplier[k][product] * base_price and epsilon i.i.d. standard Gumbel,
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
    """Deterministic utility (w - p - delta*age) / sigma for each option, in the order given."""
    raise NotImplementedError


def choice_probabilities(
    scenario: Scenario,
    customer_type: CustomerType,
    options: list[OptionKey],
    prices: dict[OptionKey, float],
) -> dict[OptionKey | None, float]:
    """MNL probabilities for each option plus None (leave); sums to 1."""
    raise NotImplementedError


def choose_option(
    scenario: Scenario,
    customer: Customer,
    pick_number: int,
    available_options: list[OptionKey],
    prices: dict[OptionKey, float],
) -> OptionKey | None:
    """The option the customer picks on this pick (None = leave), using their pre-drawn noise."""
    raise NotImplementedError
