"""
Shared contracts between modules.

- DayCustomers / Customer: everything random about a day's customers, drawn BEFORE any
  price is known (arrivals.py + choice.py produce it). Two pricing policies run on the
  same DayCustomers face identical people with identical tastes (common random numbers).
- PricingContext / PricingPolicy: what a policy sees each hour and what it returns.
- TABLE_COLUMNS: exact column order of the four output tables.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from bakery_sim.config import Scenario

OptionKey = tuple[str, int]  # (product_id, age); age 0 = fresh, 1 = day-old


@dataclass(frozen=True)
class Customer:
    """One arriving customer and all of their pre-drawn randomness."""

    arrival_minute: float  # minutes since opening
    hour: int  # clock hour of arrival
    type_id: str
    planned_basket_size: int  # N >= 1
    pick_noise: np.ndarray  # shape (planned_basket_size, 1 + len(scenario.option_keys())); column 0 = leave


@dataclass(frozen=True)
class DayCustomers:
    """All customers of one day, sorted by arrival time."""

    day_index: int
    day_of_week: str
    demand_multiplier: float  # M_d
    customers: list[Customer]


@dataclass(frozen=True)
class PricingContext:
    """What a pricing policy may observe when setting prices for one open hour."""

    scenario: Scenario
    day_index: int
    day_of_week: str
    hour: int
    inventory: dict[OptionKey, int]  # units on the shelf at the start of the hour


@dataclass(frozen=True)
class PriceDecision:
    """A price plus its ladder level and why it was set (logged in prices.csv)."""

    price: float  # dollars
    level: float  # fraction of the item's current menu price, e.g. 0.70
    reason: str  # e.g. "base", "end_of_day", "promo", "day_old", "policy"


class PricingPolicy(Protocol):
    """Sets the price of every (product_id, age) for one open hour."""

    name: str

    def prices(self, context: PricingContext) -> dict[OptionKey, PriceDecision]: ...


TABLE_COLUMNS = {
    "days": ["day_index", "date", "day_of_week", "split", "demand_multiplier"],
    "prices": ["day_index", "hour", "product_id", "age", "price", "price_level", "price_reason"],
    "choices": [
        "day_index",
        "visit_id",
        "arrival_minute",
        "hour",
        "type_id",
        "pick_number",
        "options",
        "chosen_product_id",
        "chosen_age",
        "price_paid",
    ],
    "inventory": [
        "day_index",
        "product_id",
        "age",
        "start_units",
        "baked_units",
        "sold_units",
        "carried_over_units",
        "wasted_units",
        "sellout_minute",
    ],
}

# choices.options format: "croissant@0:5.00;sourdough@1:3.85" = every (product@age:price) on the
# shelf at that pick. A walk-out/stop is chosen_product_id="none", chosen_age=-1, price_paid=0.0.
# inventory.sellout_minute is NaN if that (product, age) never ran out that day.
OPTION_SEPARATOR = ";"
NO_PURCHASE = "none"

# Columns hidden from the bakery (dropped in door-counter and register views).
TRUTH_ONLY_COLUMNS = {"days": ["demand_multiplier"], "choices": ["type_id"]}
