"""
Pricing policies (Subagent B).

A policy sets a price for every stocked (product_id, age) each open hour, always on the
scenario's price ladder (fractions of the item's current menu price). The historical
policy reproduces the bakery's past practice and is used only to generate the history log.
"""

from __future__ import annotations

from bakery_sim.config import Scenario
from bakery_sim.interfaces import OptionKey, PriceDecision, PricingContext


def current_menu_price(scenario: Scenario, product_id: str, day_index: int) -> float:
    """Menu price on a given day, including the historical menu price increase."""
    raise NotImplementedError


def ladder_price(scenario: Scenario, product_id: str, day_index: int, level: float) -> PriceDecision:
    """Price at a ladder level; raises ValueError if level is not on scenario.price_ladder."""
    raise NotImplementedError


class FixedPricePolicy:
    """Same ladder level all day: `fresh_level` for fresh items, `day_old_level` for day-old."""

    def __init__(self, fresh_level: float = 1.0, day_old_level: float = 0.55, name: str = "fixed_price") -> None:
        raise NotImplementedError

    def prices(self, context: PricingContext) -> dict[OptionKey, PriceDecision]:
        raise NotImplementedError


class HistoricalPricingPolicy:
    """Menu price + end-of-day discount + weekday promo + day-old level + one menu price increase."""

    name = "historical"

    def __init__(self, scenario: Scenario) -> None:
        raise NotImplementedError

    def prices(self, context: PricingContext) -> dict[OptionKey, PriceDecision]:
        raise NotImplementedError
