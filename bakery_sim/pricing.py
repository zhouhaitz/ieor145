"""
Pricing policies (Subagent B).

A policy sets a price for every stocked (product_id, age) each open hour, always on the
scenario's price ladder (fractions of the item's current menu price). The historical
policy reproduces the bakery's past practice and is used only to generate the history log.
"""

from __future__ import annotations

import math

from bakery_sim.config import Scenario
from bakery_sim.interfaces import OptionKey, PriceDecision, PricingContext


def current_menu_price(scenario: Scenario, product_id: str, day_index: int) -> float:
    """Menu price on a given day, including the historical menu price increase."""
    base_price = scenario.product(product_id).base_price
    historical = scenario.historical_pricing
    if day_index >= historical.menu_increase_day_index:
        return base_price * (1 + historical.menu_increase_fraction)
    return base_price


def ladder_price(scenario: Scenario, product_id: str, day_index: int, level: float) -> PriceDecision:
    """Price at a ladder level; raises ValueError if level is not on scenario.price_ladder."""
    if not any(math.isclose(level, ladder_level) for ladder_level in scenario.price_ladder):
        raise ValueError(f"Price level {level} is not on the price ladder {scenario.price_ladder}")
    price = round(current_menu_price(scenario, product_id, day_index) * level, 2)
    return PriceDecision(price=price, level=level, reason="policy")


def _with_reason(decision: PriceDecision, reason: str) -> PriceDecision:
    return PriceDecision(price=decision.price, level=decision.level, reason=reason)


def _stocked_options(context: PricingContext) -> list[OptionKey]:
    return [option for option, units in context.inventory.items() if units > 0]


class FixedPricePolicy:
    """Same ladder level all day: `fresh_level` for fresh items, `day_old_level` for day-old."""

    def __init__(self, fresh_level: float = 1.0, day_old_level: float = 0.55, name: str = "fixed_price") -> None:
        self.fresh_level = fresh_level
        self.day_old_level = day_old_level
        self.name = name

    def prices(self, context: PricingContext) -> dict[OptionKey, PriceDecision]:
        decisions: dict[OptionKey, PriceDecision] = {}
        for product_id, age in _stocked_options(context):
            if age >= 1:
                decision = _with_reason(
                    ladder_price(context.scenario, product_id, context.day_index, self.day_old_level), "day_old"
                )
            else:
                fresh_decision = ladder_price(context.scenario, product_id, context.day_index, self.fresh_level)
                reason = "base" if math.isclose(self.fresh_level, 1.0) else "fixed_discount"
                decision = _with_reason(fresh_decision, reason)
            decisions[(product_id, age)] = decision
        return decisions


class HistoricalPricingPolicy:
    """Menu price + end-of-day discount + weekday promo + day-old level + one menu price increase."""

    name = "historical"

    def __init__(self, scenario: Scenario) -> None:
        self.historical = scenario.historical_pricing

    def prices(self, context: PricingContext) -> dict[OptionKey, PriceDecision]:
        historical = self.historical
        scenario = context.scenario
        first_end_of_day_hour = scenario.close_hour - historical.end_of_day_hours
        is_end_of_day = context.hour >= first_end_of_day_hour
        is_promo_time = context.day_of_week in historical.promo_days_of_week and context.hour in historical.promo_hours

        decisions: dict[OptionKey, PriceDecision] = {}
        for product_id, age in _stocked_options(context):
            if age >= 1:
                level, reason = historical.day_old_level, "day_old"
            elif is_end_of_day:
                level, reason = historical.end_of_day_level, "end_of_day"
            elif is_promo_time and product_id == historical.promo_product_id:
                level, reason = historical.promo_level, "promo"
            else:
                level, reason = 1.0, "base"
            decisions[(product_id, age)] = _with_reason(
                ladder_price(scenario, product_id, context.day_index, level), reason
            )
        return decisions
