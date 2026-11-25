"""
Example pricing policy built from the MNL model (a starting point for the team's policies).

MyopicMnlPolicy: each open hour, for each item on the shelf, pick the ladder price that
maximizes this hour's expected revenue,

    sum over types k of  lambda_k(h) * sum over items j of  p_j * P_k(j | prices),

using the true arrival rates and MNL choice probabilities (the bakery knows each type's
parameters but not who walks in). It ignores remaining inventory, so it discounts even when
stock would sell out at full price later; an inventory-aware policy (DP, LP / bid prices)
should beat it on profit.
"""

from __future__ import annotations

from bakery_sim.arrivals import hourly_arrival_rates
from bakery_sim.choice import choice_probabilities
from bakery_sim.config import Scenario
from bakery_sim.interfaces import OptionKey, PriceDecision, PricingContext
from bakery_sim.pricing import ladder_price


class MyopicMnlPolicy:
    """Best ladder price per item for the current hour's expected revenue; ignores inventory."""

    name = "myopic_mnl"

    def prices(self, context: PricingContext) -> dict[OptionKey, PriceDecision]:
        scenario = context.scenario
        stocked_options = [option for option, units in context.inventory.items() if units > 0]
        expected_arrivals = self._expected_arrivals(scenario, context.day_of_week, context.hour)
        chosen_levels = dict.fromkeys(stocked_options, 1.0)
        for option in stocked_options:  # coordinate search: improve one item while holding the others fixed
            chosen_levels[option] = max(
                scenario.price_ladder,
                key=lambda level, option=option: self._expected_revenue(
                    context, stocked_options, {**chosen_levels, option: level}, expected_arrivals
                ),
            )
        return {
            option: PriceDecision(
                price=ladder_price(scenario, option[0], context.day_index, level).price, level=level, reason="policy"
            )
            for option, level in chosen_levels.items()
        }

    @staticmethod
    def _expected_arrivals(scenario: Scenario, day_of_week: str, hour: int) -> dict[str, float]:
        """Expected customers of each type in this hour."""
        hour_position = scenario.open_hours.index(hour)
        rates = hourly_arrival_rates(scenario, day_of_week)
        return {type_id: float(type_rates[hour_position]) for type_id, type_rates in rates.items()}

    @staticmethod
    def _expected_revenue(
        context: PricingContext,
        stocked_options: list[OptionKey],
        levels: dict[OptionKey, float],
        expected_arrivals: dict[str, float],
    ) -> float:
        """Expected revenue this hour if items are priced at the given ladder levels."""
        scenario = context.scenario
        prices = {
            option: ladder_price(scenario, option[0], context.day_index, level).price
            for option, level in levels.items()
        }
        revenue = 0.0
        for customer_type in scenario.customer_types:
            probabilities = choice_probabilities(scenario, customer_type, stocked_options, prices)
            revenue_per_customer = sum(prices[option] * probabilities[option] for option in stocked_options)
            revenue += expected_arrivals[customer_type.type_id] * revenue_per_customer
        return revenue
