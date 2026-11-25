"""
Day-by-day bakery simulation (Subagent B).

For each day: day-old stock carries in (if enabled), scheduled batches are baked, each
customer arrives in time order and picks items one at a time from what is on the shelf at
the current hour's prices, and at closing leftovers are carried over (multi-day items, if
enabled) or salvaged. Customers come from arrivals.draw_day_customers, so results depend
on the policy only through prices.
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import date, timedelta

import pandas as pd

from bakery_sim import arrivals, choice
from bakery_sim.config import Scenario
from bakery_sim.dataset import Dataset
from bakery_sim.interfaces import (
    NO_PURCHASE,
    OPTION_SEPARATOR,
    TABLE_COLUMNS,
    Customer,
    OptionKey,
    PriceDecision,
    PricingContext,
    PricingPolicy,
)
from bakery_sim.pricing import HistoricalPricingPolicy

UNLIMITED_STOCK = 10**9
VISIT_ID_STRIDE = 100_000  # visit_id = day_index * stride + customer sequence within the day


def split_bake_quantity(total_units: int, shares: list[float]) -> list[int]:
    """Split a total into integer batch sizes by share (largest remainder); sizes sum exactly to total."""
    exact_sizes = [total_units * share for share in shares]
    batch_sizes = [math.floor(exact_size) for exact_size in exact_sizes]
    by_largest_remainder = sorted(
        range(len(shares)), key=lambda index: exact_sizes[index] - batch_sizes[index], reverse=True
    )
    for index in by_largest_remainder[: total_units - sum(batch_sizes)]:
        batch_sizes[index] += 1
    return batch_sizes


def plan_batches(scenario: Scenario, day_of_week: str) -> dict[int, dict[str, int]]:
    """Units to bake by open hour and product for one day of week."""
    units_by_hour: dict[int, dict[str, int]] = defaultdict(dict)
    for product in scenario.products:
        batches = [batch for batch in scenario.bake_schedule if batch.product_id == product.product_id]
        total_units = scenario.bake_quantities[day_of_week][product.product_id]
        batch_sizes = split_bake_quantity(total_units, [batch.share for batch in batches])
        for batch, batch_size in zip(batches, batch_sizes, strict=True):
            hour_plan = units_by_hour[batch.hour]
            hour_plan[product.product_id] = hour_plan.get(product.product_id, 0) + batch_size
    return units_by_hour


def format_options(available_options: list[OptionKey], prices: dict[OptionKey, float]) -> str:
    """Options string such as 'croissant@0:5.00;sourdough@1:3.85'."""
    return OPTION_SEPARATOR.join(
        f"{product_id}@{age}:{prices[(product_id, age)]:.2f}" for product_id, age in available_options
    )


def check_decisions(
    scenario: Scenario,
    policy: PricingPolicy,
    inventory: dict[OptionKey, int],
    decisions: dict[OptionKey, PriceDecision],
) -> None:
    """Raise ValueError if a policy skips a stocked option or uses a price level off the ladder."""
    for option, units in inventory.items():
        if units > 0 and option not in decisions:
            raise ValueError(f"Policy {policy.name!r} set no price for stocked option {option}")
    for option, decision in decisions.items():
        if not any(math.isclose(decision.level, ladder_level) for ladder_level in scenario.price_ladder):
            raise ValueError(f"Policy {policy.name!r} used level {decision.level} for {option}, not on the ladder")


def serve_customer(
    scenario: Scenario,
    customer: Customer,
    day_index: int,
    visit_id: int,
    decisions: dict[OptionKey, PriceDecision],
    inventory: dict[OptionKey, int],
    sold_units: dict[OptionKey, int],
    sellout_minutes: dict[OptionKey, float],
    choice_rows: list[tuple],
    unlimited_inventory: bool,
) -> None:
    """Run one customer's picks, updating inventory and appending one choices row per pick."""
    for pick_number in range(customer.planned_basket_size):
        available_options = [
            option for option in scenario.option_keys() if inventory[option] > 0 and option in decisions
        ]
        prices = {option: decisions[option].price for option in available_options}
        chosen_option = None
        if available_options:
            chosen_option = choice.choose_option(
                scenario=scenario,
                customer=customer,
                pick_number=pick_number,
                available_options=available_options,
                prices=prices,
            )
        pick_description = (
            day_index,
            visit_id,
            customer.arrival_minute,
            customer.hour,
            customer.type_id,
            pick_number,
            format_options(available_options, prices),
        )
        if chosen_option is None:
            choice_rows.append((*pick_description, NO_PURCHASE, -1, 0.0))
            return
        choice_rows.append((*pick_description, chosen_option[0], chosen_option[1], prices[chosen_option]))
        sold_units[chosen_option] += 1
        if not unlimited_inventory:
            inventory[chosen_option] -= 1
            if inventory[chosen_option] == 0:
                sellout_minutes.setdefault(chosen_option, customer.arrival_minute)


def simulate(
    scenario: Scenario,
    policy: PricingPolicy,
    day_indices: list[int],
    split: str,
    unlimited_inventory: bool = False,
) -> Dataset:
    """Run consecutive days under one policy; carryover state starts empty at the first day."""
    day_rows: list[tuple] = []
    price_rows: list[tuple] = []
    choice_rows: list[tuple] = []
    inventory_rows: list[tuple] = []
    first_date = date.fromisoformat(scenario.start_date)
    option_keys = scenario.option_keys()
    carryover: dict[OptionKey, int] = {}

    for day_index in day_indices:
        day_of_week = arrivals.day_of_week_for(scenario, day_index)
        day_customers = arrivals.draw_day_customers(scenario, day_index)
        day_date = (first_date + timedelta(days=day_index)).isoformat()
        day_rows.append((day_index, day_date, day_of_week, split, day_customers.demand_multiplier))

        inventory = dict.fromkeys(option_keys, 0)
        for (product_id, age), units in carryover.items():
            inventory[(product_id, age + 1)] += units
        start_units = dict(inventory)
        if unlimited_inventory:
            # demand measurement: unlimited fresh stock only, so discounted day-old never competes with fresh
            inventory = {option: UNLIMITED_STOCK if option[1] == 0 else 0 for option in option_keys}

        planned_batches = plan_batches(scenario, day_of_week)
        baked_units = dict.fromkeys(option_keys, 0)
        for units_by_product in planned_batches.values():
            for product_id, units in units_by_product.items():
                baked_units[(product_id, 0)] += units
        sold_units = dict.fromkeys(option_keys, 0)
        sellout_minutes: dict[OptionKey, float] = {}

        customers_by_hour: dict[int, list[tuple[int, Customer]]] = defaultdict(list)
        for customer_sequence, customer in enumerate(day_customers.customers):
            customers_by_hour[customer.hour].append((customer_sequence, customer))

        for hour in scenario.open_hours:
            if not unlimited_inventory:
                for product_id, units in planned_batches.get(hour, {}).items():
                    inventory[(product_id, 0)] += units
            context = PricingContext(
                scenario=scenario,
                day_index=day_index,
                day_of_week=day_of_week,
                hour=hour,
                inventory=dict(inventory),
            )
            decisions = policy.prices(context)
            check_decisions(scenario=scenario, policy=policy, inventory=inventory, decisions=decisions)
            for (product_id, age), decision in decisions.items():
                price_rows.append((day_index, hour, product_id, age, decision.price, decision.level, decision.reason))
            for customer_sequence, customer in customers_by_hour[hour]:
                serve_customer(
                    scenario=scenario,
                    customer=customer,
                    day_index=day_index,
                    visit_id=day_index * VISIT_ID_STRIDE + customer_sequence,
                    decisions=decisions,
                    inventory=inventory,
                    sold_units=sold_units,
                    sellout_minutes=sellout_minutes,
                    choice_rows=choice_rows,
                    unlimited_inventory=unlimited_inventory,
                )

        carryover = {}
        for option in option_keys:
            product_id, age = option
            leftover_units = 0 if unlimited_inventory else inventory[option]
            can_carry = scenario.carryover_enabled and age + 1 < scenario.product(product_id).shelf_life_days
            carried_units = leftover_units if can_carry else 0
            if carried_units > 0:
                carryover[option] = carried_units
            inventory_rows.append(
                (
                    day_index,
                    product_id,
                    age,
                    start_units[option],
                    baked_units[option],
                    sold_units[option],
                    carried_units,
                    leftover_units - carried_units,
                    sellout_minutes.get(option, float("nan")),
                )
            )

    return Dataset(
        scenario_id=scenario.scenario_id,
        days=pd.DataFrame(day_rows, columns=TABLE_COLUMNS["days"]),
        prices=pd.DataFrame(price_rows, columns=TABLE_COLUMNS["prices"]),
        choices=pd.DataFrame(choice_rows, columns=TABLE_COLUMNS["choices"]),
        inventory=pd.DataFrame(inventory_rows, columns=TABLE_COLUMNS["inventory"]),
    )


def simulate_history(scenario: Scenario) -> Dataset:
    """History days (0..n_history_days-1) under HistoricalPricingPolicy, split='history'."""
    return simulate(
        scenario=scenario,
        policy=HistoricalPricingPolicy(scenario),
        day_indices=list(range(scenario.n_history_days)),
        split="history",
    )


def simulate_test(scenario: Scenario, policy: PricingPolicy) -> Dataset:
    """Test days (n_history_days..n_days-1) under the given policy, split='test'."""
    return simulate(
        scenario=scenario,
        policy=policy,
        day_indices=list(range(scenario.n_history_days, scenario.n_days)),
        split="test",
    )
