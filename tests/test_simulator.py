"""Simulator mechanics, tested against fake arrivals and choice so they do not depend on those modules."""

import math

import numpy as np
import pytest

from bakery_sim import arrivals, choice
from bakery_sim.config import DAYS_OF_WEEK, BakeBatch, HistoricalPricing, Product, Scenario
from bakery_sim.interfaces import Customer, DayCustomers, PriceDecision
from bakery_sim.pricing import FixedPricePolicy, HistoricalPricingPolicy
from bakery_sim.simulator import (
    UNLIMITED_STOCK,
    simulate,
    simulate_history,
    simulate_test,
    split_bake_quantity,
)

LEAVER = "leaver"
LEAVES_AFTER_FIRST_PICK = "once:croissant"
OPEN_HOUR = 7


@pytest.fixture
def tiny_scenario(baseline_scenario) -> Scenario:
    """Three open hours, a same-day croissant (two batches) and a two-day bread (one batch)."""
    products = [
        Product(product_id="croissant", name="Croissant", base_price=5.0, unit_cost=1.5, shelf_life_days=1),
        Product(product_id="bread", name="Bread", base_price=4.0, unit_cost=1.0, shelf_life_days=2),
    ]
    schedule = [
        BakeBatch(product_id="croissant", hour=7, share=0.6),
        BakeBatch(product_id="croissant", hour=8, share=0.4),
        BakeBatch(product_id="bread", hour=7, share=1.0),
    ]
    quantities = {day_of_week: {"croissant": 7, "bread": 4} for day_of_week in DAYS_OF_WEEK}
    historical = HistoricalPricing(
        end_of_day_hours=1,
        end_of_day_level=0.7,
        day_old_level=0.55,
        promo_days_of_week=["Tue"],
        promo_product_id="croissant",
        promo_level=0.85,
        promo_hours=[8],
        menu_increase_day_index=100,
        menu_increase_fraction=0.05,
    )
    return baseline_scenario.with_overrides(
        n_history_days=2,
        n_test_days=1,
        open_hour=OPEN_HOUR,
        close_hour=OPEN_HOUR + 3,
        products=products,
        bake_schedule=schedule,
        bake_quantities=quantities,
        historical_pricing=historical,
        carryover_enabled=True,
    )


def make_customer(minute: float, wants: str, basket_size: int = 1) -> Customer:
    """A customer whose type_id tells the fake choice what to buy."""
    return Customer(
        arrival_minute=minute,
        hour=OPEN_HOUR + int(minute // 60),
        type_id=wants,
        planned_basket_size=basket_size,
        pick_noise=np.zeros((basket_size, 1)),
    )


def fake_choose_option(scenario, customer, pick_number, available_options, prices):
    """Buy the first available option of the wanted product; the 'leaver' types never buy."""
    if customer.type_id == LEAVER:
        return None
    if customer.type_id == LEAVES_AFTER_FIRST_PICK:
        wanted_product = "croissant" if pick_number == 0 else None
    else:
        wanted_product = customer.type_id
    for option in available_options:
        if option[0] == wanted_product:
            return option
    return None


@pytest.fixture
def customers_by_day(monkeypatch) -> dict[int, list[Customer]]:
    """Patch arrivals and choice with deterministic fakes; tests fill the returned dict."""
    scripted_customers: dict[int, list[Customer]] = {}

    def fake_draw_day_customers(scenario, day_index):
        return DayCustomers(
            day_index=day_index,
            day_of_week=DAYS_OF_WEEK[day_index % 7],
            demand_multiplier=1.0 + day_index,
            customers=scripted_customers.get(day_index, []),
        )

    def fake_day_of_week_for(scenario, day_index):
        return DAYS_OF_WEEK[day_index % 7]

    monkeypatch.setattr(arrivals, "draw_day_customers", fake_draw_day_customers)
    monkeypatch.setattr(arrivals, "day_of_week_for", fake_day_of_week_for)
    monkeypatch.setattr(choice, "choose_option", fake_choose_option)
    return scripted_customers


def inventory_row(dataset, day_index, product_id, age):
    table = dataset.inventory
    return table[(table["day_index"] == day_index) & (table["product_id"] == product_id) & (table["age"] == age)].iloc[
        0
    ]


def run_days(scenario, day_indices, policy=None, **kwargs):
    policy = policy or FixedPricePolicy()
    return simulate(scenario, policy, day_indices, split="test", **kwargs)


def test_split_bake_quantity_sums_exactly():
    assert split_bake_quantity(7, [0.6, 0.4]) == [4, 3]
    assert split_bake_quantity(10, [1 / 3, 1 / 3, 1 / 3]) == [4, 3, 3]
    for total_units in range(0, 60):
        assert sum(split_bake_quantity(total_units, [0.5, 0.3, 0.2])) == total_units


def test_baked_units_match_plan_and_batches_appear_at_their_hour(tiny_scenario, customers_by_day):
    # Six croissant buyers in hour 7: only the 4-unit first batch exists. One more buyer in hour 8.
    customers_by_day[0] = [make_customer(minute, "croissant") for minute in range(6)]
    customers_by_day[0].append(make_customer(65, "croissant"))
    dataset = run_days(tiny_scenario, [0])
    assert inventory_row(dataset, 0, "croissant", 0)["baked_units"] == 7
    hour_seven = dataset.choices[dataset.choices["hour"] == 7]
    assert (hour_seven["chosen_product_id"] == "croissant").sum() == 4
    assert (hour_seven["chosen_product_id"] == "none").sum() == 2
    hour_eight = dataset.choices[dataset.choices["hour"] == 8]
    assert list(hour_eight["chosen_product_id"]) == ["croissant"]
    assert inventory_row(dataset, 0, "croissant", 0)["sold_units"] == 5


def test_first_sellout_minute_is_recorded_once(tiny_scenario, customers_by_day):
    customers_by_day[0] = [make_customer(minute, "croissant") for minute in range(4)]
    customers_by_day[0].append(make_customer(70, "croissant"))
    dataset = run_days(tiny_scenario, [0])
    assert inventory_row(dataset, 0, "croissant", 0)["sellout_minute"] == 3
    assert math.isnan(inventory_row(dataset, 0, "bread", 0)["sellout_minute"])


def test_inventory_invariants_hold_over_several_days(tiny_scenario, customers_by_day):
    for day_index in range(3):
        customers_by_day[day_index] = [make_customer(minute * 7, "croissant", basket_size=2) for minute in range(20)]
        customers_by_day[day_index] += [make_customer(minute * 5, "bread") for minute in range(3 * (day_index + 1))]
    dataset = run_days(tiny_scenario, [0, 1, 2])
    inventory = dataset.inventory
    assert (inventory["start_units"] + inventory["baked_units"] - inventory["sold_units"]).tolist() == (
        inventory["carried_over_units"] + inventory["wasted_units"]
    ).tolist()
    numeric_columns = ["start_units", "baked_units", "sold_units", "carried_over_units", "wasted_units"]
    assert (inventory[numeric_columns] >= 0).all().all()
    assert len(inventory) == 3 * len(tiny_scenario.option_keys())


def test_sold_units_equal_non_none_choices(tiny_scenario, customers_by_day):
    customers_by_day[0] = [make_customer(minute * 3, "croissant" if minute % 2 else "bread") for minute in range(30)]
    dataset = run_days(tiny_scenario, [0])
    bought = dataset.choices[dataset.choices["chosen_product_id"] != "none"]
    for product_id, age in tiny_scenario.option_keys():
        picks = ((bought["chosen_product_id"] == product_id) & (bought["chosen_age"] == age)).sum()
        assert inventory_row(dataset, 0, product_id, age)["sold_units"] == picks


def test_carryover_moves_leftovers_to_age_one(tiny_scenario, customers_by_day):
    customers_by_day[0] = [make_customer(0, "bread")]
    dataset = run_days(tiny_scenario, [0, 1])
    assert inventory_row(dataset, 0, "bread", 0)["carried_over_units"] == 3
    assert inventory_row(dataset, 1, "bread", 1)["start_units"] == 3
    assert inventory_row(dataset, 0, "croissant", 0)["carried_over_units"] == 0
    assert inventory_row(dataset, 0, "croissant", 0)["wasted_units"] == 7
    day_one_prices = dataset.prices[dataset.prices["day_index"] == 1]
    day_old_prices = day_one_prices[(day_one_prices["product_id"] == "bread") & (day_one_prices["age"] == 1)]
    assert set(day_old_prices["price_reason"]) == {"day_old"}


def test_day_old_leftovers_are_wasted_not_carried_twice(tiny_scenario, customers_by_day):
    dataset = run_days(tiny_scenario, [0, 1, 2])
    assert inventory_row(dataset, 1, "bread", 1)["carried_over_units"] == 0
    assert inventory_row(dataset, 1, "bread", 1)["wasted_units"] == 4
    assert inventory_row(dataset, 2, "bread", 1)["start_units"] == 4  # day 1's fresh bread, not day 0's


def test_disabled_carryover_has_no_day_old_stock(tiny_scenario, customers_by_day):
    scenario = tiny_scenario.with_overrides(carryover_enabled=False)
    dataset = run_days(scenario, [0, 1])
    assert inventory_row(dataset, 0, "bread", 0)["carried_over_units"] == 0
    assert inventory_row(dataset, 0, "bread", 0)["wasted_units"] == 4
    assert (dataset.inventory[dataset.inventory["age"] == 1]["start_units"] == 0).all()
    assert (dataset.prices["age"] == 0).all()


def test_carryover_starts_empty_at_first_day_of_call(tiny_scenario, customers_by_day):
    dataset = run_days(tiny_scenario, [1])
    assert (dataset.inventory["start_units"] == 0).all()


def test_unlimited_inventory_never_sells_out(tiny_scenario, customers_by_day):
    customers_by_day[0] = [make_customer(minute, "croissant") for minute in range(40)]
    dataset = run_days(tiny_scenario, [0], unlimited_inventory=True)
    croissant_row = inventory_row(dataset, 0, "croissant", 0)
    assert croissant_row["sold_units"] == 40
    assert croissant_row["baked_units"] == 7
    assert dataset.inventory["sellout_minute"].isna().all()
    assert (dataset.choices["chosen_product_id"] == "croissant").all()
    assert UNLIMITED_STOCK > 40


def test_visit_ends_after_a_none_pick(tiny_scenario, customers_by_day):
    customers_by_day[0] = [make_customer(1, LEAVER, basket_size=3), make_customer(2, "croissant", basket_size=3)]
    customers_by_day[0].append(make_customer(3, LEAVES_AFTER_FIRST_PICK, basket_size=3))
    dataset = run_days(tiny_scenario, [0])
    rows_per_visit = dataset.choices.groupby("visit_id").size().to_dict()
    assert rows_per_visit == {0: 1, 1: 3, 2: 2}
    leaver_row = dataset.choices[dataset.choices["visit_id"] == 0].iloc[0]
    assert (leaver_row["chosen_product_id"], leaver_row["chosen_age"], leaver_row["price_paid"]) == ("none", -1, 0.0)


def test_options_string_format_and_empty_when_sold_out(tiny_scenario, customers_by_day):
    customers_by_day[0] = [make_customer(1, "croissant")]
    customers_by_day[1] = [make_customer(1, LEAVER)]
    dataset = run_days(tiny_scenario, [0, 1])
    first_row = dataset.choices.iloc[0]
    assert first_row["options"] == "croissant@0:5.00;bread@0:4.00"
    assert first_row["price_paid"] == 5.0
    day_one_row = dataset.choices[dataset.choices["day_index"] == 1].iloc[0]
    assert day_one_row["options"] == "croissant@0:5.00;bread@0:4.00;bread@1:2.20"
    assert day_one_row["visit_id"] == 100_000

    customers_by_day[2] = [make_customer(minute, "croissant") for minute in range(7)]
    customers_by_day[2] += [make_customer(10 + minute, "bread") for minute in range(4)] + [make_customer(30, LEAVER)]
    dataset = run_days(tiny_scenario.with_overrides(carryover_enabled=False), [2])
    last_pick = dataset.choices.iloc[-1]
    assert last_pick["options"] == ""
    assert last_pick["chosen_product_id"] == "none"


def test_prices_logged_on_the_ladder_for_stocked_options(tiny_scenario, customers_by_day):
    customers_by_day[1] = [make_customer(1, "croissant")]
    dataset = run_days(tiny_scenario, [0, 1], policy=HistoricalPricingPolicy(tiny_scenario))
    ladder = tiny_scenario.price_ladder
    assert all(any(math.isclose(level, rung) for rung in ladder) for level in dataset.prices["price_level"])
    base_prices = {product.product_id: product.base_price for product in tiny_scenario.products}
    for row in dataset.prices.itertuples():
        assert row.price == round(base_prices[row.product_id] * row.price_level, 2)
    # Day 1 is a Tuesday: croissant promo at hour 8, end-of-day level at the last hour (9).
    day_one = dataset.prices[(dataset.prices["day_index"] == 1) & (dataset.prices["product_id"] == "croissant")]
    reasons_by_hour = day_one.set_index("hour")["price_reason"].to_dict()
    assert reasons_by_hour == {7: "base", 8: "promo", 9: "end_of_day"}
    # Day 0 has no carryover, so no day-old prices are logged.
    assert set(dataset.prices[dataset.prices["day_index"] == 0]["age"]) == {0}


def test_days_table_and_column_order(tiny_scenario, customers_by_day):
    dataset = run_days(tiny_scenario, [0, 1])
    assert dataset.days["date"].tolist() == ["2026-09-07", "2026-09-08"]
    assert dataset.days["day_of_week"].tolist() == ["Mon", "Tue"]
    assert dataset.days["split"].tolist() == ["test", "test"]
    assert dataset.days["demand_multiplier"].tolist() == [1.0, 2.0]
    from bakery_sim.interfaces import TABLE_COLUMNS

    for table_name, columns in TABLE_COLUMNS.items():
        assert list(getattr(dataset, table_name).columns) == columns


def test_history_and_test_wrappers_use_the_right_days(tiny_scenario, customers_by_day):
    history = simulate_history(tiny_scenario)
    test = simulate_test(tiny_scenario, FixedPricePolicy())
    assert history.days["day_index"].tolist() == [0, 1]
    assert set(history.days["split"]) == {"history"}
    assert test.days["day_index"].tolist() == [2]
    assert set(test.days["split"]) == {"test"}


class OffLadderPolicy:
    name = "off_ladder"

    def prices(self, context):
        return {option: PriceDecision(price=1.0, level=0.123, reason="policy") for option in context.inventory}


class MissingOptionPolicy:
    name = "missing_option"

    def prices(self, context):
        return {}


def test_policy_with_off_ladder_level_raises(tiny_scenario, customers_by_day):
    with pytest.raises(ValueError):
        run_days(tiny_scenario, [0], policy=OffLadderPolicy())


def test_policy_missing_a_stocked_option_raises(tiny_scenario, customers_by_day):
    with pytest.raises(ValueError):
        run_days(tiny_scenario, [0], policy=MissingOptionPolicy())
