"""Tests for choice.py: MNL probabilities, substitution, logistic reduction, and noise-based choice."""

import numpy as np
import pytest

from bakery_sim.choice import choice_probabilities, choose_option, mean_utilities
from bakery_sim.config import Scenario
from bakery_sim.interfaces import Customer, OptionKey

FRESH_CROISSANT: OptionKey = ("croissant", 0)
FRESH_SOURDOUGH: OptionKey = ("sourdough", 0)
DAY_OLD_SOURDOUGH: OptionKey = ("sourdough", 1)


def base_prices(scenario: Scenario) -> dict[OptionKey, float]:
    return {
        (product.product_id, age): product.base_price
        for product in scenario.products
        for age in range(product.shelf_life_days)
    }


def test_probabilities_sum_to_one(baseline_scenario: Scenario) -> None:
    options = baseline_scenario.option_keys()
    for customer_type in baseline_scenario.customer_types:
        probabilities = choice_probabilities(
            baseline_scenario, customer_type, options, base_prices(baseline_scenario)
        )
        assert sum(probabilities.values()) == pytest.approx(1.0)
        assert set(probabilities) == {None, *options}


def test_raising_one_price_shifts_demand_to_others(baseline_scenario: Scenario) -> None:
    customer_type = baseline_scenario.customer_type("regular")
    options = baseline_scenario.option_keys()
    prices = base_prices(baseline_scenario)
    before = choice_probabilities(baseline_scenario, customer_type, options, prices)
    raised_prices = {**prices, FRESH_CROISSANT: prices[FRESH_CROISSANT] + 1.0}
    after = choice_probabilities(baseline_scenario, customer_type, options, raised_prices)
    assert after[FRESH_CROISSANT] < before[FRESH_CROISSANT]
    for other_option in [None, *(option for option in options if option != FRESH_CROISSANT)]:
        assert after[other_option] > before[other_option]


def test_day_old_is_less_likely_than_fresh_at_same_price(baseline_scenario: Scenario) -> None:
    customer_type = baseline_scenario.customer_type("regular")
    same_price = baseline_scenario.product("sourdough").base_price
    prices = {FRESH_SOURDOUGH: same_price, DAY_OLD_SOURDOUGH: same_price}
    probabilities = choice_probabilities(
        baseline_scenario, customer_type, [FRESH_SOURDOUGH, DAY_OLD_SOURDOUGH], prices
    )
    assert probabilities[DAY_OLD_SOURDOUGH] < probabilities[FRESH_SOURDOUGH]


def test_single_option_reduces_to_logistic(baseline_scenario: Scenario) -> None:
    customer_type = baseline_scenario.customer_type("commuter")
    price = 4.25
    probabilities = choice_probabilities(baseline_scenario, customer_type, [FRESH_CROISSANT], {FRESH_CROISSANT: price})
    willingness_to_pay = customer_type.wtp_multiplier["croissant"] * baseline_scenario.product("croissant").base_price
    expected_buy_probability = 1 / (1 + np.exp(-(willingness_to_pay - price) / customer_type.wtp_spread))
    assert probabilities[FRESH_CROISSANT] == pytest.approx(expected_buy_probability)
    assert probabilities[None] == pytest.approx(1 - expected_buy_probability)


def test_mean_utilities_follow_given_order(baseline_scenario: Scenario) -> None:
    customer_type = baseline_scenario.customer_type("regular")
    prices = base_prices(baseline_scenario)
    forward = mean_utilities(baseline_scenario, customer_type, [FRESH_CROISSANT, DAY_OLD_SOURDOUGH], prices)
    backward = mean_utilities(baseline_scenario, customer_type, [DAY_OLD_SOURDOUGH, FRESH_CROISSANT], prices)
    np.testing.assert_allclose(forward, backward[::-1])


def make_customer(scenario: Scenario, type_id: str, pick_noise: np.ndarray) -> Customer:
    return Customer(
        arrival_minute=0.0,
        hour=scenario.open_hour,
        type_id=type_id,
        planned_basket_size=pick_noise.shape[0],
        pick_noise=pick_noise,
    )


def test_choose_option_with_no_options_returns_none(baseline_scenario: Scenario) -> None:
    noise_columns = 1 + len(baseline_scenario.option_keys())
    customer = make_customer(baseline_scenario, "regular", np.zeros((1, noise_columns)))
    assert choose_option(baseline_scenario, customer, 0, available_options=[], prices={}) is None


def test_choose_option_with_forced_noise(baseline_scenario: Scenario) -> None:
    option_keys = baseline_scenario.option_keys()
    prices = base_prices(baseline_scenario)
    available_options = [FRESH_CROISSANT, FRESH_SOURDOUGH]
    huge_noise = 1000.0

    sourdough_forced = np.zeros((1, 1 + len(option_keys)))
    sourdough_forced[0, 1 + option_keys.index(FRESH_SOURDOUGH)] = huge_noise
    customer = make_customer(baseline_scenario, "regular", sourdough_forced)
    assert choose_option(baseline_scenario, customer, 0, available_options, prices) == FRESH_SOURDOUGH

    leave_forced = np.zeros((1, 1 + len(option_keys)))
    leave_forced[0, 0] = huge_noise
    customer = make_customer(baseline_scenario, "regular", leave_forced)
    assert choose_option(baseline_scenario, customer, 0, available_options, prices) is None

    unavailable_forced = np.zeros((1, 1 + len(option_keys)))
    unavailable_forced[0, 1 + option_keys.index(DAY_OLD_SOURDOUGH)] = huge_noise
    customer = make_customer(baseline_scenario, "regular", unavailable_forced)
    assert choose_option(baseline_scenario, customer, 0, available_options, prices) != DAY_OLD_SOURDOUGH


def test_choose_option_uses_requested_pick_number(baseline_scenario: Scenario) -> None:
    option_keys = baseline_scenario.option_keys()
    noise = np.zeros((2, 1 + len(option_keys)))
    noise[0, 0] = 1000.0
    noise[1, 1 + option_keys.index(FRESH_CROISSANT)] = 1000.0
    customer = make_customer(baseline_scenario, "regular", noise)
    prices = base_prices(baseline_scenario)
    assert choose_option(baseline_scenario, customer, 0, [FRESH_CROISSANT], prices) is None
    assert choose_option(baseline_scenario, customer, 1, [FRESH_CROISSANT], prices) == FRESH_CROISSANT


@pytest.mark.slow
def test_empirical_choice_frequencies_match_probabilities(baseline_scenario: Scenario) -> None:
    rng = np.random.default_rng(7)
    option_keys = baseline_scenario.option_keys()
    prices = base_prices(baseline_scenario)
    available_options = [FRESH_CROISSANT, FRESH_SOURDOUGH, DAY_OLD_SOURDOUGH]
    customer_type = baseline_scenario.customer_type("regular")
    simulation_count = 20000
    counts: dict[OptionKey | None, int] = {None: 0, **{option: 0 for option in available_options}}
    for _ in range(simulation_count):
        noise = rng.gumbel(size=(1, 1 + len(option_keys)))
        customer = make_customer(baseline_scenario, customer_type.type_id, noise)
        counts[choose_option(baseline_scenario, customer, 0, available_options, prices)] += 1
    probabilities = choice_probabilities(baseline_scenario, customer_type, available_options, prices)
    for option, probability in probabilities.items():
        assert counts[option] / simulation_count == pytest.approx(probability, abs=0.015)
