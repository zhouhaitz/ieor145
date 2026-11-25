"""Pricing policies set ladder prices with the right reasons at the right hours and days."""

import math

import pytest

from bakery_sim.interfaces import PricingContext
from bakery_sim.pricing import FixedPricePolicy, HistoricalPricingPolicy, current_menu_price, ladder_price

MONDAY_INDEX = 0
TUESDAY_INDEX = 1


def stocked_context(scenario, day_index, day_of_week, hour, units=1):
    inventory = dict.fromkeys(scenario.option_keys(), units)
    return PricingContext(
        scenario=scenario, day_index=day_index, day_of_week=day_of_week, hour=hour, inventory=inventory
    )


def test_menu_price_rises_on_increase_day(baseline_scenario):
    historical = baseline_scenario.historical_pricing
    base_price = baseline_scenario.product("croissant").base_price
    before = current_menu_price(baseline_scenario, "croissant", historical.menu_increase_day_index - 1)
    after = current_menu_price(baseline_scenario, "croissant", historical.menu_increase_day_index)
    assert before == base_price
    assert after == pytest.approx(base_price * (1 + historical.menu_increase_fraction))


def test_ladder_price_rounds_to_cents(baseline_scenario):
    decision = ladder_price(baseline_scenario, "croissant", day_index=30, level=0.85)
    assert decision.price == 4.46  # 5.25 * 0.85 = 4.4625
    assert decision.level == 0.85
    assert decision.reason == "policy"


def test_ladder_price_rejects_off_ladder_level(baseline_scenario):
    with pytest.raises(ValueError):
        ladder_price(baseline_scenario, "croissant", day_index=0, level=0.9)


def test_fixed_policy_reasons_and_levels(baseline_scenario):
    policy = FixedPricePolicy(fresh_level=0.85, day_old_level=0.4)
    decisions = policy.prices(stocked_context(baseline_scenario, MONDAY_INDEX, "Mon", hour=10))
    for (_product_id, age), decision in decisions.items():
        if age == 0:
            assert (decision.level, decision.reason) == (0.85, "fixed_discount")
        else:
            assert (decision.level, decision.reason) == (0.4, "day_old")


def test_fixed_policy_full_price_is_base(baseline_scenario):
    decisions = FixedPricePolicy().prices(stocked_context(baseline_scenario, MONDAY_INDEX, "Mon", hour=10))
    fresh_reasons = {decision.reason for (_, age), decision in decisions.items() if age == 0}
    assert fresh_reasons == {"base"}


def test_policies_skip_unstocked_options(baseline_scenario):
    context = stocked_context(baseline_scenario, MONDAY_INDEX, "Mon", hour=10)
    context.inventory[("croissant", 0)] = 0
    for policy in (FixedPricePolicy(), HistoricalPricingPolicy(baseline_scenario)):
        decisions = policy.prices(context)
        assert ("croissant", 0) not in decisions
        assert len(decisions) == len(baseline_scenario.option_keys()) - 1


def test_historical_end_of_day_only_in_last_hours(baseline_scenario):
    policy = HistoricalPricingPolicy(baseline_scenario)
    last_hour = baseline_scenario.close_hour - 1
    end_of_day = policy.prices(stocked_context(baseline_scenario, MONDAY_INDEX, "Mon", hour=last_hour))
    earlier = policy.prices(stocked_context(baseline_scenario, MONDAY_INDEX, "Mon", hour=last_hour - 1))
    expected_level = baseline_scenario.historical_pricing.end_of_day_level
    assert {d.reason for (_, age), d in end_of_day.items() if age == 0} == {"end_of_day"}
    assert {d.level for (_, age), d in end_of_day.items() if age == 0} == {expected_level}
    assert {d.reason for (_, age), d in earlier.items() if age == 0} == {"base"}


def test_historical_day_old_level_all_day(baseline_scenario):
    policy = HistoricalPricingPolicy(baseline_scenario)
    expected_level = baseline_scenario.historical_pricing.day_old_level
    for hour in baseline_scenario.open_hours:
        decisions = policy.prices(stocked_context(baseline_scenario, MONDAY_INDEX, "Mon", hour=hour))
        for (_, age), decision in decisions.items():
            if age >= 1:
                assert (decision.reason, decision.level) == ("day_old", expected_level)


def test_historical_promo_only_on_promo_days_hours_and_product(baseline_scenario):
    historical = baseline_scenario.historical_pricing
    policy = HistoricalPricingPolicy(baseline_scenario)
    promo_hour = historical.promo_hours[0]
    promo_day = policy.prices(stocked_context(baseline_scenario, TUESDAY_INDEX, "Tue", hour=promo_hour))
    assert promo_day[(historical.promo_product_id, 0)].reason == "promo"
    assert promo_day[(historical.promo_product_id, 0)].level == historical.promo_level
    assert promo_day[("croissant", 0)].reason == "base"

    other_day = policy.prices(stocked_context(baseline_scenario, MONDAY_INDEX, "Mon", hour=promo_hour))
    assert other_day[(historical.promo_product_id, 0)].reason == "base"

    outside_hours = policy.prices(
        stocked_context(baseline_scenario, TUESDAY_INDEX, "Tue", hour=baseline_scenario.open_hour)
    )
    assert outside_hours[(historical.promo_product_id, 0)].reason == "base"


def test_end_of_day_beats_promo(baseline_scenario):
    last_hour = baseline_scenario.close_hour - 1
    historical = baseline_scenario.historical_pricing
    overlapping = baseline_scenario.with_overrides(
        historical_pricing=type(historical)(**{**historical.__dict__, "promo_hours": [last_hour]})
    )
    decisions = HistoricalPricingPolicy(overlapping).prices(
        stocked_context(overlapping, TUESDAY_INDEX, "Tue", hour=last_hour)
    )
    assert decisions[(historical.promo_product_id, 0)].reason == "end_of_day"


def test_historical_prices_follow_menu_increase(baseline_scenario):
    increase_day = baseline_scenario.historical_pricing.menu_increase_day_index
    policy = HistoricalPricingPolicy(baseline_scenario)
    before = policy.prices(stocked_context(baseline_scenario, increase_day - 1, "Mon", hour=8))
    after = policy.prices(stocked_context(baseline_scenario, increase_day, "Mon", hour=8))
    base_price = baseline_scenario.product("croissant").base_price
    assert before[("croissant", 0)].price == base_price
    assert math.isclose(after[("croissant", 0)].price, round(base_price * 1.05, 2))


def test_all_historical_levels_are_on_ladder(baseline_scenario):
    policy = HistoricalPricingPolicy(baseline_scenario)
    for hour in baseline_scenario.open_hours:
        decisions = policy.prices(stocked_context(baseline_scenario, TUESDAY_INDEX, "Tue", hour=hour))
        for decision in decisions.values():
            assert any(math.isclose(decision.level, level) for level in baseline_scenario.price_ladder)


def test_daily_deal_is_deterministic_and_avoids_end_of_day(baseline_scenario):
    from bakery_sim.pricing import daily_deal

    historical = baseline_scenario.historical_pricing
    first_end_of_day_hour = baseline_scenario.close_hour - historical.end_of_day_hours
    deals = [daily_deal(baseline_scenario, day_index) for day_index in range(200)]
    assert deals == [daily_deal(baseline_scenario, day_index) for day_index in range(200)]
    active_deals = [deal for deal in deals if deal is not None]
    assert abs(len(active_deals) / 200 - historical.daily_deal_probability) < 0.1
    for _product_id, hours, level in active_deals:
        assert max(hours) < first_end_of_day_hour and min(hours) >= baseline_scenario.open_hour
        assert level in historical.daily_deal_levels


def test_no_deals_when_probability_is_zero(baseline_scenario):
    from bakery_sim.pricing import daily_deal

    historical = type(baseline_scenario.historical_pricing)(
        **{**baseline_scenario.historical_pricing.__dict__, "daily_deal_probability": 0.0}
    )
    no_deal_scenario = baseline_scenario.with_overrides(historical_pricing=historical)
    assert all(daily_deal(no_deal_scenario, day_index) is None for day_index in range(50))
