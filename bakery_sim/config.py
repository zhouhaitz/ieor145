"""
Scenario configuration for the synthetic bakery.

A Scenario holds every input the generator uses: products, customer types,
hourly arrival patterns, bake schedule, pricing ladder, the historical pricing
practice, and the report-card targets. Nothing in the generator is hard-coded;
change a number here (or in configs/*.json) and rerun.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

DAYS_OF_WEEK = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
OUTSIDE_OPTION = "none"


@dataclass(frozen=True)
class Product:
    """One baked item. Ages 0..shelf_life_days-1 can be sold; age 1 is 'day-old'."""

    product_id: str
    name: str
    base_price: float  # menu price in dollars
    unit_cost: float  # cost to bake one unit, dollars
    shelf_life_days: int  # 1 = same day only, 2 = can be sold day-old


@dataclass(frozen=True)
class CustomerType:
    """A customer segment; each arrival belongs to exactly one type (hidden from the bakery)."""

    type_id: str
    name: str
    wtp_multiplier: dict[str, float]  # average willingness to pay = multiplier * base_price, per product
    wtp_spread: float  # sigma_k in dollars; larger = less price sensitive
    day_old_penalty: float  # delta_k in dollars subtracted from value of a day-old item
    basket_extra_mean: float  # m_k; planned basket size N = 1 + Poisson(m_k)
    hourly_shape: list[float]  # relative arrival intensity per open hour (normalized in arrivals.py)
    weekend_hourly_shape: list[float]  # same for Sat/Sun


@dataclass(frozen=True)
class BakeBatch:
    """A scheduled batch: `share` of the day's planned quantity of `product_id` arrives at `hour`."""

    product_id: str
    hour: int
    share: float


@dataclass(frozen=True)
class HistoricalPricing:
    """The bakery's past pricing practice, used only to generate the history log."""

    end_of_day_hours: int  # last N open hours get the end-of-day discount
    end_of_day_level: float  # ladder level used at end of day, e.g. 0.70
    day_old_level: float  # ladder level for day-old items all day
    promo_days_of_week: list[str]  # weekdays with a promo
    promo_product_id: str
    promo_level: float
    promo_hours: list[int]  # open hours the promo applies (not near closing, to avoid confounding)
    menu_increase_day_index: int  # day index from which base prices rise
    menu_increase_fraction: float  # e.g. 0.05 = +5%
    daily_deal_probability: float = 0.0  # chance a day has a "deal of the day" on one random fresh item
    daily_deal_levels: list[float] = field(default_factory=list)  # ladder levels a deal can use
    daily_deal_hours: int = 2  # consecutive hours a deal lasts (never in the end-of-day hours)


@dataclass(frozen=True)
class Scenario:
    """Every input to one synthetic-data run."""

    scenario_id: str
    seed: int
    start_date: str  # ISO date of day index 0
    n_history_days: int
    n_test_days: int
    open_hour: int  # first open hour, 24h clock
    close_hour: int  # closing time, 24h clock (exclusive)
    products: list[Product]
    customer_types: list[CustomerType]
    daily_customers: dict[str, float]  # expected arrivals per day of week (all types)
    type_share: dict[str, dict[str, float]]  # day of week -> type_id -> share of arrivals
    observed_hourly_shape: dict[str, list[float]]  # day of week -> real traffic share per open hour (Popular Times)
    demand_multiplier_cv: float  # coefficient of variation of the daily multiplier M_d (Gamma, mean 1)
    price_ladder: list[float]  # allowed price levels as fractions of base price
    bake_schedule: list[BakeBatch]
    bake_quantities: dict[str, dict[str, int]]  # day of week -> product_id -> units baked that day
    supply_ratio: float  # baked / expected full-price demand (used by calibrate.py to set bake_quantities)
    salvage_fraction: float  # leftover value as fraction of base price (0 = discard, ~0.33 = resale app)
    carryover_enabled: bool  # sell leftover multi-day items as day-old tomorrow
    historical_pricing: HistoricalPricing
    targets: dict[str, list[float]] = field(default_factory=dict)  # report-card metric -> [low, high]

    @property
    def open_hours(self) -> list[int]:
        return list(range(self.open_hour, self.close_hour))

    @property
    def n_days(self) -> int:
        return self.n_history_days + self.n_test_days

    def product(self, product_id: str) -> Product:
        return next(product for product in self.products if product.product_id == product_id)

    def customer_type(self, type_id: str) -> CustomerType:
        return next(customer_type for customer_type in self.customer_types if customer_type.type_id == type_id)

    def option_keys(self) -> list[tuple[str, int]]:
        """All sellable (product_id, age) pairs, in a fixed order used for noise columns."""
        return [(product.product_id, age) for product in self.products for age in range(product.shelf_life_days)]

    def with_overrides(self, **changes) -> Scenario:
        return replace(self, **changes)


def load_scenario(path: str | Path) -> Scenario:
    """Load and validate a Scenario from a JSON file."""
    raw = json.loads(Path(path).read_text())
    scenario = scenario_from_dict(raw)
    validate_scenario(scenario)
    return scenario


def validate_scenario(scenario: Scenario) -> None:
    """Raise ValueError listing every inconsistency in the inputs (caught at load time, not mid-simulation)."""
    problems = []
    n_hours = len(scenario.open_hours)
    product_ids = {product.product_id for product in scenario.products}
    type_ids = {customer_type.type_id for customer_type in scenario.customer_types}
    for customer_type in scenario.customer_types:
        if len(customer_type.hourly_shape) != n_hours or len(customer_type.weekend_hourly_shape) != n_hours:
            problems.append(f"{customer_type.type_id}: hourly shapes need {n_hours} entries (one per open hour)")
        if set(customer_type.wtp_multiplier) != product_ids:
            problems.append(f"{customer_type.type_id}: wtp_multiplier must cover exactly {sorted(product_ids)}")
        if customer_type.wtp_spread <= 0:
            problems.append(f"{customer_type.type_id}: wtp_spread must be positive")
    for day_of_week in DAYS_OF_WEEK:
        shares = scenario.type_share.get(day_of_week, {})
        if set(shares) != type_ids or abs(sum(shares.values()) - 1) > 1e-6:
            problems.append(f"type_share[{day_of_week}] must cover every type and sum to 1")
        if len(scenario.observed_hourly_shape.get(day_of_week, [])) != n_hours:
            problems.append(f"observed_hourly_shape[{day_of_week}] needs {n_hours} entries")
        if set(scenario.bake_quantities.get(day_of_week, {})) != product_ids:
            problems.append(f"bake_quantities[{day_of_week}] must cover every product")
    for product_id in product_ids:
        shares = [batch.share for batch in scenario.bake_schedule if batch.product_id == product_id]
        if abs(sum(shares) - 1) > 1e-6:
            problems.append(f"bake_schedule shares for {product_id} must sum to 1")
    if any(batch.hour not in scenario.open_hours for batch in scenario.bake_schedule):
        problems.append("every bake batch hour must be an open hour")
    pricing = scenario.historical_pricing
    fixed_levels = [1.0, pricing.end_of_day_level, pricing.day_old_level, pricing.promo_level]
    used_levels = fixed_levels + list(pricing.daily_deal_levels)
    for level in used_levels:
        if not any(abs(level - ladder_level) < 1e-9 for ladder_level in scenario.price_ladder):
            problems.append(f"price level {level} is not on price_ladder")
    if problems:
        raise ValueError("Invalid scenario:\n- " + "\n- ".join(problems))


def scenario_from_dict(raw: dict) -> Scenario:
    raw = dict(raw)
    raw["products"] = [Product(**product) for product in raw["products"]]
    raw["customer_types"] = [CustomerType(**customer_type) for customer_type in raw["customer_types"]]
    raw["bake_schedule"] = [BakeBatch(**batch) for batch in raw["bake_schedule"]]
    raw["historical_pricing"] = HistoricalPricing(**raw["historical_pricing"])
    raw.pop("_comment", None)
    return Scenario(**raw)


def save_scenario(scenario: Scenario, path: str | Path) -> None:
    Path(path).write_text(json.dumps(asdict(scenario), indent=2))
