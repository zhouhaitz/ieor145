"""
Calibration (Subagent C): set unobservable inputs so outputs match sourced targets.

Only inputs listed in CALIBRATED_INPUTS may be tuned; observable inputs (prices, hours,
arrival shapes) stay fixed from sources, and outputs are never edited. Every candidate
run is logged with its seed so the tuning history can be audited.
"""

from __future__ import annotations

import pandas as pd

from bakery_sim.config import Scenario

CALIBRATED_INPUTS = ["wtp_level_scale", "wtp_spread_scale"]


def estimate_full_price_demand(scenario: Scenario, n_days: int = 28) -> pd.DataFrame:
    """Mean units demanded per day_of_week x product at menu price with unlimited stock."""
    raise NotImplementedError


def set_bake_quantities(scenario: Scenario, supply_ratio: float | None = None) -> Scenario:
    """Scenario with bake_quantities = round(supply_ratio * full-price demand) per day of week."""
    raise NotImplementedError


def apply_calibration(scenario: Scenario, wtp_level_scale: float, wtp_spread_scale: float) -> Scenario:
    """Scale every type's wtp_multiplier and wtp_spread by the given factors."""
    raise NotImplementedError


def calibrate(scenario: Scenario, n_days: int = 28) -> tuple[Scenario, pd.DataFrame]:
    """Search the calibrated inputs to land metrics inside scenario.targets; returns (scenario, log)."""
    raise NotImplementedError
