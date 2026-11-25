"""
MNL estimation by maximum likelihood (Subagent C; Lectures 14-15).

Utility (w - p - delta*age)/sigma is linear in beta_j = w_j/sigma, alpha = 1/sigma,
gamma = delta/sigma, so the log-likelihood is concave. Fitted values are mapped back to
dollars (w, sigma, delta) to compare with the scenario's true inputs (the recovery check).
"""

from __future__ import annotations

import pandas as pd

from bakery_sim.config import Scenario


def parse_options(options: str) -> dict[tuple[str, int], float]:
    """'croissant@0:5.00;sourdough@1:3.85' -> {('croissant', 0): 5.0, ('sourdough', 1): 3.85}."""
    raise NotImplementedError


def fit_mnl(choices: pd.DataFrame, scenario: Scenario, type_id: str | None = None) -> pd.DataFrame:
    """Fit one MNL (for one type, or pooled if type_id is None); columns: parameter, product_id, estimate."""
    raise NotImplementedError


def recovery_table(choices: pd.DataFrame, scenario: Scenario) -> pd.DataFrame:
    """True vs. estimated w, sigma, delta per type; columns: type_id, parameter, product_id, true, estimate."""
    raise NotImplementedError
