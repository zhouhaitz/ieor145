"""
The four output tables and their views (Subagent B).

Dataset holds days, prices, choices, inventory (columns in interfaces.TABLE_COLUMNS).
Views: "truth" (everything), "door_counter" (hidden columns dropped, walk-outs kept),
"register" (hidden columns dropped, only purchases). Derived tables (visits, receipts,
daily summary) are computed, never stored.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class Dataset:
    scenario_id: str
    days: pd.DataFrame
    prices: pd.DataFrame
    choices: pd.DataFrame
    inventory: pd.DataFrame

    def view(self, view_name: str) -> Dataset:
        """Return the 'truth', 'door_counter' or 'register' view."""
        raise NotImplementedError

    def visits(self) -> pd.DataFrame:
        """One row per visit: day_index, visit_id, arrival_minute, hour, items_bought, spend."""
        raise NotImplementedError

    def receipts(self) -> pd.DataFrame:
        """One row per visit x item bought: day_index, visit_id, product_id, age, quantity, unit_price."""
        raise NotImplementedError

    def daily_summary(self, unit_costs: dict[str, float], salvage_values: dict[str, float]) -> pd.DataFrame:
        """Per day x product: baked, sold, revenue, wasted, salvage_revenue, cost, profit."""
        raise NotImplementedError

    def save(self, directory: str | Path) -> None:
        """Write days.csv, prices.csv, choices.csv, inventory.csv to `directory`."""
        raise NotImplementedError

    def to_zip(self, path: str | Path) -> None:
        """Write the four CSVs into one zip file."""
        raise NotImplementedError

    @staticmethod
    def concat(datasets: list[Dataset]) -> Dataset:
        """Stack datasets (e.g. history + test) into one."""
        raise NotImplementedError
