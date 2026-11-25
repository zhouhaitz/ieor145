"""
The four output tables and their views (Subagent B).

Dataset holds days, prices, choices, inventory (columns in interfaces.TABLE_COLUMNS).
Views: "truth" (everything), "door_counter" (hidden columns dropped, walk-outs kept),
"register" (hidden columns dropped, only purchases). Derived tables (visits, receipts,
daily summary) are computed, never stored.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from bakery_sim.interfaces import NO_PURCHASE, TRUTH_ONLY_COLUMNS

TABLE_NAMES = ["days", "prices", "choices", "inventory"]


@dataclass
class Dataset:
    scenario_id: str
    days: pd.DataFrame
    prices: pd.DataFrame
    choices: pd.DataFrame
    inventory: pd.DataFrame

    def _tables(self) -> dict[str, pd.DataFrame]:
        return {table_name: getattr(self, table_name) for table_name in TABLE_NAMES}

    def view(self, view_name: str) -> Dataset:
        """Return the 'truth', 'door_counter' or 'register' view."""
        if view_name not in ("truth", "door_counter", "register"):
            raise ValueError(f"Unknown view {view_name!r}; use 'truth', 'door_counter' or 'register'")
        tables = {table_name: table.copy() for table_name, table in self._tables().items()}
        if view_name != "truth":
            for table_name, hidden_columns in TRUTH_ONLY_COLUMNS.items():
                tables[table_name] = tables[table_name].drop(columns=hidden_columns)
        if view_name == "register":
            purchases_only = tables["choices"]["chosen_product_id"] != NO_PURCHASE
            tables["choices"] = tables["choices"][purchases_only].reset_index(drop=True)
        return Dataset(scenario_id=self.scenario_id, **tables)

    def _purchases(self) -> pd.DataFrame:
        return self.choices[self.choices["chosen_product_id"] != NO_PURCHASE]

    def visits(self) -> pd.DataFrame:
        """One row per visit: day_index, visit_id, arrival_minute, hour, items_bought, spend."""
        bought = (self.choices["chosen_product_id"] != NO_PURCHASE).astype(int)
        choices = self.choices.assign(items_bought=bought, spend=self.choices["price_paid"])
        return choices.groupby(["day_index", "visit_id"], as_index=False).agg(
            arrival_minute=("arrival_minute", "first"),
            hour=("hour", "first"),
            items_bought=("items_bought", "sum"),
            spend=("spend", "sum"),
        )

    def receipts(self) -> pd.DataFrame:
        """One row per visit x item bought: day_index, visit_id, product_id, age, quantity, unit_price."""
        purchases = self._purchases().rename(columns={"chosen_product_id": "product_id", "chosen_age": "age"})
        return purchases.groupby(["day_index", "visit_id", "product_id", "age"], as_index=False).agg(
            quantity=("price_paid", "size"),
            unit_price=("price_paid", "mean"),
        )

    def daily_summary(self, unit_costs: dict[str, float], salvage_values: dict[str, float]) -> pd.DataFrame:
        """Per day x product: baked, sold, revenue, wasted, salvage_revenue, cost, profit."""
        keys = ["day_index", "product_id"]
        stock = self.inventory.groupby(keys, as_index=False)[["baked_units", "sold_units", "wasted_units"]].sum()
        revenue = (
            self._purchases()
            .rename(columns={"chosen_product_id": "product_id"})
            .groupby(keys, as_index=False)
            .agg(revenue=("price_paid", "sum"))
        )
        summary = stock.merge(revenue, on=keys, how="left")
        summary["revenue"] = summary["revenue"].fillna(0.0)
        summary["salvage_revenue"] = summary["wasted_units"] * summary["product_id"].map(salvage_values)
        summary["cost"] = summary["baked_units"] * summary["product_id"].map(unit_costs)
        summary["profit"] = summary["revenue"] + summary["salvage_revenue"] - summary["cost"]
        column_order = [
            "day_index",
            "product_id",
            "baked_units",
            "sold_units",
            "revenue",
            "wasted_units",
            "salvage_revenue",
            "cost",
            "profit",
        ]
        return summary[column_order]

    def save(self, directory: str | Path) -> None:
        """Write days.csv, prices.csv, choices.csv, inventory.csv to `directory`."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for table_name, table in self._tables().items():
            table.to_csv(directory / f"{table_name}.csv", index=False)

    def to_zip(self, path: str | Path) -> None:
        """Write the four CSVs into one zip file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            for table_name, table in self._tables().items():
                archive.writestr(f"{table_name}.csv", table.to_csv(index=False))

    @staticmethod
    def concat(datasets: list[Dataset]) -> Dataset:
        """Stack datasets (e.g. history + test) into one."""
        stacked = {
            table_name: pd.concat([getattr(dataset, table_name) for dataset in datasets], ignore_index=True)
            for table_name in TABLE_NAMES
        }
        return Dataset(scenario_id=datasets[0].scenario_id, **stacked)
