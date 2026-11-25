"""
Write one synthetic dataset to disk.

generate() simulates the history log and writes, per scenario, the three views (truth,
door_counter, register) as four CSVs each, the scenario inputs (params.json), the expected
arrivals table, and the report card, then zips the whole folder. generate_carryover_pair()
does this twice, with and without carryover of leftovers, on the same seed.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from bakery_sim.arrivals import expected_arrivals_table
from bakery_sim.config import Scenario, save_scenario
from bakery_sim.dataset import Dataset
from bakery_sim.report_card import report_card
from bakery_sim.simulator import simulate_history

VIEW_NAMES = ["truth", "door_counter", "register"]


def generate(scenario: Scenario, output_directory: str | Path) -> tuple[Dataset, pd.DataFrame]:
    """Simulate the history, write all files under output_directory/<scenario_id>(.zip), return (dataset, card)."""
    output_directory = Path(output_directory)
    scenario_directory = output_directory / scenario.scenario_id
    if scenario_directory.exists():
        shutil.rmtree(scenario_directory)
    scenario_directory.mkdir(parents=True)

    dataset = simulate_history(scenario)
    for view_name in VIEW_NAMES:
        dataset.view(view_name).save(scenario_directory / view_name)
    save_scenario(scenario, scenario_directory / "params.json")
    expected_arrivals_table(scenario).to_csv(scenario_directory / "expected_arrivals.csv", index=False)
    card = report_card(dataset, scenario)
    card.to_csv(scenario_directory / "report_card.csv", index=False)

    archive_base = output_directory / scenario.scenario_id
    shutil.make_archive(str(archive_base), "zip", root_dir=output_directory, base_dir=scenario.scenario_id)
    return dataset, card


def generate_carryover_pair(
    scenario: Scenario, output_directory: str | Path
) -> dict[str, tuple[Dataset, pd.DataFrame]]:
    """Run generate() with carryover on and off (same seed); keys 'carryover_on' and 'carryover_off'."""
    results = {}
    for suffix, carryover_enabled in [("carryover_on", True), ("carryover_off", False)]:
        variant = scenario.with_overrides(
            scenario_id=f"{scenario.scenario_id}_{suffix}", carryover_enabled=carryover_enabled
        )
        results[suffix] = generate(variant, output_directory)
    return results
