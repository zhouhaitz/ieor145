"""generate(): files for every view, view contents, zip, carryover pair, and config writing."""

import json
import zipfile

import pandas as pd

from bakery_sim.__main__ import write_config
from bakery_sim.config import load_scenario
from bakery_sim.generate import VIEW_NAMES, generate, generate_carryover_pair

TABLE_FILES = ["days.csv", "prices.csv", "choices.csv", "inventory.csv"]


def test_generate_writes_every_view_and_extra_files(small_scenario, tmp_path):
    dataset, card = generate(small_scenario, tmp_path)
    scenario_directory = tmp_path / small_scenario.scenario_id
    for view_name in VIEW_NAMES:
        for file_name in TABLE_FILES:
            assert (scenario_directory / view_name / file_name).exists()
    for file_name in ["params.json", "expected_arrivals.csv", "report_card.csv"]:
        assert (scenario_directory / file_name).exists()
    assert load_scenario(scenario_directory / "params.json") == small_scenario
    assert len(pd.read_csv(scenario_directory / "report_card.csv")) == len(card)
    assert len(dataset.days) == small_scenario.n_history_days


def test_views_hide_what_a_real_bakery_would_not_see(small_scenario, tmp_path):
    generate(small_scenario, tmp_path)
    scenario_directory = tmp_path / small_scenario.scenario_id
    truth = pd.read_csv(scenario_directory / "truth" / "choices.csv")
    door_counter = pd.read_csv(scenario_directory / "door_counter" / "choices.csv")
    register = pd.read_csv(scenario_directory / "register" / "choices.csv")
    assert "type_id" in truth.columns and "type_id" not in door_counter.columns
    assert (door_counter["chosen_product_id"] == "none").any()
    assert not (register["chosen_product_id"] == "none").any()


def test_zip_holds_the_whole_scenario_folder(small_scenario, tmp_path):
    generate(small_scenario, tmp_path)
    archive_path = tmp_path / f"{small_scenario.scenario_id}.zip"
    assert archive_path.exists()
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
    assert f"{small_scenario.scenario_id}/register/choices.csv" in names
    assert f"{small_scenario.scenario_id}/params.json" in names


def test_carryover_pair_uses_same_seed_and_flips_carryover(small_scenario, tmp_path):
    results = generate_carryover_pair(small_scenario, tmp_path)
    assert set(results) == {"carryover_on", "carryover_off"}
    for suffix, expected_flag in [("carryover_on", True), ("carryover_off", False)]:
        folder_name = f"{small_scenario.scenario_id}_{suffix}"
        saved = load_scenario(tmp_path / folder_name / "params.json")
        assert saved.carryover_enabled is expected_flag and saved.seed == small_scenario.seed
        assert (tmp_path / f"{folder_name}.zip").exists()


def test_write_config_keeps_the_comment_and_round_trips(baseline_scenario, tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"_comment": "keep me"}))
    write_config(baseline_scenario, path)
    assert json.loads(path.read_text())["_comment"] == "keep me"
    assert load_scenario(path) == baseline_scenario
