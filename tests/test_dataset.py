"""Dataset views, derived tables and file round trips on a tiny hand-built dataset."""

import zipfile

import pandas as pd
import pytest

from bakery_sim.dataset import Dataset
from bakery_sim.interfaces import TABLE_COLUMNS, TRUTH_ONLY_COLUMNS


@pytest.fixture
def tiny_dataset() -> Dataset:
    days = pd.DataFrame([(0, "2026-09-07", "Mon", "history", 1.1)], columns=TABLE_COLUMNS["days"])
    prices = pd.DataFrame(
        [(0, 7, "croissant", 0, 5.0, 1.0, "base"), (0, 7, "bread", 1, 2.2, 0.55, "day_old")],
        columns=TABLE_COLUMNS["prices"],
    )
    choice_rows = [
        # visit 1: buys two croissants then leaves
        (0, 1, 3.0, 7, "commuter", 0, "croissant@0:5.00;bread@1:2.20", "croissant", 0, 5.0),
        (0, 1, 3.0, 7, "commuter", 1, "croissant@0:5.00;bread@1:2.20", "croissant", 0, 5.0),
        (0, 1, 3.0, 7, "commuter", 2, "croissant@0:5.00;bread@1:2.20", "none", -1, 0.0),
        # visit 2: buys the day-old bread
        (0, 2, 10.0, 7, "bargain", 0, "croissant@0:5.00;bread@1:2.20", "bread", 1, 2.2),
        # visit 3: walks out
        (0, 3, 70.0, 8, "regular", 0, "croissant@0:5.00", "none", -1, 0.0),
    ]
    choices = pd.DataFrame(choice_rows, columns=TABLE_COLUMNS["choices"])
    inventory_rows = [
        (0, "croissant", 0, 0, 10, 2, 0, 8, float("nan")),
        (0, "bread", 1, 3, 0, 1, 0, 2, float("nan")),
    ]
    inventory = pd.DataFrame(inventory_rows, columns=TABLE_COLUMNS["inventory"])
    return Dataset(scenario_id="tiny", days=days, prices=prices, choices=choices, inventory=inventory)


def test_truth_view_is_a_copy(tiny_dataset):
    truth = tiny_dataset.view("truth")
    truth.choices.loc[0, "price_paid"] = 99.0
    assert tiny_dataset.choices.loc[0, "price_paid"] == 5.0
    assert list(truth.choices.columns) == TABLE_COLUMNS["choices"]


@pytest.mark.parametrize("view_name", ["door_counter", "register"])
def test_hidden_views_drop_truth_only_columns(tiny_dataset, view_name):
    view = tiny_dataset.view(view_name)
    for table_name, hidden_columns in TRUTH_ONLY_COLUMNS.items():
        for column in hidden_columns:
            assert column not in getattr(view, table_name).columns
    assert "type_id" in tiny_dataset.choices.columns


def test_door_counter_keeps_walkouts_register_does_not(tiny_dataset):
    assert (tiny_dataset.view("door_counter").choices["chosen_product_id"] == "none").sum() == 2
    assert (tiny_dataset.view("register").choices["chosen_product_id"] == "none").sum() == 0
    assert len(tiny_dataset.view("register").choices) == 3


def test_unknown_view_raises(tiny_dataset):
    with pytest.raises(ValueError):
        tiny_dataset.view("secret")


def test_visits(tiny_dataset):
    visits = tiny_dataset.visits().set_index("visit_id")
    assert list(visits.index) == [1, 2, 3]
    assert list(visits["items_bought"]) == [2, 1, 0]
    assert list(visits["spend"]) == pytest.approx([10.0, 2.2, 0.0])
    assert list(visits["hour"]) == [7, 7, 8]
    assert list(visits["arrival_minute"]) == [3.0, 10.0, 70.0]


def test_receipts(tiny_dataset):
    receipts = tiny_dataset.receipts()
    assert len(receipts) == 2
    croissant_row = receipts[receipts["product_id"] == "croissant"].iloc[0]
    assert (croissant_row["visit_id"], croissant_row["quantity"], croissant_row["unit_price"]) == (1, 2, 5.0)
    bread_row = receipts[receipts["product_id"] == "bread"].iloc[0]
    assert (bread_row["age"], bread_row["quantity"], bread_row["unit_price"]) == (1, 1, 2.2)


def test_daily_summary_arithmetic(tiny_dataset):
    summary = tiny_dataset.daily_summary(
        unit_costs={"croissant": 1.5, "bread": 1.0}, salvage_values={"croissant": 1.0, "bread": 0.5}
    ).set_index("product_id")
    croissant = summary.loc["croissant"]
    assert (croissant["baked_units"], croissant["sold_units"], croissant["wasted_units"]) == (10, 2, 8)
    assert croissant["revenue"] == pytest.approx(10.0)
    assert croissant["salvage_revenue"] == pytest.approx(8.0)
    assert croissant["cost"] == pytest.approx(15.0)
    assert croissant["profit"] == pytest.approx(10.0 + 8.0 - 15.0)
    bread = summary.loc["bread"]
    assert bread["revenue"] == pytest.approx(2.2)
    assert bread["profit"] == pytest.approx(2.2 + 1.0 - 0.0)


def test_save_round_trip(tiny_dataset, tmp_path):
    output_directory = tmp_path / "nested" / "out"
    tiny_dataset.save(output_directory)
    for table_name in ["days", "prices", "choices", "inventory"]:
        reloaded = pd.read_csv(output_directory / f"{table_name}.csv")
        assert list(reloaded.columns) == TABLE_COLUMNS[table_name]
        assert len(reloaded) == len(getattr(tiny_dataset, table_name))
    reloaded_choices = pd.read_csv(output_directory / "choices.csv")
    assert reloaded_choices["options"].iloc[0] == "croissant@0:5.00;bread@1:2.20"


def test_to_zip_round_trip(tiny_dataset, tmp_path):
    zip_path = tmp_path / "bundle" / "bakery.zip"
    tiny_dataset.to_zip(zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        assert sorted(archive.namelist()) == ["choices.csv", "days.csv", "inventory.csv", "prices.csv"]
        with archive.open("inventory.csv") as inventory_file:
            reloaded = pd.read_csv(inventory_file)
    assert list(reloaded.columns) == TABLE_COLUMNS["inventory"]
    assert len(reloaded) == 2


def test_concat_stacks_tables(tiny_dataset):
    combined = Dataset.concat([tiny_dataset, tiny_dataset])
    assert combined.scenario_id == "tiny"
    assert len(combined.choices) == 2 * len(tiny_dataset.choices)
    assert list(combined.choices.index) == list(range(len(combined.choices)))
