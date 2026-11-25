"""
Command line: `python -m bakery_sim generate` and `python -m bakery_sim calibrate`.

generate writes the with/without-carryover dataset pair and prints each report card;
calibrate fits the unobservable inputs, optionally writes them back to a config file, and
saves the full candidate log.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from bakery_sim.calibrate import calibrate
from bakery_sim.config import Scenario, load_scenario
from bakery_sim.generate import generate_carryover_pair
from bakery_sim.report_card import format_report_card, report_card
from bakery_sim.simulator import simulate_history


def write_config(scenario: Scenario, path: Path) -> None:
    """Save a scenario as JSON, keeping the original file's _comment line if it had one."""
    comment = json.loads(path.read_text()).get("_comment") if path.exists() else None
    payload = {"_comment": comment, **asdict(scenario)} if comment else asdict(scenario)
    path.write_text(json.dumps(payload, indent=2))


def run_generate(config_path: Path, output_directory: Path) -> None:
    scenario = load_scenario(config_path)
    for suffix, (_, card) in generate_carryover_pair(scenario, output_directory).items():
        print(f"\n{scenario.scenario_id}_{suffix} -> {output_directory / (scenario.scenario_id + '_' + suffix)}.zip")
        print(format_report_card(card))


def run_calibrate(config_path: Path, write_path: Path | None, log_path: Path | None) -> None:
    scenario = load_scenario(config_path)
    calibrated, log = calibrate(scenario)
    chosen = log[log["chosen"]].iloc[0]
    print(
        f"Chosen scales: wtp_level={chosen['wtp_level_scale']}, wtp_spread={chosen['wtp_spread_scale']}, "
        f"basket={chosen['basket_scale']} ({len(log)} candidates)"
    )
    print(format_report_card(report_card(simulate_history(calibrated), calibrated)))
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log.to_csv(log_path, index=False)
    if write_path is not None:
        write_config(calibrated, write_path)


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m bakery_sim", description="Synthetic bakery data generator")
    subcommands = parser.add_subparsers(dest="command", required=True)

    generate_parser = subcommands.add_parser("generate", help="write the carryover on/off dataset pair")
    generate_parser.add_argument("--config", type=Path, required=True, help="scenario JSON")
    generate_parser.add_argument("--out", type=Path, required=True, help="output directory")

    calibrate_parser = subcommands.add_parser("calibrate", help="fit unobservable inputs to the report-card bands")
    calibrate_parser.add_argument("--config", type=Path, required=True, help="scenario JSON to calibrate")
    calibrate_parser.add_argument("--write", type=Path, default=None, help="write calibrated scenario JSON here")
    calibrate_parser.add_argument("--log", type=Path, default=None, help="write the candidate log CSV here")

    arguments = parser.parse_args()
    if arguments.command == "generate":
        run_generate(arguments.config, arguments.out)
    else:
        run_calibrate(arguments.config, arguments.write, arguments.log)


if __name__ == "__main__":
    main()
