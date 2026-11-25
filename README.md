# IEOR 145 — Dynamic pricing for a bakery

Synthetic data generator and experiments for our IEOR 145 project: when does per-item,
hour-by-hour pricing of baked goods beat fixed prices and fixed markdowns?

**Plan and experiment questions:** [project doc](https://claude.ai/artifact/LcG7B8nQT2kfi8N5SjYxUu)

## Quickstart
- **Teammates:** open `notebooks/generate_synthetic_data.ipynb` in Colab, run top to bottom, download the zip.
- **Command line:** `pip install -r requirements.txt`, then `python -m bakery_sim generate --config configs/baseline.json --out data/`.
- **Before pushing a change:** `make check` (fast, run after every edit) and `make check-all` (adds statistical checks and runs the notebook).

## Layout
| Path | What it is |
| --- | --- |
| `bakery_sim/` | Generator code (config, arrivals, choice, pricing, simulator, dataset, calibration, estimation, report card) |
| `configs/baseline_inputs.json` | Starting assumptions (edit these) |
| `configs/baseline.json` | Calibrated baseline, produced by `python -m bakery_sim calibrate --config configs/baseline_inputs.json --write configs/baseline.json --log docs/calibration_log.csv` |
| `docs/sources.md` | Where each number comes from (sourced, assumption, or calibrated) |
| `notebooks/` | Teammate-facing notebook |
| `tests/` | `make check` (fast, run after every change) and `make check-all` (adds slow statistical checks) |

## Output tables
`days.csv`, `prices.csv`, `choices.csv` (one row per pick, with what was on the shelf), `inventory.csv`.
Views: `truth` (everything), `door_counter` (customer type hidden, walk-outs kept), `register` (purchases only).
