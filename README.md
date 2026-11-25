# IEOR 145 — Dynamic pricing for a bakery

Synthetic data generator and experiments for our IEOR 145 project: when does per-item,
hour-by-hour pricing of baked goods beat fixed prices and fixed markdowns?

**Plan and experiment questions:** [project doc](https://claude.ai/artifact/LcG7B8nQT2kfi8N5SjYxUu)

## Quickstart
- **Teammates:** open `notebooks/generate_synthetic_data.ipynb` in Colab, run top to bottom, download the zip.
- **Command line:** `pip install -r requirements.txt`, then see the notebook's "Generate" cell for the Python API.

## Layout
| Path | What it is |
| --- | --- |
| `bakery_sim/` | Generator code (config, arrivals, choice, pricing, simulator, dataset, calibration, estimation, report card) |
| `configs/baseline.json` | Every input number for the baseline bakery |
| `docs/sources.md` | Where each number comes from (sourced, assumption, or calibrated) |
| `notebooks/` | Teammate-facing notebook |
| `tests/` | `make check` (fast, run after every change) and `make check-all` (adds slow statistical checks) |

## Output tables
`days.csv`, `prices.csv`, `choices.csv` (one row per pick, with what was on the shelf), `inventory.csv`.
Views: `truth` (everything), `door_counter` (customer type hidden, walk-outs kept), `register` (purchases only).
