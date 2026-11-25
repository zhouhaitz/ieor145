# Sources for every input

Status: **SOURCED** (link), **ASSUMPTION** (reasoned; varied in experiments), **CALIBRATED** (tuned so outputs match sourced targets), **TO SOURCE**.
Rule: only CALIBRATED inputs are ever tuned, and only by changing inputs and rerunning the generator. Outputs are never edited.

## Inputs
| Input | Value | Status | Source / reasoning |
| --- | --- | --- | --- |
| Croissant price | $5.00 | SOURCED | SF bakery croissant average $5.80, range $4.00–$7.70 ([Square](https://squareup.com/us/en/menu-pricing/bakery-and-coffee-shop-prices-in-san-francisco)); East Bay neighborhood shops slightly lower |
| Cookie price | $3.50 | SOURCED | Starter Bakery salted chocolate chip $3.50 ([Toast](https://www.toasttab.com/local/order/starterbakery-rockridge/item-espresso-cappuccino_544b85bc-5948-4c55-a4be-2e235cd72b2a)); Berkeley cafés $2.50–$3.00 ([Toast](https://www.toasttab.com/local/berkeley-ca-restaurants/dish/cookies)) |
| Sourdough price | $7.00 | SOURCED | Arizmendi classic sourdough round $6.25 ([menu](https://store.arizmendibakery.com/menu-today)) |
| Baguette price | $4.50 | SOURCED | Starter Bakery baguette $4.00 ([Toast](https://www.toasttab.com/local/order/starterbakery-rockridge/item-croissant-ham-and-cheese-twice-baked_2d76d16b-fa7f-4e67-a3bd-2af45efa82f6)); Bouchon $5.50 |
| Unit cost | 30% of price | SOURCED | Toast: 30% food cost is the common bakery target ([Toast](https://pos.toasttab.com/blog/on-the-line/bakery-menu-pricing)) |
| Sourdough shelf life | 2 days (sold day-old on day 2) | SOURCED | Acme: consume within 3 days ([Acme](https://acmebread.com/contact)) |
| Pastries, baguette shelf life | 1 day | ASSUMPTION | Stale within a day |
| Hours | 7am–5pm | SOURCED | Acme Berkeley Mon–Wed & Sat 7am–4pm, Thu–Fri 7am–5pm; Arizmendi Emeryville to ~7pm; traffic falls off after 4pm in all three (Popular Times) |
| Hourly traffic shape by day | `observed_hourly_shape` | SOURCED | Google Popular Times for Acme (San Pablo), Fournée, Arizmendi Emeryville, read 2026-11-24 → `docs/data/popular_times_raw.json`, derived by `docs/data/derive_popular_times_targets.py` |
| Relative daily volume | Fri +7%, Sat +19%, Sun +1% vs. mid-week | SOURCED | Same Popular Times data (bars share one scale per bakery) |
| Customers per mid-week day | 300 | ASSUMPTION | Average successful neighborhood bakery; varied |
| Day-to-day variability (cv) | 0.15 | ASSUMPTION | Varied in experiment Q2 |
| Salvage value | 1/3 of price | SOURCED | Too Good To Go bags ≈ 1/3 of retail ([Axios](https://www.axios.com/2025/11/12/whole-foods-too-good-to-go-surprise-bags)); Acme Berkeley lists $18 value for $5.99 ([TGTG](https://www.toogoodtogo.com/en-us/find/berkeley/acmebread-berkeley/bakedgoods/1489908)) |
| Day-old value share (share of value an item keeps on day 2) | commuter 0.50, regular 0.65, bargain 0.80 | ASSUMPTION | Applies to any item sold day-old (only sourdough today); to be replaced by survey answers; varied in experiment Q5 |
| Customer-type hourly shapes | see config | ASSUMPTION | Commuter early, regular midday, bargain late; type shares are then CALIBRATED so the total matches Popular Times |
| Willingness to pay, spread, basket sizes | see config | CALIBRATED | Start from `configs/baseline_inputs.json`; scales tuned so report-card metrics land in the bands below; ties go to scales nearest 1 (log: `docs/calibration_log.csv`) |
| Past pricing practice | menu price; 30% off last hour; Tue/Wed cookie promo; deal of the day on 40% of days; +5% menu increase on day 30 | ASSUMPTION | Common bakery practices; the varied prices are what make price sensitivity estimable (Lecture 16 on confounders) |

## Report-card targets (what calibration aims for)
| Metric | Band | Status | Source / reasoning |
| --- | --- | --- | --- |
| Hourly traffic shape error (model's expected share vs. Popular Times, max abs.) | ≤ 0.02 | SOURCED | Popular Times shape above |
| Price elasticity (all fresh prices −15%, arc elasticity of units) | 0.8–2.0 | SOURCED lower bound | Category elasticity for food away from home ≈ 0.81 ([Andreyeva et al. 2010](https://pmc.ncbi.nlm.nih.gov/articles/PMC2804646)); a single bakery is at least that elastic because customers can switch stores. Upper bound is an assumption |
| Items per purchase | 1.2–2.2 | SOURCED | Café/bakery visits: 1–2 items ([DojoBusiness](https://dojobusiness.com/blogs/news/coffee-shop-average-ticket)); bakeries skew higher (loaf + pastry) |
| Average ticket (baked goods only, no drinks) | $5–$14 | SOURCED | Coffee-shop ticket $7.81–$12 incl. drinks ([DojoBusiness](https://dojobusiness.com/blogs/news/coffee-shop-average-ticket)) |
| Purchase rate (walk-ins who buy) | 60–90% | ASSUMPTION | No public source found; varied |
| Leftover share of baked units | 5–20% | ASSUMPTION | "Typical bakery with leftovers"; TGTG participation implies regular surplus |
| Busiest-hour customers | ≤ 60 | ASSUMPTION | About one customer per minute at a single register |
| Largest item revenue share | ≤ 50% | ASSUMPTION | Balanced four-item menu |

## Context (not an input)
Net profit margins for food businesses run about 3–5% (range 0–15%) ([Toast](https://pos.toasttab.com/blog/on-the-line/7-critical-restaurant-benchmarks)). Our profit metric counts ingredient cost only (no labor or rent), so it is a contribution margin, much higher than net margin.
