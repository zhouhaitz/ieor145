# Sources for every input

Status: **SOURCED** (link), **ASSUMPTION** (reasoned, varied in experiments), **CALIBRATED** (tuned so outputs match sourced targets), **TO SOURCE**.

| Input | Value | Status | Source / reasoning |
| --- | --- | --- | --- |
| Croissant price | $5.00 | SOURCED | SF bakery croissant average $5.80, range $4.00–$7.70 ([Square](https://squareup.com/us/en/menu-pricing/bakery-and-coffee-shop-prices-in-san-francisco)); East Bay neighborhood shops slightly lower |
| Cookie price | $3.50 | SOURCED | Starter Bakery salted chocolate chip $3.50 ([Toast](https://www.toasttab.com/local/order/starterbakery-rockridge/item-espresso-cappuccino_544b85bc-5948-4c55-a4be-2e235cd72b2a)); Berkeley cafés $2.50–$3.00 ([Toast](https://www.toasttab.com/local/berkeley-ca-restaurants/dish/cookies)) |
| Sourdough price | $7.00 | SOURCED | Arizmendi classic sourdough round $6.25 ([menu](https://store.arizmendibakery.com/menu-today)) |
| Baguette price | $4.50 | SOURCED | Starter Bakery baguette $4.00 ([Toast](https://www.toasttab.com/local/order/starterbakery-rockridge/item-croissant-ham-and-cheese-twice-baked_2d76d16b-fa7f-4e67-a3bd-2af45efa82f6)); Bouchon $5.50 |
| Sourdough sold day-old (2-day shelf life) | 2 days | SOURCED | Acme: consume within 3 days ([Acme](https://acmebread.com/contact)) |
| Baguette, pastries 1-day | 1 day | ASSUMPTION | Stale within a day |
| Hours | 7am–6pm | SOURCED | Acme retail Mon–Fri 7am–6pm ([Acme](https://acmebread.com/retail-locations.html)) |
| Salvage value | 1/3 of price | SOURCED | Too Good To Go bags ≈ 1/3 of retail ([Axios](https://www.axios.com/2025/11/12/whole-foods-too-good-to-go-surprise-bags)); Acme Berkeley $18 value for $5.99 ([TGTG](https://www.toogoodtogo.com/en-us/find/berkeley/acmebread-berkeley/bakedgoods/1489908)) |
| Unit cost | 30% of price | TO SOURCE | Bakery food-cost share from POS/industry reports |
| Hourly arrival shapes | see config | TO SOURCE | Google Maps Popular Times for comparable Berkeley bakeries |
| Customers per day | 260–420 | ASSUMPTION | Average successful neighborhood bakery; varied |
| Willingness to pay, spread, day-old penalty | see config | CALIBRATED | Tuned so report-card metrics fall in target bands |
| Basket sizes | see config | TO SOURCE | Items per transaction from POS reports |
| Report-card target bands | see config | TO SOURCE | POS/industry benchmarks |
