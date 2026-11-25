"""
Turns raw Google Popular Times bars into the target hourly shape and relative daily volume per day of week.

For each day and hour, average each open bakery's share of that day's traffic (bakeries closed at that
hour are skipped, not counted as zero), then renormalize. Relative volume = a bakery's daily bar total
divided by its own Wed-Fri average, averaged across bakeries. Output: docs/data/popular_times_targets.json
"""

import json
from pathlib import Path

import numpy as np

OPEN_HOUR, CLOSE_HOUR = 7, 17  # simulated bakery hours (7am-5pm)
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
here = Path(__file__).parent
raw = json.loads((here / "popular_times_raw.json").read_text())
raw.pop("_comment")
n_hours = CLOSE_HOUR - OPEN_HOUR

shapes, volumes = {}, {}
for day in DAYS:
    hourly_shares = [[] for _ in range(n_hours)]
    day_volumes = []
    for bakery in raw.values():
        if day not in bakery:
            continue
        bars = np.array(bakery[day][:n_hours], dtype=float)
        day_share = bars / bars.sum()
        for hour_position, share in enumerate(day_share):
            if bars[hour_position] > 0:  # bakery open this hour
                hourly_shares[hour_position].append(share)
        reference = np.mean([np.sum(bakery[ref][:n_hours]) for ref in ["Wed", "Thu", "Fri"] if ref in bakery])
        day_volumes.append(bars.sum() / reference)
    mean_shares = np.array([np.mean(values) if values else 0.0 for values in hourly_shares])
    shapes[day] = (mean_shares / mean_shares.sum()).round(4).tolist()
    volumes[day] = round(float(np.mean(day_volumes)), 3)

output = {"open_hour": OPEN_HOUR, "close_hour": CLOSE_HOUR, "hourly_shape": shapes, "relative_daily_volume": volumes}
(here / "popular_times_targets.json").write_text(json.dumps(output, indent=2))
print(json.dumps(output))
