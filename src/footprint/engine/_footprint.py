"""Core footprint array construction.

This is the hot path. All accumulation must be vectorized via np.add.at.
Python for-loops here would destroy throughput for high-frequency tick streams.
"""

from __future__ import annotations

import numpy as np

from footprint._config import FootprintConfig
from footprint.engine._bucket import TimeBucketAggregator
from footprint.engine._price_grid import PriceGrid


class FootprintBuilder:
    """Builds a 4-channel footprint array from tick data.

    Pre-allocates (4, time_buckets, price_levels) zeros.
    Channel 0 accumulates bid volume (side == -1).
    Channel 1 accumulates ask volume (side == 1).
    Channels 2 and 3 are derived after accumulation (delta, total).

    Critical: No Python for-loops in the accumulation path.
    Use np.add.at with boolean masks for vectorized aggregation.
    """

    def __init__(self, config: FootprintConfig) -> None:
        self._config = config

    def build(
        self,
        ticks: np.ndarray,
        mid_price: float,
    ) -> np.ndarray:
        bucket_agg = TimeBucketAggregator(self._config)
        price_grid = PriceGrid(self._config, mid_price)

        buckets = bucket_agg.assign_buckets(ticks["timestamp_ns"])
        levels = price_grid.assign_levels(ticks["price"])

        footprint = np.zeros((4, self._config.time_buckets, self._config.price_levels), dtype=np.float32)

        bid_mask = ticks["side"] == -1
        ask_mask = ticks["side"] == 1

        # np.add.at handles duplicate (bucket, level) pairs by accumulating all matching tick sizes.
        if bid_mask.any():
            np.add.at(footprint[0], (buckets[bid_mask], levels[bid_mask]), ticks["size"][bid_mask])

        if ask_mask.any():
            np.add.at(footprint[1], (buckets[ask_mask], levels[ask_mask]), ticks["size"][ask_mask])

        # Delta and total are derived after accumulation so they are consistent with the raw volumes.
        footprint[2] = footprint[1] - footprint[0]
        footprint[3] = footprint[0] + footprint[1]

        return footprint
