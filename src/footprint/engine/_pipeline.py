"""End-to-end pipeline: raw ticks → normalized 4×64×64 footprint array.

Orchestrates PriceGrid → TimeBucketAggregator → FootprintBuilder →
FootprintNormalizer in sequence. This is the single public entry point
for the footprint engine.
"""

from __future__ import annotations

import numpy as np

from footprint._config import FootprintConfig
from footprint.engine._footprint import FootprintBuilder
from footprint.engine._normalizer import FootprintNormalizer


class FootprintPipeline:
    """Batch-only, stateless pipeline.

    process(ticks) takes one candle's worth of ticks and returns one
    (4, 64, 64) array. No internal state persists between calls.
    Deterministic: same ticks → same output.
    """

    def __init__(self, config: FootprintConfig) -> None:
        self._config = config
        self._builder = FootprintBuilder(config)
        self._normalizer = FootprintNormalizer(config)

    def process(self, ticks: np.ndarray) -> np.ndarray:
        mid_price = self._compute_mid_price(ticks)
        footprint = self._builder.build(ticks, mid_price)
        return self._normalizer.normalize(footprint)

    def _compute_mid_price(self, ticks: np.ndarray) -> float:
        # Use VWAP (not simple midpoint) because it reflects actual traded volume center.
        vwap = np.average(ticks["price"], weights=ticks["size"].astype("f8"))
        return float(vwap)
