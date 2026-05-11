from __future__ import annotations

import numpy as np

from footprint._config import FootprintConfig
from footprint.engine._footprint import FootprintBuilder
from footprint.engine._normalizer import FootprintNormalizer


class FootprintPipeline:
    """End-to-end pipeline: raw ticks → normalized 4×64×64 footprint array."""

    def __init__(self, config: FootprintConfig) -> None:
        self._config = config
        self._builder = FootprintBuilder(config)
        self._normalizer = FootprintNormalizer(config)

    def process(self, ticks: np.ndarray) -> np.ndarray:
        mid_price = self._compute_mid_price(ticks)
        footprint = self._builder.build(ticks, mid_price)
        return self._normalizer.normalize(footprint)

    def _compute_mid_price(self, ticks: np.ndarray) -> float:
        vwap = np.average(ticks["price"], weights=ticks["size"].astype("f8"))
        return float(vwap)
