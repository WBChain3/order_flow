from __future__ import annotations

import numpy as np

from footprint._config import FootprintConfig


class PriceGrid:
    """Maps prices to price levels centered on a mid price.

    Grid span is controlled by `price_range_ticks`, not `price_levels`.
    `price_levels` is the fixed output resolution (CNN input dimension).
    `price_range_ticks / price_levels` determines ticks-per-level (must be >= 1).
    """

    def __init__(self, config: FootprintConfig, mid_price: float) -> None:
        self._config = config
        self._mid_price = mid_price
        self._tick_size = config.tick_size
        self._num_levels = config.price_levels
        half_range_ticks = config.price_range_ticks // 2

        self._min_price = mid_price - (half_range_ticks * self._tick_size)
        self._max_price = mid_price + (half_range_ticks * self._tick_size) - self._tick_size
        self._ticks_per_level = config.price_range_ticks / config.price_levels

        self._level_prices = self._min_price + np.arange(self._num_levels) * (self._ticks_per_level * self._tick_size)

    def assign_levels(self, prices: np.ndarray) -> np.ndarray:
        level_size = self._ticks_per_level * self._tick_size
        levels = np.floor((prices - self._min_price) / level_size).astype(np.intp)
        levels = np.clip(levels, 0, self._num_levels - 1)
        return levels

    def price_at_level(self, level: int) -> float:
        return float(self._level_prices[level])

    @property
    def min_price(self) -> float:
        return self._min_price

    @property
    def max_price(self) -> float:
        return self._max_price

    @property
    def level_prices(self) -> np.ndarray:
        return self._level_prices.copy()
