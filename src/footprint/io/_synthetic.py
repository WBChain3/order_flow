"""Deterministic synthetic tick generator for testing.

All output matches the canonical TICK_SCHEMA. Seeds are fixed so
unit tests assert exact array equality across runs.
"""

from __future__ import annotations

import numpy as np

from footprint._config import FootprintConfig
from footprint.io._schema import numpy_tick_dtype


class SyntheticTickGenerator:
    """Generates synthetic tick data for pipeline testing.

    Deterministic with a fixed seed. Produces structured NumPy arrays
    matching the standard tick schema. Random walk price is centered
    on a configurable mid (default 20000.0 for NQ).

    generate_candle() wraps generate() and folds timestamps into a single
    candle window, ensuring the output is suitable for FootprintPipeline.process().
    """

    def __init__(self, config: FootprintConfig, seed: int = 42) -> None:
        self._config = config
        self._rng = np.random.default_rng(seed)

    def generate(self, num_ticks: int, mid_price: float = 20000.0) -> np.ndarray:
        dtype = numpy_tick_dtype()
        arr = np.empty(num_ticks, dtype=dtype)

        # exponential(scale=1ms) simulates Poisson process tick arrivals.
        start_ns = 0
        timestamps_ns = start_ns + np.cumsum(
            self._rng.exponential(scale=1_000_000, size=num_ticks).astype("i8")
        )
        arr["timestamp_ns"] = timestamps_ns

        price_walk = np.cumsum(self._rng.normal(0, 1.0, num_ticks)).round(2)
        arr["price"] = mid_price + price_walk

        arr["size"] = self._rng.poisson(5, num_ticks).clip(1).astype("i4")
        arr["side"] = self._rng.choice([-1, 0, 1], num_ticks, p=[0.45, 0.10, 0.45]).astype("i1")
        arr["sequence"] = np.arange(1, num_ticks + 1, dtype="i8")
        arr["symbol"] = f"{self._config.instrument}F1"

        return arr

    def generate_candle(self, mid_price: float = 20000.0) -> np.ndarray:
        num_ticks = int(self._rng.integers(500, 2000))
        arr = self.generate(num_ticks, mid_price=mid_price)
        candle_duration_ns = self._config.candle_duration_seconds * 1_000_000_000
        # Fold timestamps into [0, candle_duration) then sort to maintain monotonicity.
        arr["timestamp_ns"] = arr["timestamp_ns"] % candle_duration_ns
        arr["timestamp_ns"] = np.sort(arr["timestamp_ns"])
        return arr
