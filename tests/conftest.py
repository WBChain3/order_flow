from __future__ import annotations

from typing import Generator

import numpy as np
import pytest

from footprint._config import FootprintConfig


@pytest.fixture
def default_config() -> FootprintConfig:
    return FootprintConfig()


@pytest.fixture
def alt_config() -> FootprintConfig:
    return FootprintConfig(tick_size=0.5, candle_duration_seconds=30, instrument="ES")


@pytest.fixture
def synthetic_ticks() -> Generator[np.ndarray, None, None]:
    """Small batch of structured tick data for quick tests."""
    n = 100
    dtype = np.dtype([
        ("timestamp_ns", "i8"),
        ("price", "f8"),
        ("size", "i4"),
        ("side", "i1"),
        ("sequence", "i8"),
        ("symbol", "U10"),
    ])
    arr = np.zeros(n, dtype=dtype)
    arr["timestamp_ns"] = np.arange(0, n * 1_000_000, 1_000_000, dtype="i8")
    arr["price"] = 20000.0 + np.random.default_rng(42).normal(0, 2.0, n).round(2)
    arr["size"] = np.random.default_rng(42).poisson(5, n).clip(1).astype("i4")
    arr["side"] = np.random.default_rng(42).choice([-1, 0, 1], n).astype("i1")
    arr["sequence"] = np.arange(1, n + 1, dtype="i8")
    arr["symbol"] = "NQ"
    yield arr
