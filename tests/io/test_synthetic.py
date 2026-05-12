"""Determinism, monotonicity, and schema compliance tests for SyntheticTickGenerator.

All tests assert exact array equality where deterministic (same seed)
and range checks where stochastic (different seeds).
"""

from __future__ import annotations

import numpy as np
import pytest

from footprint._config import FootprintConfig
from footprint.io._synthetic import SyntheticTickGenerator


class TestSyntheticTickGenerator:
    """Determinism, monotonicity, and schema compliance tests."""

    @pytest.fixture
    def config(self) -> FootprintConfig:
        return FootprintConfig()

    def test_determinism(self, config: FootprintConfig) -> None:
        gen1 = SyntheticTickGenerator(config, seed=42)
        gen2 = SyntheticTickGenerator(config, seed=42)
        arr1 = gen1.generate(1000)
        arr2 = gen2.generate(1000)
        assert np.array_equal(arr1["timestamp_ns"], arr2["timestamp_ns"])
        assert np.array_equal(arr1["price"], arr2["price"])
        assert np.array_equal(arr1["size"], arr2["size"])
        assert np.array_equal(arr1["side"], arr2["side"])
        assert np.array_equal(arr1["sequence"], arr2["sequence"])

    def test_monotonic_timestamps(self, config: FootprintConfig) -> None:
        gen = SyntheticTickGenerator(config, seed=42)
        arr = gen.generate(5000)
        diffs = np.diff(arr["timestamp_ns"])
        assert np.all(diffs > 0), "Timestamps must be strictly increasing"

    def test_monotonic_sequences(self, config: FootprintConfig) -> None:
        gen = SyntheticTickGenerator(config, seed=42)
        arr = gen.generate(5000)
        assert np.array_equal(arr["sequence"], np.arange(1, 5001))

    def test_generate_candle_fits_duration(self, config: FootprintConfig) -> None:
        gen = SyntheticTickGenerator(config, seed=42)
        arr = gen.generate_candle()
        duration_ns = config.candle_duration_seconds * 1_000_000_000
        assert len(arr) >= 1
        assert arr["timestamp_ns"][-1] < duration_ns
        assert arr["timestamp_ns"][0] >= 0

    def test_generate_candle_sorted(self, config: FootprintConfig) -> None:
        gen = SyntheticTickGenerator(config, seed=42)
        arr = gen.generate_candle()
        assert np.all(np.diff(arr["timestamp_ns"]) >= 0)

    def test_generate_candle_variable_size(self, config: FootprintConfig) -> None:
        gen1 = SyntheticTickGenerator(config, seed=42)
        gen2 = SyntheticTickGenerator(config, seed=99)
        arr1 = gen1.generate_candle()
        arr2 = gen2.generate_candle()
        assert len(arr1) >= 500
        assert len(arr2) >= 500

    def test_output_dtype(self, config: FootprintConfig) -> None:
        gen = SyntheticTickGenerator(config, seed=42)
        arr = gen.generate(100)
        assert arr.dtype.names == (
            "timestamp_ns", "price", "size", "side", "sequence", "symbol"
        )

    def test_symbol_uses_instrument(self) -> None:
        cfg = FootprintConfig(instrument="ES")
        gen = SyntheticTickGenerator(cfg, seed=42)
        arr = gen.generate(10)
        assert all(s == "ESF1" for s in arr["symbol"])
