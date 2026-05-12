"""End-to-end pipeline tests using synthetic candles.

Covers output shape/dtype, value ranges, channel consistency,
determinism, and seed sensitivity.
"""

from __future__ import annotations

import numpy as np
import pytest

from footprint._config import FootprintConfig
from footprint.engine._pipeline import FootprintPipeline
from footprint.io._synthetic import SyntheticTickGenerator


class TestFootprintPipeline:
    """End-to-end pipeline tests using synthetic candles."""

    @pytest.fixture
    def config(self) -> FootprintConfig:
        return FootprintConfig()

    @pytest.fixture
    def pipeline(self, config: FootprintConfig) -> FootprintPipeline:
        return FootprintPipeline(config)

    @pytest.fixture
    def candle_ticks(self, config: FootprintConfig) -> np.ndarray:
        gen = SyntheticTickGenerator(config, seed=42)
        return gen.generate_candle()

    def test_e2e_output_shape_and_dtype(self, pipeline: FootprintPipeline, candle_ticks: np.ndarray) -> None:
        result = pipeline.process(candle_ticks)
        assert result.shape == (4, 64, 64)
        assert result.dtype == np.float32
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_e2e_values_in_range(self, pipeline: FootprintPipeline, candle_ticks: np.ndarray) -> None:
        result = pipeline.process(candle_ticks)
        assert result[0].min() >= 0.0
        assert result[0].max() <= 1.0
        assert result[1].min() >= 0.0
        assert result[1].max() <= 1.0
        assert result[2].min() >= -1.0
        assert result[2].max() <= 1.0
        assert result[3].min() >= 0.0
        assert result[3].max() <= 2.0

    def test_deterministic(self, config: FootprintConfig) -> None:
        """Two separate FootprintPipeline instances with identical config and ticks must produce bit-exact arrays."""
        gen = SyntheticTickGenerator(config, seed=42)
        ticks = gen.generate_candle()
        p1 = FootprintPipeline(config)
        p2 = FootprintPipeline(config)
        result1 = p1.process(ticks)
        result2 = p2.process(ticks)
        assert np.array_equal(result1, result2)

    def test_channel_relationships(self, pipeline: FootprintPipeline, candle_ticks: np.ndarray) -> None:
        result = pipeline.process(candle_ticks)
        assert np.allclose(result[2], result[1] - result[0], atol=1e-6)
        assert np.allclose(result[3], result[0] + result[1], atol=1e-6)

    def test_different_seed_produces_different_result(self, config: FootprintConfig) -> None:
        gen1 = SyntheticTickGenerator(config, seed=42)
        gen2 = SyntheticTickGenerator(config, seed=99)
        ticks1 = gen1.generate_candle()
        ticks2 = gen2.generate_candle()
        p = FootprintPipeline(config)
        r1 = p.process(ticks1)
        r2 = p.process(ticks2)
        assert not np.array_equal(r1, r2)
