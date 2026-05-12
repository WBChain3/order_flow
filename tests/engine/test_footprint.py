"""FootprintBuilder tests covering shape, dtype, channel relationships,
and side-based separation.
"""

from __future__ import annotations

import numpy as np
import pytest

from footprint._config import FootprintConfig
from footprint.engine._footprint import FootprintBuilder
from footprint.io._synthetic import SyntheticTickGenerator


class TestFootprintBuilder:
    """FootprintBuilder tests covering shape, dtype, channel relationships,
    and side-based separation."""

    @pytest.fixture
    def config(self) -> FootprintConfig:
        return FootprintConfig()

    @pytest.fixture
    def builder(self, config: FootprintConfig) -> FootprintBuilder:
        return FootprintBuilder(config)

    @pytest.fixture
    def mixed_ticks(self, config: FootprintConfig) -> np.ndarray:
        gen = SyntheticTickGenerator(config, seed=42)
        return gen.generate_candle()

    def test_output_shape_and_dtype(self, builder: FootprintBuilder, mixed_ticks: np.ndarray) -> None:
        footprint = builder.build(mixed_ticks, mid_price=20000.0)
        assert footprint.shape == (4, 64, 64)
        assert footprint.dtype == np.float32

    def test_channel_relationships(self, builder: FootprintBuilder, mixed_ticks: np.ndarray) -> None:
        footprint = builder.build(mixed_ticks, mid_price=20000.0)
        assert np.allclose(footprint[2], footprint[1] - footprint[0])
        assert np.allclose(footprint[3], footprint[0] + footprint[1])

    def test_no_side_zero(self, config: FootprintConfig, builder: FootprintBuilder) -> None:
        gen = SyntheticTickGenerator(config, seed=42)
        ticks = gen.generate_candle()
        ticks["side"] = 0
        footprint = builder.build(ticks, mid_price=20000.0)
        assert np.all(footprint[0] == 0)
        assert np.all(footprint[1] == 0)
        assert np.all(footprint[2] == 0)
        assert np.all(footprint[3] == 0)

    def test_bid_ask_separation(self, config: FootprintConfig, builder: FootprintBuilder) -> None:
        gen = SyntheticTickGenerator(config, seed=42)
        ticks = gen.generate_candle()
        ticks["side"] = 1
        footprint = builder.build(ticks, mid_price=20000.0)
        assert np.all(footprint[0] == 0)
        assert footprint[1].sum() > 0
        assert footprint[3].sum() > 0

    def test_only_bid_ticks(self, config: FootprintConfig, builder: FootprintBuilder) -> None:
        gen = SyntheticTickGenerator(config, seed=42)
        ticks = gen.generate_candle()
        ticks["side"] = -1
        footprint = builder.build(ticks, mid_price=20000.0)
        assert np.all(footprint[1] == 0)
        assert footprint[0].sum() > 0

    def test_non_zero_footprint(self, builder: FootprintBuilder, mixed_ticks: np.ndarray) -> None:
        footprint = builder.build(mixed_ticks, mid_price=20000.0)
        assert footprint[3].sum() > 0
