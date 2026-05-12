"""FootprintNormalizer tests for minmax, zscore, none modes,
zero-channel handling, and shape preservation.
"""

from __future__ import annotations

import numpy as np
import pytest

from footprint._config import FootprintConfig
from footprint.engine._normalizer import FootprintNormalizer


class TestFootprintNormalizer:
    """FootprintNormalizer tests for minmax, zscore, none modes,
    zero-channel handling, and shape preservation."""

    @pytest.fixture
    def config(self) -> FootprintConfig:
        return FootprintConfig()

    @pytest.fixture
    def footprint(self) -> np.ndarray:
        f = np.zeros((4, 64, 64), dtype=np.float32)
        f[0, 0, 0] = 10.0
        f[0, 31, 31] = 50.0
        f[1, 0, 0] = 5.0
        f[1, 31, 31] = 25.0
        f[2] = f[1] - f[0]
        f[3] = f[0] + f[1]
        return f

    def test_minmax_range(self, config: FootprintConfig, footprint: np.ndarray) -> None:
        normalizer = FootprintNormalizer(config)
        result = normalizer.normalize(footprint)
        for c in range(2):
            assert result[c].min() >= 0.0
            assert result[c].max() <= 1.0
        assert result[2].min() >= -1.0
        assert result[2].max() <= 1.0
        assert result[3].min() >= 0.0
        assert result[3].max() <= 2.0

    def test_minmax_active_channel(self, config: FootprintConfig, footprint: np.ndarray) -> None:
        normalizer = FootprintNormalizer(config)
        result = normalizer.normalize(footprint)
        assert float(result[0].min()) == pytest.approx(0.0, abs=1e-6)
        assert float(result[0].max()) == pytest.approx(1.0, abs=1e-6)

    def test_handles_zero_channel(self, config: FootprintConfig) -> None:
        footprint = np.zeros((4, 64, 64), dtype=np.float32)
        normalizer = FootprintNormalizer(config)
        result = normalizer.normalize(footprint)
        assert np.all(result == 0.0)

    def test_zscore_normalization(self) -> None:
        """Only two non-zero values in ch0; std will be non-zero so z-score is well-defined."""
        cfg = FootprintConfig(normalization="per_candle_zscore")
        footprint = np.zeros((4, 64, 64), dtype=np.float32)
        footprint[0, 0, 0] = 100.0
        footprint[0, 0, 1] = 200.0
        normalizer = FootprintNormalizer(cfg)
        result = normalizer.normalize(footprint)
        assert abs(float(result[0].mean())) < 1e-6
        assert float(result[0].std()) == pytest.approx(1.0, abs=1e-5)

    def test_none_normalization(self) -> None:
        cfg = FootprintConfig(normalization="none")
        footprint = np.random.default_rng(42).random((4, 64, 64)).astype(np.float32)
        normalizer = FootprintNormalizer(cfg)
        result = normalizer.normalize(footprint)
        assert np.array_equal(result, footprint)

    def test_preserves_shape(self, config: FootprintConfig, footprint: np.ndarray) -> None:
        normalizer = FootprintNormalizer(config)
        result = normalizer.normalize(footprint)
        assert result.shape == (4, 64, 64)
