from __future__ import annotations

import numpy as np
import pytest

from footprint._config import FootprintConfig
from footprint._exceptions import PipelineError
from footprint.dataset._window import SlidingWindowExtractor


class TestSlidingWindowExtractor:
    @pytest.fixture
    def config(self) -> FootprintConfig:
        return FootprintConfig()

    @pytest.fixture
    def footprints(self) -> np.ndarray:
        return np.arange(10 * 4 * 64 * 64, dtype=np.float32).reshape(10, 4, 64, 64)

    def test_basic(self, config: FootprintConfig, footprints: np.ndarray) -> None:
        extractor = SlidingWindowExtractor(config, lookback=3)
        result = extractor.extract(footprints, stride=1)
        assert result.shape == (8, 3, 4, 64, 64)
        assert np.array_equal(result[0, 0], footprints[0])
        assert np.array_equal(result[0, 1], footprints[1])
        assert np.array_equal(result[0, 2], footprints[2])

    def test_no_overlap(self, config: FootprintConfig, footprints: np.ndarray) -> None:
        extractor = SlidingWindowExtractor(config, lookback=3)
        result = extractor.extract(footprints, stride=3)
        assert result.shape == (3, 3, 4, 64, 64)
        assert np.array_equal(result[0, 0], footprints[0])
        assert np.array_equal(result[1, 0], footprints[3])
        assert np.array_equal(result[2, 0], footprints[6])

    def test_insufficient_data(self, config: FootprintConfig, footprints: np.ndarray) -> None:
        extractor = SlidingWindowExtractor(config, lookback=15)
        with pytest.raises(PipelineError):
            extractor.extract(footprints, stride=1)

    def test_lookback_1(self, config: FootprintConfig, footprints: np.ndarray) -> None:
        extractor = SlidingWindowExtractor(config, lookback=1)
        result = extractor.extract(footprints, stride=1)
        assert result.shape == (10, 1, 4, 64, 64)

    def test_stride_2(self, config: FootprintConfig, footprints: np.ndarray) -> None:
        extractor = SlidingWindowExtractor(config, lookback=2)
        result = extractor.extract(footprints, stride=2)
        assert result.shape == (5, 2, 4, 64, 64)
