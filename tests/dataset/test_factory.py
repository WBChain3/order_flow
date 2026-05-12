"""DatasetFactory integration tests combining windowing, augmentation, and weighting."""

from __future__ import annotations

import numpy as np
import pytest

from footprint._config import FootprintConfig
from footprint.dataset._augment import AugmentationPipeline
from footprint.dataset._factory import DatasetFactory


class TestDatasetFactory:
    """DatasetFactory integration tests."""

    @pytest.fixture
    def config(self) -> FootprintConfig:
        return FootprintConfig()

    @pytest.fixture
    def footprints(self) -> np.ndarray:
        return np.arange(10 * 4 * 64 * 64, dtype=np.float32).reshape(10, 4, 64, 64)

    def test_creates_samples_and_weights(self, config: FootprintConfig, footprints: np.ndarray) -> None:
        factory = DatasetFactory(config)
        samples, weights = factory.create(footprints, stride=1, lookback=1)
        assert samples.shape == (10, 1, 4, 64, 64)
        assert weights.shape == (10,)
        assert np.all(weights == 1.0)

    def test_with_lookback(self, config: FootprintConfig, footprints: np.ndarray) -> None:
        factory = DatasetFactory(config)
        samples, weights = factory.create(footprints, stride=1, lookback=3)
        assert samples.shape == (8, 3, 4, 64, 64)
        assert weights.shape == (8,)

    def test_with_stride(self, config: FootprintConfig, footprints: np.ndarray) -> None:
        factory = DatasetFactory(config)
        samples, weights = factory.create(footprints, stride=2, lookback=1)
        assert samples.shape == (5, 1, 4, 64, 64)
        assert weights.shape == (5,)

    def test_with_augmentation(self, config: FootprintConfig, footprints: np.ndarray) -> None:
        augment = AugmentationPipeline(enabled=True)
        augment.add_transform(lambda x: x + 1.0)
        factory = DatasetFactory(config, augment=augment)
        samples, _ = factory.create(footprints, stride=1, lookback=1)
        expected = footprints + 1.0
        assert np.allclose(samples[:, 0], expected)

    def test_output_dtype(self, config: FootprintConfig, footprints: np.ndarray) -> None:
        factory = DatasetFactory(config)
        samples, weights = factory.create(footprints, stride=1, lookback=1)
        assert samples.dtype == np.float32
        assert weights.dtype == np.float64
