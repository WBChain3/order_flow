"""AugmentationPipeline tests for identity mode, transform application, and chain ordering."""

from __future__ import annotations

import numpy as np

from footprint.dataset._augment import AugmentationPipeline


class TestAugmentationPipeline:
    """AugmentationPipeline tests for identity, transform, and ordering."""

    def test_disabled_returns_identity(self) -> None:
        pipeline = AugmentationPipeline(enabled=False)
        sample = np.random.default_rng(42).random((4, 64, 64)).astype(np.float32)
        result = pipeline(sample)
        assert np.array_equal(result, sample)

    def test_enabled_applies_transform(self) -> None:
        pipeline = AugmentationPipeline(enabled=True)
        pipeline.add_transform(lambda x: x + 1.0)
        sample = np.zeros((4, 64, 64), dtype=np.float32)
        result = pipeline(sample)
        assert np.allclose(result, 1.0)

    def test_transform_chain_order(self) -> None:
        pipeline = AugmentationPipeline(enabled=True)
        pipeline.add_transform(lambda x: x + 1.0)
        pipeline.add_transform(lambda x: x * 2.0)
        sample = np.zeros((4, 64, 64), dtype=np.float32)
        result = pipeline(sample)
        assert np.allclose(result, 2.0)

    def test_empty_transform_list(self) -> None:
        pipeline = AugmentationPipeline(enabled=True)
        sample = np.random.default_rng(42).random((4, 64, 64)).astype(np.float32)
        result = pipeline(sample)
        assert np.array_equal(result, sample)

    def test_no_mutation_of_original(self) -> None:
        pipeline = AugmentationPipeline(enabled=True)
        pipeline.add_transform(lambda x: x + 1.0)
        sample = np.zeros((4, 64, 64), dtype=np.float32)
        result = pipeline(sample)
        assert np.all(sample == 0.0)
        assert np.all(result == 1.0)
