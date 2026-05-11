from __future__ import annotations

import numpy as np

from footprint._config import FootprintConfig
from footprint.dataset._augment import AugmentationPipeline
from footprint.dataset._weighter import SampleWeighter
from footprint.dataset._window import SlidingWindowExtractor


class DatasetFactory:
    """Produces labeled/unlabeled samples and weights from footprint arrays."""

    def __init__(
        self,
        config: FootprintConfig,
        augment: AugmentationPipeline | None = None,
        weighter: SampleWeighter | None = None,
    ) -> None:
        self._config = config
        self._augment = augment or AugmentationPipeline(enabled=False)
        self._weighter = weighter or SampleWeighter()

    def create(
        self,
        footprints: np.ndarray,
        stride: int = 1,
        lookback: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        extractor = SlidingWindowExtractor(self._config, lookback=lookback)
        windows = extractor.extract(footprints, stride=stride)

        N = windows.shape[0]
        samples = np.empty((N, lookback, 4, 64, 64), dtype=np.float32)
        weights = np.empty(N, dtype=np.float64)

        for i in range(N):
            sample = windows[i]
            sample = self._augment(sample)
            samples[i] = sample
            weights[i] = self._weighter(sample)

        return samples, weights
