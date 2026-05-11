from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


class AugmentationPipeline:
    """Applies a chain of augmentation transforms to footprint samples.

    Off by default. When disabled, returns identity.
    """

    def __init__(self, enabled: bool = False) -> None:
        self._enabled = enabled
        self._transforms: list[Callable[[np.ndarray], np.ndarray]] = []

    def __call__(self, sample: np.ndarray) -> np.ndarray:
        if not self._enabled or not self._transforms:
            return sample
        result = sample.copy()
        for transform in self._transforms:
            result = transform(result)
        return result

    def add_transform(self, transform: Callable[[np.ndarray], np.ndarray]) -> None:
        self._transforms.append(transform)
