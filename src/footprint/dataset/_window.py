from __future__ import annotations

from collections.abc import Callable

import numpy as np

from footprint._config import FootprintConfig
from footprint._exceptions import PipelineError


class SlidingWindowExtractor:
    """Extracts sliding windows from a sequence of footprint arrays."""

    def __init__(self, config: FootprintConfig, lookback: int = 1) -> None:
        self._config = config
        self._lookback = lookback

    def extract(self, footprints: np.ndarray, stride: int = 1) -> np.ndarray:
        if footprints.shape[0] < self._lookback:
            raise PipelineError(
                f"Need at least {self._lookback} footprints, got {footprints.shape[0]}"
            )
        T = footprints.shape[0]
        windows = np.lib.stride_tricks.sliding_window_view(footprints, self._lookback, axis=0)
        result = windows[::stride]
        axes = list(range(result.ndim))
        axes.pop(-1)
        axes.insert(1, result.ndim - 1)
        result = result.transpose(axes)
        return result
