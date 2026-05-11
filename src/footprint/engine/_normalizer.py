from __future__ import annotations

import numpy as np

from footprint._config import FootprintConfig


class FootprintNormalizer:
    """Normalizes a 4-channel footprint array."""

    def __init__(self, config: FootprintConfig) -> None:
        self._config = config
        self._mode = config.normalization

    def normalize(self, footprint: np.ndarray) -> np.ndarray:
        if self._mode == "none":
            return footprint

        result = footprint.copy()

        for c in range(2):
            channel = result[c]
            cmin = channel.min()
            cmax = channel.max()

            if cmax == cmin:
                result[c] = 0.0
                continue

            if self._mode == "per_candle_minmax":
                result[c] = (channel - cmin) / (cmax - cmin)
            elif self._mode == "per_candle_zscore":
                mean = channel.mean()
                std = channel.std()
                if std > 0:
                    result[c] = (channel - mean) / std
                else:
                    result[c] = 0.0

        result[2] = result[1] - result[0]
        result[3] = result[0] + result[1]

        return result
