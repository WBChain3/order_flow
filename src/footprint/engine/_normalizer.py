"""Per-candle normalization for footprint arrays.

Only channels 0 and 1 are normalized directly. Channels 2 (delta) and
3 (total) are recomputed from the normalized channels to maintain
semantic consistency (delta = ask - bid, total = ask + bid).
"""

from __future__ import annotations

import numpy as np

from footprint._config import FootprintConfig


class FootprintNormalizer:
    """Normalizes a 4-channel footprint array.

    per_candle_minmax: scales ch0, ch1 to [0, 1] per candle.
    per_candle_zscore: z-scores ch0, ch1 per candle.
    none: identity pass-through.

    Zero-volume channels are set to all zeros (no division by zero).
    Delta and total channels are always derived from normalized bid/ask,
    not normalized independently.
    """

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

            # If max == min, the channel is all-constant; set to 0 to avoid 0/0.
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
