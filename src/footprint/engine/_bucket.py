from __future__ import annotations

import numpy as np

from footprint._config import FootprintConfig
from footprint._exceptions import DataError


class TimeBucketAggregator:
    """Assigns nanosecond timestamps to time buckets within a candle window."""

    def __init__(self, config: FootprintConfig) -> None:
        self._config = config
        self._bucket_count = config.time_buckets
        self._candle_duration_ns = config.candle_duration_seconds * 1_000_000_000

    def assign_buckets(self, timestamps_ns: np.ndarray) -> np.ndarray:
        if len(timestamps_ns) == 0:
            return np.array([], dtype=np.intp)

        candle_start = self._candle_start(timestamps_ns)

        bucket_size_ns = self._candle_duration_ns // self._bucket_count
        elapsed = timestamps_ns.astype("i8") - candle_start
        buckets = elapsed // bucket_size_ns

        if np.any(buckets < 0) or np.any(buckets >= self._bucket_count):
            out_of_range = np.where((buckets < 0) | (buckets >= self._bucket_count))[0]
            raise DataError(
                f"{len(out_of_range)} tick(s) fall outside the candle window "
                f"(bucket range 0..{self._bucket_count - 1})"
            )

        return buckets.astype(np.intp)

    def bucket_boundaries(self) -> tuple[int, int]:
        return (0, self._bucket_count - 1)

    def _candle_start(self, timestamps_ns: np.ndarray) -> int:
        first_ts = int(timestamps_ns[0])
        return (first_ts // self._candle_duration_ns) * self._candle_duration_ns
