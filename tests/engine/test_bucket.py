from __future__ import annotations

import numpy as np
import pytest

from footprint._config import FootprintConfig
from footprint._exceptions import DataError
from footprint.engine._bucket import TimeBucketAggregator


class TestTimeBucketAggregator:
    @pytest.fixture
    def config(self) -> FootprintConfig:
        return FootprintConfig()

    def test_assign_buckets_range(self, config: FootprintConfig) -> None:
        aggregator = TimeBucketAggregator(config)
        n = 5000
        candle_duration_ns = config.candle_duration_seconds * 1_000_000_000
        timestamps = np.arange(0, n * 1_000_000, 1_000_000, dtype="i8")[:n]
        timestamps = timestamps % candle_duration_ns
        timestamps = np.sort(timestamps)
        buckets = aggregator.assign_buckets(timestamps)
        assert np.all(buckets >= 0)
        assert np.all(buckets < config.time_buckets)

    def test_assign_buckets_dense(self, config: FootprintConfig) -> None:
        aggregator = TimeBucketAggregator(config)
        n = 5000
        candle_duration_ns = config.candle_duration_seconds * 1_000_000_000
        timestamps = np.linspace(0, candle_duration_ns - 1, n, dtype="i8")
        buckets = aggregator.assign_buckets(timestamps)
        unique_buckets = np.unique(buckets)
        assert len(unique_buckets) == config.time_buckets

    def test_assign_buckets_empty(self, config: FootprintConfig) -> None:
        aggregator = TimeBucketAggregator(config)
        buckets = aggregator.assign_buckets(np.array([], dtype="i8"))
        assert len(buckets) == 0

    def test_rejects_outside_window(self, config: FootprintConfig) -> None:
        aggregator = TimeBucketAggregator(config)
        candle_duration_ns = config.candle_duration_seconds * 1_000_000_000
        timestamps = np.array([0, candle_duration_ns + 1], dtype="i8")
        with pytest.raises(DataError):
            aggregator.assign_buckets(timestamps)

    def test_bucket_boundaries(self, config: FootprintConfig) -> None:
        aggregator = TimeBucketAggregator(config)
        lo, hi = aggregator.bucket_boundaries()
        assert lo == 0
        assert hi == config.time_buckets - 1

    def test_bucket_order(self, config: FootprintConfig) -> None:
        aggregator = TimeBucketAggregator(config)
        candle_duration_ns = config.candle_duration_seconds * 1_000_000_000
        timestamps = np.array([0, candle_duration_ns // 4, candle_duration_ns // 2], dtype="i8")
        buckets = aggregator.assign_buckets(timestamps)
        assert buckets[0] < buckets[1]
        assert buckets[1] < buckets[2]
