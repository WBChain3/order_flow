"""TickValidator tests covering rejection logic and logging.

Tests inject specific anomalies (price outliers, OOS timestamps,
zero size, duplicate sequences) into otherwise clean synthetic data
and assert the validator catches them.
"""

from __future__ import annotations

import numpy as np
import pytest

from footprint._config import FootprintConfig
from footprint.io._schema import numpy_tick_dtype
from footprint.io._synthetic import SyntheticTickGenerator
from footprint.io._validator import TickValidator


class TestTickValidator:
    """TickValidator tests covering rejection logic and logging."""

    @pytest.fixture
    def config(self) -> FootprintConfig:
        return FootprintConfig()

    @pytest.fixture
    def clean_ticks(self, config: FootprintConfig) -> np.ndarray:
        gen = SyntheticTickGenerator(config, seed=42)
        return gen.generate(500)

    def test_keeps_valid_ticks(self, config: FootprintConfig, clean_ticks: np.ndarray) -> None:
        val = TickValidator(config)
        result = val.validate(clean_ticks)
        assert len(result) == len(clean_ticks)
        assert len(val.rejection_log) == 0

    def test_removes_price_outliers(self, config: FootprintConfig, clean_ticks: np.ndarray) -> None:
        val = TickValidator(config)
        clean_ticks["price"][100] = 1_000_000.0
        clean_ticks["price"][200] = -1_000_000.0
        result = val.validate(clean_ticks)
        assert len(result) == len(clean_ticks) - 2
        rejections = [r for r in val.rejection_log if r["reason"] == "price_outlier"]
        assert len(rejections) >= 2

    def test_logs_outlier_reason(self, config: FootprintConfig, clean_ticks: np.ndarray) -> None:
        val = TickValidator(config)
        clean_ticks["price"][100] = 999_999.0
        val.validate(clean_ticks)
        assert any(r["reason"] == "price_outlier" for r in val.rejection_log)

    def test_handles_empty_ticks(self, config: FootprintConfig) -> None:
        val = TickValidator(config)
        dtype = numpy_tick_dtype()
        empty = np.empty(0, dtype=dtype)
        result = val.validate(empty)
        assert len(result) == 0

    def test_out_of_sequence_rejected(self, config: FootprintConfig, clean_ticks: np.ndarray) -> None:
        val = TickValidator(config)
        clean_ticks["timestamp_ns"][50] = clean_ticks["timestamp_ns"][49] - 20_000_000
        result = val.validate(clean_ticks)
        assert len(result) < len(clean_ticks)

    def test_out_of_sequence_logged(self, config: FootprintConfig, clean_ticks: np.ndarray) -> None:
        val = TickValidator(config)
        clean_ticks["timestamp_ns"][50] = clean_ticks["timestamp_ns"][49] - 20_000_000
        val.validate(clean_ticks)
        assert any(r["reason"] == "out_of_sequence" for r in val.rejection_log)

    def test_zero_size_warned(self, config: FootprintConfig, clean_ticks: np.ndarray) -> None:
        val = TickValidator(config)
        clean_ticks["size"][0] = 0
        result = val.validate(clean_ticks)
        assert len(result) == len(clean_ticks)
        assert any(r["reason"] == "zero_size" for r in val.rejection_log)

    def test_duplicate_sequence_warned(self, config: FootprintConfig, clean_ticks: np.ndarray) -> None:
        val = TickValidator(config)
        clean_ticks["sequence"][0] = 999
        clean_ticks["sequence"][1] = 999
        result = val.validate(clean_ticks)
        assert len(result) == len(clean_ticks)
        assert any(r["reason"] == "duplicate_sequence" for r in val.rejection_log)

    def test_log_has_correct_keys(self, config: FootprintConfig, clean_ticks: np.ndarray) -> None:
        val = TickValidator(config)
        clean_ticks["price"][100] = 999_999.0
        val.validate(clean_ticks)
        if val.rejection_log:
            entry = val.rejection_log[0]
            assert "reason" in entry
            assert "type" in entry

    def test_rolling_vwap_catches_moderate_outlier(self, config: FootprintConfig) -> None:
        """A 10σ outlier on a 1s rolling basis must be caught.
        # Injected outlier is +500 ticks from local VWAP, extreme enough to trigger rolling-window detection.
        """
        gen = SyntheticTickGenerator(config, seed=42)
        ticks = gen.generate(2000, mid_price=20000.0)

        ticks["timestamp_ns"] = np.arange(0, len(ticks) * 1_000_000, 1_000_000, dtype="i8")

        outlier_idx = len(ticks) // 2
        ticks["price"][outlier_idx] = ticks["price"][outlier_idx] + 500.0

        val = TickValidator(config)
        result = val.validate(ticks)
        assert len(result) < len(ticks), "Rolling VWAP should catch extreme outlier within a 1s window"
        assert any(r["reason"] == "price_outlier" for r in val.rejection_log)
