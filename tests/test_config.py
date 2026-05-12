"""Validation tests for FootprintConfig dataclass.

Covers all __post_init__ guards: power-of-2 dimensions, square grid,
positive tick size, valid normalization, positive schema_version,
and price_range_ticks >= price_levels.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from footprint._config import FootprintConfig
from footprint._exceptions import ConfigError


class TestFootprintConfig:
    """Validation tests for FootprintConfig dataclass."""

    def test_creates_with_defaults(self) -> None:
        cfg = FootprintConfig()
        assert cfg.candle_duration_seconds == 60
        assert cfg.price_levels == 64
        assert cfg.time_buckets == 64
        assert cfg.tick_size == 0.25
        assert cfg.price_range_ticks == 64
        assert cfg.normalization == "per_candle_minmax"
        assert cfg.schema_version == 1
        assert cfg.instrument == "NQ"

    def test_creates_with_custom_values(self) -> None:
        cfg = FootprintConfig(tick_size=0.5, candle_duration_seconds=30, instrument="ES")
        assert cfg.tick_size == 0.5
        assert cfg.candle_duration_seconds == 30
        assert cfg.instrument == "ES"

    @pytest.mark.parametrize("bad_dim", [0, -1, 50, 63, 65, 100])
    def test_rejects_invalid_price_levels(self, bad_dim: int) -> None:
        # 0, -1 test non-positives; 50, 63, 65, 100 test non-powers-of-2 near valid values.
        with pytest.raises(ConfigError):
            FootprintConfig(price_levels=bad_dim)

    @pytest.mark.parametrize("bad_dim", [0, -1, 50, 63, 65, 100])
    def test_rejects_invalid_time_buckets(self, bad_dim: int) -> None:
        with pytest.raises(ConfigError):
            FootprintConfig(time_buckets=bad_dim)

    @pytest.mark.parametrize("good_dim,range_ticks", [(32, 32), (64, 64), (128, 128), (256, 256)])
    def test_accepts_valid_dimensions(self, good_dim: int, range_ticks: int) -> None:
        # Powers of 2 that match the _POWERS_OF_2 frozenset in _config.py.
        cfg = FootprintConfig(price_levels=good_dim, time_buckets=good_dim, price_range_ticks=range_ticks)
        assert cfg.price_levels == good_dim
        assert cfg.time_buckets == good_dim

    @pytest.mark.parametrize("bad_ts", [0, -0.25, -1.0])
    def test_rejects_non_positive_tick_size(self, bad_ts: float) -> None:
        with pytest.raises(ConfigError):
            FootprintConfig(tick_size=bad_ts)

    def test_rejects_mismatched_dimensions(self) -> None:
        with pytest.raises(ConfigError, match="square grid"):
            FootprintConfig(price_levels=64, time_buckets=32)

    @pytest.mark.parametrize("bad_norm", ["", "invalid", "minmax", "zscore"])
    def test_rejects_invalid_normalization(self, bad_norm: str) -> None:
        with pytest.raises(ConfigError):
            FootprintConfig(normalization=bad_norm)

    @pytest.mark.parametrize("good_norm", ["per_candle_minmax", "per_candle_zscore", "none"])
    def test_accepts_valid_normalization(self, good_norm: str) -> None:
        cfg = FootprintConfig(normalization=good_norm)
        assert cfg.normalization == good_norm

    def test_rejects_non_positive_duration(self) -> None:
        with pytest.raises(ConfigError):
            FootprintConfig(candle_duration_seconds=0)
        with pytest.raises(ConfigError):
            FootprintConfig(candle_duration_seconds=-10)

    def test_is_frozen(self) -> None:
        cfg = FootprintConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.tick_size = 0.5  # type: ignore[misc]

    def test_schema_version_is_positive_int(self) -> None:
        with pytest.raises(ConfigError):
            FootprintConfig(schema_version=0)
        with pytest.raises(ConfigError):
            FootprintConfig(schema_version=-1)

    def test_repr_is_readable(self) -> None:
        cfg = FootprintConfig()
        r = repr(cfg)
        assert "tick_size" in r
        assert "normalization" in r
        assert "schema_version" in r
        assert "instrument" in r


class TestPriceRangeTicksValidation:
    """Tests for the price_range_ticks >= price_levels guard."""

    def test_rejects_range_less_than_levels(self) -> None:
        with pytest.raises(ConfigError):
            FootprintConfig(price_range_ticks=32, price_levels=64)

    def test_accepts_range_equal_to_levels(self) -> None:
        cfg = FootprintConfig(price_range_ticks=64, price_levels=64)
        assert cfg.price_range_ticks == 64
        assert cfg.price_levels == 64

    def test_accepts_range_greater_than_levels(self) -> None:
        cfg = FootprintConfig(price_range_ticks=128, price_levels=64)
        assert cfg.price_range_ticks == 128
        assert cfg.price_levels == 64
