from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass
from math import log2

_VALID_NORMALIZATIONS = frozenset({"per_candle_minmax", "per_candle_zscore", "none"})

_POWERS_OF_2 = frozenset({32, 64, 128, 256})


@dataclass(frozen=True)
class FootprintConfig:
    candle_duration_seconds: int = 60
    price_levels: int = 64
    time_buckets: int = 64
    tick_size: float = 0.25
    price_range_ticks: int = 64
    normalization: str = "per_candle_minmax"
    schema_version: int = 1
    instrument: str = "NQ"

    def __post_init__(self) -> None:
        if self.candle_duration_seconds <= 0:
            raise ConfigError(
                f"candle_duration_seconds must be > 0, got {self.candle_duration_seconds}"
            )
        if self.tick_size <= 0:
            raise ConfigError(f"tick_size must be > 0, got {self.tick_size}")
        if self.price_levels not in _POWERS_OF_2:
            raise ConfigError(
                f"price_levels must be a power of 2 in {sorted(_POWERS_OF_2)}, "
                f"got {self.price_levels}"
            )
        if self.time_buckets not in _POWERS_OF_2:
            raise ConfigError(
                f"time_buckets must be a power of 2 in {sorted(_POWERS_OF_2)}, "
                f"got {self.time_buckets}"
            )
        if self.price_levels != self.time_buckets:
            raise ConfigError(
                f"price_levels ({self.price_levels}) must equal "
                f"time_buckets ({self.time_buckets}) — square grid required"
            )
        if self.price_range_ticks <= 0:
            raise ConfigError(
                f"price_range_ticks must be > 0, got {self.price_range_ticks}"
            )
        if self.price_range_ticks < self.price_levels:
            raise ConfigError(
                f"price_range_ticks ({self.price_range_ticks}) must be >= "
                f"price_levels ({self.price_levels}) so each level covers at least one tick"
            )
        if self.normalization not in _VALID_NORMALIZATIONS:
            raise ConfigError(
                f"normalization must be one of {sorted(_VALID_NORMALIZATIONS)}, "
                f"got '{self.normalization}'"
            )
        if not isinstance(self.schema_version, int) or self.schema_version < 1:
            raise ConfigError(
                f"schema_version must be a positive int, got {self.schema_version}"
            )


# Re-export for convenience
from footprint._exceptions import ConfigError
