"""Exception hierarchy for the footprint package.

All errors inherit from FootprintError so callers can catch everything
with a single except clause.
"""


class FootprintError(Exception):
    """Base exception for all footprint-related errors."""


class ConfigError(FootprintError):
    """Raised when FootprintConfig fails __post_init__ validation."""


class DataError(FootprintError):
    """Raised when tick data fails schema validation or contains anomalies."""


class PipelineError(FootprintError):
    """Raised when an internal pipeline step encounters an unrecoverable state
    (e.g., insufficient data for windowing)."""
