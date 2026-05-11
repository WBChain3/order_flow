class FootprintError(Exception):
    """Base exception for all footprint-related errors."""


class ConfigError(FootprintError):
    """Raised when a FootprintConfig is invalid."""


class DataError(FootprintError):
    """Raised when tick data is malformed or fails validation."""


class PipelineError(FootprintError):
    """Raised when a processing pipeline encounters an unrecoverable state."""
