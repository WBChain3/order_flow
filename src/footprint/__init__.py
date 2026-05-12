"""Footprint chart tensor construction for order-flow analysis.

Public API:
  FootprintConfig         — immutable pipeline configuration
  FootprintPipeline       — batch tick processor → (4, 64, 64) array
  DatasetFactory          — sliding-window dataset from footprint arrays
  ParquetTickReader       — parquet tick data loader
  SyntheticTickGenerator  — deterministic test data generator
  run_pipeline            — CLI entry point

Internal modules (src/footprint/_*.py) are not part of the public contract.
"""

from footprint._config import FootprintConfig
from footprint._version import __version__
from footprint._exceptions import FootprintError, ConfigError, DataError, PipelineError
from footprint.engine._pipeline import FootprintPipeline
from footprint.dataset._factory import DatasetFactory
from footprint.io._parquet_reader import ParquetTickReader
from footprint.io._synthetic import SyntheticTickGenerator
from footprint.pipeline import run_pipeline

__all__ = [
    "FootprintConfig",
    "FootprintPipeline",
    "DatasetFactory",
    "ParquetTickReader",
    "SyntheticTickGenerator",
    "run_pipeline",
    "__version__",
    "FootprintError",
    "ConfigError",
    "DataError",
    "PipelineError",
]
