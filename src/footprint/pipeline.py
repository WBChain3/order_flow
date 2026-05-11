from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import structlog

from footprint._config import FootprintConfig
from footprint.engine._pipeline import FootprintPipeline
from footprint.io._parquet_reader import ParquetTickReader
from footprint.io._validator import TickValidator
from footprint.dataset._factory import DatasetFactory


logger = structlog.get_logger()


def run_pipeline(
    tick_path: str,
    output_dir: str,
    config: FootprintConfig | None = None,
) -> tuple[Path, Path]:
    if config is None:
        config = FootprintConfig()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    logger.info("Reading ticks", path=tick_path)
    reader = ParquetTickReader(tick_path)
    ticks = reader.read_all()
    logger.info("Read ticks", count=len(ticks))

    logger.info("Validating ticks")
    validator = TickValidator(config)
    ticks = validator.validate(ticks)
    if validator.rejection_log:
        logger.warning("Ticks rejected", count=len(validator.rejection_log))
    logger.info("Valid ticks remaining", count=len(ticks))

    logger.info("Building pipeline")
    pipeline = FootprintPipeline(config)

    logger.info("Building footprints")
    footprint = pipeline.process(ticks)
    footprints_path = out / "footprints.npy"
    np.save(footprints_path, footprint)
    logger.info("Footprints saved", path=str(footprints_path))

    logger.info("Creating dataset")
    factory = DatasetFactory(config)
    samples, weights = factory.create(
        np.expand_dims(footprint, axis=0), stride=1, lookback=1
    )
    dataset_path = out / "dataset.npz"
    np.savez(dataset_path, samples=samples, weights=weights)
    logger.info("Dataset saved", path=str(dataset_path))

    logger.info("Pipeline complete")
    return footprints_path, dataset_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build footprint tensors from tick data parquet files."
    )
    parser.add_argument("--input", required=True, help="Path to input parquet tick file")
    parser.add_argument("--output", required=True, help="Path to output directory")
    parser.add_argument("--tick-size", type=float, default=None, help="Instrument tick size")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config override")
    args = parser.parse_args()

    config = FootprintConfig()
    if args.tick_size is not None:
        config = FootprintConfig(tick_size=args.tick_size)
    if args.config is not None:
        print("Warning: --config is not yet implemented. Ignoring provided path.", file=sys.stderr)

    run_pipeline(tick_path=args.input, output_dir=args.output, config=config)


if __name__ == "__main__":
    main()
