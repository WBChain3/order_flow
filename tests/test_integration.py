from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from footprint._config import FootprintConfig
from footprint.io._schema import TICK_SCHEMA, numpy_tick_dtype
from footprint.io._synthetic import SyntheticTickGenerator
from footprint.pipeline import run_pipeline


class TestIntegration:
    @pytest.fixture
    def config(self) -> FootprintConfig:
        return FootprintConfig()

    @pytest.fixture
    def synthetic_parquet(self, config: FootprintConfig, tmp_path: Path) -> Path:
        gen = SyntheticTickGenerator(config, seed=42)
        ticks = gen.generate_candle()
        table = pa.table({
            "timestamp_ns": pa.array(ticks["timestamp_ns"], type=pa.int64()),
            "price": pa.array(ticks["price"], type=pa.float64()),
            "size": pa.array(ticks["size"], type=pa.int32()),
            "side": pa.array(ticks["side"], type=pa.int8()),
            "sequence": pa.array(ticks["sequence"], type=pa.int64()),
            "symbol": pa.array(ticks["symbol"].astype(str), type=pa.string()),
        }, schema=TICK_SCHEMA)
        path = tmp_path / "ticks.parquet"
        pq.write_table(table, path)
        return path

    def test_end_to_end(self, config: FootprintConfig, synthetic_parquet: Path, tmp_path: Path) -> None:
        output_dir = tmp_path / "output"
        footprints_path, dataset_path = run_pipeline(
            tick_path=str(synthetic_parquet),
            output_dir=str(output_dir),
            config=config,
        )
        assert footprints_path.exists()
        assert dataset_path.exists()

        footprints = np.load(footprints_path, allow_pickle=False)
        assert footprints.shape == (4, 64, 64)
        assert footprints.dtype == np.float32

        data = np.load(dataset_path, allow_pickle=False)
        assert "samples" in data
        assert "weights" in data
        assert data["samples"].shape[0] >= 1
        assert data["samples"].shape[-3:] == (4, 64, 64)

    def test_cli_help(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "footprint.pipeline", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--input" in result.stdout
        assert "--output" in result.stdout

    def test_cli_basic_run(self, config: FootprintConfig, synthetic_parquet: Path, tmp_path: Path) -> None:
        output_dir = tmp_path / "cli_output"
        result = subprocess.run(
            [
                sys.executable, "-m", "footprint.pipeline",
                "--input", str(synthetic_parquet),
                "--output", str(output_dir),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert (output_dir / "footprints.npy").exists()
        assert (output_dir / "dataset.npz").exists()
