from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from footprint._config import FootprintConfig
from footprint._exceptions import DataError
from footprint.io._parquet_reader import ParquetTickReader
from footprint.io._schema import TICK_SCHEMA, numpy_tick_dtype


class TestParquetTickReader:
    @pytest.fixture
    def synthetic_ticks(self) -> np.ndarray:
        dtype = numpy_tick_dtype()
        n = 100
        arr = np.empty(n, dtype=dtype)
        arr["timestamp_ns"] = np.arange(0, n * 1_000_000, 1_000_000, dtype="i8")
        arr["price"] = 20000.0 + np.random.default_rng(42).normal(0, 2.0, n).round(2)
        arr["size"] = np.random.default_rng(42).poisson(5, n).clip(1).astype("i4")
        arr["side"] = np.random.default_rng(42).choice([-1, 0, 1], n).astype("i1")
        arr["sequence"] = np.arange(1, n + 1, dtype="i8")
        arr["symbol"] = "NQF1"
        return arr

    def _write_parquet(self, path, table):
        pq.write_table(table, path)

    @pytest.fixture
    def parquet_path(self, synthetic_ticks, tmp_path):
        path = tmp_path / "ticks.parquet"
        table = pa.table({
            "timestamp_ns": pa.array(synthetic_ticks["timestamp_ns"], type=pa.int64()),
            "price": pa.array(synthetic_ticks["price"], type=pa.float64()),
            "size": pa.array(synthetic_ticks["size"], type=pa.int32()),
            "side": pa.array(synthetic_ticks["side"], type=pa.int8()),
            "sequence": pa.array(synthetic_ticks["sequence"], type=pa.int64()),
            "symbol": pa.array(synthetic_ticks["symbol"].astype(str), type=pa.string()),
        }, schema=TICK_SCHEMA)
        self._write_parquet(path, table)
        return path

    def test_roundtrip(self, parquet_path, synthetic_ticks):
        reader = ParquetTickReader(parquet_path)
        result = reader.read_all()
        assert len(result) == len(synthetic_ticks)
        assert np.array_equal(result["timestamp_ns"], synthetic_ticks["timestamp_ns"])
        assert np.allclose(result["price"], synthetic_ticks["price"])
        assert np.array_equal(result["size"], synthetic_ticks["size"])
        assert np.array_equal(result["side"], synthetic_ticks["side"])
        assert np.array_equal(result["sequence"], synthetic_ticks["sequence"])

    def test_num_rows(self, parquet_path):
        reader = ParquetTickReader(parquet_path)
        assert reader.num_rows == 100

    def test_rejects_missing_column(self, synthetic_ticks, tmp_path):
        path = tmp_path / "bad.parquet"
        table = pa.table({
            "timestamp_ns": pa.array(synthetic_ticks["timestamp_ns"], type=pa.int64()),
            "price": pa.array(synthetic_ticks["price"], type=pa.float64()),
            "size": pa.array(synthetic_ticks["size"], type=pa.int32()),
        })
        self._write_parquet(path, table)
        with pytest.raises(DataError, match="Schema field mismatch"):
            ParquetTickReader(path)

    def test_rejects_wrong_dtype(self, tmp_path):
        """Write parquet with wrong column types using a different schema."""
        path = tmp_path / "bad_dtype.parquet"
        wrong_schema = pa.schema([
            pa.field("timestamp_ns", pa.float64(), nullable=False),
            pa.field("price", pa.float64(), nullable=False),
            pa.field("size", pa.int32(), nullable=False),
            pa.field("side", pa.int8(), nullable=False),
            pa.field("sequence", pa.int64(), nullable=True),
            pa.field("symbol", pa.string(), nullable=False),
        ])
        n = 10
        table = pa.table({
            "timestamp_ns": pa.array(np.arange(n, dtype="f8"), type=pa.float64()),
            "price": pa.array(np.full(n, 20000.0), type=pa.float64()),
            "size": pa.array(np.full(n, 5, dtype="i4"), type=pa.int32()),
            "side": pa.array(np.full(n, 1, dtype="i1"), type=pa.int8()),
            "sequence": pa.array(np.arange(1, n + 1, dtype="i8"), type=pa.int64()),
            "symbol": pa.array(["NQF1"] * n, type=pa.string()),
        }, schema=wrong_schema)
        self._write_parquet(path, table)
        with pytest.raises(DataError, match="expected type"):
            ParquetTickReader(path)

    def test_chunked_matches_full(self, parquet_path, synthetic_ticks):
        reader = ParquetTickReader(parquet_path)
        full = reader.read_all()
        chunks = list(reader.read_chunked(chunk_size=30))
        combined = np.concatenate(chunks)
        assert len(combined) == len(full)
        assert np.array_equal(combined["timestamp_ns"], full["timestamp_ns"])

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            ParquetTickReader("/nonexistent/path.parquet")

    def test_read_all_on_empty_file(self, tmp_path):
        path = tmp_path / "empty.parquet"
        table = pa.table({
            "timestamp_ns": pa.array([], type=pa.int64()),
            "price": pa.array([], type=pa.float64()),
            "size": pa.array([], type=pa.int32()),
            "side": pa.array([], type=pa.int8()),
            "sequence": pa.array([], type=pa.int64()),
            "symbol": pa.array([], type=pa.string()),
        }, schema=TICK_SCHEMA)
        self._write_parquet(path, table)
        reader = ParquetTickReader(path)
        result = reader.read_all()
        assert len(result) == 0
