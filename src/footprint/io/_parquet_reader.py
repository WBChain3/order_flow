from __future__ import annotations

from pathlib import Path
from typing import Generator

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from footprint._exceptions import DataError
from footprint.io._schema import TICK_SCHEMA, numpy_tick_dtype, validate_tick_schema


class ParquetTickReader:
    """Reads tick data from a Parquet file into structured NumPy arrays."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        if not self._path.exists():
            raise FileNotFoundError(f"Tick parquet not found: {self._path}")
        self._parquet_file = pq.ParquetFile(self._path)
        self._validate_schema()
        self._num_rows: int | None = None

    def _validate_schema(self) -> None:
        schema = self._parquet_file.schema_arrow
        validate_tick_schema(pa.schema(schema))

    def read_all(self) -> np.ndarray:
        table = self._parquet_file.read()
        return self._table_to_array(table)

    def read_chunked(self, chunk_size: int = 500_000) -> Generator[np.ndarray, None, None]:
        for batch in self._parquet_file.iter_batches(batch_size=chunk_size):
            yield self._table_to_array(pa.Table.from_batches([batch]))

    @property
    def num_rows(self) -> int:
        if self._num_rows is None:
            self._num_rows = self._parquet_file.metadata.num_rows
        return self._num_rows

    @property
    def schema(self) -> pa.schema:
        return self._parquet_file.schema_arrow

    @staticmethod
    def _table_to_array(table: pa.Table) -> np.ndarray:
        dtype = numpy_tick_dtype()
        n = len(table)
        arr = np.empty(n, dtype=dtype)
        arr["timestamp_ns"] = table.column("timestamp_ns").to_numpy()
        arr["price"] = table.column("price").to_numpy()
        arr["size"] = table.column("size").to_numpy()
        arr["side"] = table.column("side").to_numpy()
        seq_col = table.column("sequence")
        arr["sequence"] = np.where(seq_col.is_null(), 0, seq_col.to_numpy()).astype("i8")
        arr["symbol"] = table.column("symbol").to_numpy().astype("U10")
        return arr
