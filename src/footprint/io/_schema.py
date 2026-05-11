from __future__ import annotations

import pyarrow as pa

TICK_SCHEMA = pa.schema([
    pa.field("timestamp_ns", pa.int64(), nullable=False),
    pa.field("price", pa.float64(), nullable=False),
    pa.field("size", pa.int32(), nullable=False),
    pa.field("side", pa.int8(), nullable=False),
    pa.field("sequence", pa.int64(), nullable=True),
    pa.field("symbol", pa.string(), nullable=False),
])

TICK_NAMES = [f.name for f in TICK_SCHEMA]
TICK_TYPES = [f.type for f in TICK_SCHEMA]


def numpy_tick_dtype() -> np.dtype:
    import numpy as np
    return np.dtype([
        ("timestamp_ns", "i8"),
        ("price", "f8"),
        ("size", "i4"),
        ("side", "i1"),
        ("sequence", "i8"),
        ("symbol", "U10"),
    ])


def validate_tick_schema(schema: pa.Schema) -> bool:
    from footprint._exceptions import DataError

    names = schema.names
    expected_names = TICK_NAMES
    if names != expected_names:
        raise DataError(
            f"Schema field mismatch: expected {expected_names}, got {names}"
        )
    for field in schema:
        expected_type = dict(zip(TICK_NAMES, TICK_TYPES))[field.name]
        if field.type != expected_type:
            raise DataError(
                f"Field '{field.name}': expected type {expected_type}, got {field.type}"
            )
    return True
