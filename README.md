# order-flow

A fast, config-driven toolkit for building volume footprint charts from tick-level market data.

---

## What it does

`order-flow` turns raw trade and order-book tick data into fixed-shape, normalized tensor arrays suitable for quantitative research and downstream modeling.

The public API is intentionally narrow: **a config object goes in, a `numpy` array comes out.**

- **Data ingestion:** adapters for real-time tick feeds, L2 snapshots, and deterministic historical replay
- **Footprint engine:** time-bucketed volume matrices with bid/ask/delta/total channels
- **Dataset factory:** sliding-window extraction, optional geometric/temporal transforms, unlabeled output
- **Zero bundled data:** BYOD — bring your own historical data in Parquet or binary format

---

## Quick start

### Installation

```bash
pip install order-flow
```

Requires Python >= 3.11 and NumPy.

### Basic usage

```python
from footprint import FootprintConfig, FootprintPipeline

config = FootprintConfig(
    candle_duration_seconds=60,
    price_levels=64,
    time_buckets=64,
    tick_size=0.25,          # NQ E-mini
    price_range_ticks=64,
    normalization="per_candle_minmax",
)

pipeline = FootprintPipeline(config)

# Pass a structured NumPy array of ticks (see Input data format below)
footprint = pipeline.process(ticks)   # shape: (4, 64, 64), dtype float32
```

### Channels

Each channel is a 64×64 grid: **price levels (y-axis) × time buckets (x-axis)** within the candle window.

| Channel | Content |
|---------|---------|
| 0 | Bid volume per price × time cell |
| 1 | Ask volume per price × time cell |
| 2 | Delta (ask − bid) per cell |
| 3 | Total volume per cell |

---

## Design philosophy

- **Deterministic replay:** same input always produces the same array
- **Latency-conscious:** hot paths use pre-allocated NumPy buffers; Pandas is not used in the construction loop
- **Schema-stable:** output shape and dtype are fixed; only the config object changes
- **Unlabeled by design:** this library produces raw arrays. Labeling, modeling, and inference are out of scope

---

## Input data format

The replay engine and tick adapters expect normalized tick data with the following schema:

| Column | Type | Description |
|--------|------|-------------|
| `timestamp_ns` | int64 | nanosecond epoch UTC |
| `price`         | float64 | trade price |
| `size`          | int32   | contract quantity |
| `side`          | int8    | `1` = buy/initiator, `-1` = sell/initiator, `0` = unknown |
| `sequence`      | int64   | exchange sequence number (nullable) |
| `symbol`        | string  | e.g. `"NQH5"` |

Supply your own historical data as Parquet or a packed binary format. No sample data is bundled.

---

## Development

```bash
git clone https://github.com/YOUR_USERNAME/order-flow.git
cd order-flow
pip install -e ".[dev]"
pytest
```

---

## License

MIT
