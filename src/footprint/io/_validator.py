"""Validates and filters tick data before engine ingestion.

Price outliers are detected via rolling 1s VWAP window (not global),
because a single candle can contain regime shifts where a global std
would be too wide to catch subtle outliers.

Zero-size trades are warned, not rejected — they may indicate legitimate
cancelled or adjusted trades from the exchange.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from footprint._config import FootprintConfig


class TickValidator:
    """Validates and filters tick data before engine ingestion.

    Price outliers are detected via rolling 1s VWAP window (not global).
    """

    def __init__(self, config: FootprintConfig, clock_skew_window_ns: int = 10_000_000) -> None:
        self._config = config
        self._clock_skew_window_ns = clock_skew_window_ns
        self._rejection_log: list[dict[str, Any]] = []

    def validate(self, ticks: np.ndarray) -> np.ndarray:
        self._rejection_log.clear()
        mask = np.ones(len(ticks), dtype=bool)

        mask &= self._check_price_outliers(ticks)
        mask &= self._check_zero_size(ticks)
        mask &= self._check_out_of_sequence(ticks)
        mask &= self._check_duplicate_sequence(ticks)

        return ticks[mask]

    @property
    def rejection_log(self) -> list[dict[str, Any]]:
        return list(self._rejection_log)

    def _check_price_outliers(self, ticks: np.ndarray) -> np.ndarray:
        if len(ticks) < 10:
            return np.ones(len(ticks), dtype=bool)

        window_ns = 1_000_000_000
        ts = ticks["timestamp_ns"]
        prices = ticks["price"]
        sizes = ticks["size"].astype("f8")

        mask = np.ones(len(ticks), dtype=bool)
        left = 0
        # Two-pointer sliding window: maintain [left, right-1] of ticks within 1s of current tick.
        for right in range(len(ticks)):
            while ts[right] - ts[left] > window_ns:
                left += 1

            if right - left < 5:
                continue

            left_idx = left
            right_idx = right

            if right_idx > left_idx:
                pre_left = left_idx
                pre_right = right_idx - 1
            else:
                continue

            pre_count = pre_right - pre_left + 1
            if pre_count < 5:
                continue

            # Exclude the current tick from its own window so VWAP is purely historical.
            window_prices = prices[pre_left:pre_right + 1]
            window_sizes = sizes[pre_left:pre_right + 1]
            size_sum = window_sizes.sum()
            if size_sum == 0:
                continue
            vwap = np.average(window_prices, weights=window_sizes)
            variance = np.average((window_prices - vwap) ** 2, weights=window_sizes)
            std = np.sqrt(variance)
            if std == 0:
                continue
            threshold = 5.0 * std
            if abs(prices[right] - vwap) > threshold:
                mask[right] = False
                # Cap rejection log at 100 entries per reason to prevent memory bloat on corrupt files.
                if len(self._rejection_log) < 100:
                    self._rejection_log.append({
                        "reason": "price_outlier",
                        "type": "reject",
                        "tick_index": int(right),
                        "price": float(prices[right]),
                        "vwap": float(vwap),
                        "std": float(std),
                        "timestamp_ns": int(ts[right]),
                    })
        return mask

    def _check_zero_size(self, ticks: np.ndarray) -> np.ndarray:
        zero_mask = ticks["size"] <= 0
        rejected = np.where(zero_mask)[0]
        for idx in rejected[:100]:
            self._rejection_log.append({
                "reason": "zero_size",
                "type": "warn",
                "tick_index": int(idx),
                "size": int(ticks[idx]["size"]),
                "timestamp_ns": int(ticks[idx]["timestamp_ns"]),
            })
        mask = np.ones(len(ticks), dtype=bool)
        return mask

    def _check_out_of_sequence(self, ticks: np.ndarray) -> np.ndarray:
        if len(ticks) < 2:
            return np.ones(len(ticks), dtype=bool)

        diffs = np.diff(ticks["timestamp_ns"])
        bad = np.where(diffs < -self._clock_skew_window_ns)[0]
        mask = np.ones(len(ticks), dtype=bool)
        for idx in bad:
            self._rejection_log.append({
                "reason": "out_of_sequence",
                "type": "reject",
                "tick_index": int(idx + 1),
                "timestamp_ns": int(ticks[idx + 1]["timestamp_ns"]),
                "prev_timestamp_ns": int(ticks[idx]["timestamp_ns"]),
            })
            mask[idx + 1] = False
        return mask

    def _check_duplicate_sequence(self, ticks: np.ndarray) -> np.ndarray:
        if len(ticks) < 2:
            return np.ones(len(ticks), dtype=bool)
        seqs = ticks["sequence"]
        unique, counts = np.unique(seqs[seqs != 0], return_counts=True)
        dupe_seqs = unique[counts > 1]
        for seq in dupe_seqs[:100]:
            indices = np.where(seqs == seq)[0]
            self._rejection_log.append({
                "reason": "duplicate_sequence",
                "type": "warn",
                "sequence": int(seq),
                "tick_indices": indices.tolist(),
            })
        return np.ones(len(ticks), dtype=bool)
