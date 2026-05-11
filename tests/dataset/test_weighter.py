from __future__ import annotations

import numpy as np

from footprint.dataset._weighter import SampleWeighter


class TestSampleWeighter:
    def test_returns_one(self) -> None:
        weighter = SampleWeighter()
        sample = np.random.default_rng(42).random((4, 64, 64)).astype(np.float32)
        assert weighter(sample) == 1.0

    def test_always_one(self) -> None:
        weighter = SampleWeighter()
        sample1 = np.zeros((4, 64, 64), dtype=np.float32)
        sample2 = np.ones((4, 64, 64), dtype=np.float32) * 100.0
        assert weighter(sample1) == 1.0
        assert weighter(sample2) == 1.0
