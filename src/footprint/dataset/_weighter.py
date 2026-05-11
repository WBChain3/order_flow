from __future__ import annotations

import numpy as np


class SampleWeighter:
    """Interface placeholder for sample weighting.

    Always returns 1.0 (uniform weighting).
    Private repo will override with curriculum-based weighting.
    """

    def __call__(self, sample: np.ndarray) -> float:
        return 1.0
