from __future__ import annotations

import numpy as np
import pytest

from footprint._config import FootprintConfig
from footprint.engine._price_grid import PriceGrid


class TestPriceGrid:
    @pytest.fixture
    def config(self) -> FootprintConfig:
        return FootprintConfig()

    def test_centering(self, config: FootprintConfig) -> None:
        grid = PriceGrid(config, mid_price=20000.0)
        level_32_price = grid.price_at_level(32)
        level_31_price = grid.price_at_level(31)
        level_33_price = grid.price_at_level(33)
        assert abs(level_32_price - 20000.0) <= config.tick_size
        assert level_31_price < 20000.0
        assert level_33_price > 20000.0

    def test_min_max_price_default(self, config: FootprintConfig) -> None:
        grid = PriceGrid(config, mid_price=20000.0)
        half_range = config.price_range_ticks // 2
        expected_min = 20000.0 - half_range * config.tick_size
        expected_max = 20000.0 + half_range * config.tick_size - config.tick_size
        assert grid.min_price == pytest.approx(expected_min)
        assert grid.max_price == pytest.approx(expected_max)

    def test_min_max_price_wider_range(self) -> None:
        cfg = FootprintConfig(price_range_ticks=128, price_levels=64)
        grid = PriceGrid(cfg, mid_price=20000.0)
        half_range = 64
        expected_min = 20000.0 - half_range * cfg.tick_size
        expected_max = 20000.0 + half_range * cfg.tick_size - cfg.tick_size
        assert grid.min_price == pytest.approx(expected_min)
        assert grid.max_price == pytest.approx(expected_max)

    def test_level_assignment_default(self, config: FootprintConfig) -> None:
        grid = PriceGrid(config, mid_price=20000.0)
        prices = np.array([
            grid.min_price,
            grid.min_price + config.tick_size * 0.5,
            grid.min_price + config.tick_size * 2.0,
            grid.max_price,
        ])
        levels = grid.assign_levels(prices)
        assert levels[0] == 0
        assert levels[1] == 0
        assert levels[2] == 2
        assert levels[3] == config.price_levels - 1

    def test_level_assignment_wider_range(self) -> None:
        cfg = FootprintConfig(price_range_ticks=128, price_levels=64)
        grid = PriceGrid(cfg, mid_price=20000.0)
        level_size_ticks = cfg.price_range_ticks / cfg.price_levels
        level_size_price = level_size_ticks * cfg.tick_size
        prices = np.array([
            grid.min_price,
            grid.min_price + level_size_price * 0.5,
            grid.min_price + level_size_price * 2.0,
            grid.max_price,
        ])
        levels = grid.assign_levels(prices)
        assert levels[0] == 0
        assert levels[1] == 0
        assert levels[2] == 2
        assert levels[3] == 63

    def test_clamps_outside_range(self, config: FootprintConfig) -> None:
        grid = PriceGrid(config, mid_price=20000.0)
        prices = np.array([grid.min_price - 100.0, grid.max_price + 100.0])
        levels = grid.assign_levels(prices)
        assert levels[0] == 0
        assert levels[1] == config.price_levels - 1

    def test_price_at_level_consistency(self, config: FootprintConfig) -> None:
        grid = PriceGrid(config, mid_price=20000.0)
        for level in [0, 15, 31, 32, 48, 63]:
            price = grid.price_at_level(level)
            assigned = grid.assign_levels(np.array([price]))
            assert assigned[0] == level

    def test_level_prices_length(self, config: FootprintConfig) -> None:
        grid = PriceGrid(config, mid_price=20000.0)
        assert len(grid.level_prices) == config.price_levels
