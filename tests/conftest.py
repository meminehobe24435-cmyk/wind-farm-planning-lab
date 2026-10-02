"""pytest 公共配置：把仓库根目录加入 sys.path，保证 ``import windlab`` 可用。

同时统一设置随机种子与工作目录，让测试可复现。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _fixed_seed() -> None:
    """每个测试前后固定 numpy 全局随机种子，避免偶发失败。"""
    np.random.seed(20240501)
    yield
    np.random.seed(20240501)


@pytest.fixture()
def repo_root() -> Path:
    """仓库根目录。"""
    return ROOT


@pytest.fixture()
def example_curve_fixture():
    """内置示例功率曲线（5.0 MW / 132 m）。"""
    from windlab.power_curve import example_curve
    return example_curve()


@pytest.fixture()
def example_resource_fixture():
    """内置示例风资源。"""
    from windlab.resource import example_wind_resource
    return example_wind_resource()


@pytest.fixture()
def example_site_fixture():
    """3.6 km × 2.4 km 矩形示例场址。"""
    from windlab.layout import Site
    return Site.rectangle(3600.0, 2400.0, name="测试场址")
