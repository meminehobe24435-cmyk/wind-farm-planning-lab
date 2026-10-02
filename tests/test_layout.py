"""排布模块测试：场址几何、约束检查（锚点 8）、排布生成与修复。"""

from __future__ import annotations

import math

import numpy as np
import pytest

from windlab.layout import (
    LayoutReport,
    Site,
    check_layout,
    convex_hull,
    footprint_area,
    grid_layout,
    layout_summary,
    min_pairwise_spacing,
    random_layout,
    repair_layout,
    staggered_layout,
)

D = 100.0


# ---------------------------------------------------------------- 场址几何


def test_rectangle_area_and_centroid() -> None:
    """矩形场址面积与形心。"""
    s = Site.rectangle(2000.0, 1000.0)
    assert abs(s.area - 2e6) < 1e-6
    assert abs(s.area_km2 - 2.0) < 1e-9
    c = s.centroid
    assert abs(c[0] - 1000.0) < 1e-9
    assert abs(c[1] - 500.0) < 1e-9


def test_bbox_and_local_extent() -> None:
    """外接矩形与沿主风向的投影跨度。"""
    s = Site.rectangle(2000.0, 1000.0)
    assert s.bbox == (0.0, 0.0, 2000.0, 1000.0)
    l_main, l_cross = s.local_extent(270.0)      # 主风向为西风 -> 主方向为东（x）
    assert abs(l_main - 2000.0) < 1e-6
    assert abs(l_cross - 1000.0) < 1e-6
    l_main2, l_cross2 = s.local_extent(0.0)      # 主风向为北风 -> 主方向为南（-y）
    assert abs(l_main2 - 1000.0) < 1e-6
    assert abs(l_cross2 - 2000.0) < 1e-6


def test_polygon_orientation_normalized() -> None:
    """顺时针输入的多边形会被自动整理成逆时针（面积为正）。"""
    cw = np.array([[0.0, 0.0], [0.0, 1000.0], [2000.0, 1000.0], [2000.0, 0.0]])
    s = Site(cw)
    assert Site.signed_area(s.polygon) > 0
    assert abs(s.area - 2e6) < 1e-6


def test_contains_inside_outside_and_tolerance() -> None:
    """点在多边形内外判定 + 边界容差。"""
    s = Site.rectangle(2000.0, 1000.0)
    pts = np.array([[500.0, 500.0], [-10.0, 500.0], [2500.0, 500.0], [1000.0, 1500.0]])
    inside = s.contains(pts)
    assert bool(inside[0]) is True
    assert bool(inside[1]) is False
    assert bool(inside[2]) is False
    assert bool(inside[3]) is False
    # 容差 50 m：边界外 10 m 的点算作在内
    assert bool(s.contains(np.array([-10.0, 500.0]), tolerance=50.0)[0]) is True
    assert bool(s.contains(np.array([-100.0, 500.0]), tolerance=50.0)[0]) is False


def test_distance_to_boundary() -> None:
    """点到边界距离。"""
    s = Site.rectangle(2000.0, 1000.0)
    d = s.distance_to_boundary(np.array([[1000.0, 500.0], [1000.0, 10.0], [1000.0, 900.0]]))
    assert abs(d[0] - 500.0) < 1e-6
    assert abs(d[1] - 10.0) < 1e-6
    assert abs(d[2] - 100.0) < 1e-6


def test_site_validation() -> None:
    with pytest.raises(ValueError):
        Site(np.array([[0.0, 0.0], [1.0, 1.0]]))


# ---------------------------------------------------------------- 凸包与占地


def test_convex_hull_and_footprint() -> None:
    """凸包面积：4 个点构成 1000×1000 正方形，内部点不改变结果。"""
    pts = np.array([[0.0, 0.0], [1000.0, 0.0], [1000.0, 1000.0], [0.0, 1000.0], [500.0, 500.0]])
    hull = convex_hull(pts)
    assert hull.shape[0] == 4
    assert abs(footprint_area(pts) - 1e6) < 1e-6
    assert footprint_area(np.array([[0.0, 0.0], [10.0, 0.0]])) == 0.0


def test_min_pairwise_spacing() -> None:
    pts = np.array([[0.0, 0.0], [300.0, 0.0], [0.0, 400.0]])
    assert abs(min_pairwise_spacing(pts) - 300.0) < 1e-9
    assert min_pairwise_spacing(np.array([[0.0, 0.0]])) == float("inf")


# ---------------------------------------------------------------- 约束检查（锚点 8）


def test_check_layout_flags_too_close_pair() -> None:
    """锚点 8：两台机位仅相距 2D，必须被判为违反最小间距。"""
    s = Site.rectangle(2000.0, 2000.0)
    pos = np.array([[500.0, 500.0], [700.0, 500.0]])       # 相距 2D = 200 m
    rep = check_layout(s, pos, D, min_spacing_main=5.0, min_spacing_cross=3.0,
                       main_dir_deg=270.0)
    assert rep.ok is False
    assert rep.n_violations == 1
    assert rep.n_pairs == 1
    assert rep.min_pair_distance == pytest.approx(200.0)
    assert rep.min_pair_distance_d == pytest.approx(2.0)
    assert "不通过" in rep.summary()


def test_check_layout_passes_valid_spacing() -> None:
    """沿主风向 5D、横风向 3D 的排布应通过。"""
    s = Site.rectangle(2000.0, 2000.0)
    pos = np.array([[500.0, 500.0], [500.0 + 5 * D, 500.0], [500.0, 500.0 + 3 * D],
                    [500.0 + 5 * D, 500.0 + 3 * D]])
    rep = check_layout(s, pos, D, main_dir_deg=270.0)
    assert rep.ok is True
    assert rep.n_violations == 0
    assert rep.min_pair_distance_d > 2.9
    assert "通过" in rep.summary()


def test_ellipse_exclusion_zone_semantics() -> None:
    """椭圆禁区判据：(Δmain/5D)² + (Δcross/3D)² >= 1。"""
    s = Site.rectangle(4000.0, 4000.0)
    # 沿主风向 5D：恰好满足
    ok = np.array([[1000.0, 1000.0], [1000.0 + 5 * D, 1000.0]])
    assert check_layout(s, ok, D, main_dir_deg=270.0).ok is True
    # 沿主风向 3D：不满足（要 5D）
    bad = np.array([[1000.0, 1000.0], [1000.0 + 3 * D, 1000.0]])
    assert check_layout(s, bad, D, main_dir_deg=270.0).ok is False
    # 沿横风向 3D：满足
    cross = np.array([[1000.0, 1000.0], [1000.0, 1000.0 + 3 * D]])
    assert check_layout(s, cross, D, main_dir_deg=270.0).ok is True
    # 斜向 (3D, 2D)：椭圆值 = 0.36 + 0.444 = 0.804 < 1 -> 不满足
    diag = np.array([[1000.0, 1000.0], [1000.0 + 3 * D, 1000.0 + 2 * D]])
    assert check_layout(s, diag, D, main_dir_deg=270.0).ok is False


def test_check_layout_flags_outside_turbines() -> None:
    """越界机位被识别（锚点 8 的另一半）。"""
    s = Site.rectangle(1000.0, 1000.0)
    pos = np.array([[500.0, 500.0], [1500.0, 500.0]])
    rep = check_layout(s, pos, D, main_dir_deg=270.0)
    assert rep.ok is False
    assert rep.outside_indices == [1]
    assert "越界" in rep.summary()


def test_check_layout_absolute_minimum_spacing() -> None:
    """横风向拉开但直线距离小于 2D 时仍违规。"""
    s = Site.rectangle(2000.0, 2000.0)
    pos = np.array([[1000.0, 1000.0], [1000.0 + 3 * D, 1000.0], [1000.0 + 1.5 * D, 1000.0]])
    rep = check_layout(s, pos, D, main_dir_deg=270.0)
    assert rep.ok is False
    assert rep.n_violations >= 1


def test_check_layout_empty_and_single() -> None:
    """单台机组没有机位对，报告应通过且最小值为占位值。"""
    s = Site.rectangle(1000.0, 1000.0)
    rep = check_layout(s, np.array([[500.0, 500.0]]), D, main_dir_deg=270.0)
    assert rep.n_pairs == 0
    assert rep.ok is True
    assert rep.n_turbines == 1


# ---------------------------------------------------------------- 排布生成


def test_grid_layout_spacing_and_count() -> None:
    """规则网格：行列数与间距正确。"""
    s = Site.rectangle(4000.0, 3000.0)
    pos = grid_layout(s, 3, 4, 5 * D, 3 * D, main_dir_deg=270.0, margin=100.0)
    assert pos.shape == (12, 2)
    assert np.all(s.contains(pos))
    rep = check_layout(s, pos, D, main_dir_deg=270.0)
    assert rep.ok is True
    # 主风向（东西）相邻列间距为 5D
    xs = np.unique(np.round(pos[:, 0], 6))
    assert len(xs) == 4
    assert np.allclose(np.diff(xs), 5 * D)
    ys = np.unique(np.round(pos[:, 1], 6))
    assert len(ys) == 3
    assert np.allclose(np.diff(ys), 3 * D)


def test_grid_layout_raises_when_outside() -> None:
    """网格超出场址时报错（不静默丢点）。"""
    s = Site.rectangle(800.0, 800.0)
    with pytest.raises(ValueError):
        grid_layout(s, 3, 4, 5 * D, 3 * D, main_dir_deg=270.0, margin=50.0)
    with pytest.raises(ValueError):
        grid_layout(s, 0, 4, 5 * D, 3 * D)


def test_staggered_layout_offsets_alternate_lines() -> None:
    """错列排布：沿主风向奇数位的机组在横风向上偏移 0.5×3D，主风向间距仍为 5D。"""
    s = Site.rectangle(6000.0, 3000.0)
    pos = staggered_layout(s, 4, 4, 5 * D, 3 * D, main_dir_deg=270.0, stagger=0.5)
    assert pos.shape[0] == 16
    xs = np.unique(np.round(pos[:, 0], 6))
    assert len(xs) == 4, "主风向（东西）位置应保持 4 个"
    assert np.allclose(np.diff(xs), 5 * D), "主风向间距应为 5D"
    # 同一横风向位置的相邻两列，y 相差 0.5×3D（相邻实例横向错开）
    y00 = pos[np.isclose(pos[:, 0], xs[0])][0, 1]
    y10 = pos[np.isclose(pos[:, 0], xs[1])][0, 1]
    assert abs(abs(y10 - y00) - 1.5 * D) < 1e-6
    rep = check_layout(s, pos, D, main_dir_deg=270.0)
    assert rep.ok is True


def test_staggered_layout_lower_mainwind_wake() -> None:
    """错列应打断主风向的串内对齐，显著降低主流向（西风）的尾流损失。"""
    from windlab.aep import evaluate_aep
    from windlab.power_curve import example_curve
    from windlab.resource import N_SECTORS, WindResource, WindRose
    from windlab.wake import WakeModel

    curve = example_curve()
    d = curve.diameter
    s = Site.rectangle(6000.0, 3000.0)
    grid = grid_layout(s, 4, 4, 5 * d, 3 * d, main_dir_deg=270.0)
    stag = staggered_layout(s, 4, 4, 5 * d, 3 * d, main_dir_deg=270.0, stagger=0.5)
    f = np.zeros(N_SECTORS)
    f[12] = 1.0                      # 只有西风（主风向）
    res = WindResource(rose=WindRose(freqs=f, mean_speeds=np.full(N_SECTORS, 8.0)), k=2.0)
    model = WakeModel(diameter=d, k_wake=0.075, rotor_grid=10)
    a = evaluate_aep(grid, res, curve, model, bin_width=4.0)
    b = evaluate_aep(stag, res, curve, model, bin_width=4.0)
    assert a.wake_loss_pct > 10.0, "规则网格在主流向上应有明显尾流"
    assert b.wake_loss_pct < a.wake_loss_pct - 5.0, \
        "错列未降低主流向尾流：网格 %.2f%%，错列 %.2f%%" % (a.wake_loss_pct, b.wake_loss_pct)


def test_stagger_validation() -> None:
    s = Site.rectangle(3000.0, 3000.0)
    with pytest.raises(ValueError):
        staggered_layout(s, 2, 2, 5 * D, 3 * D, stagger=1.5)


def test_random_layout_satisfies_constraints() -> None:
    """随机排布必须满足最小间距与边界约束。"""
    s = Site.rectangle(4000.0, 3000.0)
    rng = np.random.default_rng(3)
    pos = random_layout(s, 12, D, rng, main_dir_deg=270.0)
    assert pos.shape == (12, 2)
    rep = check_layout(s, pos, D, main_dir_deg=270.0)
    assert rep.ok is True
    assert min_pairwise_spacing(pos) >= 3 * D - 1e-6


def test_random_layout_reproducible_and_seed_sensitive() -> None:
    """同种子结果一致，不同种子不同。"""
    s = Site.rectangle(4000.0, 3000.0)
    a = random_layout(s, 8, D, np.random.default_rng(5), main_dir_deg=270.0)
    b = random_layout(s, 8, D, np.random.default_rng(5), main_dir_deg=270.0)
    c = random_layout(s, 8, D, np.random.default_rng(6), main_dir_deg=270.0)
    assert np.allclose(a, b)
    assert not np.allclose(a, c)


def test_random_layout_fails_on_impossible_request() -> None:
    """场址太小 / 台数太多时抛错或（strict=False）返回部分结果。"""
    s = Site.rectangle(300.0, 300.0)
    with pytest.raises(RuntimeError):
        random_layout(s, 20, D, np.random.default_rng(0), max_attempts=500)
    part = random_layout(s, 20, D, np.random.default_rng(0), max_attempts=500, strict=False)
    assert 0 < part.shape[0] < 20


# ---------------------------------------------------------------- 约束修复


def test_repair_layout_fixes_violations() -> None:
    """把违规排布修复成可行排布。"""
    s = Site.rectangle(4000.0, 3000.0)
    bad = np.array([[1000.0, 1000.0], [1050.0, 1000.0], [1100.0, 1000.0], [1150.0, 1000.0]])
    assert check_layout(s, bad, D, main_dir_deg=270.0).ok is False
    fixed = repair_layout(s, bad, D, main_dir_deg=270.0, iterations=400)
    rep = check_layout(s, fixed, D, main_dir_deg=270.0)
    assert rep.ok is True, rep.summary()


def test_repair_layout_keeps_points_inside() -> None:
    """修复过程会把跑出场址的机位拉回场内。"""
    s = Site.rectangle(1000.0, 1000.0)
    bad = np.array([[-500.0, 500.0], [1500.0, 500.0], [500.0, -900.0]])
    fixed = repair_layout(s, bad, D, main_dir_deg=270.0, iterations=100)
    assert np.all(s.contains(fixed))


def test_repair_layout_leaves_valid_layout_unchanged() -> None:
    """已经可行的排布不应被改动。"""
    s = Site.rectangle(4000.0, 3000.0)
    pos = np.array([[1000.0, 1000.0], [1000.0 + 5 * D, 1000.0], [1000.0, 1000.0 + 3 * D]])
    fixed = repair_layout(s, pos, D, main_dir_deg=270.0)
    assert np.allclose(fixed, pos)


def test_layout_summary_text() -> None:
    """排布摘要包含台数、最小间距与占地。"""
    s = Site.rectangle(4000.0, 3000.0)
    pos = grid_layout(s, 3, 4, 5 * D, 3 * D, main_dir_deg=270.0, margin=100.0)
    text = layout_summary(pos, D)
    assert "12 台机组" in text
    assert "最小间距" in text
    assert "km²" in text
    assert layout_summary(np.zeros((0, 2)), D) == "空排布"


def test_report_dataclass_defaults() -> None:
    """LayoutReport 默认值不会导致误判。"""
    rep = LayoutReport(ok=True, n_turbines=0, n_pairs=0, n_violations=0)
    assert "排布为空" in rep.summary()
