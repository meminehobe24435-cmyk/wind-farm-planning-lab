"""尾流模块测试：Jensen 解析解、单调性、平方和叠加、部分遮挡、风向依赖性。

这些是本项目最关键的物理正确性锚点。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from windlab.wake import (
    DEFAULT_WAKE_DECAY,
    WakeModel,
    direction_from_vector,
    effective_speeds,
    jensen_deficit,
    loss_matrix,
    wake_loss_at_direction,
    wake_radius,
    wind_unit_vector,
)

D = 100.0
K = 0.075


def analytic_deficit(x: float, ct: float, d: float = D, k: float = K) -> float:
    """独立重写一遍解析式，用于和被测代码比对（避免自我引用）。"""
    return (1.0 - math.sqrt(1.0 - ct)) * (d / (d + 2.0 * k * x)) ** 2


# ---------------------------------------------------------------- 解析解（锚点 4）


def test_jensen_deficit_hand_computed() -> None:
    """锚点 4：手算 x=500 m、Ct=0.8、D=100 m、k=0.075 处的亏损。

    1 - sqrt(1-0.8) = 1 - 0.4472136 = 0.5527864
    100 / (100 + 2*0.075*500) = 100/175 = 0.5714286
    deficit = 0.5527864 * 0.5714286^2 = 0.1805017
    """
    got = jensen_deficit(500.0, 0.8, D, K)
    assert abs(got - 0.1805017) < 1e-6, "实测 %.8f" % got
    base = 1.0 - math.sqrt(0.2)
    assert abs(got - base * (100.0 / 175.0) ** 2) < 1e-12


@pytest.mark.parametrize("x", [100.0, 250.0, 500.0, 1000.0, 2000.0])
def test_jensen_matches_reference_formula(x: float) -> None:
    """与独立重写的解析式逐点一致。"""
    for ct in (0.2, 0.5, 0.8):
        assert abs(jensen_deficit(x, ct, D, K) - analytic_deficit(x, ct)) < 1e-12


def test_jensen_zero_or_negative_x() -> None:
    """x <= 0（风轮平面及其上游）亏损为 0。"""
    assert jensen_deficit(0.0, 0.8, D, K) == 0.0
    assert jensen_deficit(-100.0, 0.8, D, K) == 0.0
    assert np.all(jensen_deficit(np.array([-1.0, 0.0]), 0.8, D, K) == 0.0)


def test_jensen_zero_ct_no_deficit() -> None:
    """Ct=0 表示不吸收动量，不产生尾流。"""
    assert jensen_deficit(500.0, 0.0, D, K) == 0.0


def test_jensen_zero_wake_decay_is_cylinder() -> None:
    """k=0 时尾流不扩张，亏损不随距离衰减（圆柱尾流）。"""
    a = jensen_deficit(100.0, 0.8, D, 0.0)
    b = jensen_deficit(5000.0, 0.8, D, 0.0)
    assert abs(a - b) < 1e-12
    assert abs(a - (1.0 - math.sqrt(0.2))) < 1e-12


def test_jensen_validation() -> None:
    with pytest.raises(ValueError):
        jensen_deficit(100.0, 0.8, 0.0, K)
    with pytest.raises(ValueError):
        jensen_deficit(100.0, 0.8, D, -0.1)


# ---------------------------------------------------------------- 单调性（锚点 5）


def test_deficit_decreases_with_downstream_distance() -> None:
    """锚点 5：下游距离越大，亏损越小（严格单调递减）。"""
    x = np.array([50.0, 100.0, 200.0, 400.0, 800.0, 1600.0, 3200.0])
    d = jensen_deficit(x, 0.8, D, K)
    assert np.all(np.diff(d) < 0.0)
    assert d[0] > d[-1] * 5.0


def test_deficit_increases_with_ct() -> None:
    """锚点 5：推力系数越大，亏损越大。"""
    cts = np.array([0.1, 0.3, 0.5, 0.7, 0.85])
    d = jensen_deficit(500.0, cts, D, K)
    assert np.all(np.diff(d) > 0.0)
    assert d[-1] > d[0]


def test_deficit_increases_with_wake_decay() -> None:
    """尾流衰减系数越小，尾流越"硬"，近场亏损越大。"""
    d_small = jensen_deficit(500.0, 0.8, D, 0.04)
    d_large = jensen_deficit(500.0, 0.8, D, 0.12)
    assert d_small > d_large


def test_wake_radius_grows_linearly() -> None:
    """尾流半径 R(x) = D/2 + k x。"""
    assert abs(wake_radius(0.0, D, K) - 50.0) < 1e-12
    assert abs(wake_radius(1000.0, D, K) - (50.0 + 75.0)) < 1e-12
    assert abs(wake_radius(-10.0, D, K) - 50.0) < 1e-12


# ---------------------------------------------------------------- 风向约定


def test_wind_unit_vector_convention() -> None:
    """风向 -> 下游方向矢量：北风向南、东风向西、西风向东。"""
    u_n, _ = wind_unit_vector(0.0)
    assert abs(u_n[0]) < 1e-12 and abs(u_n[1] + 1.0) < 1e-12
    u_e, _ = wind_unit_vector(90.0)
    assert abs(u_e[0] + 1.0) < 1e-12 and abs(u_e[1]) < 1e-12
    u_w, _ = wind_unit_vector(270.0)
    assert abs(u_w[0] - 1.0) < 1e-12 and abs(u_w[1]) < 1e-12


def test_direction_roundtrip() -> None:
    """下游矢量与风向可互相还原。"""
    for deg in (0.0, 45.0, 135.0, 200.0, 270.0, 359.0):
        u, _ = wind_unit_vector(deg)
        assert abs(direction_from_vector(u) - deg) < 1e-6


def test_crosswind_vector_is_perpendicular() -> None:
    """横风向矢量与下游矢量正交且为单位矢量。"""
    for deg in (0.0, 37.0, 180.0, 271.5):
        u, n = wind_unit_vector(deg)
        assert abs(float(np.dot(u, n))) < 1e-12
        assert abs(float(np.linalg.norm(n)) - 1.0) < 1e-12
        assert abs(float(np.linalg.norm(u)) - 1.0) < 1e-12


# ---------------------------------------------------------------- 风轮离散


def test_rotor_points_inside_disk_and_count() -> None:
    """风轮离散点覆盖圆盘，数量与网格分辨率相符（20×20 -> 约 π/4·400 ≈ 316）。"""
    m = WakeModel(diameter=D, k_wake=K, rotor_grid=20)
    pts, w = m.rotor_samples()
    r = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
    cell = D / 20
    assert np.all(r <= D / 2 + cell)          # 允许边界单元中心略微出圈
    assert pts.shape[0] == m.rotor_point_count()
    assert 300 <= pts.shape[0] <= 340
    assert w.shape == (pts.shape[0],)
    assert abs(float(np.sum(w)) - 1.0) < 1e-12
    assert np.all(w > 0.0)
    coarse = WakeModel(diameter=D, k_wake=K, rotor_grid=8)
    assert coarse.rotor_point_count() < pts.shape[0]


def test_rotor_area_weights_are_accurate() -> None:
    """面积权重必须能还原圆盘的解析面积性质：加权平均半径 = 2R/3。"""
    for grid in (8, 12, 20, 28):
        m = WakeModel(diameter=D, k_wake=K, rotor_grid=grid)
        pts, w = m.rotor_samples()
        r = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
        mean_r = float(np.sum(w * r))
        assert abs(mean_r - 2.0 * (D / 2) / 3.0) / (D / 2) < 0.02, \
            "网格 %d 的加权平均半径 %.3f 偏离解析值 %.3f" % (grid, mean_r, 2 * (D / 2) / 3)


def test_rotor_grid_resolution_convergence() -> None:
    """不同风轮离散分辨率下的尾流损失应收敛（相邻分辨率差异 < 0.2 个百分点）。

    这是对「面积加权离散」的收敛性检查：未加权的等权网格采样会因为边界单元
    被过度代表而出现非单调抖动。
    """
    pos = np.array([[0.0, 0.0], [5.0 * D, 0.0], [10.0 * D, 0.0]])
    ct = 0.8
    losses = []
    for grid in (8, 12, 20, 28):
        m = WakeModel(diameter=D, k_wake=K, rotor_grid=grid)
        losses.append(float(np.max(m.loss_matrix(pos, 8.0, 270.0, ct))))
    spread = max(losses) - min(losses)
    assert spread < 0.002, "风轮离散未收敛，尾流损失跨度 %.4f：%s" % (spread, losses)


def test_array_aep_insensitive_to_rotor_grid() -> None:
    """整场 AEP 对风轮离散分辨率不敏感：6×6 ~ 28×28 的极差应小于净 AEP 的 1%。

    部分遮挡的面积估计精度受限于风轮离散的几何分辨率（圆盘边界上的单元宽度），
    因此 AEP 会有 ~0.3% 量级的抖动，且**不严格单调**。这个测试把抖动上界钉住，
    同时说明：方案之间的 AEP 差异（10% 量级）远大于该数值不确定度。
    """
    from windlab.aep import evaluate_aep
    from windlab.power_curve import example_curve
    from windlab.resource import example_wind_resource

    curve = example_curve()
    res = example_wind_resource()
    pos = np.array([[0.0, 0.0], [660.0, 0.0], [1320.0, 0.0], [0.0, 396.0], [660.0, 396.0]])
    values = []
    for grid in (6, 8, 12, 20, 28):
        m = WakeModel(diameter=curve.diameter, k_wake=K, rotor_grid=grid)
        values.append(evaluate_aep(pos, res, curve, m, bin_width=2.0).net_gwh)
    spread_rel = (max(values) - min(values)) / float(np.mean(values))
    assert spread_rel < 0.01, "风轮离散导致 AEP 抖动过大：%.4f%%（%s）" % (100 * spread_rel, values)


def test_rotor_grid_validation() -> None:
    with pytest.raises(ValueError):
        WakeModel(diameter=0.0)
    with pytest.raises(ValueError):
        WakeModel(diameter=D, k_wake=-1.0)
    with pytest.raises(ValueError):
        WakeModel(diameter=D, rotor_grid=1)


# ---------------------------------------------------------------- 损失矩阵


def test_loss_matrix_shape_and_diagonal() -> None:
    """损失矩阵对角为 0，非负。"""
    m = WakeModel(diameter=D, k_wake=K, rotor_grid=16)
    pos = np.array([[0.0, 0.0], [500.0, 0.0], [1000.0, 0.0]])
    mat = m.loss_matrix(pos, 8.0, 270.0, 0.8)
    assert mat.shape == (3, 3)
    assert np.allclose(np.diag(mat), 0.0)
    assert np.all(mat >= 0.0)


def test_aligned_pair_matches_analytic_deficit() -> None:
    """锚点 4（全场口径）：完全对风的两台机组，间距 5D 时下游机组的等效亏损
    应等于解析式在 x=5D 的值（整个风轮都在尾流锥内，且每点 sp 相同）。"""
    m = WakeModel(diameter=D, k_wake=K, rotor_grid=20)
    pos = np.array([[0.0, 0.0], [5.0 * D, 0.0]])
    ct = 0.8
    mat = m.loss_matrix(pos, 8.0, 270.0, ct)      # 西风 -> 向东吹
    expected = analytic_deficit(5.0 * D, ct)
    assert abs(mat[0, 1] - expected) < 1e-9, "%.8f vs %.8f" % (mat[0, 1], expected)
    assert mat[1, 0] == 0.0, "上游机组不应受下游影响"


def test_upwind_turbine_has_no_deficit() -> None:
    """逆风方向没有尾流。"""
    m = WakeModel(diameter=D, k_wake=K, rotor_grid=16)
    pos = np.array([[0.0, 0.0], [800.0, 0.0]])
    # 东风（风来自东，向东吹到 -x 方向）-> (0,0) 在下游
    mat = m.loss_matrix(pos, 8.0, 90.0, 0.8)
    assert mat[0, 0] == 0.0
    assert mat[1, 0] > 0.0
    assert mat[0, 1] == 0.0


def test_perpendicular_wind_no_wake() -> None:
    """锚点 9（对级别）：风向垂直于连线时没有尾流。"""
    m = WakeModel(diameter=D, k_wake=K, rotor_grid=16)
    pos = np.array([[0.0, 0.0], [1000.0, 0.0]])
    mat = m.loss_matrix(pos, 8.0, 0.0, 0.8)       # 北风 -> 向南吹
    assert float(np.max(mat)) == 0.0


def test_partial_overlap_between_full_and_none() -> None:
    """部分遮挡：横向偏移越大，等效亏损越小，最终为 0。

    5D 下游处尾流半径 = D/2 + k·5D = 50 + 37.5 = 87.5 m，风轮半径 50 m，
    因此横向偏移超过 137.5 m 后就完全没有遮挡。
    """
    m = WakeModel(diameter=D, k_wake=K, rotor_grid=20)
    ct = 0.8
    full = m.loss_matrix(np.array([[0.0, 0.0], [5.0 * D, 0.0]]), 8.0, 270.0, ct)[0, 1]
    half = m.loss_matrix(np.array([[0.0, 0.0], [5.0 * D, 0.75 * D]]), 8.0, 270.0, ct)[0, 1]
    edge = m.loss_matrix(np.array([[0.0, 0.0], [5.0 * D, 1.2 * D]]), 8.0, 270.0, ct)[0, 1]
    far = m.loss_matrix(np.array([[0.0, 0.0], [5.0 * D, 10.0 * D]]), 8.0, 270.0, ct)[0, 1]
    assert 0.0 < half < full
    assert 0.0 < edge < half
    assert far == 0.0


def test_partial_overlap_not_simplified_to_full_blockage() -> None:
    """明确验证没有退化成"整台机全遮挡"：偏移 1 倍半径时的亏损必须明显小于对齐工况。"""
    m = WakeModel(diameter=D, k_wake=K, rotor_grid=20)
    ct = 0.8
    full = m.loss_matrix(np.array([[0.0, 0.0], [5.0 * D, 0.0]]), 8.0, 270.0, ct)[0, 1]
    offset = m.loss_matrix(np.array([[0.0, 0.0], [5.0 * D, 1.0 * D]]), 8.0, 270.0, ct)[0, 1]
    assert offset < 0.5 * full


def test_deficit_decreases_with_downstream_spacing_in_field() -> None:
    """全场口径：间距越大，下游机组亏损越小。"""
    m = WakeModel(diameter=D, k_wake=K, rotor_grid=20)
    vals = []
    for s in (2.0, 3.0, 5.0, 8.0, 12.0):
        pos = np.array([[0.0, 0.0], [s * D, 0.0]])
        vals.append(float(m.loss_matrix(pos, 8.0, 270.0, 0.8)[0, 1]))
    assert all(vals[i] > vals[i + 1] for i in range(len(vals) - 1))
    assert vals[0] > 0.25


def test_loss_matrix_requires_ct() -> None:
    m = WakeModel(diameter=D, k_wake=K, rotor_grid=8)
    with pytest.raises(ValueError):
        m.loss_matrix(np.array([[0.0, 0.0], [500.0, 0.0]]), 8.0, 270.0, None)
    with pytest.raises(ValueError):
        m.effective_speeds(np.array([[0.0, 0.0], [500.0, 0.0]]), 8.0, 270.0, None)


# ---------------------------------------------------------------- 等效风速与叠加


def test_single_turbine_effective_speed_equals_free_stream() -> None:
    """单台机组没有尾流，等效风速等于自由来流。"""
    m = WakeModel(diameter=D, k_wake=K, rotor_grid=16)
    pos = np.array([[123.0, 456.0]])
    eff = m.effective_speeds(pos, 9.0, 270.0, 0.8)
    assert abs(float(eff[0]) - 9.0) < 1e-12


def test_aligned_chain_effective_speed_matches_analytic() -> None:
    """锚点 4/5（全场口径）：三台沿主风向等间距 5D 排列时的等效风速。"""
    m = WakeModel(diameter=D, k_wake=K, rotor_grid=20)
    ct = 0.8
    pos = np.array([[0.0, 0.0], [5.0 * D, 0.0], [10.0 * D, 0.0]])
    u0 = 8.0
    eff = m.effective_speeds(pos, u0, 270.0, ct)
    d1 = analytic_deficit(5.0 * D, ct)
    d2 = analytic_deficit(10.0 * D, ct)
    combined = math.sqrt(d1 ** 2 + d2 ** 2)      # 平方和叠加（Katic）
    assert abs(eff[0] - u0) < 1e-12
    assert abs(eff[1] - u0 * (1.0 - d1)) < 1e-9
    assert abs(eff[2] - u0 * (1.0 - combined)) < 1e-9
    # 线性叠加会明显更大，验证用的确实是平方和而不是简单相加
    assert abs(eff[2] - u0 * (1.0 - (d1 + d2))) > 0.05


def test_sum_of_squares_superposition_explicit() -> None:
    """直接验证 (1-u/u0)² = Σ(1-u_i/u0)²。"""
    m = WakeModel(diameter=D, k_wake=K, rotor_grid=12)
    ct = 0.8
    pair = np.array([[0.0, 0.0], [5.0 * D, 0.0]])
    d_single = float(m.loss_matrix(pair, 8.0, 270.0, ct)[0, 1])
    assert d_single > 0.0
    # 两台完全重合的上游机组（物理上不存在，但可用于验证叠加公式）
    two = np.array([[0.0, 0.0], [0.0, 0.0]])
    target = np.array([[5.0 * D, 0.0]])
    eff = m.effective_speeds(np.vstack([two, target]), 8.0, 270.0, ct)
    expected = 8.0 * (1.0 - math.sqrt(2.0) * d_single)
    assert abs(eff[2] - expected) < 1e-9
    # 线性叠加会得到明显更低的等效风速，证明用的是平方和
    linear = 8.0 * (1.0 - 2.0 * d_single)
    assert eff[2] > linear + 0.1


def test_effective_speed_bounded() -> None:
    """等效风速必须落在 (0, 自由来流]。"""
    m = WakeModel(diameter=D, k_wake=K, rotor_grid=12)
    pos = np.array([[0.0, 0.0], [1.2 * D, 0.0], [2.4 * D, 0.0]])
    eff = m.effective_speeds(pos, 10.0, 270.0, 0.85)
    assert np.all(eff <= 10.0 + 1e-12)
    assert np.all(eff > 0.0)


def test_zero_wind_speed_returns_zero() -> None:
    m = WakeModel(diameter=D, k_wake=K, rotor_grid=8)
    pos = np.array([[0.0, 0.0], [500.0, 0.0]])
    assert np.all(m.effective_speeds(pos, 0.0, 270.0, 0.8) == 0.0)


def test_empty_layout() -> None:
    m = WakeModel(diameter=D, k_wake=K, rotor_grid=8)
    assert m.effective_speeds(np.zeros((0, 2)), 8.0, 270.0, 0.8).shape == (0,)
    assert m.array_efficiency(np.zeros((0, 2)), 8.0, 270.0, 0.8) == 1.0


def test_array_efficiency_range() -> None:
    """整场速度效率在 (0,1]，排布越密越低；横向完全错开时为 1。"""
    m = WakeModel(diameter=D, k_wake=K, rotor_grid=12)
    dense = np.array([[0.0, 0.0], [2.0 * D, 0.0], [4.0 * D, 0.0]])
    # 横向拉开 20D，风向为西风（向东吹）时完全没有尾流遮挡
    sparse = np.array([[0.0, 0.0], [0.0, 20.0 * D], [0.0, 40.0 * D]])
    e_dense = m.array_efficiency(dense, 9.0, 270.0, 0.8)
    e_sparse = m.array_efficiency(sparse, 9.0, 270.0, 0.8)
    assert 0.0 < e_dense < e_sparse
    assert abs(e_sparse - 1.0) < 1e-9


def test_deficit_cap_prevents_negative_speed() -> None:
    """极端密集排布下等效风速仍为正（数值保护）。"""
    m = WakeModel(diameter=D, k_wake=K, rotor_grid=8, deficit_cap=0.99)
    pos = np.array([[0.0, 0.0], [0.05 * D, 0.0], [0.1 * D, 0.0]])
    eff = m.effective_speeds(pos, 10.0, 270.0, 0.9)
    assert np.all(eff >= 0.0)
    assert np.all(np.isfinite(eff))


def test_module_level_wrappers() -> None:
    """模块级便捷函数与类方法一致。"""
    m = WakeModel(diameter=D, k_wake=K, rotor_grid=8)
    pos = np.array([[0.0, 0.0], [500.0, 0.0]])
    assert np.allclose(effective_speeds(pos, 8.0, 270.0, 0.8, m),
                       m.effective_speeds(pos, 8.0, 270.0, 0.8))
    assert np.allclose(loss_matrix(pos, 8.0, 270.0, 0.8, m),
                       m.loss_matrix(pos, 8.0, 270.0, 0.8))


def test_wake_loss_at_direction_report() -> None:
    """单风向诊断字典的字段与取值范围。"""
    m = WakeModel(diameter=D, k_wake=K, rotor_grid=8)
    pos = np.array([[0.0, 0.0], [5.0 * D, 0.0]])
    rep = wake_loss_at_direction(pos, 8.0, 270.0, 0.8, m)
    assert set(rep.keys()) == {"speed_deficit_ratio", "u_min", "u_max", "u_mean"}
    assert 0.0 < rep["speed_deficit_ratio"] < 1.0
    assert abs(rep["u_max"] - 8.0) < 1e-9
    assert rep["u_min"] < rep["u_max"]
    assert abs(rep["u_mean"] - 0.5 * (rep["u_min"] + rep["u_max"])) < 1e-9


def test_default_wake_decay_is_onshore_typical() -> None:
    """默认尾流衰减系数取陆上常用的 0.075。"""
    assert abs(DEFAULT_WAKE_DECAY - 0.075) < 1e-12
