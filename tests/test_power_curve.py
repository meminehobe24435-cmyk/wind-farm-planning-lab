"""功率曲线模块测试：边界条件、单调性、PCHIP 保形插值、容量系数、CSV 导入导出。"""

from __future__ import annotations

import math

import numpy as np
import pytest

from windlab.power_curve import (
    PowerCurve,
    builtin_curves,
    default_ct_table,
    example_curve,
    load_curve_csv,
    pchip_eval,
    pchip_slopes,
    save_curve_csv,
    synthetic_curve,
    trapezoid,
)


# ---------------------------------------------------------------- 示例曲线形态


def test_example_curve_parameters() -> None:
    """内置示例曲线的关键参数（明确标注为构造数据）。"""
    c = example_curve()
    assert c.rated_power == 5000.0
    assert c.diameter == 132.0
    assert c.cut_in == 3.0
    assert c.cut_out == 25.0
    assert abs(float(c.rated_speed) - 11.0) < 0.2, "额定风速应约为 11 m/s"
    assert "示例曲线" in c.source
    assert "非厂商实测" in c.source


def test_builtin_curves_three_types() -> None:
    """内置三条示例曲线，比功率（W/m²）越低额定风速越低。"""
    curves = builtin_curves()
    assert len(curves) == 3
    sp = [c.specific_power for c in curves]
    rs = [float(c.rated_speed) for c in curves]
    order = np.argsort(sp)
    assert np.all(np.diff(np.array(rs)[order]) > 0), "比功率越低额定风速应越低"
    for c in curves:
        assert all("示例曲线" in c.source for c in curves)


def test_specific_power_value() -> None:
    """单位扫风面积装机容量 = P_rated / (πR²)。"""
    c = example_curve()
    expected = 5000.0 * 1000.0 / (math.pi * 66.0 ** 2)
    assert abs(c.specific_power - expected) < 1e-9
    assert abs(c.specific_power - 365.4) < 0.5


# ---------------------------------------------------------------- 边界条件（锚点 3）


def test_power_zero_below_cut_in() -> None:
    """锚点 3：切入风速以下功率为 0。"""
    c = example_curve()
    assert c.power(0.0) == 0.0
    assert c.power(1.0) == 0.0
    assert c.power(2.999) == 0.0
    assert c.power(c.cut_in) == 0.0
    assert np.all(c.power(np.array([0.0, 1.0, 2.5, 2.99])) == 0.0)


def test_power_zero_at_and_above_cut_out() -> None:
    """锚点 3：切出风速及以上功率为 0（机组顺桨停机）。"""
    c = example_curve()
    assert c.power(25.0) == 0.0
    assert c.power(25.001) == 0.0
    assert c.power(30.0) == 0.0
    assert c.power(100.0) == 0.0
    assert np.all(c.power(np.array([25.0, 26.0, 40.0])) == 0.0)


def test_power_just_below_cut_out_is_rated() -> None:
    """切出风速前一档仍在额定功率上。"""
    c = example_curve()
    assert abs(c.power(24.9) - c.rated_power) < 1e-6
    assert abs(c.power(20.0) - c.rated_power) < 1e-6


def test_rated_plateau_is_constant() -> None:
    """锚点 3：额定段（额定风速 ~ 切出风速）功率恒为额定值，一点不差。"""
    c = example_curve()
    v = np.linspace(float(c.rated_speed), c.cut_out - 1e-9, 200)
    p = c.power(v)
    assert np.allclose(p, c.rated_power, atol=1e-9)
    assert abs(float(np.max(p)) - c.rated_power) < 1e-9
    # 额定风速就是功率表里第一个达到额定功率的风速
    assert abs(float(c.rated_speed) - 10.985) < 0.01


def test_power_monotone_non_decreasing() -> None:
    """功率曲线在 [0, 切出) 上单调不减（物理上不可能出现负斜率）。

    切出风速处会「断崖式」跳到 0，那是机组顺桨停机，属于模型边界而非曲线缺陷，
    因此单调性只在 [0, 切出) 上要求（下面单独验证切出跳变）。
    """
    c = example_curve()
    v = np.linspace(0.0, c.cut_out - 1e-9, 3000)
    p = c.power(v)
    assert np.all(np.diff(p) >= -1e-9), "功率曲线出现下降段"
    assert float(np.max(p)) <= c.rated_power + 1e-9
    # 切出跳变：24.999 m/s 满发，25 m/s 停机
    assert c.power(c.cut_out - 1e-3) == pytest.approx(c.rated_power)
    assert c.power(c.cut_out) == 0.0


def test_power_negative_speed_is_zero() -> None:
    c = example_curve()
    assert c.power(-3.0) == 0.0
    assert np.all(c.power(np.array([-1.0, -10.0])) == 0.0)


def test_power_scalar_and_array_agree() -> None:
    """标量与数组调用结果一致。"""
    c = example_curve()
    v = np.array([2.0, 4.0, 6.0, 8.0, 11.0, 15.0, 25.0])
    arr = c.power(v)
    for i, vv in enumerate(v):
        assert abs(c.power(float(vv)) - arr[i]) < 1e-12


def test_power_smooth_in_transition_region() -> None:
    """过渡段平滑：相邻 0.05 m/s 之间的功率跳跃不应超过额定功率的 3%。"""
    c = example_curve()
    v = np.arange(c.cut_in, float(c.rated_speed) + 1e-9, 0.05)
    d = np.diff(c.power(v))
    assert float(np.max(d)) < 0.03 * c.rated_power, "过渡段存在阶梯"


# ---------------------------------------------------------------- PCHIP


def test_pchip_interpolates_nodes_exactly() -> None:
    """PCHIP 在节点上取给定值。"""
    x = np.array([0.0, 1.0, 2.0, 4.0, 7.0])
    y = np.array([1.0, 3.0, 2.5, 5.0, 4.0])
    out = pchip_eval(x, y, x)
    assert np.allclose(out, y)


def test_pchip_does_not_overshoot() -> None:
    """保形插值不应超出节点值的范围（单调数据不产生过冲）。"""
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = np.array([0.0, 0.0, 1.0, 5.0, 5.0])
    xq = np.linspace(0.0, 4.0, 2000)
    out = pchip_eval(x, y, xq)
    assert float(np.min(out)) >= -1e-9
    assert float(np.max(out)) <= 5.0 + 1e-9


def test_pchip_monotone_on_monotone_data() -> None:
    """单调数据的 PCHIP 结果单调。"""
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([0.0, 0.1, 1.0, 2.0, 4.0, 4.0])
    xq = np.linspace(0.0, 5.0, 1000)
    out = pchip_eval(x, y, xq)
    assert np.all(np.diff(out) >= -1e-9)


def test_pchip_slopes_zero_at_local_extrema() -> None:
    """局部极值点处斜率应为 0（避免过冲）。"""
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([0.0, 2.0, 2.0, 0.0])
    m = pchip_slopes(x, y)
    assert abs(m[1]) < 1e-12
    assert abs(m[2]) < 1e-12


def test_pchip_requires_increasing_x() -> None:
    with pytest.raises(ValueError):
        pchip_slopes(np.array([0.0, 0.0, 1.0]), np.array([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError):
        pchip_eval(np.array([0.0]), np.array([1.0]), np.array([0.5]))


def test_dense_table_matches_ideal_cubic_below_rated() -> None:
    """示例曲线在额定风速以下应等于 0.5·ρ·A·Cp·v³（额定上限内）。"""
    c = example_curve()
    area = math.pi * (c.diameter / 2.0) ** 2
    for v in (5.0, 6.0, 7.0, 8.0, 9.0, 10.0):
        ideal = 0.5 * 1.225 * area * 0.45 * v ** 3 / 1000.0
        assert abs(c.power(v) - ideal) < 0.02 * ideal, "v=%.1f 偏差过大" % v


# ---------------------------------------------------------------- 推力系数


def test_ct_boundaries() -> None:
    """切入以下、切出及以上 Ct 为 0；区间内为正且不超过 0.95。"""
    c = example_curve()
    assert c.ct(2.0) == 0.0
    assert c.ct(3.0) == 0.0
    assert c.ct(25.0) == 0.0
    assert c.ct(30.0) == 0.0
    for v in np.linspace(4.0, 24.0, 60):
        assert 0.0 < float(c.ct(v)) <= 0.95


def test_ct_decreases_above_rated() -> None:
    """额定风速以上变桨限功率，Ct 单调下降。"""
    c = example_curve()
    v = np.linspace(float(c.rated_speed) + 0.5, c.cut_out - 0.5, 40)
    ct = c.ct(v)
    assert np.all(np.diff(ct) <= 1e-9)
    assert float(c.ct(24.0)) < 0.3
    assert float(c.ct(12.0)) < float(c.ct(10.0))


def test_default_ct_table_validates_order() -> None:
    with pytest.raises(ValueError):
        default_ct_table(10.0, 5.0, 25.0)
    v, ct = default_ct_table(3.0, 11.0, 25.0)
    assert v.shape == ct.shape
    assert np.all(np.diff(v) > 0)
    assert np.all((ct >= 0.0) & (ct <= 0.95))


# ---------------------------------------------------------------- 容量系数


def test_capacity_factor_range_and_monotonicity() -> None:
    """容量系数在 (0,1)，且随 c 增大而增大。"""
    c = example_curve()
    cf1 = c.capacity_factor(2.0, 6.0)
    cf2 = c.capacity_factor(2.0, 8.0)
    cf3 = c.capacity_factor(2.0, 10.0)
    assert 0.0 < cf1 < cf2 < cf3 < 1.0
    assert abs(cf2 - 0.3647) < 0.005, "k=2/c=8 时示例曲线毛容量系数约 36.5%%（实算 %.4f）" % cf2


def test_capacity_factor_matches_mean_power() -> None:
    """容量系数 = 期望功率 / 额定功率。"""
    c = example_curve()
    mp = c.mean_power(2.0, 8.0)
    assert abs(c.capacity_factor(2.0, 8.0) - mp / c.rated_power) < 1e-12


def test_average_power_density() -> None:
    """单位扫风面积年均电功率 = 期望功率 / 扫风面积。"""
    c = example_curve()
    mp = c.mean_power(2.0, 8.0)
    expected = mp * 1000.0 / c.rotor_area
    assert abs(c.average_power_density(2.0, 8.0) - expected) < 1e-9
    assert 100.0 < c.average_power_density(2.0, 8.0) < 250.0


def test_trapezoid_matches_analytic() -> None:
    """自写梯形积分：∫x²dx 在 [0,3] 上 = 9。"""
    x = np.linspace(0.0, 3.0, 200001)
    assert abs(trapezoid(x ** 2, x) - 9.0) < 1e-4
    with pytest.raises(ValueError):
        trapezoid(np.array([1.0]), np.array([1.0]))


# ---------------------------------------------------------------- 曲线校验与 CSV


def test_curve_rejects_invalid_tables() -> None:
    """功率表降序 / 超额定 / 长度不一致都应报错。"""
    with pytest.raises(ValueError):
        PowerCurve("bad", np.array([0.0, 2.0, 1.0]), np.array([0.0, 100.0, 200.0]),
                   cut_in=3.0, cut_out=25.0, rated_power=5000.0, diameter=132.0)
    with pytest.raises(ValueError):
        PowerCurve("bad", np.array([3.0, 25.0]), np.array([0.0, 6000.0]),
                   cut_in=3.0, cut_out=25.0, rated_power=5000.0, diameter=132.0)
    with pytest.raises(ValueError):
        PowerCurve("bad", np.array([3.0, 4.0]), np.array([0.0]),
                   cut_in=3.0, cut_out=25.0, rated_power=5000.0, diameter=132.0)
    with pytest.raises(ValueError):
        PowerCurve("bad", np.array([3.0, 4.0]), np.array([0.0, 100.0]),
                   cut_in=30.0, cut_out=25.0, rated_power=5000.0, diameter=132.0)


def test_synthetic_curve_rejects_inconsistent_params() -> None:
    """参数不自洽（Cp 推出的额定风速超出区间）应报错。"""
    with pytest.raises(ValueError):
        synthetic_curve("bad", 5000.0, 30.0, cut_in=3.0, cut_out=25.0, cp_max=0.45)


def test_curve_csv_roundtrip(tmp_path) -> None:
    """CSV 导出再导入后，功率插值结果基本不变。"""
    c = example_curve()
    path = str(tmp_path / "curve.csv")
    save_curve_csv(c, path)
    back = load_curve_csv(path, name="back", diameter=c.diameter,
                          rated_power=c.rated_power, cut_in=c.cut_in, cut_out=c.cut_out)
    v = np.linspace(0.0, 25.0, 250)
    assert np.max(np.abs(back.power(v) - c.power(v))) < 1.0
    assert back.rated_power == c.rated_power
    assert abs(back.specific_power - c.specific_power) < 1e-9


def test_load_curve_csv_skips_comments_and_header(tmp_path) -> None:
    """CSV 里的注释行与表头行被跳过。"""
    p = tmp_path / "t.csv"
    p.write_text("# 注释\n风速,功率\n0,0\n3,0\n5,500\n8,2000\n11,5000\n25,5000\n",
                 encoding="utf-8")
    c = load_curve_csv(str(p), diameter=100.0)
    assert c.power(5.0) == 500.0
    assert abs(c.power(8.0) - 2000.0) < 1e-9
    assert c.rated_power == 5000.0


def test_load_curve_csv_errors(tmp_path) -> None:
    """空文件报错。"""
    p = tmp_path / "empty.csv"
    p.write_text("# 只有注释\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_curve_csv(str(p))


def test_curve_summary_mentions_source() -> None:
    """摘要应包含来源说明（诚实标注）。"""
    c = example_curve()
    assert "示例曲线" in c.summary()
    assert "kW" in c.summary()
