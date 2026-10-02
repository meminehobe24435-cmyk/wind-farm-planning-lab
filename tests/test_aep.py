"""发电量模块测试：无尾流期望值（锚点 6）、分仓一致性、折减系数、风向影响（锚点 9）。"""

from __future__ import annotations

import math

import numpy as np
import pytest

from windlab.aep import (
    AEPResult,
    HOURS_PER_YEAR,
    compare_with_without_wake,
    directional_wake_loss,
    evaluate_aep,
    histogram_bins,
    power_speed,
    weibull_energy_bins,
)
from windlab.power_curve import example_curve, trapezoid
from windlab.resource import (
    N_SECTORS,
    WindResource,
    WindRose,
    c_from_mean,
    example_wind_resource,
    weibull_pdf,
)
from windlab.wake import WakeModel

D = 132.0


def single_sector_resource(index: int, mean_speed: float = 8.0, k: float = 2.0) -> WindResource:
    """只有一个扇区有风的测试用风资源。"""
    f = np.zeros(N_SECTORS)
    f[index] = 1.0
    rose = WindRose(freqs=f, mean_speeds=np.full(N_SECTORS, mean_speed), name="单扇区测试")
    return WindResource(rose=rose, k=k, name="单扇区测试场地")


def delta_bins(speeds_probs, n_sectors: int = N_SECTORS):
    """构造「只在给定风速上取到概率」的分仓（用于手算场景）。"""
    out = []
    for i in range(n_sectors):
        if i == 0 and speeds_probs:
            out.append((np.array([v for v, _ in speeds_probs], dtype=np.float64),
                        np.array([p for _, p in speeds_probs], dtype=np.float64)))
        else:
            out.append((np.array([0.0]), np.array([0.0])))
    return out


# ---------------------------------------------------------------- 分仓


def test_energy_bins_probability_and_range() -> None:
    """分仓概率和为 1（尾部扣掉切出以上），代表风速落在 [0, 切出] 内。"""
    c = example_curve()
    speeds, probs = weibull_energy_bins(2.0, 8.5, c, bin_width=1.0)
    assert speeds.shape == probs.shape
    assert 0.99 < float(np.sum(probs)) <= 1.0
    assert float(np.min(speeds)) >= 0.0
    assert float(np.max(speeds)) <= c.cut_out + 1e-9
    assert np.all(probs > 0.0)


def test_energy_bins_representative_speed_is_energy_equivalent() -> None:
    """每个仓的代表风速满足 P(v_rep) = 该仓的条件期望功率。"""
    c = example_curve()
    k, cc = 2.0, 8.0
    bin_width = 2.0
    speeds, probs = weibull_energy_bins(k, cc, c, bin_width=bin_width)
    for b in range(len(speeds)):
        lo = b * bin_width
        hi = min(lo + bin_width, c.cut_out)
        v = np.linspace(lo, hi, 501)
        f = weibull_pdf(v, k, cc)
        e_cond = trapezoid(c.power_integrand(v) * f, v) / float(probs[b])
        # 用功率曲线反查代表风速：P(v_rep) 必须还原该仓的条件期望功率。
        # 误差最大出现在跨切入风速的低风速仓（那里功率曲线最陡），实测 < 0.2%。
        assert abs(float(c.power(speeds[b])) - e_cond) <= 5e-3 * e_cond + 1e-9, \
            "第 %d 仓不满足能量等效" % b


def test_energy_bins_requires_positive_width() -> None:
    with pytest.raises(ValueError):
        weibull_energy_bins(2.0, 8.0, example_curve(), bin_width=0.0)


def test_power_speed_inverse() -> None:
    """功率反查风速：与功率曲线互逆。"""
    c = example_curve()
    for v in (4.0, 6.0, 8.0, 10.0):
        p = float(c.power(v))
        assert abs(power_speed(c, p) - v) < 0.02
    assert power_speed(c, 0.0) == 0.0
    assert abs(power_speed(c, c.rated_power * 2) - float(c.rated_speed)) < 1e-9


def test_histogram_bins_from_measured_data() -> None:
    """实测风频直方图可直接转成计算分仓。"""
    c = example_curve()
    centers = np.array([0.5, 1.5, 2.5, 3.5, 5.5, 7.5, 9.5, 11.5])
    freqs = np.array([0.02, 0.04, 0.06, 0.08, 0.20, 0.25, 0.20, 0.15])
    speeds, probs = histogram_bins(centers, freqs, c)
    assert abs(float(np.sum(probs)) - 1.0) < 1e-12
    assert len(speeds) == len(centers)
    with pytest.raises(ValueError):
        histogram_bins(centers, np.zeros_like(freqs), c)
    with pytest.raises(ValueError):
        histogram_bins(centers, freqs[:3], c)


# ---------------------------------------------------------------- 手算锚点（锚点 6）


def test_gross_aep_hand_computable() -> None:
    """锚点 6：风全部集中在 8 m/s 时，无尾流毛发电量 = N · P(8) · 8760。

    这是一个可以完全手算的场景：单扇区、单一风速、无尾流、可用率与其他折减都取 1。
    """
    c = example_curve()
    n = 10
    pos = np.zeros((n, 2))
    pos[:, 0] = np.arange(n) * 1000.0
    res = single_sector_resource(0, mean_speed=8.0)
    bins = delta_bins([(8.0, 1.0)])

    out = evaluate_aep(pos, res, c, None, availability=1.0, other_loss_factor=1.0,
                       include_wake=False, bins_by_sector=bins)
    expected_gwh = float(c.power(8.0)) * n * HOURS_PER_YEAR / 1e6
    assert abs(out.gross_gwh - expected_gwh) < 1e-9
    assert abs(out.net_gwh - expected_gwh) < 1e-9
    assert out.wake_loss_pct == 0.0
    assert abs(out.net_capacity_factor - expected_gwh * 1e6 / (n * c.rated_power * HOURS_PER_YEAR)) < 1e-12
    # 手算数值：P(8) = 1931.2 kW -> 10 台 -> 169.2 GWh
    assert abs(out.net_gwh - 169.2) < 0.5


def test_no_wake_equals_wind_frequency_expectation() -> None:
    """锚点 6（分布版）：无尾流毛发电量 = N·8760·Σ_s f_s·E[P | 扇区 s]，
    其中 E[P] 用细网格独立数值积分得到。"""
    c = example_curve()
    res = example_wind_resource()
    n = 6
    pos = np.zeros((n, 2))
    pos[:, 0] = np.arange(n) * 700.0

    out = evaluate_aep(pos, res, c, None, availability=1.0, other_loss_factor=1.0,
                       include_wake=False, bin_width=1.0)

    v = np.linspace(0.0, c.cut_out, 40001)
    p = c.power(v)
    manual = 0.0
    for s in range(N_SECTORS):
        f_s = float(res.rose.freqs[s])
        if f_s <= 0:
            continue
        k_s, c_s = res.sector_weibull(s)
        manual += f_s * trapezoid(p * weibull_pdf(v, k_s, c_s), v)
    expected_gwh = manual * n * HOURS_PER_YEAR / 1e6
    assert abs(out.gross_gwh - expected_gwh) / expected_gwh < 0.005, \
        "%.4f vs %.4f" % (out.gross_gwh, expected_gwh)


def test_coarse_bins_match_fine_bins() -> None:
    """能耗等效分仓的威力：4 m/s 粗分仓与 0.5 m/s 细分仓结果差异应 < 1%。"""
    c = example_curve()
    res = example_wind_resource()
    pos = np.zeros((5, 2))
    pos[:, 0] = np.arange(5) * 800.0
    fine = evaluate_aep(pos, res, c, None, include_wake=False, bin_width=0.5)
    coarse = evaluate_aep(pos, res, c, None, include_wake=False, bin_width=4.0)
    assert abs(coarse.gross_gwh - fine.gross_gwh) / fine.gross_gwh < 0.01


# ---------------------------------------------------------------- 尾流与折减


def test_wake_reduces_aep() -> None:
    """计入尾流后发电量必然下降（锚点 6 的对照）。"""
    c = example_curve()
    res = example_wind_resource()
    pos = np.zeros((6, 2))
    pos[:, 0] = np.arange(6) * 5 * D
    model = WakeModel(diameter=D, k_wake=0.075, rotor_grid=16)
    with_wake = evaluate_aep(pos, res, c, model, bin_width=2.0)
    no_wake = evaluate_aep(pos, res, c, None, include_wake=False, bin_width=2.0)
    assert with_wake.wake_gwh < no_wake.gross_gwh
    assert abs(with_wake.gross_gwh - no_wake.gross_gwh) < 1e-9
    assert with_wake.wake_loss_pct > 5.0
    assert with_wake.wake_loss_pct < 60.0


def test_availability_and_other_loss_scaling() -> None:
    """净发电量 = 尾流后发电量 × 可用率 × 其他折减系数。"""
    c = example_curve()
    res = example_wind_resource()
    pos = np.zeros((4, 2))
    pos[:, 0] = np.arange(4) * 900.0
    model = WakeModel(diameter=D, k_wake=0.075, rotor_grid=10)
    out = evaluate_aep(pos, res, c, model, availability=0.95, other_loss_factor=0.90,
                       bin_width=2.0)
    assert abs(out.net_gwh - out.wake_gwh * 0.95 * 0.90) < 1e-9
    assert abs(out.other_loss_pct - 10.0) < 1e-9
    assert out.total_loss_pct > out.wake_loss_pct


def test_per_turbine_energy_sums_to_array_energy() -> None:
    """单机发电量之和 = 尾流后发电量。"""
    c = example_curve()
    res = example_wind_resource()
    pos = np.array([[0.0, 0.0], [660.0, 0.0], [1320.0, 0.0], [0.0, 396.0]])
    model = WakeModel(diameter=D, k_wake=0.075, rotor_grid=10)
    out = evaluate_aep(pos, res, c, model, bin_width=2.0)
    assert out.per_turbine_gwh.shape == (4,)
    assert abs(float(np.sum(out.per_turbine_gwh)) - out.wake_gwh) < 1e-6
    assert float(out.per_turbine_gwh[0]) > float(out.per_turbine_gwh[2])


def test_no_wake_per_turbine_is_equal() -> None:
    """无尾流时各机组发电量完全相同（位置不影响毛发电量）。"""
    c = example_curve()
    res = example_wind_resource()
    pos = np.array([[0.0, 0.0], [660.0, 0.0], [1320.0, 0.0]])
    out = evaluate_aep(pos, res, c, None, include_wake=False, bin_width=2.0)
    assert np.allclose(out.per_turbine_gwh, out.per_turbine_gwh[0])


def test_results_scale_linearly_with_turbine_count() -> None:
    """无尾流时毛发电量与机组数成正比（线性叠加检查）。"""
    c = example_curve()
    res = example_wind_resource()
    base = np.array([[0.0, 0.0], [1000.0, 0.0]])
    twice = np.vstack([base, base + np.array([0.0, 1000.0])])
    r1 = evaluate_aep(base, res, c, None, include_wake=False, bin_width=2.0)
    r2 = evaluate_aep(twice, res, c, None, include_wake=False, bin_width=2.0)
    assert abs(r2.gross_gwh - 2 * r1.gross_gwh) < 1e-9


def test_compare_with_without_wake_helper() -> None:
    """含/不含尾流对照辅助函数的一致性。"""
    c = example_curve()
    res = example_wind_resource()
    pos = np.zeros((4, 2))
    pos[:, 0] = np.arange(4) * 700.0
    model = WakeModel(diameter=D, k_wake=0.075, rotor_grid=8)
    a, b = compare_with_without_wake(pos, res, c, model, bin_width=4.0)
    assert a.include_wake is True
    assert b.include_wake is False
    assert a.gross_gwh == b.gross_gwh
    assert b.net_gwh > a.net_gwh


# ---------------------------------------------------------------- 风向影响（锚点 9）


def test_direction_changes_wake_loss() -> None:
    """锚点 9：同一排布，风沿机位连线吹与垂直于连线吹，尾流损失差别巨大。"""
    c = example_curve()
    n = 6
    pos = np.zeros((n, 2))
    pos[:, 0] = np.arange(n) * 5.0 * D        # 沿东西方向一列，间距 5D
    model = WakeModel(diameter=D, k_wake=0.075, rotor_grid=16)

    along = evaluate_aep(pos, single_sector_resource(12, 8.0), c, model, bin_width=2.0)
    across = evaluate_aep(pos, single_sector_resource(0, 8.0), c, model, bin_width=2.0)

    assert along.wake_loss_pct > 5.0, "顺排时尾流损失应显著"
    assert across.wake_loss_pct < 0.5, "垂直吹时不应有尾流"
    assert along.wake_loss_pct > across.wake_loss_pct + 5.0
    assert along.net_gwh < across.net_gwh
    assert abs(along.gross_gwh - across.gross_gwh) < 1e-9, "毛发电量只取决于风资源"


def test_upwind_downwind_pair_asymmetry() -> None:
    """同一对机位，正吹与反吹的尾流损失相同（对称性），但受影响的机组互换。"""
    c = example_curve()
    pos = np.array([[0.0, 0.0], [5.0 * D, 0.0]])
    model = WakeModel(diameter=D, k_wake=0.075, rotor_grid=12)
    west = evaluate_aep(pos, single_sector_resource(12, 8.0), c, model, bin_width=2.0)
    east = evaluate_aep(pos, single_sector_resource(4, 8.0), c, model, bin_width=2.0)
    assert abs(west.wake_loss_pct - east.wake_loss_pct) < 1e-6
    assert west.wake_loss_pct > 0.0


def test_directional_wake_loss_sweep() -> None:
    """风向扫描：沿排布方向损失最大，垂直方向接近 0。"""
    c = example_curve()
    pos = np.zeros((5, 2))
    pos[:, 0] = np.arange(5) * 5.0 * D
    model = WakeModel(diameter=D, k_wake=0.075, rotor_grid=10)
    rows = directional_wake_loss(pos, c, model, 8.0, step_deg=10.0)
    assert len(rows) == 36
    by_dir = {r["direction_deg"]: r for r in rows}
    along = by_dir[270.0]["power_loss_pct"]
    across = by_dir[0.0]["power_loss_pct"]
    assert along > 20.0
    assert across < 0.5
    assert along > across
    for r in rows:
        assert 0.0 <= r["speed_deficit_pct"] < 100.0
        assert r["u_min"] <= 8.0


# ---------------------------------------------------------------- 校验与输出


def test_evaluate_aep_validation() -> None:
    c = example_curve()
    res = example_wind_resource()
    pos = np.array([[0.0, 0.0], [700.0, 0.0]])
    with pytest.raises(ValueError):
        evaluate_aep(pos, res, c, None, availability=0.0)
    with pytest.raises(ValueError):
        evaluate_aep(pos, res, c, None, other_loss_factor=1.5)
    with pytest.raises(ValueError):
        evaluate_aep(pos, res, c, None, bins_by_sector=[(np.array([8.0]), np.array([1.0]))])
    with pytest.raises(ValueError):
        evaluate_aep(np.zeros((3, 3)), res, c, None)


def test_empty_layout_gives_zero_aep() -> None:
    """空排布不报错，发电量为 0。"""
    c = example_curve()
    res = example_wind_resource()
    out = evaluate_aep(np.zeros((0, 2)), res, c, None, include_wake=False, bin_width=4.0)
    assert out.net_gwh == 0.0
    assert out.n_turbines == 0
    assert out.net_capacity_factor == 0.0


def test_result_summary_and_sector_table() -> None:
    """中文摘要与逐扇区表结构。"""
    c = example_curve()
    res = example_wind_resource()
    pos = np.zeros((4, 2))
    pos[:, 0] = np.arange(4) * 700.0
    model = WakeModel(diameter=D, k_wake=0.075, rotor_grid=8)
    out = evaluate_aep(pos, res, c, model, bin_width=4.0)
    text = out.summary()
    assert "尾流损失" in text
    assert "净容量系数" in text
    assert "年满发小时" in text
    rows = out.sector_table_rows()
    assert len(rows) == N_SECTORS
    assert set(rows[0].keys()) == {"扇区", "风向_deg", "频率", "平均风速_m_s",
                                   "毛发电量_GWh", "尾流后_GWh", "净发电量_GWh", "尾流损失_%"}
    total_gross = sum(float(r["毛发电量_GWh"]) for r in rows)
    assert abs(total_gross - out.gross_gwh) < 1e-6
    assert isinstance(out, AEPResult)


def test_capacity_factor_and_full_load_hours_consistency() -> None:
    """满发小时 = 净容量系数 × 8760。"""
    c = example_curve()
    res = example_wind_resource()
    pos = np.zeros((3, 2))
    pos[:, 0] = np.arange(3) * 700.0
    out = evaluate_aep(pos, res, c, None, include_wake=False, bin_width=4.0)
    assert abs(out.full_load_hours - out.net_capacity_factor * HOURS_PER_YEAR) < 1e-6
    assert 0.0 < out.net_capacity_factor < 1.0


def test_bins_by_sector_requires_all_sectors() -> None:
    c = example_curve()
    res = example_wind_resource()
    with pytest.raises(ValueError):
        evaluate_aep(np.zeros((2, 2)), res, c, None,
                     bins_by_sector=[(np.array([1.0]), np.array([1.0]))] * 4)
