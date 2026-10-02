"""发电量（AEP）模块：按风向扇区加权的年发电量、容量系数、尾流损失。

计算框架
--------
行业通用的「风向扇区 × 风速分仓」方法：

1. 把风向分成 16 个扇区，每个扇区有出现频率 ``f_s`` 与自己的 Weibull 分布；
2. 每个扇区把风速离散成若干仓（bin），每仓有概率 ``w_b`` 与一个**代表风速**；
3. 对每个 (扇区, 风速仓) 组合求一次尾流场，得到各机组的等效来流风速，
   用功率曲线换算成功率；
4. 能量 = Σ_s Σ_b f_s · w_b · P_array(v_b) · 8760 h。

**代表风速的取法**（关键，决定了粗分仓还能不能算准）：
    不是取仓中点，而是取「能量等效风速」——使 ``P(v_rep) = E[P(v) | v∈bin]`` 的那个风速，
    其中 ``E[P|bin]`` 用细网格数值积分得到。这样即使把分仓放宽到 4~5 m/s，
    毛发电量的误差依然很小（见 ``tests/test_aep.py`` 中粗/细分仓一致性测试）。

损失口径
--------
::

    E_gross   自由来流（不考虑尾流）的理论发电量
    E_wake    只考虑尾流折减后的发电量
    E_net     = E_wake × 可用率 × 其他综合折减系数
    尾流损失率 = (E_gross - E_wake) / E_gross
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .power_curve import PowerCurve, trapezoid
from .resource import (
    N_SECTORS,
    WindResource,
    sector_direction_deg,
    sector_name,
    weibull_cdf,
    weibull_pdf,
)
from .wake import WakeModel

#: 全年小时数（不考虑闰年）
HOURS_PER_YEAR = 8760.0

#: 典型可用率
DEFAULT_AVAILABILITY = 0.95

#: 典型其他综合折减（电气损耗、限电、叶片/功率曲线衰减、场用电等）
DEFAULT_OTHER_LOSS_FACTOR = 0.90


# ---------------------------------------------------------------- 风速分仓


def power_speed(curve: PowerCurve, power: float) -> float:
    """由功率反查风速（在 [切入, 额定] 上单调段求逆）；超出范围时截断到端点。"""
    p = float(power)
    if p <= 0.0:
        return 0.0
    if p >= curve.rated_power:
        return float(curve.rated_speed)
    # 只取单调上升段：切入 ~ 额定风速
    mask = (curve.speeds >= curve.cut_in) & (curve.speeds <= float(curve.rated_speed))
    xs = curve.speeds[mask]
    ys = curve.powers[mask]
    order = np.argsort(ys)
    return float(np.interp(p, ys[order], xs[order]))


def weibull_energy_bins(
    k: float,
    c: float,
    curve: PowerCurve,
    bin_width: float = 1.0,
    n_fine: int = 21,
) -> Tuple[np.ndarray, np.ndarray]:
    """把 Weibull(k, c) 离散成能量等效风速分仓。

    返回 ``(speeds, probs)``：``speeds`` 是每仓的能量等效代表风速，
    ``probs`` 是该仓的概率质量（即 ``F(hi) - F(lo)``）。概率和略小于 1，
    差值来自切出风速以上的尾部（那部分功率本来就是 0，不影响能量）。
    """
    if bin_width <= 0:
        raise ValueError("bin_width 必须 > 0")
    if n_fine < 5:
        raise ValueError("n_fine 太小")
    v_max = float(curve.cut_out)
    n_bin = int(math.ceil(v_max / bin_width))
    speeds: List[float] = []
    probs: List[float] = []
    for b in range(n_bin):
        lo = b * bin_width
        hi = min(lo + bin_width, v_max)
        if hi <= lo:
            continue
        mass = float(weibull_cdf(hi, k, c) - weibull_cdf(lo, k, c))
        if mass <= 1e-12:
            continue
        fine = np.linspace(lo, hi, int(n_fine))
        f = weibull_pdf(fine, k, c)
        # 用 power_integrand：切出风速点取左极限，避免分仓边界上的虚假 0 功率
        p = curve.power_integrand(fine)
        e_cond = trapezoid(p * f, fine) / mass          # 条件期望功率 kW
        speeds.append(power_speed(curve, e_cond))
        probs.append(mass)
    return np.array(speeds, dtype=np.float64), np.array(probs, dtype=np.float64)


def histogram_bins(
    centers: Sequence[float],
    freqs: Sequence[float],
    curve: PowerCurve,
) -> Tuple[np.ndarray, np.ndarray]:
    """把实测风频直方图（区间中点 + 频率）转成能量等效分仓，可直接喂给 :func:`evaluate_aep`。

    每个仓代表风速取 ``curve`` 上能量等效的风速，保证 ``Σ w·P(v_rep) = Σ w·P(中点)`` 的
    功率期望尽量一致（对窄仓两者几乎相同）。
    """
    c = np.asarray(centers, dtype=np.float64).ravel()
    f = np.asarray(freqs, dtype=np.float64).ravel()
    if c.size != f.size:
        raise ValueError("centers 与 freqs 长度必须一致")
    s = float(np.sum(f))
    if s <= 0:
        raise ValueError("频率之和必须 > 0")
    probs = f / s
    speeds = np.array([power_speed(curve, float(curve.power(v))) for v in c], dtype=np.float64)
    return speeds, probs


# ---------------------------------------------------------------- 结果对象


@dataclass
class AEPResult:
    """发电量评估结果。能量单位 GWh，功率单位 kW。"""

    n_turbines: int
    capacity_mw: float
    gross_gwh: float
    wake_gwh: float
    net_gwh: float
    wake_loss_pct: float
    availability: float
    other_loss_factor: float
    other_loss_pct: float
    gross_capacity_factor: float
    net_capacity_factor: float
    full_load_hours: float
    per_sector: List[Dict[str, float]] = field(default_factory=list)
    per_turbine_gwh: np.ndarray = field(default_factory=lambda: np.zeros(0))
    include_wake: bool = True
    bin_width: float = 1.0
    hours: float = HOURS_PER_YEAR

    @property
    def total_loss_pct(self) -> float:
        """从毛发电量到净发电量的总折减率（%）。"""
        if self.gross_gwh <= 0:
            return 0.0
        return 100.0 * (self.gross_gwh - self.net_gwh) / self.gross_gwh

    @property
    def net_capacity_factor_corrected(self) -> float:
        """净容量系数（= 净发电量 / 额定电量）。"""
        return self.net_capacity_factor

    def sector_table_rows(self) -> List[Dict[str, object]]:
        """逐扇区表（用于 CSV / Markdown 输出）。"""
        rows = []
        for s in self.per_sector:
            rows.append({
                "扇区": s["name"],
                "风向_deg": s["direction_deg"],
                "频率": s["freq"],
                "平均风速_m_s": s["mean_speed"],
                "毛发电量_GWh": s["gross_gwh"],
                "尾流后_GWh": s["wake_gwh"],
                "净发电量_GWh": s["net_gwh"],
                "尾流损失_%": s["wake_loss_pct"],
            })
        return rows

    def summary(self) -> str:
        """中文多行摘要。"""
        return (
            "机组 %d 台（装机 %.1f MW）\n"
            "  毛发电量      %8.2f GWh/年\n"
            "  尾流后        %8.2f GWh/年（尾流损失 %.2f%%）\n"
            "  净发电量      %8.2f GWh/年（可用率 %.2f × 其他折减 %.2f）\n"
            "  毛容量系数    %8.2f%%\n"
            "  净容量系数    %8.2f%%\n"
            "  年满发小时    %8.0f h\n"
            "  总折减率      %8.2f%%"
            % (
                self.n_turbines, self.capacity_mw,
                self.gross_gwh, self.wake_gwh, self.wake_loss_pct,
                self.net_gwh, self.availability, self.other_loss_factor,
                100.0 * self.gross_capacity_factor, 100.0 * self.net_capacity_factor,
                self.full_load_hours, self.total_loss_pct,
            )
        )


# ---------------------------------------------------------------- 主计算


def evaluate_aep(
    layout: np.ndarray,
    resource: WindResource,
    curve: PowerCurve,
    wake_model: Optional[WakeModel] = None,
    availability: float = DEFAULT_AVAILABILITY,
    other_loss_factor: float = DEFAULT_OTHER_LOSS_FACTOR,
    bin_width: float = 1.0,
    include_wake: bool = True,
    bins_by_sector: Optional[Sequence[Tuple[np.ndarray, np.ndarray]]] = None,
    hours: float = HOURS_PER_YEAR,
    collect_per_turbine: bool = True,
) -> AEPResult:
    """按风向扇区加权计算年发电量。

    Args:
        layout: ``(N, 2)`` 机位坐标（m）。
        resource: 风资源（风玫瑰 + Weibull 形状参数）。
        curve: 功率曲线（含 Ct）。
        wake_model: Jensen 尾流模型；``include_wake=False`` 时可为 None。
        availability: 可用率，典型 0.95。
        other_loss_factor: 其他综合折减系数，典型 0.90。
        bin_width: 风速分仓宽度（m/s）。1.0 为标准精度，3~5 为优化用的快速近似。
        include_wake: 是否计入尾流；False 时得到「无尾流对照组」。
        bins_by_sector: 自定义逐扇区分仓 ``[(speeds, probs)] * 16``；
            默认由各扇区 Weibull 分布生成能量等效分仓。
        hours: 年小时数。
        collect_per_turbine: 是否统计单机发电量。

    Returns:
        :class:`AEPResult`
    """
    if not (0.0 < availability <= 1.0):
        raise ValueError("可用率必须在 (0, 1] 内")
    if not (0.0 < other_loss_factor <= 1.0):
        raise ValueError("其他折减系数必须在 (0, 1] 内")
    pos = np.atleast_2d(np.asarray(layout, dtype=np.float64))
    if pos.ndim != 2 or pos.shape[1] != 2:
        raise ValueError("layout 形状必须为 (N, 2)")
    n_t = pos.shape[0]

    rated_total = float(n_t) * float(curve.rated_power)          # kW
    gross_kwh = 0.0
    wake_kwh = 0.0
    per_sector: List[Dict[str, float]] = []
    per_turbine_kwh = np.zeros(n_t, dtype=np.float64)

    if bins_by_sector is not None and len(bins_by_sector) != N_SECTORS:
        raise ValueError("bins_by_sector 必须包含 %d 个扇区" % N_SECTORS)

    for s in range(N_SECTORS):
        f_s = float(resource.rose.freqs[s])
        if f_s <= 0.0:
            per_sector.append({
                "sector": s, "name": sector_name(s), "direction_deg": sector_direction_deg(s),
                "freq": 0.0, "mean_speed": float(resource.rose.mean_speeds[s]),
                "gross_gwh": 0.0, "wake_gwh": 0.0, "net_gwh": 0.0, "wake_loss_pct": 0.0,
            })
            continue
        k_s, c_s = resource.sector_weibull(s)
        if bins_by_sector is not None:
            speeds = np.asarray(bins_by_sector[s][0], dtype=np.float64)
            probs = np.asarray(bins_by_sector[s][1], dtype=np.float64)
        else:
            speeds, probs = weibull_energy_bins(k_s, c_s, curve, bin_width)
        direction = sector_direction_deg(s)

        sec_gross = 0.0
        sec_wake = 0.0
        for v_b, w_b in zip(speeds, probs):
            if w_b <= 0.0:
                continue
            v_b = float(v_b)
            weight = f_s * float(w_b) * hours
            p_free = float(curve.power(v_b))
            p_array_gross = p_free * n_t
            sec_gross += weight * p_array_gross
            if include_wake and wake_model is not None and v_b > 0.0:
                ct = float(curve.ct(v_b))
                if ct > 1e-9 and n_t > 0:
                    u_eff = wake_model.effective_speeds(pos, v_b, direction, ct)
                    p_each = np.asarray(curve.power(u_eff), dtype=np.float64)
                    sec_wake += weight * float(np.sum(p_each))
                    if collect_per_turbine:
                        per_turbine_kwh += weight * p_each
                else:
                    sec_wake += weight * p_array_gross
                    if collect_per_turbine:
                        per_turbine_kwh += weight * p_free
            else:
                sec_wake += weight * p_array_gross
                if collect_per_turbine:
                    per_turbine_kwh += weight * p_free

        gross_kwh += sec_gross
        wake_kwh += sec_wake
        sec_loss = 0.0 if sec_gross <= 0 else 100.0 * (sec_gross - sec_wake) / sec_gross
        per_sector.append({
            "sector": s,
            "name": sector_name(s),
            "direction_deg": direction,
            "freq": f_s,
            "mean_speed": float(resource.rose.mean_speeds[s]),
            "gross_gwh": sec_gross / 1e6,
            "wake_gwh": sec_wake / 1e6,
            "net_gwh": sec_wake * availability * other_loss_factor / 1e6,
            "wake_loss_pct": sec_loss,
        })

    net_kwh = wake_kwh * availability * other_loss_factor
    wake_loss_pct = 0.0 if gross_kwh <= 0 else 100.0 * (gross_kwh - wake_kwh) / gross_kwh
    rated_kwh = rated_total * hours

    return AEPResult(
        n_turbines=n_t,
        capacity_mw=rated_total / 1000.0,
        gross_gwh=gross_kwh / 1e6,
        wake_gwh=wake_kwh / 1e6,
        net_gwh=net_kwh / 1e6,
        wake_loss_pct=wake_loss_pct,
        availability=availability,
        other_loss_factor=other_loss_factor,
        other_loss_pct=100.0 * (1.0 - other_loss_factor),
        gross_capacity_factor=(0.0 if rated_kwh <= 0 else gross_kwh / rated_kwh),
        net_capacity_factor=(0.0 if rated_kwh <= 0 else net_kwh / rated_kwh),
        full_load_hours=(0.0 if rated_total <= 0 else net_kwh / rated_total),
        per_sector=per_sector,
        per_turbine_gwh=per_turbine_kwh / 1e6,
        include_wake=include_wake,
        bin_width=float(bin_width),
        hours=hours,
    )


def directional_wake_loss(
    layout: np.ndarray,
    curve: PowerCurve,
    wake_model: WakeModel,
    wind_speed: float,
    step_deg: float = 10.0,
) -> List[Dict[str, float]]:
    """扫一遍风向（0~360°），给出每个风向下的速度亏损比与功率损失比。

    发电量口径的功率损失比用功率曲线换算：``1 - ΣP(u_eff) / (N·P(u0))``。
    """
    pos = np.atleast_2d(np.asarray(layout, dtype=np.float64))
    n_t = pos.shape[0]
    if n_t == 0:
        return []
    v = float(wind_speed)
    ct = float(curve.ct(v))
    rows: List[Dict[str, float]] = []
    for direction in np.arange(0.0, 360.0, step_deg):
        if ct <= 1e-9:
            u_eff = np.full(n_t, v)
        else:
            u_eff = wake_model.effective_speeds(pos, v, float(direction), ct)
        p_free = float(curve.power(v)) * n_t
        p_net = float(np.sum(curve.power(u_eff)))
        rows.append({
            "direction_deg": float(direction),
            "speed_deficit_pct": 100.0 * (1.0 - float(np.mean(u_eff)) / v) if v > 0 else 0.0,
            "power_loss_pct": 0.0 if p_free <= 0 else 100.0 * (p_free - p_net) / p_free,
            "u_min": float(np.min(u_eff)),
        })
    return rows


def compare_with_without_wake(
    layout: np.ndarray,
    resource: WindResource,
    curve: PowerCurve,
    wake_model: WakeModel,
    **kwargs,
) -> Tuple[AEPResult, AEPResult]:
    """同时算「含尾流」与「无尾流对照」两组结果，用于单独量化尾流损失。"""
    with_wake = evaluate_aep(layout, resource, curve, wake_model, include_wake=True, **kwargs)
    no_wake = evaluate_aep(layout, resource, curve, None, include_wake=False, **kwargs)
    return with_wake, no_wake
