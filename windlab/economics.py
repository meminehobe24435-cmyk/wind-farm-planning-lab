"""经济性模块：CAPEX / OPEX / LCOE / NPV / 投资回收期 / 单因素敏感性。

LCOE 定义（本工具采用的现金流口径）
-----------------------------------
::

              Σ_{t=0}^{N-1} (CAPEX_t + OPEX_t) / (1+r)^t
    LCOE  =  ---------------------------------------------
                    Σ_{t=1}^{N}  AEP_t / (1+r)^t

    - 建设投资 CAPEX 全部发生在第 0 年（不折现）；
    - 运营成本 OPEX 与发电量 AEP 发生在第 1 ~ N 年；
    - r 为折现率，N 为项目寿命（默认 20 年）；
    - 分子分母同时约掉年金系数 AF(r,N) 后，在 OPEX 与 AEP 恒定的简化情形下退化为
      标准等额年金（CRF）形式::

          LCOE = CAPEX * CRF(r, N) / AEP + OPEX / AEP
          CRF  = r / (1 - (1+r)^-N)

      这是 ``tests/test_economics.py`` 中的闭式验证锚点。

⚠️ 参数均为**典型值**，不是任何实际项目的报价或中标电价。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

#: GWh -> kWh
GWH_TO_KWH = 1e6


# ---------------------------------------------------------------- 假设


@dataclass
class Economics:
    """项目经济性假设（默认值都是**典型取值范围**，非实际报价）。

    Attributes:
        capex_per_kw: 单位千瓦静态投资（元/kW），陆上典型 5000~8000。
        opex_per_kw_year: 年运营成本（元/kW/年），典型 150~300。
        discount_rate: 折现率（税后加权资金成本），典型 0.06~0.10。
        lifetime_years: 项目寿命（年），典型 20。
        electricity_price: 上网电价（元/kWh），典型平价项目 0.30~0.45。
        annual_degradation: 年发电量衰减率（%/年，取小数），典型 0.005。
        capex_year0_fraction: 第 0 年支付的投资比例（默认 1.0）。
    """

    capex_per_kw: float = 5800.0
    opex_per_kw_year: float = 200.0
    discount_rate: float = 0.08
    lifetime_years: int = 20
    electricity_price: float = 0.35
    annual_degradation: float = 0.005
    capex_year0_fraction: float = 1.0

    def __post_init__(self) -> None:
        if self.capex_per_kw < 0 or self.opex_per_kw_year < 0:
            raise ValueError("造价与运营成本不能为负")
        if self.discount_rate <= -1.0:
            raise ValueError("折现率必须 > -100%")
        if self.lifetime_years < 1:
            raise ValueError("项目寿命至少 1 年")
        if not (0.0 <= self.annual_degradation < 1.0):
            raise ValueError("年衰减率必须在 [0, 1) 内")

    # -- 派生量 ---------------------------------------------------

    def capex(self, capacity_kw: float) -> float:
        """静态总投资（元）。"""
        return float(capacity_kw) * float(self.capex_per_kw)

    def opex_year(self, capacity_kw: float, year: int = 1) -> float:
        """第 year 年的运营成本（元）。"""
        return float(capacity_kw) * float(self.opex_per_kw_year)

    def annuity_factor(self) -> float:
        """年金现值系数 AF = (1 - (1+r)^-N) / r；r=0 时退化为 N。"""
        r = float(self.discount_rate)
        n = int(self.lifetime_years)
        if abs(r) < 1e-12:
            return float(n)
        return (1.0 - (1.0 + r) ** (-n)) / r

    def crf(self) -> float:
        """资本回收系数 CRF = 1 / AF = r / (1 - (1+r)^-N)。"""
        return 1.0 / self.annuity_factor()

    def discount(self, year: int) -> float:
        """第 year 年的折现因子 1/(1+r)^year（year=0 时为 1）。"""
        return 1.0 / (1.0 + float(self.discount_rate)) ** int(year)

    def to_dict(self) -> Dict[str, float]:
        return {
            "capex_per_kw": self.capex_per_kw,
            "opex_per_kw_year": self.opex_per_kw_year,
            "discount_rate": self.discount_rate,
            "lifetime_years": self.lifetime_years,
            "electricity_price": self.electricity_price,
            "annual_degradation": self.annual_degradation,
            "capex_year0_fraction": self.capex_year0_fraction,
        }


# ---------------------------------------------------------------- 现金流


@dataclass
class CashFlow:
    """逐年现金流（单位：元 / kWh）。"""

    years: np.ndarray
    capex: np.ndarray
    opex: np.ndarray
    energy_kwh: np.ndarray
    revenue: np.ndarray

    @property
    def net(self) -> np.ndarray:
        """净现金流（不含初始投资年）：收入 - 运营成本。"""
        return self.revenue - self.opex

    def net_with_capex(self) -> np.ndarray:
        """含初始投资的净现金流。"""
        return self.net - self.capex


def cashflow(
    econ: Economics,
    capacity_kw: float,
    net_aep_gwh: float,
    degradation: Optional[float] = None,
) -> CashFlow:
    """构造项目逐年现金流。

    CAPEX 在第 0 年一次性支付；OPEX 与发电量出现在第 1~N 年；
    发电量按 ``(1 - 衰减率)^(t-1)`` 逐年递减。
    """
    n = int(econ.lifetime_years)
    deg = econ.annual_degradation if degradation is None else float(degradation)
    years = np.arange(0, n + 1, dtype=np.int64)

    capex = np.zeros(n + 1, dtype=np.float64)
    capex[0] = econ.capex(capacity_kw) * float(econ.capex_year0_fraction)

    opex = np.zeros(n + 1, dtype=np.float64)
    energy = np.zeros(n + 1, dtype=np.float64)
    for t in range(1, n + 1):
        opex[t] = econ.opex_year(capacity_kw, t)
        energy[t] = float(net_aep_gwh) * GWH_TO_KWH * (1.0 - deg) ** (t - 1)
    revenue = energy * float(econ.electricity_price)

    return CashFlow(years=years, capex=capex, opex=opex, energy_kwh=energy, revenue=revenue)


# ---------------------------------------------------------------- LCOE


def lcoe(
    econ: Economics,
    capacity_kw: float,
    net_aep_gwh: float,
    degradation: Optional[float] = None,
    lifetime_years: Optional[int] = None,
) -> float:
    """平准化度电成本（元/kWh），按模块开头给出的折现现金流公式计算。"""
    e = econ
    if lifetime_years is not None and int(lifetime_years) != e.lifetime_years:
        e = Economics(
            capex_per_kw=e.capex_per_kw, opex_per_kw_year=e.opex_per_kw_year,
            discount_rate=e.discount_rate, lifetime_years=int(lifetime_years),
            electricity_price=e.electricity_price, annual_degradation=e.annual_degradation,
            capex_year0_fraction=e.capex_year0_fraction,
        )
    cf = cashflow(e, capacity_kw, net_aep_gwh, degradation=degradation)
    num = float(np.sum((cf.capex + cf.opex) * np.array([e.discount(int(t)) for t in cf.years])))
    den = float(np.sum(cf.energy_kwh * np.array([e.discount(int(t)) for t in cf.years])))
    if den <= 0:
        raise ValueError("发电量为 0，无法计算 LCOE")
    return num / den


def lcoe_closed_form(
    econ: Economics,
    capacity_kw: float,
    net_aep_gwh: float,
) -> float:
    """等额年金（CRF）闭式 LCOE，仅适用于 OPEX 与发电量恒定、无衰减的情形。

        LCOE = CAPEX * CRF(r, N) / AEP + OPEX / AEP

    用作 :func:`lcoe` 的正确性锚点（对应测试里的闭式验证）。
    """
    aep_kwh = float(net_aep_gwh) * GWH_TO_KWH
    if aep_kwh <= 0:
        raise ValueError("发电量为 0，无法计算 LCOE")
    capex_part = econ.capex(capacity_kw) * econ.crf() / aep_kwh
    opex_part = econ.opex_year(capacity_kw) / aep_kwh
    return capex_part + opex_part


# ---------------------------------------------------------------- 财务指标


def npv(
    econ: Economics,
    capacity_kw: float,
    net_aep_gwh: float,
    price: Optional[float] = None,
    discount_rate: Optional[float] = None,
) -> float:
    """项目净现值（元）。"""
    rate = float(econ.discount_rate if discount_rate is None else discount_rate)
    cf = cashflow(econ, capacity_kw, net_aep_gwh)
    revenue = cf.energy_kwh * float(econ.electricity_price if price is None else price)
    total = 0.0
    for t in range(0, int(econ.lifetime_years) + 1):
        cash = revenue[t] - cf.opex[t] - cf.capex[t]
        total += cash / (1.0 + rate) ** t
    return float(total)


def irr(
    econ: Economics,
    capacity_kw: float,
    net_aep_gwh: float,
    price: Optional[float] = None,
    lo: float = -0.9,
    hi: float = 3.0,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> float:
    """内部收益率（二分法，返回小数；找不到符号变化时返回 nan）。"""

    def f(rate: float) -> float:
        return npv(econ, capacity_kw, net_aep_gwh, price=price, discount_rate=rate)

    f_lo, f_hi = f(lo), f(hi)
    if f_lo * f_hi > 0:
        return float("nan")
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = f(mid)
        if abs(f_mid) < tol or (hi - lo) < tol:
            return float(mid)
        if f_lo * f_mid <= 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return float(0.5 * (lo + hi))


def payback_period(
    econ: Economics,
    capacity_kw: float,
    net_aep_gwh: float,
    price: Optional[float] = None,
    discounted: bool = False,
) -> float:
    """投资回收期（年）。``discounted=True`` 时为折现回收期。

    线性插值到现金流由负转正的那一年；项目期内无法回收时返回 ``inf``。
    """
    rate = float(econ.discount_rate) if discounted else 0.0
    cf = cashflow(econ, capacity_kw, net_aep_gwh)
    revenue = cf.energy_kwh * float(econ.electricity_price if price is None else price)
    cumulative = -cf.capex[0]
    if cumulative >= 0:
        return 0.0
    for t in range(1, int(econ.lifetime_years) + 1):
        cash = (revenue[t] - cf.opex[t]) / (1.0 + rate) ** t
        prev = cumulative
        cumulative += cash
        if cumulative >= 0:
            if abs(cash) < 1e-12:
                return float(t)
            return float(t - 1 + (-prev) / cash)
    return float("inf")


def simple_metrics(
    econ: Economics,
    capacity_kw: float,
    net_aep_gwh: float,
    price: Optional[float] = None,
) -> Dict[str, float]:
    """一组常用财务指标汇总。"""
    p = float(econ.electricity_price if price is None else price)
    aep_kwh = float(net_aep_gwh) * GWH_TO_KWH
    return {
        "lcoe_yuan_per_kwh": lcoe(econ, capacity_kw, net_aep_gwh),
        "capex_yuan": econ.capex(capacity_kw),
        "opex_yuan_per_year": econ.opex_year(capacity_kw),
        "annual_revenue_yuan": aep_kwh * p,
        "npv_yuan": npv(econ, capacity_kw, net_aep_gwh, price=p),
        "irr": irr(econ, capacity_kw, net_aep_gwh, price=p),
        "payback_years": payback_period(econ, capacity_kw, net_aep_gwh, price=p),
        "discounted_payback_years": payback_period(
            econ, capacity_kw, net_aep_gwh, price=p, discounted=True),
    }


# ---------------------------------------------------------------- 敏感性分析


#: 默认的单因素敏感范围（相对基准值的乘子）
DEFAULT_FACTORS: Tuple[Tuple[str, float, float], ...] = (
    ("单位造价 CAPEX", 0.8, 1.2),
    ("年运营成本 OPEX", 0.7, 1.3),
    ("折现率 r", 0.75, 1.25),
    ("年发电量 AEP", 0.9, 1.1),
    ("电量年衰减率", 0.0, 3.0),
)


def _rebuilt(econ: Economics, **kw):
    """按字段覆盖生成一个新的 :class:`Economics`。"""
    data = econ.to_dict()
    data.update(kw)
    return Economics(**data)


def sensitivity_lcoe(
    econ: Economics,
    capacity_kw: float,
    net_aep_gwh: float,
    factors: Optional[Sequence[Tuple[str, float, float]]] = None,
) -> List[Dict[str, float]]:
    """LCOE 单因素敏感性（龙卷风图数据）。

    每个因素给出「低值 / 基准 / 高值」三种情形下的 LCOE，以及变化幅度（元/kWh）。
    注意：**上网电价对 LCOE 没有影响**（LCOE 是成本口径指标，与电价无关），
    电价的影响体现在 :func:`sensitivity_npv` 里。
    """
    base = lcoe(econ, capacity_kw, net_aep_gwh)
    out: List[Dict[str, float]] = []
    for name, lo_mul, hi_mul in (factors or DEFAULT_FACTORS):
        if name.startswith("单位造价"):
            lo = _rebuilt(econ, capex_per_kw=econ.capex_per_kw * lo_mul)
            hi = _rebuilt(econ, capex_per_kw=econ.capex_per_kw * hi_mul)
        elif name.startswith("年运营成本"):
            lo = _rebuilt(econ, opex_per_kw_year=econ.opex_per_kw_year * lo_mul)
            hi = _rebuilt(econ, opex_per_kw_year=econ.opex_per_kw_year * hi_mul)
        elif name.startswith("折现率"):
            lo = _rebuilt(econ, discount_rate=econ.discount_rate * lo_mul)
            hi = _rebuilt(econ, discount_rate=econ.discount_rate * hi_mul)
        elif name.startswith("年发电量"):
            lo_v = net_aep_gwh * lo_mul
            hi_v = net_aep_gwh * hi_mul
            lo = hi = econ
        elif name.startswith("电量年衰减"):
            lo = _rebuilt(econ, annual_degradation=econ.annual_degradation * lo_mul)
            hi = _rebuilt(econ, annual_degradation=econ.annual_degradation * hi_mul)
        else:
            raise ValueError("未知的敏感性因素：%s" % name)

        aep_lo = net_aep_gwh * lo_mul if name.startswith("年发电量") else net_aep_gwh
        aep_hi = net_aep_gwh * hi_mul if name.startswith("年发电量") else net_aep_gwh
        v_lo = lcoe(lo, capacity_kw, aep_lo)
        v_hi = lcoe(hi, capacity_kw, aep_hi)
        span = abs(v_hi - v_lo)
        out.append({
            "factor": name,
            "low_multiplier": float(lo_mul),
            "high_multiplier": float(hi_mul),
            "low_value": v_lo,
            "high_value": v_hi,
            "base": base,
            "span": span,
            "low_pct": 100.0 * (v_lo - base) / base if base else 0.0,
            "high_pct": 100.0 * (v_hi - base) / base if base else 0.0,
        })
    out.sort(key=lambda d: -d["span"])
    return out


def sensitivity_npv(
    econ: Economics,
    capacity_kw: float,
    net_aep_gwh: float,
    factors: Optional[Sequence[Tuple[str, float, float]]] = None,
) -> List[Dict[str, float]]:
    """NPV 单因素敏感性（含上网电价）。

    这才是「电价」应该出现的地方：电价只影响收益，不影响 LCOE。
    """
    default = (
        ("上网电价", 0.8, 1.2),
        ("单位造价 CAPEX", 0.8, 1.2),
        ("年运营成本 OPEX", 0.7, 1.3),
        ("折现率 r", 0.75, 1.25),
        ("年发电量 AEP", 0.9, 1.1),
    )
    base = npv(econ, capacity_kw, net_aep_gwh)
    out: List[Dict[str, float]] = []
    for name, lo_mul, hi_mul in (factors or default):
        if name.startswith("上网电价"):
            lo_p = econ.electricity_price * lo_mul
            hi_p = econ.electricity_price * hi_mul
            v_lo = npv(econ, capacity_kw, net_aep_gwh, price=lo_p)
            v_hi = npv(econ, capacity_kw, net_aep_gwh, price=hi_p)
        elif name.startswith("单位造价"):
            v_lo = npv(_rebuilt(econ, capex_per_kw=econ.capex_per_kw * lo_mul), capacity_kw, net_aep_gwh)
            v_hi = npv(_rebuilt(econ, capex_per_kw=econ.capex_per_kw * hi_mul), capacity_kw, net_aep_gwh)
        elif name.startswith("年运营成本"):
            v_lo = npv(_rebuilt(econ, opex_per_kw_year=econ.opex_per_kw_year * lo_mul), capacity_kw, net_aep_gwh)
            v_hi = npv(_rebuilt(econ, opex_per_kw_year=econ.opex_per_kw_year * hi_mul), capacity_kw, net_aep_gwh)
        elif name.startswith("折现率"):
            v_lo = npv(_rebuilt(econ, discount_rate=econ.discount_rate * lo_mul), capacity_kw, net_aep_gwh)
            v_hi = npv(_rebuilt(econ, discount_rate=econ.discount_rate * hi_mul), capacity_kw, net_aep_gwh)
        elif name.startswith("年发电量"):
            v_lo = npv(econ, capacity_kw, net_aep_gwh * lo_mul)
            v_hi = npv(econ, capacity_kw, net_aep_gwh * hi_mul)
        else:
            raise ValueError("未知的敏感性因素：%s" % name)
        out.append({
            "factor": name,
            "low_multiplier": float(lo_mul),
            "high_multiplier": float(hi_mul),
            "low_value": v_lo,
            "high_value": v_hi,
            "base": base,
            "span": abs(v_hi - v_lo),
        })
    out.sort(key=lambda d: -d["span"])
    return out


def economic_summary(econ: Economics, capacity_kw: float, net_aep_gwh: float) -> str:
    """中文经济性摘要。"""
    m = simple_metrics(econ, capacity_kw, net_aep_gwh)
    return (
        "装机 %.1f MW，净发电量 %.2f GWh/年\n"
        "  CAPEX        %10.0f 万元（%.0f 元/kW）\n"
        "  OPEX         %10.0f 万元/年（%.0f 元/kW/年）\n"
        "  年售电收入   %10.0f 万元（电价 %.3f 元/kWh）\n"
        "  LCOE         %10.4f 元/kWh\n"
        "  NPV          %10.0f 万元（r=%.1f%%，%d 年）\n"
        "  IRR          %10.2f%%\n"
        "  投资回收期   %10.1f 年（折现 %.1f 年）"
        % (
            capacity_kw / 1000.0, net_aep_gwh,
            m["capex_yuan"] / 1e4, econ.capex_per_kw,
            m["opex_yuan_per_year"] / 1e4, econ.opex_per_kw_year,
            m["annual_revenue_yuan"] / 1e4, econ.electricity_price,
            m["lcoe_yuan_per_kwh"],
            m["npv_yuan"] / 1e4, econ.discount_rate * 100, econ.lifetime_years,
            100.0 * m["irr"] if m["irr"] == m["irr"] else float("nan"),
            m["payback_years"], m["discounted_payback_years"],
        )
    )
