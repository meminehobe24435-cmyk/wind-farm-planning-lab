"""经济性模块测试：LCOE 闭式验证（锚点 7）、NPV/IRR/回收期、单因素敏感性。"""

from __future__ import annotations

import math

import numpy as np
import pytest

from windlab.economics import (
    DEFAULT_FACTORS,
    Economics,
    cashflow,
    economic_summary,
    irr,
    lcoe,
    lcoe_closed_form,
    npv,
    payback_period,
    sensitivity_lcoe,
    sensitivity_npv,
    simple_metrics,
)

#: 测试基准：6 台 5 MW 机组 = 60 MW，净发电量 150 GWh/年
CAPACITY_KW = 60000.0
AEP_GWH = 150.0


def crf(r: float, n: int) -> float:
    """独立实现的资本回收系数 CRF = r(1+r)^n / ((1+r)^n - 1)。"""
    return r * (1.0 + r) ** n / ((1.0 + r) ** n - 1.0)


# ---------------------------------------------------------------- 基础参数


def test_economics_defaults_are_typical() -> None:
    """默认经济性参数落在陆上风电的典型区间内。"""
    e = Economics()
    assert 4000.0 <= e.capex_per_kw <= 8000.0
    assert 100.0 <= e.opex_per_kw_year <= 350.0
    assert 0.04 <= e.discount_rate <= 0.12
    assert e.lifetime_years == 20
    assert 0.2 <= e.electricity_price <= 0.5


def test_economics_validation() -> None:
    with pytest.raises(ValueError):
        Economics(capex_per_kw=-1.0)
    with pytest.raises(ValueError):
        Economics(opex_per_kw_year=-1.0)
    with pytest.raises(ValueError):
        Economics(discount_rate=-1.0)
    with pytest.raises(ValueError):
        Economics(lifetime_years=0)
    with pytest.raises(ValueError):
        Economics(annual_degradation=1.5)


def test_annuity_factor_and_crf() -> None:
    """年金系数与资本回收系数互为倒数，且与手算一致。"""
    e = Economics(discount_rate=0.08, lifetime_years=20)
    assert abs(e.annuity_factor() - 9.818147) < 1e-5
    assert abs(e.crf() - 0.1018522) < 1e-6
    assert abs(e.crf() - 1.0 / e.annuity_factor()) < 1e-12
    assert abs(e.crf() - crf(0.08, 20)) < 1e-12


def test_zero_discount_rate_annuity_factor() -> None:
    """折现率为 0 时年金系数退化为年数。"""
    e = Economics(discount_rate=0.0, lifetime_years=20)
    assert abs(e.annuity_factor() - 20.0) < 1e-12
    assert abs(e.crf() - 0.05) < 1e-12


def test_capex_opex_amounts() -> None:
    """CAPEX / OPEX 的绝对金额。"""
    e = Economics(capex_per_kw=5800.0, opex_per_kw_year=200.0)
    assert abs(e.capex(CAPACITY_KW) - 3.48e8) < 1e-6
    assert abs(e.opex_year(CAPACITY_KW) - 1.2e7) < 1e-6


# ---------------------------------------------------------------- LCOE（锚点 7）


def test_lcoe_matches_closed_form_crf() -> None:
    """锚点 7：无衰减、CAPEX 在第 0 年时，折现现金流公式必须等于等额年金闭式解。

        LCOE = CAPEX·CRF(r,N)/AEP + OPEX/AEP
    """
    e = Economics(capex_per_kw=5800.0, opex_per_kw_year=200.0,
                  discount_rate=0.08, lifetime_years=20, annual_degradation=0.0)
    got = lcoe(e, CAPACITY_KW, AEP_GWH)
    closed = lcoe_closed_form(e, CAPACITY_KW, AEP_GWH)
    hand = (5800.0 * CAPACITY_KW * crf(0.08, 20) / (AEP_GWH * 1e6)
            + 200.0 * CAPACITY_KW / (AEP_GWH * 1e6))
    assert abs(got - closed) < 1e-12, "%.10f vs %.10f" % (got, closed)
    assert abs(got - hand) < 1e-12
    assert abs(got - 0.316297) < 1e-5, "手算 LCOE 应为 0.3163 元/kWh，实测 %.6f" % got


def test_lcoe_components_additivity() -> None:
    """LCOE = 投资分项 + 运营分项，两项分别可手算。"""
    e = Economics(capex_per_kw=5000.0, opex_per_kw_year=150.0,
                  discount_rate=0.06, lifetime_years=20, annual_degradation=0.0)
    capex_part = 5000.0 * CAPACITY_KW * crf(0.06, 20) / (AEP_GWH * 1e6)
    opex_part = 150.0 * CAPACITY_KW / (AEP_GWH * 1e6)
    assert abs(lcoe(e, CAPACITY_KW, AEP_GWH) - (capex_part + opex_part)) < 1e-12
    assert capex_part > opex_part


def test_lcoe_uses_discounted_energy_denominator() -> None:
    """有衰减时，分母必须逐年折现（不能简单用 AEP × 年数）。"""
    e = Economics(discount_rate=0.08, lifetime_years=20, annual_degradation=0.005)
    simple = Economics(discount_rate=0.08, lifetime_years=20, annual_degradation=0.0)
    assert lcoe(e, CAPACITY_KW, AEP_GWH) > lcoe(simple, CAPACITY_KW, AEP_GWH)


def test_lcoe_increases_with_capex_opex_discount() -> None:
    """LCOE 随造价、运营成本、折现率单调上升，随发电量单调下降。"""
    base = Economics(annual_degradation=0.0)
    v0 = lcoe(base, CAPACITY_KW, AEP_GWH)
    assert lcoe(Economics(capex_per_kw=7000.0, annual_degradation=0.0), CAPACITY_KW, AEP_GWH) > v0
    assert lcoe(Economics(opex_per_kw_year=300.0, annual_degradation=0.0), CAPACITY_KW, AEP_GWH) > v0
    assert lcoe(Economics(discount_rate=0.12, annual_degradation=0.0), CAPACITY_KW, AEP_GWH) > v0
    assert lcoe(base, CAPACITY_KW, AEP_GWH * 1.2) < v0


def test_lcoe_independent_of_electricity_price() -> None:
    """LCOE 是成本口径指标，与上网电价无关（诚实的能力边界之一）。"""
    a = Economics(electricity_price=0.2, annual_degradation=0.0)
    b = Economics(electricity_price=0.55, annual_degradation=0.0)
    assert abs(lcoe(a, CAPACITY_KW, AEP_GWH) - lcoe(b, CAPACITY_KW, AEP_GWH)) < 1e-15


def test_lcoe_lifetime_scaling() -> None:
    """寿命越长，单位电量摊到的投资越少，LCOE 越低。"""
    e = Economics(annual_degradation=0.0)
    v20 = lcoe(e, CAPACITY_KW, AEP_GWH, lifetime_years=20)
    v25 = lcoe(e, CAPACITY_KW, AEP_GWH, lifetime_years=25)
    assert v25 < v20


def test_lcoe_zero_energy_raises() -> None:
    with pytest.raises(ValueError):
        lcoe(Economics(), CAPACITY_KW, 0.0)
    with pytest.raises(ValueError):
        lcoe_closed_form(Economics(), CAPACITY_KW, 0.0)


def test_lcoe_realistic_magnitude() -> None:
    """陆上风电平价项目的 LCOE 应落在 0.2~0.5 元/kWh 量级。"""
    e = Economics()
    v = lcoe(e, CAPACITY_KW, AEP_GWH)
    assert 0.2 < v < 0.5


# ---------------------------------------------------------------- 现金流


def test_cashflow_structure() -> None:
    """CAPEX 在第 0 年，OPEX 与电量在第 1~20 年。"""
    e = Economics(lifetime_years=20, annual_degradation=0.0)
    cf = cashflow(e, CAPACITY_KW, AEP_GWH)
    assert cf.years.shape == (21,)
    assert cf.capex[0] > 0.0
    assert abs(float(np.sum(cf.capex[1:]))) < 1e-9
    assert abs(float(np.sum(cf.opex[0]))) < 1e-9
    assert np.all(cf.opex[1:] > 0.0)
    assert abs(cf.energy_kwh[1] - AEP_GWH * 1e6) < 1e-6
    assert abs(cf.revenue[1] - AEP_GWH * 1e6 * e.electricity_price) < 1e-6


def test_cashflow_degradation() -> None:
    """发电量按 (1-衰减率)^(t-1) 递减。"""
    e = Economics(lifetime_years=20, annual_degradation=0.01)
    cf = cashflow(e, CAPACITY_KW, AEP_GWH)
    assert abs(cf.energy_kwh[2] / cf.energy_kwh[1] - 0.99) < 1e-12
    assert cf.energy_kwh[20] < cf.energy_kwh[1]


# ---------------------------------------------------------------- NPV / IRR / 回收期


def test_npv_at_zero_rate_equals_undiscounted_profit() -> None:
    """折现率为 0 时 NPV = 总收入 - 总支出。"""
    e = Economics(discount_rate=0.0, lifetime_years=20, annual_degradation=0.0)
    got = npv(e, CAPACITY_KW, AEP_GWH)
    expected = (AEP_GWH * 1e6 * e.electricity_price * 20
                - e.opex_year(CAPACITY_KW) * 20
                - e.capex(CAPACITY_KW))
    assert abs(got - expected) < 1e-3


def test_npv_increases_with_price_and_decreases_with_capex() -> None:
    e = Economics()
    v = npv(e, CAPACITY_KW, AEP_GWH)
    assert npv(e, CAPACITY_KW, AEP_GWH, price=0.45) > v
    assert npv(Economics(capex_per_kw=8000.0), CAPACITY_KW, AEP_GWH) < v


def test_irr_makes_npv_zero() -> None:
    """IRR 的定义性质：在该折现率下 NPV = 0。"""
    e = Economics(capex_per_kw=5800.0, opex_per_kw_year=200.0,
                  discount_rate=0.08, lifetime_years=20, annual_degradation=0.0)
    rate = irr(e, CAPACITY_KW, AEP_GWH)
    assert np.isfinite(rate)
    assert 0.0 < rate < 1.0
    assert abs(npv(e, CAPACITY_KW, AEP_GWH, discount_rate=rate)) < 1.0
    assert rate > e.discount_rate, "该基准场景应当有正收益"


def test_irr_returns_nan_when_never_profitable() -> None:
    """电价极低时项目不盈利，IRR 无法求解。"""
    e = Economics(electricity_price=0.01)
    assert not np.isfinite(irr(e, CAPACITY_KW, AEP_GWH))


def test_payback_period_basic() -> None:
    """回收期：CAPEX / 年净收益 的插值近似。"""
    e = Economics(discount_rate=0.08, lifetime_years=20, annual_degradation=0.0)
    p = payback_period(e, CAPACITY_KW, AEP_GWH)
    annual_net = AEP_GWH * 1e6 * e.electricity_price - e.opex_year(CAPACITY_KW)
    assert abs(p - e.capex(CAPACITY_KW) / annual_net) < 0.15
    assert 3.0 < p < 12.0


def test_discounted_payback_longer_than_simple() -> None:
    e = Economics()
    assert payback_period(e, CAPACITY_KW, AEP_GWH, discounted=True) > \
        payback_period(e, CAPACITY_KW, AEP_GWH, discounted=False)


def test_payback_infinite_when_never_paid_back() -> None:
    e = Economics(electricity_price=0.001)
    assert payback_period(e, CAPACITY_KW, AEP_GWH) == float("inf")


def test_simple_metrics_keys_and_consistency() -> None:
    """汇总指标齐全且与单项函数一致。"""
    e = Economics()
    m = simple_metrics(e, CAPACITY_KW, AEP_GWH)
    for key in ("lcoe_yuan_per_kwh", "capex_yuan", "opex_yuan_per_year",
                "annual_revenue_yuan", "npv_yuan", "irr", "payback_years",
                "discounted_payback_years"):
        assert key in m
    assert abs(m["lcoe_yuan_per_kwh"] - lcoe(e, CAPACITY_KW, AEP_GWH)) < 1e-12
    assert abs(m["capex_yuan"] - e.capex(CAPACITY_KW)) < 1e-9


# ---------------------------------------------------------------- 敏感性分析


def test_sensitivity_lcoe_spans_and_ordering() -> None:
    """LCOE 龙卷风数据：造价的影响最大，且各因素跨度为正。"""
    e = Economics()
    rows = sensitivity_lcoe(e, CAPACITY_KW, AEP_GWH)
    assert len(rows) == len(DEFAULT_FACTORS)
    for r in rows:
        assert r["span"] >= 0.0
        assert abs(r["base"] - lcoe(e, CAPACITY_KW, AEP_GWH)) < 1e-12
    assert rows[0]["factor"].startswith("单位造价"), "造价应为 LCOE 第一敏感因素"
    assert rows[0]["span"] > 0.0


def test_sensitivity_lcoe_manual_check() -> None:
    """单因素敏感性的数值可手算复核（造价 +20% -> LCOE 增加约 20% 的投资分项）。"""
    e = Economics(capex_per_kw=6000.0, opex_per_kw_year=200.0,
                  discount_rate=0.08, lifetime_years=20, annual_degradation=0.0)
    rows = {r["factor"]: r for r in sensitivity_lcoe(e, CAPACITY_KW, AEP_GWH)}
    row = rows["单位造价 CAPEX"]
    capex_part = 6000.0 * CAPACITY_KW * crf(0.08, 20) / (AEP_GWH * 1e6)
    assert abs((row["high_value"] - row["base"]) - 0.2 * capex_part) < 1e-9
    assert abs(row["low_value"] - (row["base"] - 0.2 * capex_part)) < 1e-9


def test_sensitivity_lcoe_has_no_price_factor() -> None:
    """诚实性：电价不影响 LCOE，因此 LCOE 敏感性里没有电价因素。"""
    rows = sensitivity_lcoe(Economics(), CAPACITY_KW, AEP_GWH)
    assert all("电价" not in r["factor"] for r in rows)


def test_sensitivity_npv_includes_price() -> None:
    """NPV 敏感性里必须有电价，且电价是重要因素。"""
    rows = sensitivity_npv(Economics(), CAPACITY_KW, AEP_GWH)
    names = [r["factor"] for r in rows]
    assert any("电价" in n for n in names)
    assert rows[0]["span"] > 0.0
    price_row = [r for r in rows if "电价" in r["factor"]][0]
    assert price_row["high_value"] > price_row["low_value"]


def test_sensitivity_unknown_factor_raises() -> None:
    with pytest.raises(ValueError):
        sensitivity_lcoe(Economics(), CAPACITY_KW, AEP_GWH,
                         factors=[("不存在因素", 0.9, 1.1)])
    with pytest.raises(ValueError):
        sensitivity_npv(Economics(), CAPACITY_KW, AEP_GWH,
                        factors=[("不存在因素", 0.9, 1.1)])


def test_economic_summary_text() -> None:
    """中文摘要包含关键指标。"""
    text = economic_summary(Economics(), CAPACITY_KW, AEP_GWH)
    assert "LCOE" in text
    assert "NPV" in text
    assert "投资回收期" in text
    assert "CAPEX" in text
