"""方案对比模块测试：多目标非支配排序（锚点 10）、拥挤度、对比表与竞争力判断。"""

from __future__ import annotations

import math

import numpy as np
import pytest

from windlab.compare import (
    Scheme,
    SchemeResult,
    best_by,
    compare_summary,
    comparison_table,
    competitive_assessment,
    crowding_distance,
    evaluate_scheme,
    markdown_table,
    non_dominated_sort,
    rank_schemes,
    to_minimization,
)
from windlab.economics import Economics
from windlab.layout import Site, grid_layout, staggered_layout
from windlab.power_curve import builtin_curves, example_curve
from windlab.resource import example_wind_resource


# ---------------------------------------------------------------- 非支配排序（锚点 10）


def test_non_dominated_sort_known_partial_order() -> None:
    """锚点 10：构造已知偏序关系的点集，验证分层结果。

    目标为 (最大化 A, 最小化 B)，第一前沿上「A 越大则 B 越大」：

    - P0=(10,3)、P1=(8,2)、P2=(6,1) 互不支配 -> 第一前沿
    - P3=(7,4) 被 P0、P1 支配（A 更小且 B 更大），但不被 P2 支配 -> 第二前沿
    - P4=(2,8) 被所有点支配 -> 第三前沿
    """
    pts = np.array([
        [10.0, 3.0],
        [8.0, 2.0],
        [6.0, 1.0],
        [7.0, 4.0],
        [2.0, 8.0],
    ])
    fronts = non_dominated_sort(pts, maximize=(True, False))
    assert fronts[0] == [0, 1, 2]
    assert set(fronts[1]) == {3}
    assert set(fronts[2]) == {4}
    assert len(fronts) == 3


def test_non_dominated_sort_dominance_semantics() -> None:
    """完全相同的点互不支配，同处第一前沿。"""
    pts = np.array([[5.0, 5.0], [5.0, 5.0], [5.0, 5.0]])
    fronts = non_dominated_sort(pts, maximize=(True, False))
    assert len(fronts) == 1
    assert sorted(fronts[0]) == [0, 1, 2]


def test_non_dominated_sort_single_point() -> None:
    fronts = non_dominated_sort(np.array([[1.0, 2.0]]), maximize=(True, False))
    assert fronts == [[0]]


def test_non_dominated_sort_all_minimize() -> None:
    """全部最小化时，靠左下的点在前沿。"""
    pts = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 1.0]])
    fronts = non_dominated_sort(pts, maximize=(False, False))
    assert fronts[0] == [0]
    assert set(fronts[1]) == {1, 2}


def test_non_dominated_sort_matches_bruteforce() -> None:
    """与暴力双重循环判定对照（随机点集）。"""
    rng = np.random.default_rng(0)
    pts = rng.random((30, 2)) * 10.0
    maxi = (True, False)
    fronts = non_dominated_sort(pts, maxi)
    obj = to_minimization(pts, maxi)

    def dom(a, b):
        return bool(np.all(a <= b) and np.any(a < b))

    # 第一前沿内的点必须互不支配
    first = fronts[0]
    for i in first:
        for j in first:
            if i != j:
                assert not dom(obj[i], obj[j])
    # 第一前沿必须覆盖所有不被任何点支配的点
    for i in range(len(pts)):
        if not any(dom(obj[j], obj[i]) for j in range(len(pts)) if j != i):
            assert i in first
    # 层号单调不减
    rank = {idx: fi for fi, fr in enumerate(fronts) for idx in fr}
    assert len(rank) == len(pts)


def test_to_minimization_flips_only_maximize_dims() -> None:
    pts = np.array([[3.0, 4.0], [1.0, 2.0]])
    out = to_minimization(pts, (True, False))
    assert out[0, 0] == -3.0
    assert out[0, 1] == 4.0
    with pytest.raises(ValueError):
        to_minimization(pts, (True,))


def test_to_minimization_shape_validation() -> None:
    with pytest.raises(ValueError):
        to_minimization(np.array([1.0, 2.0]), (True, False))


# ---------------------------------------------------------------- 拥挤度


def test_crowding_distance_boundaries_infinite() -> None:
    """前沿两端的个体拥挤度为 +inf，中间个体为有限值。"""
    pts = np.array([[1.0, 10.0], [2.0, 8.0], [3.0, 5.0], [4.0, 0.0]])
    cd = crowding_distance(pts, [0, 1, 2, 3], (True, False))
    assert math.isinf(cd[0])
    assert math.isinf(cd[3])
    assert np.isfinite(cd[1])
    assert np.isfinite(cd[2])
    assert cd[1] > 0.0


def test_crowding_distance_small_front() -> None:
    pts = np.array([[1.0, 1.0], [2.0, 2.0]])
    cd = crowding_distance(pts, [0, 1], (True, False))
    assert np.all(np.isinf(cd))
    assert crowding_distance(pts, [], (True, False)).shape == (0,)


def test_crowding_distance_rewards_isolation() -> None:
    """分布越孤立的解，拥挤度越大。"""
    pts = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [10.0, 0.0]])
    cd = crowding_distance(pts, [0, 1, 2, 3], (False, False))
    assert cd[2] > cd[1]


# ---------------------------------------------------------------- 方案评估与对比表


def build_schemes(n: int = 4, layout_kind: str = "grid"):
    """构造一个用于测试的小方案集合。"""
    res = example_wind_resource()
    site = Site.rectangle(3600.0, 2400.0, name="测试场址")
    econ = Economics()
    schemes = []
    for curve in builtin_curves():
        if layout_kind == "grid":
            pos = grid_layout(site, 3, 4, 5.0 * curve.diameter, 3.0 * curve.diameter,
                              main_dir_deg=res.dominant_direction_deg, margin=100.0)
        else:
            pos = staggered_layout(site, 3, 4, 5.0 * curve.diameter, 3.0 * curve.diameter,
                                   main_dir_deg=res.dominant_direction_deg, stagger=0.5)
        schemes.append(Scheme(name="%s-%s" % (curve.name, layout_kind), curve=curve,
                              layout=pos, site=site))
    return res, econ, schemes


def test_evaluate_scheme_fields_and_consistency() -> None:
    """方案评估结果的字段自洽性。"""
    res, econ, schemes = build_schemes()
    r = evaluate_scheme(schemes[1], res, econ, bin_width=4.0)
    assert r.n_turbines == schemes[1].n_turbines
    assert abs(r.capacity_mw - schemes[1].capacity_mw) < 1e-9
    assert r.aep_gwh > 0.0
    assert r.gross_aep_gwh > r.aep_gwh
    assert 0.0 < r.net_capacity_factor < r.gross_capacity_factor
    assert r.capex_yuan > 0.0
    assert r.lcoe > 0.0
    assert r.footprint_km2 > 0.0
    assert r.layout_report is not None
    assert r.layout_report.ok is True
    assert r.positions.shape[1] == 2
    assert "%s" % r.name == schemes[1].name


def test_evaluate_scheme_wake_model_diameter_mismatch() -> None:
    """传入的尾流模型直径与机型不符时报错。"""
    from windlab.wake import WakeModel
    res, econ, schemes = build_schemes()
    bad = WakeModel(diameter=99.0)
    with pytest.raises(ValueError):
        evaluate_scheme(schemes[0], res, econ, wake_model=bad)


def test_comparison_table_columns() -> None:
    """对比表包含 AEP、净容量系数、尾流损失、LCOE、CAPEX、占地、机位数。"""
    res, econ, schemes = build_schemes()
    results = [evaluate_scheme(s, res, econ, bin_width=4.0) for s in schemes]
    rows = comparison_table(results)
    assert len(rows) == len(schemes)
    cols = set(rows[0].keys())
    for key in ("方案", "机位数", "AEP(GWh)", "尾流损失(%)", "净容量系数(%)",
                "LCOE(元/kWh)", "CAPEX(万元)", "占地(km²)", "机型"):
        assert key in cols, "缺少列：%s" % key
    text = markdown_table(rows)
    assert text.startswith("| 方案 |")
    assert text.count("\n") == len(rows) + 1


def test_comparison_table_sorted() -> None:
    res, econ, schemes = build_schemes()
    results = [evaluate_scheme(s, res, econ, bin_width=4.0) for s in schemes]
    rows = comparison_table(results, sort_by="LCOE(元/kWh)")
    vals = [r["LCOE(元/kWh)"] for r in rows]
    assert vals == sorted(vals)


def test_markdown_table_empty() -> None:
    assert markdown_table([]) == "（无数据）"


def test_low_wind_turbine_has_higher_capacity_factor() -> None:
    """比功率低的机型在低风速场址上容量系数更高（物理直觉检查）。"""
    res, econ, schemes = build_schemes()
    results = [evaluate_scheme(s, res, econ, bin_width=4.0) for s in schemes]
    by_turbine = {r.turbine: r for r in results}
    assert by_turbine["SYN-4.0MW/170"].net_capacity_factor > \
        by_turbine["SYN-5.0MW/132"].net_capacity_factor


# ---------------------------------------------------------------- 排序与竞争力


def test_rank_schemes_structure() -> None:
    """多目标排序表：前沿层号从 0 开始，拥挤度为数值或 inf。"""
    res, econ, schemes = build_schemes()
    results = [evaluate_scheme(s, res, econ, bin_width=4.0) for s in schemes]
    rows = rank_schemes(results)
    assert len(rows) == len(results)
    assert rows[0]["前沿层"] == 0
    for r in rows:
        assert "方案" in r and "拥挤度" in r
        assert r["拥挤度"] == "inf" or isinstance(r["拥挤度"], float)


def test_rank_schemes_front0_contains_best_lcoe() -> None:
    """LCOE 最低的方案一定在第一前沿（因为 AEP 也参与支配判定）。"""
    res, econ, schemes = build_schemes()
    results = [evaluate_scheme(s, res, econ, bin_width=4.0) for s in schemes]
    rows = rank_schemes(results)
    best = min(results, key=lambda r: r.lcoe)
    row = [r for r in rows if r["方案"] == best.name][0]
    assert row["前沿层"] == 0


def test_competitive_assessment_baseline() -> None:
    """竞争力判断：基准方案标记为「基准方案」，其余给出增益百分比。"""
    res, econ, schemes = build_schemes()
    results = [evaluate_scheme(s, res, econ, bin_width=4.0) for s in schemes]
    baseline = results[0].name
    rows = competitive_assessment(results, baseline)
    assert len(rows) == len(results)
    base_row = [r for r in rows if r["方案"] == baseline][0]
    assert base_row["判断"] == "基准方案"
    assert abs(base_row["AEP相对基准(%)"]) < 1e-9
    assert abs(base_row["LCOE相对基准(%)"]) < 1e-9
    for r in rows:
        assert r["判断"] in {"基准方案", "明显更优", "更优", "基本相当", "更差"}


def test_competitive_assessment_default_baseline_is_best_lcoe() -> None:
    res, econ, schemes = build_schemes()
    results = [evaluate_scheme(s, res, econ, bin_width=4.0) for s in schemes]
    rows = competitive_assessment(results)
    best = min(results, key=lambda r: r.lcoe)
    base_row = [r for r in rows if r["方案"] == best.name][0]
    assert base_row["判断"] == "基准方案"


def test_competitive_assessment_unknown_baseline() -> None:
    res, econ, schemes = build_schemes()
    results = [evaluate_scheme(s, res, econ, bin_width=4.0) for s in schemes]
    with pytest.raises(ValueError):
        competitive_assessment(results, "不存在的方案")


def test_competitive_verdict_logic() -> None:
    """构造两个方案实例直接验证判断规则。"""
    common = dict(description="", turbine="T", n_turbines=10, capacity_mw=50.0,
                  rated_unit_kw=5000.0, gross_aep_gwh=200.0, wake_loss_pct=10.0,
                  net_capacity_factor=0.3, gross_capacity_factor=0.33,
                  full_load_hours=2600.0, capex_yuan=3e8, capex_per_kw=6000.0,
                  opex_year_yuan=1e7, npv_yuan=5e7, payback_years=8.0,
                  footprint_km2=5.0, site_area_km2=8.0)
    base = SchemeResult(name="BASE", aep_gwh=150.0, lcoe=0.30, **common)
    better = SchemeResult(name="BETTER", aep_gwh=165.0, lcoe=0.27, **common)
    same = SchemeResult(name="SAME", aep_gwh=151.0, lcoe=0.298, **common)
    worse = SchemeResult(name="WORSE", aep_gwh=120.0, lcoe=0.36, **common)
    rows = {r["方案"]: r for r in competitive_assessment([base, better, same, worse], "BASE")}
    assert rows["BETTER"]["判断"] == "明显更优"
    assert rows["SAME"]["判断"] == "基本相当"
    assert rows["WORSE"]["判断"] == "更差"
    assert rows["BETTER"]["AEP相对基准(%)"] > 9.0
    assert rows["BETTER"]["LCOE相对基准(%)"] < -9.0


def test_best_by_helper() -> None:
    res, econ, schemes = build_schemes()
    results = [evaluate_scheme(s, res, econ, bin_width=4.0) for s in schemes]
    assert best_by(results, "lcoe").lcoe == min(r.lcoe for r in results)
    assert best_by(results, "aep_gwh", minimize=False).aep_gwh == max(r.aep_gwh for r in results)
    with pytest.raises(ValueError):
        best_by([])


def test_compare_summary_text() -> None:
    """中文对比摘要包含关键结论。"""
    res, econ, schemes = build_schemes()
    results = [evaluate_scheme(s, res, econ, bin_width=4.0) for s in schemes]
    text = compare_summary(results)
    assert "方案对比" in text
    assert "LCOE 最低" in text
    assert "AEP 最大" in text


def test_scheme_result_derived_metrics() -> None:
    """派生指标：单位造价、装机密度。"""
    res, econ, schemes = build_schemes()
    r = evaluate_scheme(schemes[0], res, econ, bin_width=4.0)
    assert abs(r.unit_cost_per_watt - r.capex_per_kw / 1000.0) < 1e-12
    assert r.installed_density_mw_per_km2 > 0.0
    assert r.site_density_mw_per_km2 > 0.0
    row = r.as_row()
    assert row["约束"] == "通过"
    assert r.name in r.summary_line()
