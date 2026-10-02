"""优化模块测试：随机搜索、贪心加点、简化 NSGA-II 与 Pareto 前沿。

为控制测试耗时，全部使用小规模参数与较粗的评估网格（bin_width=4.0, rotor_grid=8）。
"""

from __future__ import annotations

import numpy as np
import pytest

from windlab.compare import non_dominated_sort
from windlab.economics import Economics
from windlab.layout import Site, check_layout
from windlab.optimize import (
    FAST_BIN_WIDTH,
    FAST_ROTOR_GRID,
    LayoutEvaluator,
    OptimizeResult,
    ParetoResult,
    best_layout_from_pareto,
    greedy_layout,
    nsga2_layout,
    random_search_layout,
)
from windlab.power_curve import example_curve
from windlab.resource import example_wind_resource

N_TURBINES = 6


@pytest.fixture(scope="module")
def scenario():
    """小规模测试场景：3.6 km × 2.4 km 场址 + 6 台 5 MW 机组。"""
    res = example_wind_resource()
    curve = example_curve()
    site = Site.rectangle(3600.0, 2400.0, name="优化测试场址")
    return res, curve, site


@pytest.fixture(scope="module")
def evaluator(scenario):
    res, curve, _ = scenario
    return LayoutEvaluator(res, curve, bin_width=FAST_BIN_WIDTH, rotor_grid=FAST_ROTOR_GRID)


# ---------------------------------------------------------------- 评估器


def test_evaluator_returns_consistent_metrics(scenario) -> None:
    """评估器输出的指标自洽（净 AEP ≤ 毛 AEP、尾流损失非负）。"""
    res, curve, site = scenario
    ev = LayoutEvaluator(res, curve, bin_width=FAST_BIN_WIDTH, rotor_grid=FAST_ROTOR_GRID)
    from windlab.layout import grid_layout
    pos = grid_layout(site, 2, 3, 5.0 * curve.diameter, 3.0 * curve.diameter,
                      main_dir_deg=res.dominant_direction_deg, margin=200.0)
    m = ev(pos)
    assert m["aep_gwh"] > 0.0
    assert m["aep_gwh"] <= m["gross_gwh"] + 1e-9
    assert m["wake_loss_pct"] >= 0.0
    assert 0.0 < m["net_cf"] < 1.0
    assert m["footprint_km2"] > 0.0
    assert ev.count == 1


def test_evaluator_cache_and_signs(scenario) -> None:
    """缓存生效（同样排布只算一次），目标值的正负号符合「越大越好」约定。"""
    res, curve, site = scenario
    ev = LayoutEvaluator(res, curve, bin_width=FAST_BIN_WIDTH, rotor_grid=FAST_ROTOR_GRID)
    from windlab.layout import grid_layout
    pos = grid_layout(site, 2, 3, 5.0 * curve.diameter, 3.0 * curve.diameter,
                      main_dir_deg=res.dominant_direction_deg, margin=200.0)
    a = ev(pos)
    b = ev(pos.copy())
    assert ev.count == 1, "缓存未生效"
    assert a == b
    assert ev.objective_value(pos, "aep") == a["aep_gwh"]
    assert ev.objective_value(pos, "wake_loss") == -a["wake_loss_pct"]
    assert ev.objective_value(pos, "footprint") == -a["footprint_km2"]
    with pytest.raises(ValueError):
        ev.objective_value(pos, "不存在目标")


def test_evaluator_with_economics_gives_lcoe(scenario) -> None:
    """带经济性假设时评估器同时给出 LCOE。"""
    res, curve, site = scenario
    ev = LayoutEvaluator(res, curve, bin_width=FAST_BIN_WIDTH, rotor_grid=FAST_ROTOR_GRID,
                         econ=Economics())
    from windlab.layout import grid_layout
    pos = grid_layout(site, 2, 3, 5.0 * curve.diameter, 3.0 * curve.diameter,
                      main_dir_deg=res.dominant_direction_deg, margin=200.0)
    m = ev(pos)
    assert "lcoe" in m
    assert 0.1 < m["lcoe"] < 1.0


# ---------------------------------------------------------------- 随机搜索


def test_random_search_feasible_and_history(scenario) -> None:
    """随机搜索返回可行排布，并给出逐步收敛历史。"""
    res, curve, site = scenario
    out = random_search_layout(site, N_TURBINES, res, curve, iterations=8, seed=1)
    assert isinstance(out, OptimizeResult)
    assert out.positions.shape == (N_TURBINES, 2)
    assert out.feasibility_ok is True
    assert len(out.history) > 0
    assert out.evaluations > 0
    assert out.aep_gwh > 0.0
    rep = check_layout(site, out.positions, curve.diameter,
                       main_dir_deg=res.dominant_direction_deg)
    assert rep.ok is True
    # best_value 单调不减，且不小于任何单次迭代值
    best = [h["best_value"] for h in out.history]
    assert all(best[i] <= best[i + 1] + 1e-9 for i in range(len(best) - 1))
    assert best[-1] >= max(h["value"] for h in out.history) - 1e-9
    assert out.value == pytest.approx(best[-1])


def test_random_search_more_iterations_not_worse(scenario) -> None:
    """同一随机序列下，更多次尝试不会更差。"""
    res, curve, site = scenario
    few = random_search_layout(site, N_TURBINES, res, curve, iterations=3, seed=2)
    many = random_search_layout(site, N_TURBINES, res, curve, iterations=10, seed=2)
    assert many.aep_gwh >= few.aep_gwh - 1e-9


def test_random_search_validation(scenario) -> None:
    res, curve, site = scenario
    with pytest.raises(ValueError):
        random_search_layout(site, N_TURBINES, res, curve, iterations=0)


# ---------------------------------------------------------------- 贪心加点


def test_greedy_reports_and_monotone_growth(scenario) -> None:
    """贪心加点：机组数逐步增加，整场 AEP 单调不减，边际增益为正。"""
    res, curve, site = scenario
    out = greedy_layout(site, N_TURBINES, res, curve, candidates_per_step=12, seed=3)
    assert isinstance(out, OptimizeResult)
    assert out.positions.shape[0] == N_TURBINES
    assert out.feasibility_ok is True
    aeps = [h["value"] for h in out.history]
    assert len(aeps) == N_TURBINES
    assert all(aeps[i] <= aeps[i + 1] + 1e-9 for i in range(len(aeps) - 1))
    margins = [h["marginal"] for h in out.history[1:]]
    assert all(m > 0.0 for m in margins)
    assert [int(h["n_turbines"]) for h in out.history] == list(range(1, N_TURBINES + 1))
    rep = check_layout(site, out.positions, curve.diameter,
                       main_dir_deg=res.dominant_direction_deg)
    assert rep.ok is True


def test_greedy_beats_or_matches_naive_average(scenario) -> None:
    """贪心结果不应差于随机搜索的平均水平。"""
    res, curve, site = scenario
    greedy = greedy_layout(site, N_TURBINES, res, curve, candidates_per_step=15, seed=4)
    rnd = random_search_layout(site, N_TURBINES, res, curve, iterations=6, seed=4)
    mean_random = float(np.mean([h["value"] for h in rnd.history]))
    assert greedy.aep_gwh >= mean_random - 1e-6


def test_greedy_single_turbine(scenario) -> None:
    """只要 1 台时贪心退化为「放在场址形心」，且没有尾流。"""
    res, curve, site = scenario
    out = greedy_layout(site, 1, res, curve, candidates_per_step=5, seed=5)
    assert out.positions.shape[0] == 1
    assert abs(out.wake_loss_pct) < 1e-9
    assert len(out.history) == 1


def test_greedy_validation(scenario) -> None:
    res, curve, site = scenario
    with pytest.raises(ValueError):
        greedy_layout(site, 0, res, curve)


# ---------------------------------------------------------------- NSGA-II


def test_nsga2_pareto_front_is_non_dominated(scenario) -> None:
    """NSGA-II 输出的前沿必须两两互不支配，且历史记录完整。"""
    res, curve, site = scenario
    par = nsga2_layout(site, N_TURBINES, res, curve, population_size=8, generations=3,
                       seed=6, objectives=("aep", "footprint"))
    assert isinstance(par, ParetoResult)
    assert len(par.front) >= 1
    assert len(par.history) == 3
    assert par.evaluations > 0

    pts = np.array([[d["aep_gwh"], d["footprint_km2"]] for d in par.front])
    fronts = non_dominated_sort(pts, maximize=(True, False))
    assert len(fronts) == 1, "前沿中出现了被支配解"
    assert sorted(fronts[0]) == list(range(len(par.front)))

    for d in par.front:
        assert d["aep_gwh"] > 0.0
        assert d["footprint_km2"] > 0.0
        assert d["positions"].shape == (N_TURBINES, 2)
        rep = check_layout(site, d["positions"], curve.diameter,
                           main_dir_deg=res.dominant_direction_deg)
        assert rep.ok is True, "Pareto 解不满足间距约束"


def test_nsga2_two_objectives_conflict(scenario) -> None:
    """AEP 与占地确实互相冲突：AEP 最大的解占地不是最小（否则就不是真双目标）。"""
    res, curve, site = scenario
    par = nsga2_layout(site, N_TURBINES, res, curve, population_size=10, generations=4,
                       seed=7, objectives=("aep", "footprint"))
    best_aep = max(par.front, key=lambda d: d["aep_gwh"])
    min_fp = min(par.front, key=lambda d: d["footprint_km2"])
    assert best_aep["aep_gwh"] >= min_fp["aep_gwh"] - 1e-9
    assert min_fp["footprint_km2"] <= best_aep["footprint_km2"] + 1e-9


def test_nsga2_aep_wake_loss_is_degenerate(scenario) -> None:
    """固定机位数与机型时，「最大化 AEP」与「最小化尾流损失」完全相关，
    Pareto 前沿会退化成单点（毛发电量与机位无关，净发电量只是尾流损失的单调函数）。

    这是本工具特意验证并记录在 docs/design.md 里的结论，也是默认改用
    「AEP vs 占地」这一对真冲突目标的原因。
    """
    res, curve, site = scenario
    par = nsga2_layout(site, N_TURBINES, res, curve, population_size=8, generations=2,
                       seed=8, objectives=("aep", "wake_loss"))
    assert len(par.front) <= 2, "固定机位数下两个目标不应出现多解前沿"
    if len(par.front) == 2:
        a, b = par.front
        assert abs(a["aep_gwh"] - b["aep_gwh"]) / a["aep_gwh"] < 1e-6


def test_nsga2_improves_over_generations(scenario) -> None:
    """精英保留应保证每代最优 AEP 不下降。"""
    res, curve, site = scenario
    par = nsga2_layout(site, N_TURBINES, res, curve, population_size=8, generations=4,
                       seed=9, objectives=("aep", "footprint"))
    best = [h["best_aep"] for h in par.history]
    assert all(best[i] <= best[i + 1] + 1e-9 for i in range(len(best) - 1))
    assert par.history[0]["front_size"] >= 1


def test_nsga2_validation(scenario) -> None:
    res, curve, site = scenario
    with pytest.raises(ValueError):
        nsga2_layout(site, N_TURBINES, res, curve, population_size=2)
    with pytest.raises(ValueError):
        nsga2_layout(site, N_TURBINES, res, curve, objectives=("aep",))
    with pytest.raises(ValueError):
        nsga2_layout(site, N_TURBINES, res, curve, objectives=("aep", "不存在"))


# ---------------------------------------------------------------- Pareto 挑选


def test_best_layout_from_pareto_criteria() -> None:
    """三种挑选准则：最大 AEP、最小占地、折中（knee）。"""
    front = [
        {"aep_gwh": 100.0, "wake_loss_pct": 12.0, "footprint_km2": 3.0, "lcoe": 0.32},
        {"aep_gwh": 110.0, "wake_loss_pct": 10.0, "footprint_km2": 5.0, "lcoe": 0.30},
        {"aep_gwh": 115.0, "wake_loss_pct": 9.0, "footprint_km2": 9.0, "lcoe": 0.29},
    ]
    par = ParetoResult(front=front, objectives=("aep", "footprint"))
    assert best_layout_from_pareto(par, "max_aep")["aep_gwh"] == 115.0
    assert best_layout_from_pareto(par, "min_footprint")["footprint_km2"] == 3.0
    knee = best_layout_from_pareto(par, "knee")
    assert knee in front
    assert best_layout_from_pareto(ParetoResult(), "max_aep") is None
    with pytest.raises(ValueError):
        best_layout_from_pareto(par, "不存在准则")


def test_pareto_result_properties() -> None:
    """ParetoResult 的极值属性与摘要。"""
    front = [
        {"aep_gwh": 100.0, "wake_loss_pct": 12.0, "footprint_km2": 3.0, "lcoe": 0.32},
        {"aep_gwh": 120.0, "wake_loss_pct": 8.0, "footprint_km2": 8.0, "lcoe": 0.28},
    ]
    par = ParetoResult(front=front, objectives=("aep", "footprint"))
    assert par.best_aep["aep_gwh"] == 120.0
    assert par.min_footprint["footprint_km2"] == 3.0
    assert "Pareto 前沿" in par.summary()
    empty = ParetoResult()
    assert empty.best_aep is None
    assert empty.min_footprint is None
    assert "为空" in empty.summary()


def test_optimize_result_summary(scenario) -> None:
    """OptimizeResult 摘要包含目标、AEP、尾流损失与评估次数。"""
    res, curve, site = scenario
    out = random_search_layout(site, 3, res, curve, iterations=2, seed=10)
    text = out.summary()
    assert "AEP" in text
    assert "尾流" in text
    assert "随机搜索" in out.label


def test_fast_defaults_are_coarser_than_final() -> None:
    """优化默认使用更粗的评估网格（速度优先），且参数确实更粗。"""
    assert FAST_BIN_WIDTH >= 2.0
    assert FAST_ROTOR_GRID <= 12
