"""排布优化模块：随机搜索、贪心加点、简化 NSGA-II（双目标 Pareto 前沿）。

三种算法（都会输出迭代收敛曲线数据）
------------------------------------
1. :func:`random_search_layout` —— 随机搜索：反复生成满足间距约束的随机排布，取最优。
   优点是简单、能跳出局部最优；缺点是没有利用问题结构。
2. :func:`greedy_layout` —— 贪心加点：从 1 台开始，每次把新机组放到「当前边际 AEP 增益最大」
   的候选位置（候选点在可行域内随机生成 + 拒绝采样）。贪心在「AEP 对机位近似可加」时
   接近最优，但会过早锁定早期机位。
3. :func:`nsga2_layout` —— 简化 NSGA-II：非支配排序 + 拥挤度 + 锦标赛选择 + 混合交叉 +
   高斯变异，输出 Pareto 前沿。

关于双目标的选取（重要）
------------------------
    若固定机位数与机型，**毛发电量与机位无关**，只由风资源决定；净发电量
    ``AEP_net = AEP_gross × (1 - 尾流损失)`` 因此是尾流损失的单调函数。
    所以在「固定机位数」的前提下，「最大化 AEP」与「最小化尾流损失」是**完全相关**的
    两个目标，Pareto 前沿会退化成单点（这本身是一个正确的结论，见 docs/design.md）。
    因此本模块默认改用真正互相冲突的一对目标：
    **最大化净 AEP + 最小化占地凸包面积**（铺得开能减尾流、但占地大）。
    ``objectives=("aep", "wake_loss")`` 依然被支持，用于验证上述退化结论。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .aep import DEFAULT_AVAILABILITY, DEFAULT_OTHER_LOSS_FACTOR, evaluate_aep
from .compare import crowding_distance, non_dominated_sort
from .economics import Economics, lcoe as lcoe_fn
from .layout import (
    Site,
    footprint_area,
    random_layout,
    repair_layout,
    check_layout,
)
from .power_curve import PowerCurve
from .resource import WindResource
from .wake import WakeModel

#: 优化用默认的快速评估参数（尾流网格更粗、风速分仓更宽）
FAST_BIN_WIDTH = 4.0
FAST_ROTOR_GRID = 8

OBJECTIVE_SIGNS: Dict[str, bool] = {
    "aep": True,          # True = 越大越好
    "wake_loss": False,
    "footprint": False,
    "lcoe": False,
}


# ---------------------------------------------------------------- 结果对象


@dataclass
class OptimizeResult:
    """单目标优化结果。"""

    label: str
    positions: np.ndarray
    objective: str
    value: float
    aep_gwh: float
    wake_loss_pct: float
    footprint_km2: float
    lcoe: Optional[float] = None
    history: List[Dict[str, float]] = field(default_factory=list)
    evaluations: int = 0
    feasibility_ok: bool = True

    def summary(self) -> str:
        return (
            "%-22s 目标(%s)=%.4f  AEP %.2f GWh  尾流 %.2f%%  占地 %.3f km²  "
            "评估 %d 次  可行 %s"
            % (self.label, self.objective, self.value, self.aep_gwh, self.wake_loss_pct,
               self.footprint_km2, self.evaluations, "是" if self.feasibility_ok else "否")
        )


@dataclass
class ParetoResult:
    """NSGA-II 结果：Pareto 前沿 + 收敛历史。"""

    front: List[Dict[str, float]] = field(default_factory=list)
    history: List[Dict[str, float]] = field(default_factory=list)
    evaluations: int = 0
    objectives: Tuple[str, str] = ("aep", "footprint")
    population_size: int = 0
    generations: int = 0

    @property
    def best_aep(self) -> Optional[Dict[str, float]]:
        if not self.front:
            return None
        return max(self.front, key=lambda d: d["aep_gwh"])

    @property
    def min_footprint(self) -> Optional[Dict[str, float]]:
        if not self.front:
            return None
        return min(self.front, key=lambda d: d["footprint_km2"])

    def summary(self) -> str:
        if not self.front:
            return "Pareto 前沿为空"
        return (
            "Pareto 前沿 %d 个非支配解（目标：%s 最大 vs %s 最小）："
            "AEP %.2f ~ %.2f GWh，占地 %.3f ~ %.3f km²，评估 %d 次"
            % (len(self.front), self.objectives[0], self.objectives[1],
               min(d["aep_gwh"] for d in self.front), max(d["aep_gwh"] for d in self.front),
               min(d["footprint_km2"] for d in self.front),
               max(d["footprint_km2"] for d in self.front), self.evaluations)
        )


# ---------------------------------------------------------------- 评估器


class LayoutEvaluator:
    """带缓存的排布评估器（优化循环里会调用几百上千次，必须快）。"""

    def __init__(
        self,
        resource: WindResource,
        curve: PowerCurve,
        wake_decay: float = 0.075,
        bin_width: float = FAST_BIN_WIDTH,
        rotor_grid: int = FAST_ROTOR_GRID,
        availability: float = DEFAULT_AVAILABILITY,
        other_loss_factor: float = DEFAULT_OTHER_LOSS_FACTOR,
        econ: Optional[Economics] = None,
        cache: bool = True,
    ) -> None:
        self.resource = resource
        self.curve = curve
        self.model = WakeModel(diameter=curve.diameter, k_wake=wake_decay, rotor_grid=rotor_grid)
        self.bin_width = float(bin_width)
        self.availability = float(availability)
        self.other_loss_factor = float(other_loss_factor)
        self.econ = econ
        self.cache = cache
        self._cache: Dict[bytes, Dict[str, float]] = {}
        self.count = 0

    def _key(self, pos: np.ndarray) -> bytes:
        return np.round(np.asarray(pos, dtype=np.float64), 3).tobytes()

    def __call__(self, pos: np.ndarray) -> Dict[str, float]:
        key = self._key(pos)
        if self.cache and key in self._cache:
            return self._cache[key]
        res = evaluate_aep(
            pos, self.resource, self.curve, self.model,
            availability=self.availability,
            other_loss_factor=self.other_loss_factor,
            bin_width=self.bin_width,
            include_wake=True,
            collect_per_turbine=False,
        )
        out = {
            "aep_gwh": float(res.net_gwh),
            "gross_gwh": float(res.gross_gwh),
            "wake_loss_pct": float(res.wake_loss_pct),
            "net_cf": float(res.net_capacity_factor),
            "footprint_km2": footprint_area(np.atleast_2d(pos)) / 1e6,
        }
        if self.econ is not None:
            out["lcoe"] = float(lcoe_fn(self.econ, res.capacity_mw * 1000.0, res.net_gwh))
        self.count += 1
        if self.cache:
            self._cache[key] = out
        return out

    def objective_value(self, pos: np.ndarray, objective: str = "aep") -> float:
        """把评估结果转成「越大越好」的目标值。"""
        m = self(pos)
        if objective == "aep":
            return m["aep_gwh"]
        if objective == "wake_loss":
            return -m["wake_loss_pct"]
        if objective == "footprint":
            return -m["footprint_km2"]
        if objective == "lcoe":
            return -float(m.get("lcoe", 0.0))
        raise ValueError("未知目标：%s" % objective)


# ---------------------------------------------------------------- 随机搜索


def random_search_layout(
    site: Site,
    n_turbines: int,
    resource: WindResource,
    curve: PowerCurve,
    iterations: int = 100,
    seed: int = 0,
    objective: str = "aep",
    bin_width: float = FAST_BIN_WIDTH,
    rotor_grid: int = FAST_ROTOR_GRID,
    wake_decay: float = 0.075,
    main_dir_deg: Optional[float] = None,
    min_spacing_main: float = 5.0,
    min_spacing_cross: float = 3.0,
    evaluator: Optional[LayoutEvaluator] = None,
) -> OptimizeResult:
    """随机搜索：多次随机生成可行排布，取目标最优的一个。"""
    if iterations < 1:
        raise ValueError("iterations 必须 >= 1")
    rng = np.random.default_rng(seed)
    direction = resource.dominant_direction_deg if main_dir_deg is None else float(main_dir_deg)
    ev = evaluator or LayoutEvaluator(resource, curve, wake_decay=wake_decay,
                                      bin_width=bin_width, rotor_grid=rotor_grid)

    best_pos: Optional[np.ndarray] = None
    best_val = -float("inf")
    history: List[Dict[str, float]] = []
    for it in range(1, int(iterations) + 1):
        try:
            pos = random_layout(
                site, n_turbines, curve.diameter, rng,
                min_spacing_main=min_spacing_main, min_spacing_cross=min_spacing_cross,
                main_dir_deg=direction,
            )
        except RuntimeError:
            continue
        val = ev.objective_value(pos, objective)
        if val > best_val:
            best_val = val
            best_pos = pos.copy()
        history.append({
            "iteration": float(it),
            "value": float(val),
            "best_value": float(best_val),
        })

    if best_pos is None:
        raise RuntimeError("随机搜索未能生成任何可行排布，请放宽间距约束或扩大场址")
    m = ev(best_pos)
    report = check_layout(site, best_pos, curve.diameter, main_dir_deg=direction)
    return OptimizeResult(
        label="随机搜索(%d 次)" % int(iterations),
        positions=best_pos,
        objective=objective,
        value=float(best_val),
        aep_gwh=m["aep_gwh"],
        wake_loss_pct=m["wake_loss_pct"],
        footprint_km2=m["footprint_km2"],
        lcoe=m.get("lcoe"),
        history=history,
        evaluations=ev.count,
        feasibility_ok=report.ok,
    )


# ---------------------------------------------------------------- 贪心加点


def greedy_layout(
    site: Site,
    n_turbines: int,
    resource: WindResource,
    curve: PowerCurve,
    candidates_per_step: int = 60,
    seed: int = 0,
    objective: str = "aep",
    bin_width: float = FAST_BIN_WIDTH,
    rotor_grid: int = FAST_ROTOR_GRID,
    wake_decay: float = 0.075,
    main_dir_deg: Optional[float] = None,
    min_spacing_main: float = 5.0,
    min_spacing_cross: float = 3.0,
    evaluator: Optional[LayoutEvaluator] = None,
) -> OptimizeResult:
    """贪心加点：逐台增加机组，每步把新机组放到当前边际目标增益最大的候选位置。

    第 1 台机组放在场址形心（单台机组无尾流，位置不影响发电量）。
    返回的 ``history`` 记录每一步的机组数、整场 AEP 与边际增益。
    """
    if n_turbines < 1:
        raise ValueError("机位数必须 >= 1")
    rng = np.random.default_rng(seed)
    direction = resource.dominant_direction_deg if main_dir_deg is None else float(main_dir_deg)
    ev = evaluator or LayoutEvaluator(resource, curve, wake_decay=wake_decay,
                                      bin_width=bin_width, rotor_grid=rotor_grid)

    placed = [site.centroid.copy()]
    history: List[Dict[str, float]] = []
    prev_val = ev.objective_value(np.array(placed), objective) if objective == "aep" else 0.0
    history.append({
        "step": 1.0, "n_turbines": 1.0,
        "value": float(prev_val), "marginal": float(prev_val),
    })

    while len(placed) < n_turbines:
        best_cand = None
        best_gain = -float("inf")
        best_val = -float("inf")
        tries = 0
        max_tries = int(candidates_per_step) * 20
        while tries < max_tries and (best_cand is None or tries < int(candidates_per_step)):
            tries += 1
            try:
                cand = random_layout(
                    site, 1, curve.diameter, rng,
                    min_spacing_main=min_spacing_main, min_spacing_cross=min_spacing_cross,
                    main_dir_deg=direction, max_attempts=200, strict=True,
                )
            except RuntimeError:
                continue
            trial = np.vstack([np.array(placed), cand])
            # 检查与已有机组的间距
            rep = check_layout(site, trial, curve.diameter,
                               min_spacing_main=min_spacing_main,
                               min_spacing_cross=min_spacing_cross,
                               main_dir_deg=direction)
            if not rep.ok:
                continue
            val = ev.objective_value(trial, objective)
            if val - prev_val > best_gain:
                best_gain = val - prev_val
                best_val = val
                best_cand = cand
        if best_cand is None:
            break
        placed.append(best_cand[0].copy())
        prev_val = best_val
        history.append({
            "step": float(len(placed)), "n_turbines": float(len(placed)),
            "value": float(best_val), "marginal": float(best_gain),
        })

    pos = np.array(placed, dtype=np.float64)
    m = ev(pos)
    report = check_layout(site, pos, curve.diameter, main_dir_deg=direction)
    return OptimizeResult(
        label="贪心加点(%d 台)" % pos.shape[0],
        positions=pos,
        objective=objective,
        value=float(ev.objective_value(pos, objective)),
        aep_gwh=m["aep_gwh"],
        wake_loss_pct=m["wake_loss_pct"],
        footprint_km2=m["footprint_km2"],
        lcoe=m.get("lcoe"),
        history=history,
        evaluations=ev.count,
        feasibility_ok=report.ok,
    )


# ---------------------------------------------------------------- NSGA-II


def _repair(site: Site, pos: np.ndarray, curve: PowerCurve, direction: float,
            min_spacing_main: float, min_spacing_cross: float, iterations: int = 40) -> np.ndarray:
    return repair_layout(
        site, pos, curve.diameter,
        min_spacing_main=min_spacing_main, min_spacing_cross=min_spacing_cross,
        main_dir_deg=direction, iterations=iterations,
    )


def nsga2_layout(
    site: Site,
    n_turbines: int,
    resource: WindResource,
    curve: PowerCurve,
    population_size: int = 24,
    generations: int = 12,
    seed: int = 0,
    objectives: Tuple[str, str] = ("aep", "footprint"),
    bin_width: float = FAST_BIN_WIDTH,
    rotor_grid: int = FAST_ROTOR_GRID,
    wake_decay: float = 0.075,
    main_dir_deg: Optional[float] = None,
    min_spacing_main: float = 5.0,
    min_spacing_cross: float = 3.0,
    crossover_rate: float = 0.9,
    mutation_scale: float = 0.6,
    evaluator: Optional[LayoutEvaluator] = None,
) -> ParetoResult:
    """简化 NSGA-II 排布优化（双目标 Pareto 前沿）。

    Args:
        objectives: 两个目标键，取值范围 ``aep``/``wake_loss``/``footprint``/``lcoe``。
        crossover_rate: 交叉概率（BLX-0.3 混合交叉）。
        mutation_scale: 变异标准差 = ``mutation_scale × 5D``（沿主风向的基准间距）。

    交叉/变异后统一做间距修复（:func:`windlab.layout.repair_layout`），保证所有个体可行。
    """
    if len(objectives) != 2:
        raise ValueError("本实现是双目标 NSGA-II")
    for o in objectives:
        if o not in OBJECTIVE_SIGNS:
            raise ValueError("未知目标：%s" % o)
    if population_size < 4:
        raise ValueError("种群规模至少 4")

    rng = np.random.default_rng(seed)
    direction = resource.dominant_direction_deg if main_dir_deg is None else float(main_dir_deg)
    ev = evaluator or LayoutEvaluator(resource, curve, wake_decay=wake_decay,
                                      bin_width=bin_width, rotor_grid=rotor_grid)
    maximize = tuple(OBJECTIVE_SIGNS[o] for o in objectives)
    sigma = float(mutation_scale) * float(min_spacing_main) * curve.diameter

    def make_individual() -> np.ndarray:
        for _ in range(50):
            try:
                return random_layout(
                    site, n_turbines, curve.diameter, rng,
                    min_spacing_main=min_spacing_main, min_spacing_cross=min_spacing_cross,
                    main_dir_deg=direction, max_attempts=3000,
                )
            except RuntimeError:
                continue
        return np.full((n_turbines, 2), np.nan)

    def objectives_of(pos: np.ndarray) -> np.ndarray:
        m = ev(pos)
        vals = []
        for o in objectives:
            if o == "aep":
                vals.append(m["aep_gwh"])
            elif o == "wake_loss":
                vals.append(m["wake_loss_pct"])
            elif o == "footprint":
                vals.append(m["footprint_km2"])
            else:
                vals.append(float(m.get("lcoe", 0.0)))
        return np.array(vals, dtype=np.float64)

    # -- 初始化种群 ------------------------------------------------
    pop: List[np.ndarray] = []
    obj: List[np.ndarray] = []
    attempts = 0
    while len(pop) < population_size:
        attempts += 1
        if attempts > 100 * population_size:
            raise RuntimeError(
                "NSGA-II 初始化失败：无法在 %d 次尝试内生成 %d 台机组的可行排布，"
                "请放宽间距约束或扩大场址" % (attempts, n_turbines)
            )
        ind = make_individual()
        if np.any(~np.isfinite(ind)):
            continue
        pop.append(ind)
        obj.append(objectives_of(ind))
    obj_arr = np.array(obj, dtype=np.float64)

    history: List[Dict[str, float]] = []

    def rank_and_crowd(pop_arr: np.ndarray, obj_m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        fronts = non_dominated_sort(obj_m, maximize)
        n_ind = int(np.asarray(obj_m).shape[0])
        rank = np.zeros(n_ind, dtype=np.int64)
        crowd = np.zeros(n_ind, dtype=np.float64)
        for fi, front in enumerate(fronts):
            cd = crowding_distance(obj_m, front, maximize)
            for k, idx in enumerate(front):
                rank[idx] = fi
                crowd[idx] = cd[k] if not math.isinf(cd[k]) else 1e12
        return rank, crowd

    for gen in range(int(generations)):
        rank, crowd = rank_and_crowd(np.array(pop), obj_arr)

        # 二元锦标赛选择
        def tournament() -> int:
            a, b = rng.integers(0, len(pop), size=2)
            if rank[a] < rank[b]:
                return int(a)
            if rank[b] < rank[a]:
                return int(b)
            return int(a if crowd[a] >= crowd[b] else b)

        offspring: List[np.ndarray] = []
        while len(offspring) < population_size:
            p1, p2 = pop[tournament()], pop[tournament()]
            if rng.random() < crossover_rate:
                alpha = 0.3
                lo = np.minimum(p1, p2) - alpha * np.abs(p1 - p2)
                hi = np.maximum(p1, p2) + alpha * np.abs(p1 - p2)
                child = rng.uniform(lo, hi)
            else:
                child = p1.copy()
            # 变异
            n_mut = max(1, int(0.3 * n_turbines))
            idx = rng.choice(n_turbines, size=n_mut, replace=False)
            child[idx] += rng.normal(0.0, sigma, size=(n_mut, 2))
            child = _repair(site, child, curve, direction, min_spacing_main, min_spacing_cross)
            if not np.all(site.contains(child)):
                child = _repair(site, child, curve, direction,
                                min_spacing_main, min_spacing_cross, iterations=80)
            if not np.all(site.contains(child)):
                child = p1.copy()
            offspring.append(child)

        off_obj = np.array([objectives_of(c) for c in offspring], dtype=np.float64)

        # 精英保留：父代 + 子代 一起做非支配排序
        merged_pop = pop + offspring
        merged_obj = np.vstack([obj_arr, off_obj])
        fronts = non_dominated_sort(merged_obj, maximize)
        new_pop: List[np.ndarray] = []
        new_obj: List[np.ndarray] = []
        for front in fronts:
            if len(new_pop) + len(front) <= population_size:
                for i in front:
                    new_pop.append(merged_pop[i])
                    new_obj.append(merged_obj[i])
            else:
                cd = crowding_distance(merged_obj, front, maximize)
                order = np.argsort(-np.array([0.0 if math.isinf(c) else c for c in cd]))
                need = population_size - len(new_pop)
                for k in order[:need]:
                    i = front[int(k)]
                    new_pop.append(merged_pop[i])
                    new_obj.append(merged_obj[i])
                break
        pop, obj_arr = new_pop, np.array(new_obj, dtype=np.float64)

        front0 = [d for d in obj_arr[non_dominated_sort(obj_arr, maximize)[0]]]
        best_aep = max(float(o[objectives.index("aep")]) for o in obj_arr) \
            if "aep" in objectives else float("nan")
        history.append({
            "generation": float(gen + 1),
            "front_size": float(len(front0)),
            "best_aep": best_aep,
            "mean_aep": float(np.mean(obj_arr[:, objectives.index("aep")])) if "aep" in objectives
            else float("nan"),
        })

    fronts = non_dominated_sort(obj_arr, maximize)
    front_rows: List[Dict[str, float]] = []
    for i in fronts[0]:
        pos = pop[i]
        m = ev(pos)
        front_rows.append({
            "aep_gwh": float(m["aep_gwh"]),
            "wake_loss_pct": float(m["wake_loss_pct"]),
            "footprint_km2": float(m["footprint_km2"]),
            "lcoe": float(m.get("lcoe", float("nan"))),
            "positions": pos.copy(),
        })
    front_rows.sort(key=lambda d: d["aep_gwh"])

    return ParetoResult(
        front=front_rows,
        history=history,
        evaluations=ev.count,
        objectives=objectives,
        population_size=int(population_size),
        generations=int(generations),
    )


def best_layout_from_pareto(
    pareto: ParetoResult,
    criterion: str = "max_aep",
) -> Optional[Dict[str, float]]:
    """从 Pareto 前沿中按偏好挑选一个方案：``max_aep`` / ``min_footprint`` / ``knee``。

    ``knee`` 用「离理想点最近」的折中解（各目标归一化后取欧氏距离最小）。
    """
    if not pareto.front:
        return None
    if criterion == "max_aep":
        return max(pareto.front, key=lambda d: d["aep_gwh"])
    if criterion == "min_footprint":
        return min(pareto.front, key=lambda d: d["footprint_km2"])
    if criterion == "knee":
        aeps = np.array([d["aep_gwh"] for d in pareto.front])
        fps = np.array([d["footprint_km2"] for d in pareto.front])
        a_norm = (aeps - aeps.min()) / max(aeps.max() - aeps.min(), 1e-12)
        f_norm = (fps - fps.min()) / max(fps.max() - fps.min(), 1e-12)
        dist = np.sqrt((1 - a_norm) ** 2 + f_norm ** 2)
        return pareto.front[int(np.argmin(dist))]
    raise ValueError("未知的挑选准则：%s" % criterion)
