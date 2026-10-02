"""方案对比模块：多方案横向对比表、多目标非支配排序（NSGA-II 前两步）、竞争力判断。

组合方式
--------
一个「方案」= 机型（功率曲线）× 机位排布 × 机位数 × 经济性假设。
本模块把每个方案跑一遍 AEP + 经济性，汇总成对比表，再做多目标排序。

多目标排序
----------
对「AEP 越大越好、LCOE 越小越好」这类混合方向的目标，先统一转成**最小化**方向，
再做非支配排序（fast non-dominated sort）得到分层前沿，然后对每一层计算拥挤度
（crowding distance）。这正是 NSGA-II 的前两步，用于在目标空间里挑出
「互不支配 + 分布均匀」的方案。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .aep import AEPResult, DEFAULT_AVAILABILITY, DEFAULT_OTHER_LOSS_FACTOR, evaluate_aep
from .economics import (
    Economics,
    lcoe,
    npv,
    payback_period,
    simple_metrics,
)
from .layout import LayoutReport, Site, check_layout, footprint_area
from .power_curve import PowerCurve
from .resource import WindResource
from .wake import WakeModel


# ---------------------------------------------------------------- 方案定义


@dataclass
class Scheme:
    """一个待评估的方案。

    Attributes:
        name: 方案名（唯一，用于对比表）。
        curve: 机型功率曲线。
        layout: ``(N, 2)`` 机位坐标（m）。
        site: 场址（可选，用于越界与占地检查）。
        description: 方案说明。
        availability: 可用率。
        other_loss_factor: 其他综合折减系数。
        wake_decay: 尾流衰减系数（陆上 0.075）。
        rotor_grid: 尾流计算的风轮离散网格（越大越准、越慢）。
    """

    name: str
    curve: PowerCurve
    layout: np.ndarray
    site: Optional[Site] = None
    description: str = ""
    availability: float = DEFAULT_AVAILABILITY
    other_loss_factor: float = DEFAULT_OTHER_LOSS_FACTOR
    wake_decay: float = 0.075
    rotor_grid: int = 20

    @property
    def n_turbines(self) -> int:
        return int(np.atleast_2d(np.asarray(self.layout)).shape[0])

    @property
    def capacity_mw(self) -> float:
        return self.n_turbines * self.curve.rated_power / 1000.0


@dataclass
class SchemeResult:
    """方案评估结果（对比表的一行）。"""

    name: str
    description: str
    turbine: str
    n_turbines: int
    capacity_mw: float
    rated_unit_kw: float
    aep_gwh: float
    gross_aep_gwh: float
    wake_loss_pct: float
    net_capacity_factor: float
    gross_capacity_factor: float
    full_load_hours: float
    lcoe: float
    capex_yuan: float
    capex_per_kw: float
    opex_year_yuan: float
    npv_yuan: float
    payback_years: float
    footprint_km2: float
    site_area_km2: float
    layout_report: Optional[LayoutReport] = None
    availability: float = DEFAULT_AVAILABILITY
    other_loss_factor: float = DEFAULT_OTHER_LOSS_FACTOR
    aep_result: Optional[AEPResult] = None
    positions: Optional[np.ndarray] = None
    row_extra: Dict[str, float] = field(default_factory=dict)

    # -- 派生指标 -------------------------------------------------

    @property
    def unit_cost_per_watt(self) -> float:
        """单位造价（元/W）。"""
        return self.capex_per_kw / 1000.0

    @property
    def installed_density_mw_per_km2(self) -> float:
        """按机位凸包占地计算的装机密度（MW/km²）。"""
        if self.footprint_km2 <= 0:
            return float("inf")
        return self.capacity_mw / self.footprint_km2

    @property
    def site_density_mw_per_km2(self) -> float:
        """按场址面积计算的装机密度（MW/km²）。"""
        if self.site_area_km2 <= 0:
            return float("inf")
        return self.capacity_mw / self.site_area_km2

    @property
    def annual_revenue_yuan(self) -> float:
        return self.aep_gwh * 1e6 * self.row_extra.get("price", 0.0)

    def as_row(self) -> Dict[str, object]:
        """对比表一行（中文列名）。"""
        return {
            "方案": self.name,
            "机型": self.turbine,
            "机位数": self.n_turbines,
            "装机_MW": round(self.capacity_mw, 2),
            "AEP_GWh": round(self.aep_gwh, 2),
            "毛AEP_GWh": round(self.gross_aep_gwh, 2),
            "尾流损失_%": round(self.wake_loss_pct, 2),
            "净容量系数_%": round(100.0 * self.net_capacity_factor, 2),
            "满发小时_h": round(self.full_load_hours, 0),
            "LCOE_元每kWh": round(self.lcoe, 4),
            "CAPEX_万元": round(self.capex_yuan / 1e4, 0),
            "单位造价_元每kW": round(self.capex_per_kw, 0),
            "占地_km2": round(self.footprint_km2, 3),
            "NPV_万元": round(self.npv_yuan / 1e4, 0),
            "回收期_年": round(self.payback_years, 1),
            "约束": ("通过" if (self.layout_report is None or self.layout_report.ok) else "不通过"),
        }

    def summary_line(self) -> str:
        return (
            "%-26s %2d 台  %6.1f MW  AEP %7.2f GWh  尾流 %4.2f%%  CF %5.2f%%  "
            "LCOE %.4f 元/kWh  CAPEX %8.0f 万元"
            % (self.name, self.n_turbines, self.capacity_mw, self.aep_gwh,
               self.wake_loss_pct, 100.0 * self.net_capacity_factor, self.lcoe,
               self.capex_yuan / 1e4)
        )


# ---------------------------------------------------------------- 方案评估


def evaluate_scheme(
    scheme: Scheme,
    resource: WindResource,
    econ: Economics,
    bin_width: float = 1.0,
    main_dir_deg: Optional[float] = None,
    wake_model: Optional[WakeModel] = None,
) -> SchemeResult:
    """跑通一个方案的 AEP + 约束检查 + 经济性，返回 :class:`SchemeResult`。"""
    pos = np.atleast_2d(np.asarray(scheme.layout, dtype=np.float64))
    curve = scheme.curve
    model = wake_model
    if model is None:
        model = WakeModel(diameter=curve.diameter, k_wake=scheme.wake_decay,
                          rotor_grid=scheme.rotor_grid)
    elif abs(model.diameter - curve.diameter) > 1e-6:
        raise ValueError("传入的尾流模型直径 %.2f 与机型直径 %.2f 不一致"
                         % (model.diameter, curve.diameter))

    aep_res = evaluate_aep(
        pos, resource, curve, model,
        availability=scheme.availability,
        other_loss_factor=scheme.other_loss_factor,
        bin_width=bin_width,
        include_wake=True,
    )
    no_wake = evaluate_aep(
        pos, resource, curve, None,
        availability=scheme.availability,
        other_loss_factor=scheme.other_loss_factor,
        bin_width=bin_width,
        include_wake=False,
        collect_per_turbine=False,
    )
    capacity_kw = float(aep_res.capacity_mw) * 1000.0
    metrics = simple_metrics(econ, capacity_kw, aep_res.net_gwh)

    report = None
    if scheme.site is not None:
        direction = resource.dominant_direction_deg if main_dir_deg is None else float(main_dir_deg)
        report = check_layout(scheme.site, pos, curve.diameter, main_dir_deg=direction)

    fp = footprint_area(pos) / 1e6
    site_area = scheme.site.area_km2 if scheme.site is not None else 0.0

    return SchemeResult(
        name=scheme.name,
        description=scheme.description,
        turbine=curve.name,
        n_turbines=aep_res.n_turbines,
        capacity_mw=aep_res.capacity_mw,
        rated_unit_kw=float(curve.rated_power),
        aep_gwh=aep_res.net_gwh,
        gross_aep_gwh=no_wake.net_gwh,
        wake_loss_pct=aep_res.wake_loss_pct,
        net_capacity_factor=aep_res.net_capacity_factor,
        gross_capacity_factor=no_wake.net_capacity_factor,
        full_load_hours=aep_res.full_load_hours,
        lcoe=metrics["lcoe_yuan_per_kwh"],
        capex_yuan=metrics["capex_yuan"],
        capex_per_kw=float(econ.capex_per_kw),
        opex_year_yuan=metrics["opex_yuan_per_year"],
        npv_yuan=metrics["npv_yuan"],
        payback_years=metrics["payback_years"],
        footprint_km2=fp,
        site_area_km2=site_area,
        layout_report=report,
        availability=scheme.availability,
        other_loss_factor=scheme.other_loss_factor,
        aep_result=aep_res,
        positions=pos,
        row_extra={"price": float(econ.electricity_price)},
    )


# ---------------------------------------------------------------- 对比表


#: 对比表的列顺序（中文列名 -> 数值键）
COMPARISON_COLUMNS: Tuple[Tuple[str, str, str], ...] = (
    ("方案", "name", "s"),
    ("机型", "turbine", "s"),
    ("机位数", "n_turbines", "d"),
    ("装机(MW)", "capacity_mw", ".2f"),
    ("AEP(GWh)", "aep_gwh", ".2f"),
    ("毛AEP(GWh)", "gross_aep_gwh", ".2f"),
    ("尾流损失(%)", "wake_loss_pct", ".2f"),
    ("净容量系数(%)", "net_cf_pct", ".2f"),
    ("满发小时(h)", "full_load_hours", ".0f"),
    ("LCOE(元/kWh)", "lcoe", ".4f"),
    ("CAPEX(万元)", "capex_wan", ".0f"),
    ("单位造价(元/kW)", "capex_per_kw", ".0f"),
    ("占地(km²)", "footprint_km2", ".3f"),
    ("NPV(万元)", "npv_wan", ".0f"),
    ("回收期(年)", "payback_years", ".1f"),
)


def _column_value(res: SchemeResult, key: str):
    if key == "net_cf_pct":
        return 100.0 * res.net_capacity_factor
    if key == "capex_wan":
        return res.capex_yuan / 1e4
    if key == "npv_wan":
        return res.npv_yuan / 1e4
    return getattr(res, key)


def comparison_table(
    results: Sequence[SchemeResult],
    columns: Optional[Sequence[Tuple[str, str, str]]] = None,
    sort_by: Optional[str] = None,
) -> List[Dict[str, object]]:
    """生成方案横向对比表（返回行字典列表）。"""
    cols = columns or COMPARISON_COLUMNS
    rows: List[Dict[str, object]] = []
    for res in results:
        row: Dict[str, object] = {}
        for label, key, fmt in cols:
            val = _column_value(res, key)
            if fmt.endswith("f"):
                row[label] = float(("%" + fmt) % float(val))
            else:
                row[label] = val
        rows.append(row)
    if sort_by:
        rows.sort(key=lambda r: float(r.get(sort_by, 0.0)))
    return rows


def markdown_table(rows: Sequence[Dict[str, object]], columns: Optional[Sequence[str]] = None) -> str:
    """把行字典列表渲染成 GitHub 风格 Markdown 表格。"""
    if not rows:
        return "（无数据）"
    cols = list(columns) if columns else list(rows[0].keys())
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, "")
            if isinstance(v, float):
                cells.append(("%.4f" % v).rstrip("0").rstrip(".") if abs(v) < 1000 else "%.0f" % v)
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------- 多目标排序


def to_minimization(points: np.ndarray, maximize: Sequence[bool]) -> np.ndarray:
    """把混合方向的目标准则统一成最小化（最大化目标取负）。"""
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2:
        raise ValueError("points 必须是 (N, M) 数组")
    if len(maximize) != p.shape[1]:
        raise ValueError("maximize 标志数必须等于目标维数")
    out = p.copy()
    for j, mx in enumerate(maximize):
        if mx:
            out[:, j] = -out[:, j]
    return out


def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """最小化准则下 a 是否支配 b（各维不劣且至少一维严格更优）。"""
    return bool(np.all(a <= b) and np.any(a < b))


def non_dominated_sort(points: np.ndarray, maximize: Sequence[bool]) -> List[List[int]]:
    """快速非支配排序，返回分层前沿（第 0 层为第一前沿），索引对应输入顺序。"""
    obj = to_minimization(points, maximize)
    n = obj.shape[0]
    dominated_by_count = np.zeros(n, dtype=np.int64)
    dominates_list: List[List[int]] = [[] for _ in range(n)]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if _dominates(obj[p], obj[q]):
                dominates_list[p].append(q)
            elif _dominates(obj[q], obj[p]):
                dominated_by_count[p] += 1

    fronts: List[List[int]] = []
    current = [int(i) for i in np.nonzero(dominated_by_count == 0)[0]]
    while current:
        fronts.append(current)
        nxt: List[int] = []
        for p in current:
            for q in dominates_list[p]:
                dominated_by_count[q] -= 1
                if dominated_by_count[q] == 0:
                    nxt.append(int(q))
        current = nxt
    return fronts


def crowding_distance(points: np.ndarray, front: Sequence[int], maximize: Sequence[bool]) -> np.ndarray:
    """计算某一前沿内每个个体的拥挤度（边界个体为 +inf）。"""
    idx = list(front)
    m = len(idx)
    dist = np.zeros(m, dtype=np.float64)
    if m == 0:
        return dist
    if m <= 2:
        dist[:] = np.inf
        return dist
    obj = to_minimization(np.asarray(points, dtype=np.float64)[idx], maximize)
    for j in range(obj.shape[1]):
        order = np.argsort(obj[:, j], kind="stable")
        dist[order[0]] = np.inf
        dist[order[-1]] = np.inf
        span = float(obj[order[-1], j] - obj[order[0], j])
        if span <= 1e-12:
            continue
        for k in range(1, m - 1):
            if np.isinf(dist[order[k]]):
                continue
            dist[order[k]] += (obj[order[k + 1], j] - obj[order[k - 1], j]) / span
    return dist


def rank_schemes(
    results: Sequence[SchemeResult],
    maximize: Sequence[bool] = (True, False),
    keys: Sequence[str] = ("aep_gwh", "lcoe"),
) -> List[Dict[str, object]]:
    """对方案做多目标排序，返回带「前沿层号 + 拥挤度」的表。"""
    n = len(results)
    if n == 0:
        return []
    pts = np.array([[float(getattr(r, k)) for k in keys] for r in results], dtype=np.float64)
    fronts = non_dominated_sort(pts, maximize)
    rows: List[Dict[str, object]] = []
    rank_of = {}
    crowd_of = {}
    for fi, front in enumerate(fronts):
        cd = crowding_distance(pts, front, maximize)
        for k, idx in enumerate(front):
            rank_of[idx] = fi
            crowd_of[idx] = float(cd[k])
    for i, r in enumerate(results):
        rows.append({
            "方案": r.name,
            "AEP(GWh)": round(r.aep_gwh, 2),
            "LCOE(元/kWh)": round(r.lcoe, 4),
            "净容量系数(%)": round(100.0 * r.net_capacity_factor, 2),
            "前沿层": rank_of.get(i, -1),
            "拥挤度": ("inf" if math.isinf(crowd_of.get(i, 0.0)) else round(crowd_of.get(i, 0.0), 4)),
        })
    rows.sort(key=lambda d: (d["前沿层"], -float(d["AEP(GWh)"])))
    return rows


# ---------------------------------------------------------------- 竞争力判断


def competitive_assessment(
    results: Sequence[SchemeResult],
    baseline: Optional[str] = None,
) -> List[Dict[str, object]]:
    """相对基准方案给出 AEP 增益与 LCOE 降幅，并给出竞争力判断。

    判断规则（本工具的**自定口径**，不是行业标准）：
        - ``明显更优``：AEP 增益 >= 2% 且 LCOE 降幅 >= 2%
        - ``更优``：AEP 或 LCOE 有一项改善 >= 2% 且另一项不劣化超过 1%
        - ``基本相当``：两项变化都在 ±2% 以内
        - ``更差``：其余情况
    """
    if not results:
        return []
    base = None
    if baseline is not None:
        for r in results:
            if r.name == baseline:
                base = r
                break
        if base is None:
            raise ValueError("找不到基准方案：%s" % baseline)
    if base is None:
        base = min(results, key=lambda r: r.lcoe)

    rows: List[Dict[str, object]] = []
    for r in results:
        aep_gain = 100.0 * (r.aep_gwh - base.aep_gwh) / base.aep_gwh if base.aep_gwh else 0.0
        lcoe_drop = 100.0 * (base.lcoe - r.lcoe) / base.lcoe if base.lcoe else 0.0
        if r.name == base.name:
            verdict = "基准方案"
        elif aep_gain >= 2.0 and lcoe_drop >= 2.0:
            verdict = "明显更优"
        elif (aep_gain >= 2.0 or lcoe_drop >= 2.0) and aep_gain > -1.0 and lcoe_drop > -1.0:
            verdict = "更优"
        elif abs(aep_gain) <= 2.0 and abs(lcoe_drop) <= 2.0:
            verdict = "基本相当"
        else:
            verdict = "更差"
        rows.append({
            "方案": r.name,
            "AEP(GWh)": round(r.aep_gwh, 2),
            "AEP相对基准(%)": round(aep_gain, 2),
            "LCOE(元/kWh)": round(r.lcoe, 4),
            "LCOE相对基准(%)": round(-lcoe_drop, 2),
            "判断": verdict,
        })
    return rows


def best_by(results: Sequence[SchemeResult], key: str = "lcoe", minimize: bool = True) -> SchemeResult:
    """按指定指标取最优方案。"""
    if not results:
        raise ValueError("方案列表为空")
    return min(results, key=lambda r: getattr(r, key)) if minimize \
        else max(results, key=lambda r: getattr(r, key))


def compare_summary(results: Sequence[SchemeResult], baseline: Optional[str] = None) -> str:
    """中文对比摘要（可直接打印）。"""
    lines = ["方案对比（共 %d 个方案）" % len(results)]
    for r in results:
        lines.append("  " + r.summary_line())
    lines.append("")
    if results:
        cheapest = min(results, key=lambda r: r.lcoe)
        biggest = max(results, key=lambda r: r.aep_gwh)
        lines.append("  LCOE 最低：%s（%.4f 元/kWh）" % (cheapest.name, cheapest.lcoe))
        lines.append("  AEP 最大 ：%s（%.2f GWh/年）" % (biggest.name, biggest.aep_gwh))
    for row in competitive_assessment(results, baseline):
        lines.append("  %-26s 相对基准 AEP %+6.2f%%，LCOE %+6.2f%%  -> %s"
                     % (row["方案"], row["AEP相对基准(%)"], row["LCOE相对基准(%)"], row["判断"]))
    return "\n".join(lines)
