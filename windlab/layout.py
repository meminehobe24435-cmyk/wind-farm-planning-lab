"""机位排布模块：场址边界、排布生成（规则网格 / 错列 / 随机）、约束检查与修复。

间距约束
--------
行业常用经验规则：沿主风向机位间距不小于 **5D**，垂直主风向不小于 **3D**（D 为风轮直径）。
本模块把它实现成**椭圆禁区**判据——对任意两台机组的间距矢量，在「主风向 / 横风向」
局部坐标系下分解为 ``(s_main, s_cross)``，要求::

    (s_main / (5D))^2 + (s_cross / (3D))^2 >= 1

即两台机组至少有一个方向上「足够分开」。另外叠加一个绝对最小间距（默认 2D）防止
极端重合。这比单纯要求「距离 > 5D」更贴近工程习惯，也允许错列排布把横风向间距
压到 3D 而沿主风向拉开。

坐标约定：``x`` 东、``y`` 北，单位 m；风向为气象习惯（风来自的方向，0°=北）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .wake import wind_unit_vector


# ---------------------------------------------------------------- 场址边界


@dataclass
class Site:
    """场址边界（多边形，单位 m）。

    Attributes:
        polygon: ``(P, 2)`` 顶点数组，自动整理为逆时针。
        name: 场址名称。
    """

    polygon: np.ndarray
    name: str = "example-site"

    def __post_init__(self) -> None:
        poly = np.asarray(self.polygon, dtype=np.float64)
        if poly.ndim != 2 or poly.shape[1] != 2 or poly.shape[0] < 3:
            raise ValueError("polygon 必须是 (P>=3, 2) 的数组")
        if self.signed_area(poly) < 0:
            poly = poly[::-1].copy()          # 统一成逆时针
        self.polygon = poly

    # -- 基本几何 -------------------------------------------------

    @staticmethod
    def signed_area(poly: np.ndarray) -> float:
        """鞋带公式计算有向面积（逆时针为正）。"""
        x = poly[:, 0]
        y = poly[:, 1]
        return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))

    @property
    def area(self) -> float:
        """场址面积（m^2）。"""
        return abs(self.signed_area(self.polygon))

    @property
    def area_km2(self) -> float:
        return self.area / 1e6

    @property
    def centroid(self) -> np.ndarray:
        """多边形形心（面积加权）。"""
        poly = self.polygon
        x, y = poly[:, 0], poly[:, 1]
        xn, yn = np.roll(x, -1), np.roll(y, -1)
        cross = x * yn - xn * y
        a = 0.5 * float(np.sum(cross))
        if abs(a) < 1e-9:
            return np.mean(poly, axis=0)
        cx = float(np.sum((x + xn) * cross)) / (6.0 * a)
        cy = float(np.sum((y + yn) * cross)) / (6.0 * a)
        return np.array([cx, cy], dtype=np.float64)

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        """``(x_min, y_min, x_max, y_max)``。"""
        return (float(np.min(self.polygon[:, 0])), float(np.min(self.polygon[:, 1])),
                float(np.max(self.polygon[:, 0])), float(np.max(self.polygon[:, 1])))

    @classmethod
    def rectangle(
        cls,
        width: float,
        height: float,
        origin: Tuple[float, float] = (0.0, 0.0),
        name: str = "rectangle",
    ) -> "Site":
        """矩形场址：以 ``origin`` 为左下角，宽 ``width``（东西）、高 ``height``（南北）。"""
        x0, y0 = float(origin[0]), float(origin[1])
        poly = np.array(
            [[x0, y0], [x0 + width, y0], [x0 + width, y0 + height], [x0, y0 + height]],
            dtype=np.float64,
        )
        return cls(polygon=poly, name=name)

    # -- 点与多边形的关系 -----------------------------------------

    def contains(self, points: np.ndarray, tolerance: float = 0.0) -> np.ndarray:
        """判断点是否在场址内（射线法）；``tolerance`` > 0 时边界外扩该距离仍算在内。"""
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        inside = self._ray_cast(pts)
        if tolerance > 0:
            inside = inside | (self.distance_to_boundary(pts) <= tolerance)
        return inside

    def _ray_cast(self, pts: np.ndarray) -> np.ndarray:
        poly = self.polygon
        x, y = pts[:, 0], pts[:, 1]
        n = poly.shape[0]
        inside = np.zeros(pts.shape[0], dtype=bool)
        j = n - 1
        for i in range(n):
            xi, yi = poly[i, 0], poly[i, 1]
            xj, yj = poly[j, 0], poly[j, 1]
            if yi != yj:
                cond = ((yi > y) != (yj > y)) & (
                    x < (xj - xi) * (y - yi) / (yj - yi) + xi
                )
                inside ^= cond
            j = i
        return inside

    def distance_to_boundary(self, points: np.ndarray) -> np.ndarray:
        """点到多边形各边的最短距离（m）。"""
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        poly = self.polygon
        n = poly.shape[0]
        best = np.full(pts.shape[0], np.inf)
        for i in range(n):
            a = poly[i]
            b = poly[(i + 1) % n]
            ab = b - a
            denom = float(np.dot(ab, ab))
            if denom < 1e-12:
                d = np.linalg.norm(pts - a, axis=1)
            else:
                t = np.clip(((pts - a) @ ab) / denom, 0.0, 1.0)
                proj = a[None, :] + t[:, None] * ab[None, :]
                d = np.linalg.norm(pts - proj, axis=1)
            best = np.minimum(best, d)
        return best

    def local_extent(self, main_dir_deg: float) -> Tuple[float, float]:
        """把多边形投影到「主风向 / 横风向」局部坐标系，返回两个方向的跨度 ``(L_main, L_cross)``。"""
        u_main, n_cross = wind_unit_vector(main_dir_deg)
        s_main = self.polygon @ u_main
        s_cross = self.polygon @ n_cross
        return (float(np.max(s_main) - np.min(s_main)),
                float(np.max(s_cross) - np.min(s_cross)))


def convex_hull(points: np.ndarray) -> np.ndarray:
    """Andrew 单调链求凸包（返回逆时针顶点）。"""
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] < 3:
        return pts.copy()

    def cross(o, a, b) -> float:
        return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

    p = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
    lower: List[np.ndarray] = []
    for pt in p:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], pt) <= 0:
            lower.pop()
        lower.append(pt)
    upper: List[np.ndarray] = []
    for pt in p[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], pt) <= 0:
            upper.pop()
        upper.append(pt)
    return np.array(lower[:-1] + upper[:-1], dtype=np.float64)


def footprint_area(points: np.ndarray) -> float:
    """机位点集的占地（凸包面积，m^2）。"""
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] < 3:
        return 0.0
    hull = convex_hull(pts)
    return abs(Site.signed_area(hull))


# ---------------------------------------------------------------- 约束检查


@dataclass
class LayoutReport:
    """排布约束检查报告。"""

    ok: bool
    n_turbines: int
    n_pairs: int
    n_violations: int
    violations: List[Dict[str, float]] = field(default_factory=list)
    outside_indices: List[int] = field(default_factory=list)
    min_pair_distance: float = 0.0
    min_pair_distance_d: float = 0.0
    min_main_separation_d: float = 0.0
    min_cross_separation_d: float = 0.0
    min_ellipse_value: float = float("inf")
    diameter: float = 0.0
    min_spacing_main: float = 5.0
    min_spacing_cross: float = 3.0
    site_name: str = ""

    def summary(self) -> str:
        """中文单行摘要。"""
        if self.n_turbines == 0:
            return "排布为空"
        head = "排布约束：%s" % ("通过" if self.ok else "不通过")
        parts = [
            "%d 台机组" % self.n_turbines,
            "机位对 %d" % self.n_pairs,
            "越界 %d 台" % len(self.outside_indices),
            "间距违规 %d 对" % self.n_violations,
            "最小间距 %.2fD" % self.min_pair_distance_d,
            "主风向最小 %.2fD（要求 >= %.1fD）" % (self.min_main_separation_d, self.min_spacing_main),
            "横风向最小 %.2fD（要求 >= %.2fD）" % (self.min_cross_separation_d, self.min_spacing_cross),
        ]
        return head + "：" + "，".join(parts)


def check_layout(
    site: Site,
    positions: np.ndarray,
    diameter: float,
    min_spacing_main: float = 5.0,
    min_spacing_cross: float = 3.0,
    main_dir_deg: float = 270.0,
    min_spacing_absolute: float = 2.0,
) -> LayoutReport:
    """检查排布是否满足场址边界与间距约束。

    间距判据为椭圆禁区：``(s_main/(a·D))^2 + (s_cross/(b·D))^2 >= 1``，
    另外要求直线距离不小于 ``min_spacing_absolute * D``。
    """
    pos = np.atleast_2d(np.asarray(positions, dtype=np.float64))
    n = pos.shape[0]
    if diameter <= 0:
        raise ValueError("风轮直径必须 > 0")

    u_main, n_cross = wind_unit_vector(main_dir_deg)
    outside = [int(i) for i in np.nonzero(~site.contains(pos))[0]]

    a = float(min_spacing_main) * diameter
    b = float(min_spacing_cross) * diameter
    d_min = float(min_spacing_absolute) * diameter

    violations: List[Dict[str, float]] = []
    n_pairs = 0
    min_dist = float("inf")
    min_main = float("inf")
    min_cross = float("inf")
    min_ellipse = float("inf")

    for i in range(n):
        for j in range(i + 1, n):
            n_pairs += 1
            dv = pos[j] - pos[i]
            s_main = abs(float(dv @ u_main))
            s_cross = abs(float(dv @ n_cross))
            dist = float(math.hypot(dv[0], dv[1]))
            min_dist = min(min_dist, dist)
            min_main = min(min_main, s_main / diameter)
            min_cross = min(min_cross, s_cross / diameter)
            ellipse = (s_main / a) ** 2 + (s_cross / b) ** 2
            min_ellipse = min(min_ellipse, ellipse)
            if ellipse < 1.0 - 1e-12 or dist < d_min:
                violations.append({
                    "i": i, "j": j,
                    "distance_m": dist, "distance_d": dist / diameter,
                    "main_m": s_main, "main_d": s_main / diameter,
                    "cross_m": s_cross, "cross_d": s_cross / diameter,
                    "ellipse": ellipse,
                })

    ok = (len(violations) == 0) and (len(outside) == 0)
    return LayoutReport(
        ok=ok,
        n_turbines=n,
        n_pairs=n_pairs,
        n_violations=len(violations),
        violations=violations,
        outside_indices=outside,
        min_pair_distance=0.0 if not np.isfinite(min_dist) else min_dist,
        min_pair_distance_d=0.0 if not np.isfinite(min_dist) else min_dist / diameter,
        min_main_separation_d=0.0 if not np.isfinite(min_main) else min_main,
        min_cross_separation_d=0.0 if not np.isfinite(min_cross) else min_cross,
        min_ellipse_value=min_ellipse,
        diameter=diameter,
        min_spacing_main=min_spacing_main,
        min_spacing_cross=min_spacing_cross,
        site_name=site.name,
    )


def min_pairwise_spacing(positions: np.ndarray) -> float:
    """点集的最小两两距离（m）。"""
    pos = np.atleast_2d(np.asarray(positions, dtype=np.float64))
    if pos.shape[0] < 2:
        return float("inf")
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return float(np.min(d))


# ---------------------------------------------------------------- 排布生成


def _to_world(origin: np.ndarray, s_main: np.ndarray, s_cross: np.ndarray, main_dir_deg: float) -> np.ndarray:
    """把局部坐标 (沿主风向, 横风向) 转成世界坐标 (x 东, y 北)。"""
    u_main, n_cross = wind_unit_vector(main_dir_deg)
    return (origin[None, :]
            + s_main[:, None] * u_main[None, :]
            + s_cross[:, None] * n_cross[None, :])


def grid_layout(
    site: Site,
    n_rows: int,
    n_cols: int,
    spacing_main: float,
    spacing_cross: float,
    main_dir_deg: float = 270.0,
    margin: float = 0.0,
) -> np.ndarray:
    """规则网格排布：以场址形心为中心，主风向为列方向、横风向为行方向。

    ``n_rows`` 为横风向行数，``n_cols`` 为主风向列数。若有机位落到场址外则抛错。
    """
    if n_rows < 1 or n_cols < 1:
        raise ValueError("行数与列数必须 >= 1")
    origin = site.centroid
    i_idx, j_idx = np.meshgrid(
        np.arange(n_cols, dtype=np.float64), np.arange(n_rows, dtype=np.float64), indexing="ij"
    )
    s_main = (i_idx.ravel() - 0.5 * (n_cols - 1)) * float(spacing_main)
    s_cross = (j_idx.ravel() - 0.5 * (n_rows - 1)) * float(spacing_cross)
    pos = _to_world(origin, s_main, s_cross, main_dir_deg)
    inside = site.contains(pos)
    if margin > 0:
        inside = inside & (site.distance_to_boundary(pos) >= margin)
    if not np.all(inside):
        bad = int(np.sum(~inside))
        raise ValueError(
            "网格排布有 %d/%d 个机位落在场址外，请减小行列数或增大场址" % (bad, pos.shape[0])
        )
    return pos


def staggered_layout(
    site: Site,
    n_rows: int,
    n_cols: int,
    spacing_main: float,
    spacing_cross: float,
    main_dir_deg: float = 270.0,
    stagger: float = 0.5,
    drop_outside: bool = True,
) -> np.ndarray:
    """错列（staggered / 梅花形）排布：沿主风向编号为**奇数**的机组在横风向上偏移
    ``stagger × spacing_cross``。

    这样同一条迎风串（沿主风向排列的机组）上的相邻机组不再正对：
    只要横向偏移超过下游机组处的尾流锥半径，串内尾流就几乎完全消失。
    代价是占地略增、横风向局部间距变小（仍满足椭圆间距约束）。

    注意本实现只对**主风向**的串内尾流有效；对垂直于主风向的串内对齐没有帮助
    （在规则格点上，任何整体平移都不能改变同一条线上的间距），
    要同时压制两个方向只能靠非格点排布（随机/优化），见 ``docs/results.md``。
    错开后落到场址外的机位默认丢弃（``drop_outside=True``）。
    """
    if not (0.0 <= stagger <= 1.0):
        raise ValueError("stagger 必须在 [0, 1] 内")
    origin = site.centroid
    pts: List[np.ndarray] = []
    for i in range(n_cols):
        offset_cross = (stagger * spacing_cross) if (i % 2 == 1) else 0.0
        for j in range(n_rows):
            s_main = (i - 0.5 * (n_cols - 1)) * float(spacing_main)
            s_cross = (j - 0.5 * (n_rows - 1)) * float(spacing_cross) + offset_cross
            pts.append((s_main, s_cross))
    arr = np.array(pts, dtype=np.float64)
    pos = _to_world(origin, arr[:, 0], arr[:, 1], main_dir_deg)
    if drop_outside:
        pos = pos[site.contains(pos)]
    return pos


def random_layout(
    site: Site,
    n_turbines: int,
    diameter: float,
    rng: np.random.Generator,
    min_spacing_main: float = 5.0,
    min_spacing_cross: float = 3.0,
    main_dir_deg: float = 270.0,
    min_spacing_absolute: float = 2.0,
    max_attempts: int = 50000,
    strict: bool = True,
) -> np.ndarray:
    """随机排布（带最小间距约束的拒绝采样）。

    在场址外接矩形内均匀撒点，只有同时满足「在场址内」与「与已接受机位满足椭圆间距约束」
    的点才被接受。``strict=True`` 时撒不满 ``n_turbines`` 台会抛错。
    """
    x0, y0, x1, y1 = site.bbox
    accepted: List[np.ndarray] = []
    u_main, n_cross = wind_unit_vector(main_dir_deg)
    a = float(min_spacing_main) * diameter
    b = float(min_spacing_cross) * diameter
    d_min = float(min_spacing_absolute) * diameter
    attempts = 0
    while len(accepted) < n_turbines and attempts < max_attempts:
        attempts += 1
        cand = np.array([rng.uniform(x0, x1), rng.uniform(y0, y1)], dtype=np.float64)
        if not bool(site.contains(cand)[0]):
            continue
        ok = True
        for p in accepted:
            dv = cand - p
            s_main = abs(float(dv @ u_main))
            s_cross = abs(float(dv @ n_cross))
            if (s_main / a) ** 2 + (s_cross / b) ** 2 < 1.0 - 1e-12:
                ok = False
                break
            if math.hypot(dv[0], dv[1]) < d_min:
                ok = False
                break
        if ok:
            accepted.append(cand)
    if len(accepted) < n_turbines and strict:
        raise RuntimeError(
            "随机撒点失败：%d 次尝试只放下 %d/%d 台机组（场址太小或间距约束太严）"
            % (attempts, len(accepted), n_turbines)
        )
    if not accepted:
        return np.zeros((0, 2), dtype=np.float64)
    return np.array(accepted, dtype=np.float64)


def repair_layout(
    site: Site,
    positions: np.ndarray,
    diameter: float,
    min_spacing_main: float = 5.0,
    min_spacing_cross: float = 3.0,
    main_dir_deg: float = 270.0,
    min_spacing_absolute: float = 2.0,
    iterations: int = 100,
    step: float = 0.5,
) -> np.ndarray:
    """把违反间距约束的排布「推」回可行域（用于优化算法中的约束修复）。

    做法：对每个违反椭圆判据的机位对，按所需比例沿间距方向互相推开；
    之后把跑出场址的机位沿着「指向形心」的方向拉回场内。
    假设场址为凸多边形（本工具的场址都是矩形/凸多边形）。
    """
    pos = np.atleast_2d(np.asarray(positions, dtype=np.float64)).copy()
    if pos.shape[0] < 2:
        return pos
    u_main, n_cross = wind_unit_vector(main_dir_deg)
    a = float(min_spacing_main) * diameter
    b = float(min_spacing_cross) * diameter
    d_min = float(min_spacing_absolute) * diameter
    centroid = site.centroid

    for _ in range(int(iterations)):
        moved = False
        for i in range(pos.shape[0]):
            for j in range(i + 1, pos.shape[0]):
                dv = pos[j] - pos[i]
                dist = float(math.hypot(dv[0], dv[1]))
                s_main = abs(float(dv @ u_main))
                s_cross = abs(float(dv @ n_cross))
                ellipse = (s_main / a) ** 2 + (s_cross / b) ** 2
                need_ellipse = ellipse < 1.0 - 1e-9
                need_dist = dist < d_min
                if not (need_ellipse or need_dist):
                    continue
                if dist < 1e-9:
                    direction = np.array([1.0, 0.0])
                    dv = direction
                    dist = 1.0
                target = dist
                if need_dist:
                    target = max(target, d_min * 1.001)
                if need_ellipse and ellipse > 1e-12:
                    target = max(target, dist / math.sqrt(ellipse) * 1.001)
                push = step * (target - dist) * 0.5
                unit = dv / dist
                pos[i] -= push * unit
                pos[j] += push * unit
                moved = True
        # 拉回场址内
        out = ~site.contains(pos)
        if np.any(out):
            for k in np.nonzero(out)[0]:
                direction = centroid - pos[k]
                norm = float(np.linalg.norm(direction))
                if norm < 1e-9:
                    pos[k] = centroid
                    continue
                direction = direction / norm
                lo, hi = 0.0, norm
                for _ in range(40):
                    mid = 0.5 * (lo + hi)
                    if bool(site.contains(pos[k] + mid * direction)[0]):
                        hi = mid
                    else:
                        lo = mid
                pos[k] = pos[k] + hi * direction
            moved = True
        if not moved:
            break
    return pos


def layout_summary(positions: np.ndarray, diameter: float) -> str:
    """机位点集的中文摘要。"""
    pos = np.atleast_2d(np.asarray(positions, dtype=np.float64))
    if pos.shape[0] == 0:
        return "空排布"
    return (
        "%d 台机组，最小间距 %.2fD，占地凸包 %.3f km²，坐标范围 x[%.0f, %.0f] y[%.0f, %.0f]"
        % (
            pos.shape[0], min_pairwise_spacing(pos) / diameter,
            footprint_area(pos) / 1e6,
            float(np.min(pos[:, 0])), float(np.max(pos[:, 0])),
            float(np.min(pos[:, 1])), float(np.max(pos[:, 1])),
        )
    )
