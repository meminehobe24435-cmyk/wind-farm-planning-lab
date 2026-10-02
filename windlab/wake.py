"""尾流模块：Jensen（Park）工程尾流模型 + 多机叠加 + 部分遮挡 + 风向。

模型要点
--------
1. 单台尾流（Jensen / Park 经典工程模型）::

        deficit(x) = (1 - sqrt(1 - Ct)) * ( D / (D + 2 * k * x) )^2

   其中 ``x`` 是下游距离（m），``Ct`` 是推力系数，``D`` 是风轮直径（m），
   ``k`` 是尾流衰减系数（陆上常用 0.075，海上常用 0.04~0.05）。
   尾流半径按锥形线性扩张：``R_w(x) = D/2 + k * x``。

2. 多台叠加用平方和（Katic 模型，也叫 RSS / sum of squares）::

        (1 - u/u0)^2 = Σ_i (1 - u_i/u0)^2

   即先把每台上游机组的**速度亏损**平方后求和，再开方，得到总亏损。

3. **部分遮挡**：把下游机组的风轮圆盘离散成网格采样点（默认 20x20 正方形覆盖圆盘、
   只保留落在圆盘内的点，等面积权重），逐点判断是否落在上游机组的尾流锥内，
   得到每个点的亏损，再对风轮面积取平均，得到该机组受到的等效亏损。
   不做「整台机全遮挡」的简化。

4. **风向**：只有位于上游（顺风方向）的机组才会对下游机组产生尾流，逆风方向不产生。

坐标与风向约定
--------------
    - 平面直角坐标 ``x`` 指向正东、``y`` 指向正北，单位 m。
    - 风向采用气象习惯 ``direction_deg``：0° = 正北，90° = 正东，表示风**来自**的方向。
    - 风的行进方向（下游方向，即尾流方向）：``u = (-sin θ, -cos θ)``。
      校验：θ=0（北风）-> u=(0,-1) 向南；θ=90（东风）-> u=(-1,0) 向西；θ=270（西风）-> u=(1,0) 向东。
    - 风轮圆盘是垂直于风向的竖直圆面。离散点在圆盘平面内取
      ``(沿水平横风向的偏移 c, 竖直偏移 z)``，在平面图上的水平位置为 ``pos + c * n``；
      点与尾流轴线的距离按 ``sqrt(c^2 + z^2)`` 计算（所有机组轮毂同高、不考虑地形）。

⚠️ 能力边界：这是 **Jensen 简化模型**，不是 CFD；不含地形、大气稳定度、湍流强度、
   偏航、尾流恢复的物理细节，近场（< 2D）精度差，多机叠加是工程近似。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

#: 陆上默认尾流衰减系数
DEFAULT_WAKE_DECAY = 0.075

#: 风轮单元面积占比估计的子采样密度（每个方向 4 个点，共 16 个）
_SUBSAMPLES = 4


def _wind_unit_vector_raw(direction_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    th = math.radians(float(direction_deg))
    u = np.array([-math.sin(th), -math.cos(th)], dtype=np.float64)
    n = np.array([-u[1], u[0]], dtype=np.float64)
    return u, n


@lru_cache(maxsize=64)
def _cached_rotor_samples(rotor_grid: int, diameter: float) -> Tuple[np.ndarray, np.ndarray]:
    """计算并缓存风轮离散采样点及其**面积权重**。

    做法：用 ``rotor_grid × rotor_grid`` 个单元覆盖 ``[-R, R]²``，取单元中心为采样点；
    每个单元再按 ``4×4`` 子采样估计「落在圆盘内的面积占比」，作为该点的权重，
    最后把权重归一化到和为 1。

    这样风轮平均亏损就是**真正的面积平均**，而不是「落在圆内的单元中心」的等权平均
    ——后者会让圆盘边界上那些只有一部分在圆内的单元被过度代表，导致
    AEP 随网格分辨率非单调抖动（实测峰峰值约 0.5%）。面积加权后收敛性显著改善。

    返回 ``(points, weights)``：``points`` 形状 ``(M, 2)``，列为 ``(横风向偏移 c, 竖直偏移 z)``；
    ``weights`` 形状 ``(M,)``，非负且和为 1。
    """
    radius = 0.5 * float(diameter)
    n = int(rotor_grid)
    step = 2.0 * radius / n
    coords = -radius + (np.arange(n, dtype=np.float64) + 0.5) * step
    cx, cz = np.meshgrid(coords, coords, indexing="ij")

    offsets = (np.arange(_SUBSAMPLES, dtype=np.float64) + 0.5) / _SUBSAMPLES - 0.5
    frac = np.zeros((n, n), dtype=np.float64)
    r2 = radius ** 2
    for dx in offsets:
        for dz in offsets:
            sx = cx + dx * step
            sz = cz + dz * step
            frac += ((sx ** 2 + sz ** 2) <= r2).astype(np.float64)
    frac /= float(_SUBSAMPLES * _SUBSAMPLES)

    keep = frac > 0.0
    weights = frac[keep]
    if float(np.sum(weights)) <= 0.0:      # pragma: no cover - 理论上不会发生
        raise ValueError("风轮离散失败：没有任何采样点落在圆盘内")
    weights = weights / float(np.sum(weights))
    points = np.stack([cx[keep], cz[keep]], axis=1)
    return points, weights


# ---------------------------------------------------------------- 方向工具


def wind_unit_vector(direction_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    """风向（度，气象习惯，风来自的方向）-> ``(下游单位矢量 u, 水平横风向单位矢量 n)``。

    ``u`` 为尾流延伸方向；``n`` 与 ``u`` 垂直（在水平面内）。
    """
    return _wind_unit_vector_raw(direction_deg)


def direction_from_vector(u: Sequence[float]) -> float:
    """下游单位矢量 -> 气象风向（度）。"""
    ux, uy = float(u[0]), float(u[1])
    return float(math.degrees(math.atan2(-ux, -uy)) % 360.0)


# ---------------------------------------------------------------- 单台尾流解析解


def jensen_deficit(
    x,
    ct,
    diameter: float,
    k_wake: float = DEFAULT_WAKE_DECAY,
):
    """Jensen 模型在下游距离 ``x`` 处的速度亏损（无量纲，0~1）。

    ``deficit = (1 - sqrt(1 - Ct)) * (D / (D + 2 k x))^2``，``x <= 0`` 时返回 0。
    ``x`` 与 ``ct`` 都支持标量或数组（按 numpy 广播规则组合）。
    """
    if diameter <= 0:
        raise ValueError("风轮直径必须 > 0")
    if k_wake < 0:
        raise ValueError("尾流衰减系数不能为负")
    x_arr = np.asarray(x, dtype=np.float64)
    ct_arr = np.asarray(ct, dtype=np.float64)
    scalar = (x_arr.ndim == 0 and ct_arr.ndim == 0)
    shape = np.broadcast_shapes(x_arr.shape, ct_arr.shape)
    xx = np.broadcast_to(x_arr, shape).astype(np.float64)
    ctb = np.clip(np.broadcast_to(ct_arr, shape).astype(np.float64), 0.0, 0.999)
    base = 1.0 - np.sqrt(np.maximum(1.0 - ctb, 0.0))
    expansion = (diameter / (diameter + 2.0 * k_wake * np.maximum(xx, 0.0))) ** 2
    out = np.where(xx > 0.0, base * expansion, 0.0)
    if scalar:
        return float(out.reshape(-1)[0])
    return out


def wake_radius(x, diameter: float, k_wake: float = DEFAULT_WAKE_DECAY):
    """尾流在下游距离 ``x`` 处的半径：``R(x) = D/2 + k x``（x<=0 时返回 D/2）。"""
    arr = np.asarray(x, dtype=np.float64)
    scalar = (arr.ndim == 0)
    xx = np.atleast_1d(arr).astype(np.float64)
    out = 0.5 * diameter + k_wake * np.maximum(xx, 0.0)
    return float(out[0]) if scalar else out


# ---------------------------------------------------------------- 尾流模型对象


@dataclass
class WakeModel:
    """Jensen 尾流模型配置（含风轮圆盘离散）。

    Attributes:
        diameter: 风轮直径（m）。
        k_wake: 尾流衰减系数，陆上 0.075 / 海上 0.05。
        rotor_grid: 风轮圆盘离散网格数 n（n x n 正方形覆盖圆盘，只取圆内点，
            等面积权重）；默认 20，得到圆内约 310 个采样点。
        deficit_cap: 单点合成亏损上限（Jensen 近场失真，超过 1 会导致负风速，
            因此截断到 0.99），仅用于数值保护。
    """

    diameter: float
    k_wake: float = DEFAULT_WAKE_DECAY
    rotor_grid: int = 20
    deficit_cap: float = 0.99

    def __post_init__(self) -> None:
        if self.diameter <= 0:
            raise ValueError("风轮直径必须 > 0")
        if self.k_wake < 0:
            raise ValueError("尾流衰减系数不能为负")
        if self.rotor_grid < 2:
            raise ValueError("rotor_grid 至少为 2")
        if not (0.0 < self.deficit_cap <= 1.0):
            raise ValueError("deficit_cap 必须在 (0, 1] 内")

    @property
    def radius(self) -> float:
        return 0.5 * self.diameter

    @property
    def rotor_area(self) -> float:
        return math.pi * self.radius ** 2

    def rotor_samples(self) -> Tuple[np.ndarray, np.ndarray]:
        """风轮离散采样点与面积权重 ``(points, weights)``，见 :func:`_cached_rotor_samples`。"""
        points, weights = _cached_rotor_samples(int(self.rotor_grid), float(self.diameter))
        return points, weights

    def rotor_points(self) -> np.ndarray:
        """风轮离散采样点，形状 ``(M, 2)``，列为 ``(横风向偏移 c, 竖直偏移 z)``，单位 m。"""
        return self.rotor_samples()[0]

    def rotor_point_count(self) -> int:
        return int(self.rotor_samples()[0].shape[0])

    # -- 单机 -----------------------------------------------------

    def single_deficit(self, x, ct: float):
        """单台机组在正下游距离 x 处的尾流中心亏损（解析式，见 :func:`jensen_deficit`）。"""
        return jensen_deficit(x, ct, self.diameter, self.k_wake)

    # -- 全场 -----------------------------------------------------

    def pair_deficits(
        self,
        positions: np.ndarray,
        wind_speed: float,
        direction_deg: float,
        ct,
        turbine_index: Optional[int] = None,
    ) -> np.ndarray:
        """计算所有「上游机组 i 对下游机组 j」的风轮逐点亏损。

        返回形状 ``(N, N, M)`` 的数组：``out[i, j, m]`` 表示机组 i 在机组 j 的第 m 个
        风轮采样点上造成的速度亏损（无量纲）；``i == j`` 恒为 0。
        指定 ``turbine_index=i`` 时只算第 i 台作为上游的那一层，返回 ``(1, N, M)``（省内存）。
        """
        pos = np.asarray(positions, dtype=np.float64)
        if pos.ndim != 2 or pos.shape[1] != 2:
            raise ValueError("positions 形状必须为 (N, 2)")
        n_t = pos.shape[0]
        pts, _ = self.rotor_samples()
        m_pts = pts.shape[0]
        c_off = pts[:, 0]                       # 横风向偏移
        z_off = pts[:, 1]                       # 竖直偏移
        u_hat, n_hat = wind_unit_vector(direction_deg)

        ct_arr = np.asarray(ct, dtype=np.float64).ravel()
        if ct_arr.size == 1:
            ct_arr = np.full(n_t, float(ct_arr[0]))
        if ct_arr.size != n_t:
            raise ValueError("Ct 必须为标量或长度 N 的数组")

        idx = range(n_t) if turbine_index is None else [int(turbine_index) % n_t]
        out = np.zeros((len(idx), n_t, m_pts))

        for layer, i in enumerate(idx):
            ct_i = float(np.clip(ct_arr[i], 0.0, 0.999))
            if ct_i <= 1e-9:
                continue
            base = 1.0 - math.sqrt(max(1.0 - ct_i, 0.0))
            # 每台下游机组 j 的风轮采样点在平面图上的水平坐标 (N, M, 2)
            horiz = pos[:, None, :] + c_off[None, :, None] * n_hat[None, None, :]
            d = horiz - pos[i][None, None, :]                      # (N, M, 2)
            sp = d[..., 0] * u_hat[0] + d[..., 1] * u_hat[1]       # 下游距离 (N, M)
            ch = d[..., 0] * n_hat[0] + d[..., 1] * n_hat[1]       # 横风向偏移 (N, M)
            rad = np.sqrt(ch ** 2 + z_off[None, :] ** 2)           # 到尾流轴的距离 (N, M)
            rw = 0.5 * self.diameter + self.k_wake * np.maximum(sp, 0.0)
            # 注意 1：sp 用一个小阈值，避免浮点噪声把「机组自身」判成它的下游
            #        （n 与 u 正交，理论上 sp=0，实际会得到 ±1e-16）。
            # 注意 2：显式排除 i == j —— 机组不处于自身尾流中。
            inside = (sp > 1e-9) & (rad < rw)
            inside[i] = False
            if not np.any(inside):
                continue
            sp_safe = np.where(sp > 0.0, sp, 1.0)
            deficit = base * (self.diameter / (self.diameter + 2.0 * self.k_wake * sp_safe)) ** 2
            out[layer] = np.where(inside, deficit, 0.0)
        return out

    def effective_speeds(
        self,
        positions: np.ndarray,
        wind_speed: float,
        direction_deg: float,
        ct=None,
        order: int = 2,
    ) -> np.ndarray:
        """给定来流风速与风向，返回各机组的**等效来流风速**（m/s）。

        ``order=2`` 用平方和叠加（Katic / RSS）：``δ_tot = sqrt(Σ δ_i^2)``；
        ``order=1`` 用线性叠加（仅用于对比展示，默认不用）。
        等效风速 = 自由来流 × (1 - 风轮面积平均亏损)。
        """
        pos = np.asarray(positions, dtype=np.float64)
        n_t = pos.shape[0]
        if n_t == 0:
            return np.zeros(0, dtype=np.float64)
        u0 = float(wind_speed)
        if u0 <= 0.0:
            return np.zeros(n_t, dtype=np.float64)
        if ct is None:
            raise ValueError("必须给出来流风速对应的推力系数 ct")
        _, weights = self.rotor_samples()
        pd = self.pair_deficits(pos, u0, direction_deg, ct)      # (N, N, M)
        if order == 2:
            total = np.sqrt(np.sum(pd ** 2, axis=0))             # (N, M)
        elif order == 1:
            total = np.sum(pd, axis=0)
        else:
            raise ValueError("order 只支持 1 或 2")
        total = np.clip(total, 0.0, self.deficit_cap)
        # 风轮面积加权平均（weights 已归一化，和为 1）
        mean_deficit = total @ weights                           # (N,)
        return u0 * (1.0 - mean_deficit)

    def loss_matrix(
        self,
        positions: np.ndarray,
        wind_speed: float,
        direction_deg: float,
        ct=None,
    ) -> np.ndarray:
        """尾流损失矩阵 ``loss[i][j]``：机组 i 对机组 j 造成的等效速度亏损（0~1）。

        对角线恒为 0；逆风方向（sp <= 0）为 0；无重叠为 0。
        """
        pos = np.asarray(positions, dtype=np.float64)
        n_t = pos.shape[0]
        if ct is None:
            raise ValueError("必须给出来流风速对应的推力系数 ct")
        _, weights = self.rotor_samples()
        pd = self.pair_deficits(pos, float(wind_speed), direction_deg, ct)
        mat = pd @ weights                                       # (N, N) 面积加权
        np.fill_diagonal(mat, 0.0)
        return mat

    def array_efficiency(
        self,
        positions: np.ndarray,
        wind_speed: float,
        direction_deg: float,
        ct=None,
    ) -> float:
        """整场等效风速比 ``u_eff / u0``（0~1），即风轮面积与机组双重平均后的剩余风速比。

        注意这只反映**速度**亏损，不含功率曲线非线性，因此不等于发电量损失率。
        """
        pos = np.asarray(positions, dtype=np.float64)
        if pos.shape[0] == 0:
            return 1.0
        u0 = float(wind_speed)
        if u0 <= 0.0:
            return 1.0
        eff = self.effective_speeds(pos, u0, direction_deg, ct)
        return float(np.mean(eff) / u0)


# ---------------------------------------------------------------- 便捷函数（模块级）


def effective_speeds(
    positions: np.ndarray,
    wind_speed: float,
    direction_deg: float,
    ct,
    model: WakeModel,
) -> np.ndarray:
    """模块级便捷封装，见 :meth:`WakeModel.effective_speeds`。"""
    return model.effective_speeds(positions, wind_speed, direction_deg, ct)


def loss_matrix(
    positions: np.ndarray,
    wind_speed: float,
    direction_deg: float,
    ct,
    model: WakeModel,
) -> np.ndarray:
    """模块级便捷封装，见 :meth:`WakeModel.loss_matrix`。"""
    return model.loss_matrix(positions, wind_speed, direction_deg, ct)


def wake_loss_at_direction(
    positions: np.ndarray,
    wind_speed: float,
    direction_deg: float,
    ct,
    model: WakeModel,
) -> Dict[str, float]:
    """某风向下的尾流统计：速度亏损比、等效风速、最小/最大机组风速。"""
    eff = model.effective_speeds(positions, wind_speed, direction_deg, ct)
    if eff.size == 0 or wind_speed <= 0:
        return {"speed_deficit_ratio": 0.0, "u_min": 0.0, "u_max": 0.0, "u_mean": 0.0}
    return {
        "speed_deficit_ratio": float(1.0 - np.mean(eff) / wind_speed),
        "u_min": float(np.min(eff)),
        "u_max": float(np.max(eff)),
        "u_mean": float(np.mean(eff)),
    }
