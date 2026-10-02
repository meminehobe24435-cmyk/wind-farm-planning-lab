"""风资源模块：Weibull 分布、参数拟合、风频分布、风功率密度、风向扇区风玫瑰。

⚠️ 数据来源声明
    本模块内置的所有风速/风向数据都是**示例（合成）数据**，由代码按给定参数生成，
    不代表任何真实测风塔、再分析数据或实际项目。真实使用时请用
    :func:`speed_histogram` 处理自己的测风数据，或用 :class:`WindRose` 直接构造。

术语约定
    - 风速单位 m/s；空气密度单位 kg/m^3（默认 1.225，即 15℃、标准大气压）。
    - 风向采用气象习惯：0° = 正北，90° = 正东，顺时针为正，表示风**来自**的方向。
    - 16 扇区，每扇区宽度 22.5°，扇区 i 的中心风向为 i * 22.5°。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------- 常量

N_SECTORS = 16
SECTOR_WIDTH_DEG = 360.0 / N_SECTORS

#: 16 扇区名称，索引 0 为北风（0°）
SECTOR_NAMES = (
    "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
)

#: 示例空气密度（kg/m^3）
DEFAULT_AIR_DENSITY = 1.225


def sector_name(index: int) -> str:
    """扇区索引 -> 名称（越界自动取模）。"""
    return SECTOR_NAMES[int(index) % N_SECTORS]


def sector_direction_deg(index: int) -> float:
    """扇区索引 -> 扇区中心风向（度）。"""
    return (int(index) % N_SECTORS) * SECTOR_WIDTH_DEG


# ---------------------------------------------------------------- Weibull 基本函数


def _check_params(k: float, c: float) -> None:
    if k <= 0:
        raise ValueError("Weibull 形状参数 k 必须 > 0，当前 k=%r" % (k,))
    if c <= 0:
        raise ValueError("Weibull 尺度参数 c 必须 > 0，当前 c=%r" % (c,))


def weibull_pdf(v: float, k: float, c: float) -> float:
    """Weibull 概率密度 f(v) = (k/c) * (v/c)^(k-1) * exp(-(v/c)^k)。

    v < 0 时返回 0；v = 0 时按极限处理（k<1 为 +inf，k=1 为 k/c，k>1 为 0）。
    支持标量与 numpy 数组（数组输入返回数组）。
    """
    _check_params(k, c)
    arr = np.asarray(v, dtype=np.float64)
    scalar = (arr.ndim == 0)
    vv = np.atleast_1d(arr).astype(np.float64)
    out = np.zeros_like(vv)

    pos = vv > 0.0
    if np.any(pos):
        vp = vv[pos]
        out[pos] = (k / c) * (vp / c) ** (k - 1.0) * np.exp(-((vp / c) ** k))

    zero = ~pos
    if np.any(zero):
        if abs(k - 1.0) < 1e-12:
            out[zero] = k / c
        elif k < 1.0:
            out[zero] = np.inf
        else:
            out[zero] = 0.0

    return float(out[0]) if scalar else out


def weibull_cdf(v: float, k: float, c: float) -> float:
    """Weibull 累积分布 F(v) = 1 - exp(-(v/c)^k)，v<0 时为 0。"""
    _check_params(k, c)
    arr = np.asarray(v, dtype=np.float64)
    scalar = (arr.ndim == 0)
    vv = np.atleast_1d(arr).astype(np.float64)
    out = np.zeros_like(vv)
    pos = vv > 0.0
    out[pos] = 1.0 - np.exp(-((vv[pos] / c) ** k))
    return float(out[0]) if scalar else out


def weibull_moment(k: float, c: float, n: float = 1.0) -> float:
    """Weibull 分布 n 阶原点矩 E[v^n] = c^n * Γ(1 + n/k)（用 lgamma 防溢出）。"""
    _check_params(k, c)
    return float(c ** n * math.exp(math.lgamma(1.0 + n / k)))


def weibull_mean(k: float, c: float) -> float:
    """平均风速 E[v] = c * Γ(1 + 1/k)。"""
    return weibull_moment(k, c, 1.0)


def weibull_std(k: float, c: float) -> float:
    """风速标准差。"""
    m1 = weibull_moment(k, c, 1.0)
    m2 = weibull_moment(k, c, 2.0)
    return float(math.sqrt(max(m2 - m1 * m1, 0.0)))


def c_from_mean(k: float, mean_speed: float) -> float:
    """由形状参数 k 与平均风速反解尺度参数 c。"""
    if mean_speed <= 0:
        raise ValueError("平均风速必须 > 0")
    return float(mean_speed / math.exp(math.lgamma(1.0 + 1.0 / k)))


def wind_power_density(
    k: float,
    c: float,
    air_density: float = DEFAULT_AIR_DENSITY,
) -> float:
    """Weibull 风速分布下的风功率密度 P/A = 0.5*ρ*E[v^3]（W/m^2）。

    注意与「某一风速下的瞬时功率密度 0.5*ρ*v^3」区分：这里是按风频分布的期望值。
    """
    return 0.5 * air_density * weibull_moment(k, c, 3.0)


# ---------------------------------------------------------------- Weibull 参数拟合


def fit_weibull_moments(
    samples: Sequence[float],
    k_min: float = 0.1,
    k_max: float = 30.0,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> Tuple[float, float]:
    """矩估计（Method of Moments）拟合 Weibull 参数，返回 ``(k, c)``。

    原理：Weibull 的变异系数 CV = σ/μ 只是 k 的函数，
        1 + CV^2 = Γ(1+2/k) / Γ(1+1/k)^2
    右端关于 k 单调递减，因此在 [k_min, k_max] 上用二分法唯一求解 k，
    再由 c = μ / Γ(1+1/k) 得到尺度参数。

    只使用正风速样本（静风 v=0 无法参与对数/矩运算，且对能量无贡献）。
    """
    v = np.asarray(samples, dtype=np.float64).ravel()
    v = v[np.isfinite(v) & (v > 0.0)]
    if v.size < 3:
        raise ValueError("拟合 Weibull 至少需要 3 个正风速样本，当前 %d 个" % v.size)

    mean = float(np.mean(v))
    std = float(np.std(v, ddof=1)) if v.size > 1 else 0.0
    if mean <= 0:
        raise ValueError("样本平均风速必须 > 0")
    cv = std / mean
    if cv <= 1e-9:
        # 风速几乎恒定 -> Weibull 退化为 δ 分布，k -> +inf
        return float(k_max), float(mean)

    def cv_of_k(k: float) -> float:
        # log 空间计算 gamma 比值，避免大 k 时 Γ(1+2/k) 溢出
        ratio = math.exp(
            math.lgamma(1.0 + 2.0 / k) - 2.0 * math.lgamma(1.0 + 1.0 / k)
        )
        return math.sqrt(max(ratio - 1.0, 0.0))

    lo, hi = float(k_min), float(k_max)
    if cv >= cv_of_k(lo):
        k_hat = lo
    elif cv <= cv_of_k(hi):
        k_hat = hi
    else:
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            if cv_of_k(mid) > cv:      # CV 随 k 递减
                lo = mid
            else:
                hi = mid
            if hi - lo < tol * max(1.0, mid):
                break
        k_hat = 0.5 * (lo + hi)

    c_hat = c_from_mean(k_hat, mean)
    return float(k_hat), float(c_hat)


def fit_weibull_mle(
    samples: Sequence[float],
    k0: float = 2.0,
    tol: float = 1e-9,
    max_iter: int = 500,
    damping: float = 0.5,
    k_min: float = 0.05,
    k_max: float = 50.0,
) -> Tuple[float, float]:
    """最大似然（MLE）拟合 Weibull 参数，返回 ``(k, c)``。

    k 的迭代公式（令 ``mean_pow = Σv_i^k / n``）::

        k_new = ( Σ v_i^k ln v_i / Σ v_i^k  -  Σ ln v_i / n )^(-1)

    收敛后用 c = (Σ v_i^k / n)^(1/k) 得到尺度参数。
    加入阻尼因子 ``damping`` 抑制大 k 时的振荡；k 被限制在 [k_min, k_max]。

    静风样本（v<=0）不参与计算，这是风电行业处理 MLE 的常规做法。
    """
    v = np.asarray(samples, dtype=np.float64).ravel()
    v = v[np.isfinite(v) & (v > 0.0)]
    if v.size < 3:
        raise ValueError("拟合 Weibull 至少需要 3 个正风速样本，当前 %d 个" % v.size)

    n = float(v.size)
    ln_v = np.log(v)
    mean_ln_v = float(np.mean(ln_v))

    k = float(min(max(k0, k_min), k_max))
    for _ in range(max_iter):
        vk = v ** k
        s_vk = float(np.sum(vk))
        if not np.isfinite(s_vk) or s_vk <= 0.0:
            k = 0.5 * (k + k0)          # 数值溢出 -> 回退
            continue
        a = float(np.sum(vk * ln_v)) / s_vk
        denom = a - mean_ln_v
        if denom <= 1e-12:
            k_new = k_max
        else:
            k_new = 1.0 / denom
        k_new = float(min(max(k_new, k_min), k_max))
        k_step = k + damping * (k_new - k)
        if abs(k_step - k) <= tol * max(1.0, k):
            k = k_step
            break
        k = k_step

    mean_pow = float(np.mean(v ** k))
    c = float(mean_pow ** (1.0 / k))
    return float(k), float(c)


# ---------------------------------------------------------------- 风频直方图


def speed_histogram(
    samples: Sequence[float],
    bin_width: float = 1.0,
    v_max: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把风速样本统计成风频直方图。

    返回 ``(centers, freq, edges)``：``freq`` 是各区间内的**频率**（和为 1），
    区间为 [0, w), [w, 2w) ...，``centers`` 取区间中点（行业惯例，0-1 m/s 记为 0.5）。
    """
    if bin_width <= 0:
        raise ValueError("bin_width 必须 > 0")
    v = np.asarray(samples, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        raise ValueError("样本为空")
    hi = float(v_max) if v_max is not None else float(np.max(v))
    hi = max(hi, bin_width)
    n_bin = int(math.ceil((hi + 1e-9) / bin_width))
    edges = np.arange(n_bin + 1, dtype=np.float64) * bin_width
    counts, _ = np.histogram(v, bins=edges)
    freq = counts.astype(np.float64) / float(v.size)
    centers = edges[:-1] + 0.5 * bin_width
    return centers, freq, edges


def histogram_moments(centers: np.ndarray, freq: np.ndarray) -> Tuple[float, float]:
    """由风频直方图（区间中点 + 频率）估算平均风速与标准差（含 Sheppard 修正的简化形式）。"""
    c = np.asarray(centers, dtype=np.float64).ravel()
    f = np.asarray(freq, dtype=np.float64).ravel()
    if c.size != f.size:
        raise ValueError("centers 与 freq 长度必须一致")
    s = float(np.sum(f))
    if s <= 0:
        raise ValueError("频率之和必须 > 0")
    fn = f / s
    mean = float(np.sum(fn * c))
    var = float(np.sum(fn * (c - mean) ** 2))
    return mean, math.sqrt(max(var, 0.0))


# ---------------------------------------------------------------- 风向 / 风玫瑰


@dataclass
class WindRose:
    """16 扇区风玫瑰：各扇区频率 + 各扇区平均风速。

    Attributes:
        freqs: 长度 16 的频率数组（自动归一化到和为 1）。
        mean_speeds: 长度 16 的各扇区平均风速（m/s）。
        name: 数据来源标签（示例数据请标注为示例）。
    """

    freqs: np.ndarray
    mean_speeds: np.ndarray
    name: str = "example-rose"

    def __post_init__(self) -> None:
        f = np.asarray(self.freqs, dtype=np.float64).ravel()
        m = np.asarray(self.mean_speeds, dtype=np.float64).ravel()
        if f.size != N_SECTORS:
            raise ValueError("风玫瑰需要 %d 个扇区频率，当前 %d 个" % (N_SECTORS, f.size))
        if m.size != N_SECTORS:
            raise ValueError("风玫瑰需要 %d 个扇区平均风速，当前 %d 个" % (N_SECTORS, m.size))
        if np.any(f < 0):
            raise ValueError("扇区频率不能为负")
        if m.size != N_SECTORS or np.any(m <= 0):
            raise ValueError("扇区平均风速必须 > 0（无风扇区请把频率设为 0）")
        total = float(np.sum(f))
        if total <= 0:
            raise ValueError("扇区频率之和必须 > 0")
        self.freqs = f / total
        self.mean_speeds = m

    # -- 便捷属性 -------------------------------------------------

    @property
    def dominant_sector_index(self) -> int:
        """主风向扇区索引。"""
        return int(np.argmax(self.freqs))

    @property
    def dominant_direction_deg(self) -> float:
        """主风向（扇区中心，度）。"""
        return sector_direction_deg(self.dominant_sector_index)

    @property
    def dominant_sector_name(self) -> str:
        return sector_name(self.dominant_sector_index)

    @property
    def mean_speed(self) -> float:
        """扇区频率加权的场地平均风速（m/s）。"""
        return float(np.sum(self.freqs * self.mean_speeds))

    @property
    def directional_spread(self) -> float:
        """风向集中度：0 = 完全均匀，1 = 全部集中在一个扇区（1 - 归一化熵）。"""
        f = self.freqs[self.freqs > 0]
        if f.size == 0:
            return 0.0
        entropy = -float(np.sum(f * np.log(f)))
        return float(1.0 - entropy / math.log(N_SECTORS))

    def resultant_direction_deg(self) -> float:
        """风矢量合成方向（度，气象习惯）：用频率加权单位矢量求和。"""
        ang = np.deg2rad(np.arange(N_SECTORS) * SECTOR_WIDTH_DEG)
        # 风来自 theta -> 单位矢量指向 theta（气象系：x=东, y=北）
        ex = float(np.sum(self.freqs * np.sin(ang)))
        ey = float(np.sum(self.freqs * np.cos(ang)))
        if abs(ex) < 1e-12 and abs(ey) < 1e-12:
            return float("nan")
        deg = math.degrees(math.atan2(ex, ey))
        return float(deg % 360.0)

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "freqs": [float(x) for x in self.freqs],
            "mean_speeds": [float(x) for x in self.mean_speeds],
            "dominant": self.dominant_sector_name,
            "mean_speed": self.mean_speed,
            "directional_spread": self.directional_spread,
        }


@dataclass
class WindResource:
    """场地风资源：风玫瑰 + 场地 Weibull 形状参数 + 空气密度。

    各扇区的 Weibull 参数由「扇区平均风速 + 场地形状参数 k」反推：
        c_s = mean_s / Γ(1 + 1/k)
    这是行业里常见的简化做法（形状参数随方向变化通常远小于尺度参数）。

    ⚠️ 内置示例是合成数据，不是实测测风塔数据。
    """

    rose: WindRose
    k: float = 2.0
    air_density: float = DEFAULT_AIR_DENSITY
    name: str = "example-site"

    def __post_init__(self) -> None:
        _check_params(self.k, 1.0)
        if self.air_density <= 0:
            raise ValueError("空气密度必须 > 0")

    # -- Weibull 参数 --------------------------------------------

    def sector_weibull(self, index: int) -> Tuple[float, float]:
        """第 index 个扇区的 ``(k, c)``。"""
        i = int(index) % N_SECTORS
        return float(self.k), c_from_mean(self.k, float(self.rose.mean_speeds[i]))

    def sector_parameters(self) -> Tuple[np.ndarray, np.ndarray]:
        """所有扇区的 ``(k_array, c_array)``。"""
        c = np.array(
            [c_from_mean(self.k, float(m)) for m in self.rose.mean_speeds],
            dtype=np.float64,
        )
        return np.full(N_SECTORS, float(self.k)), c

    def sector_power_density(self, index: int) -> float:
        """第 index 个扇区的风功率密度（W/m^2）。"""
        k_s, c_s = self.sector_weibull(index)
        return wind_power_density(k_s, c_s, self.air_density)

    # -- 场地汇总指标 --------------------------------------------

    @property
    def mean_speed(self) -> float:
        return self.rose.mean_speed

    @property
    def equivalent_c(self) -> float:
        """与场地平均风速等效的 Weibull 尺度参数。"""
        return c_from_mean(self.k, self.mean_speed)

    @property
    def power_density(self) -> float:
        """风向加权后的场地平均风功率密度（W/m^2）。"""
        return float(
            sum(
                float(self.rose.freqs[i]) * self.sector_power_density(i)
                for i in range(N_SECTORS)
            )
        )

    @property
    def dominant_direction_deg(self) -> float:
        return self.rose.dominant_direction_deg

    def summary(self) -> str:
        """一行式摘要（中文）。"""
        return (
            "场地「%s」：平均风速 %.2f m/s，Weibull k=%.2f c=%.2f，"
            "主风向 %s(%.1f°)，风功率密度 %.0f W/m²，空气密度 %.3f kg/m³"
            % (
                self.name, self.mean_speed, self.k, self.equivalent_c,
                self.rose.dominant_sector_name, self.dominant_direction_deg,
                self.power_density, self.air_density,
            )
        )


# ---------------------------------------------------------------- 示例数据生成


def sample_speeds(
    k: float,
    c: float,
    n: int = 10000,
    seed: int = 0,
    v_max: Optional[float] = None,
) -> np.ndarray:
    """生成服从 Weibull(k, c) 的**示例**风速样本（用于自检，不是实测数据）。

    使用逆变换法：u ~ U(0,1)，v = c * (-ln(1-u))^(1/k)。
    """
    if n <= 0:
        raise ValueError("样本数必须 > 0")
    rng = np.random.default_rng(seed)
    u = rng.random(n)
    v = c * (-np.log1p(-u)) ** (1.0 / k)
    if v_max is not None:
        v = np.clip(v, 0.0, float(v_max))
    return v


def example_wind_rose(
    dominant_deg: float = 270.0,
    mean_speed: float = 7.5,
    kappa: float = 2.0,
    secondary_deg: float = 45.0,
    secondary_weight: float = 0.15,
    secondary_kappa: float = 1.8,
    speed_amplitude: float = 1.7,
    speed_phase_deg: float = 250.0,
) -> WindRose:
    """构造一个**示例**风玫瑰（合成数据，非实测）。

    频率用双峰 von Mises 型分布构造：主风向 ``dominant_deg`` + 次风向 ``secondary_deg``；
    各扇区平均风速在主风向附近略高（用余弦调制），整体平均约等于 ``mean_speed``。
    ``kappa=2.0`` 时主扇区频率约 20%，与陆上风电场常见风玫瑰的集中度接近。
    """
    ang = np.arange(N_SECTORS, dtype=np.float64) * SECTOR_WIDTH_DEG
    d = np.deg2rad(ang)

    def vm(theta_deg: float, kap: float) -> np.ndarray:
        return np.exp(kap * np.cos(d - math.radians(theta_deg)))

    w = vm(dominant_deg, kappa) + secondary_weight * vm(secondary_deg, secondary_kappa)
    freqs = w / np.sum(w)

    # 各扇区平均风速：余弦调制后再整体缩放，使频率加权平均等于 mean_speed
    shape = 1.0 + 0.18 * np.cos(d - math.radians(speed_phase_deg)) + 0.05 * np.cos(2.0 * d)
    speeds = shape / float(np.sum(freqs * shape)) * mean_speed
    # speed_amplitude 用于控制方向间风速差异幅度（保留参数以支持更极端的示例）
    speeds = mean_speed + (speeds - mean_speed) * (speed_amplitude / 1.7)
    return WindRose(freqs=freqs, mean_speeds=speeds, name="示例风玫瑰（合成数据）")


def example_wind_resource(
    mean_speed: float = 7.5,
    k: float = 2.0,
    air_density: float = DEFAULT_AIR_DENSITY,
    name: str = "示例场地（合成数据）",
) -> WindResource:
    """构造一个**示例**场地风资源（合成数据，非实测）。"""
    rose = example_wind_rose(mean_speed=mean_speed)
    return WindResource(rose=rose, k=k, air_density=air_density, name=name)
