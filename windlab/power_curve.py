"""功率曲线模块：从 CSV 导入、内置示例曲线、插值、容量系数与推力系数。

⚠️ 内置曲线声明
    :func:`example_curve` / :func:`builtin_curves` 返回的都是**示例曲线**，
    按典型陆上大叶轮机组的形态**构造**（切入 2.5~3 m/s、额定 8.5~11 m/s、
    切出 25 m/s），**不是任何厂商的实测数据，也不是任何真实机型**。
    真实项目请用 :func:`load_curve_csv` 导入厂商功率曲线。

数值约定
    - 功率单位 kW，风速 m/s，直径 m，扫风面积 m^2。
    - 切入风速以下、切出风速及以上（含切出点本身，机组停机）功率为 0。
    - 额定段（额定风速 ~ 切出风速）功率为常数。
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .resource import DEFAULT_AIR_DENSITY, weibull_pdf

# ---------------------------------------------------------------- 小工具


def trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    """梯形法数值积分（自写，避免依赖 numpy 版本的 np.trapz / np.trapezoid 差异）。"""
    y = np.asarray(y, dtype=np.float64).ravel()
    x = np.asarray(x, dtype=np.float64).ravel()
    if y.size != x.size or y.size < 2:
        raise ValueError("trapezoid 需要等长的 x/y 且长度 >= 2")
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


# ---------------------------------------------------------------- PCHIP 保形插值


def pchip_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fritsch–Carlson 单调保形三次插值的节点斜率（不依赖 scipy）。"""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = x.size
    if n < 2:
        raise ValueError("PCHIP 至少需要 2 个节点")
    h = np.diff(x)
    if np.any(h <= 0):
        raise ValueError("PCHIP 节点 x 必须严格递增")
    delta = np.diff(y) / h
    m = np.zeros(n)

    if n == 2:
        m[:] = delta[0]
        return m

    for i in range(1, n - 1):
        if delta[i - 1] * delta[i] <= 0.0:
            m[i] = 0.0
        else:
            w1 = 2.0 * h[i] + h[i - 1]
            w2 = h[i] + 2.0 * h[i - 1]
            m[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i])

    def end_slope(h0: float, h1: float, d0: float, d1: float) -> float:
        s = ((2.0 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
        if s * d0 <= 0.0:
            return 0.0
        if d0 * d1 < 0.0 and abs(s) > 3.0 * abs(d0):
            return 3.0 * d0
        return s

    m[0] = end_slope(h[0], h[1], delta[0], delta[1])
    m[-1] = end_slope(h[-1], h[-2], delta[-1], delta[-2])
    return m


def pchip_eval(x: np.ndarray, y: np.ndarray, xq: np.ndarray) -> np.ndarray:
    """在节点 (x, y) 上做 Fritsch–Carlson 保形三次插值，求 xq 处的值。"""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xq = np.atleast_1d(np.asarray(xq, dtype=np.float64))
    m = pchip_slopes(x, y)

    idx = np.clip(np.searchsorted(x, xq, side="right") - 1, 0, x.size - 2)
    h = x[idx + 1] - x[idx]
    t = (xq - x[idx]) / h
    h00 = (1.0 + 2.0 * t) * (1.0 - t) ** 2
    h10 = t * (1.0 - t) ** 2
    h01 = t ** 2 * (3.0 - 2.0 * t)
    h11 = t ** 2 * (t - 1.0)
    out = (h00 * y[idx] + h10 * h * m[idx]
           + h01 * y[idx + 1] + h11 * h * m[idx + 1])
    # 端点外做常数外推，避免插值函数在区间外乱跑
    out = np.where(xq <= x[0], y[0], out)
    out = np.where(xq >= x[-1], y[-1], out)
    return out


# ---------------------------------------------------------------- 曲线对象


@dataclass
class PowerCurve:
    """一条风电机组功率曲线（含推力系数与几何参数）。

    Attributes:
        name: 机组标识。
        speeds: 功率表风速节点（严格递增）。
        powers: 功率表（kW），与 speeds 等长。
        cut_in / cut_out: 切入 / 切出风速（m/s）。
        rated_power: 额定功率（kW）。
        diameter: 风轮直径（m）。
        ct_speeds / ct_values: 推力系数表；为 None 时按 :func:`default_ct_table` 生成。
        rated_speed: 额定风速（m/s），未给出时由功率表推断。
        source: 数据来源说明（示例曲线必须写明）。
    """

    name: str
    speeds: np.ndarray
    powers: np.ndarray
    cut_in: float
    cut_out: float
    rated_power: float
    diameter: float
    ct_speeds: Optional[np.ndarray] = None
    ct_values: Optional[np.ndarray] = None
    rated_speed: Optional[float] = None
    source: str = "未标注来源"

    def __post_init__(self) -> None:
        s = np.asarray(self.speeds, dtype=np.float64).ravel()
        p = np.asarray(self.powers, dtype=np.float64).ravel()
        if s.size != p.size or s.size < 2:
            raise ValueError("功率表 speeds / powers 长度必须一致且 >= 2")
        if np.any(np.diff(s) <= 0):
            raise ValueError("功率表风速必须严格递增")
        if np.any(p < 0):
            raise ValueError("功率不能为负")
        if self.rated_power <= 0 or self.diameter <= 0:
            raise ValueError("额定功率与风轮直径必须 > 0")
        if not (0 < self.cut_in < self.cut_out):
            raise ValueError("必须满足 0 < 切入 < 切出")
        if np.max(p) > self.rated_power * 1.0001:
            raise ValueError("功率表最大值 %.1f 超过额定功率 %.1f" % (np.max(p), self.rated_power))
        self.speeds = s
        self.powers = np.minimum(p, self.rated_power)

        if self.rated_speed is None:
            mask = self.powers >= self.rated_power - 1e-9
            self.rated_speed = float(self.speeds[np.argmax(mask)]) if np.any(mask) else float(self.cut_out)

        if self.ct_values is None:
            cs, cv = default_ct_table(self.cut_in, float(self.rated_speed), self.cut_out)
            self.ct_speeds, self.ct_values = cs, cv
        else:
            cs = np.asarray(self.ct_speeds, dtype=np.float64).ravel()
            cv = np.asarray(self.ct_values, dtype=np.float64).ravel()
            if cs.size != cv.size or cs.size < 2:
                raise ValueError("推力系数表长度不一致")
            if np.any(np.diff(cs) <= 0):
                raise ValueError("推力系数表风速必须严格递增")
            self.ct_speeds, self.ct_values = cs, cv

    # -- 基本属性 -------------------------------------------------

    @property
    def rotor_area(self) -> float:
        """扫风面积（m^2）。"""
        return math.pi * (self.diameter / 2.0) ** 2

    @property
    def specific_power(self) -> float:
        """单位扫风面积装机容量（W/m^2）= P_rated / A。"""
        return self.rated_power * 1000.0 / self.rotor_area

    # -- 插值 -----------------------------------------------------

    def power(self, v):
        """风速 -> 功率（kW）。标量入参返回 float，数组入参返回 ndarray。

        边界约定（已被单元测试覆盖）::

            v <  切入        -> 0
            v == 切入         -> 功率表在切入处的值（示例曲线为 0）
            切入 < v < 切出   -> 线性插值
            v >= 切出         -> 0（含切出点本身，机组顺桨停机）
        """
        arr = np.asarray(v, dtype=np.float64)
        scalar = (arr.ndim == 0)
        vv = np.atleast_1d(arr).astype(np.float64)
        out = np.zeros_like(vv)
        mask = np.isfinite(vv) & (vv >= self.cut_in) & (vv < self.cut_out)
        if np.any(mask):
            out[mask] = np.interp(vv[mask], self.speeds, self.powers)
        return float(out[0]) if scalar else out

    def ct(self, v):
        """风速 -> 推力系数 Ct（无量纲）。切入以下、切出及以上为 0。"""
        arr = np.asarray(v, dtype=np.float64)
        scalar = (arr.ndim == 0)
        vv = np.atleast_1d(arr).astype(np.float64)
        out = np.zeros_like(vv)
        mask = np.isfinite(vv) & (vv >= self.cut_in) & (vv < self.cut_out)
        if np.any(mask):
            out[mask] = np.interp(vv[mask], self.ct_speeds, self.ct_values)
            out[mask] = np.clip(out[mask], 0.0, 0.95)
        return float(out[0]) if scalar else out

    def power_integrand(self, v):
        """数值积分专用的功率取值：**切出风速点取左极限**。

        切出风速处机组停机（``P(cut_out) = 0``），但该点在连续分布下测度为零，
        直接参与梯形积分会在分仓边界上引入虚假的 0 点（粗网格时误差可达百分级）。
        因此把积分区间右端点映射到 ``cut_out`` 的前一个可表示浮点数，取左极限值。
        """
        vv = np.asarray(v, dtype=np.float64)
        return self.power(np.minimum(vv, np.nextafter(self.cut_out, 0.0)))

    # -- 反推指标 -------------------------------------------------

    def capacity_factor(self, k: float, c: float, n_grid: int = 2000) -> float:
        """给定 Weibull(k, c) 的**毛**容量系数 = E[P(v)] / P_rated。

        用细网格数值积分 ∫P(v) f(v) dv（梯形法），上限取切出风速。
        """
        if n_grid < 10:
            raise ValueError("积分网格太粗")
        v = np.linspace(0.0, self.cut_out, int(n_grid))
        f = weibull_pdf(v, k, c)
        p = self.power_integrand(v)
        return trapezoid(p * f, v) / self.rated_power

    def mean_power(self, k: float, c: float, n_grid: int = 2000) -> float:
        """给定 Weibull(k, c) 的期望功率（kW）。"""
        v = np.linspace(0.0, self.cut_out, int(n_grid))
        f = weibull_pdf(v, k, c)
        p = self.power_integrand(v)
        return trapezoid(p * f, v)

    def average_power_density(
        self,
        k: float,
        c: float,
        air_density: float = DEFAULT_AIR_DENSITY,
        n_grid: int = 2000,
    ) -> float:
        """单位扫风面积的年均电功率（W/m^2）= E[P] / A。"""
        return self.mean_power(k, c, n_grid) * 1000.0 / self.rotor_area

    # -- 导入导出 -------------------------------------------------

    def to_csv_rows(self) -> List[Tuple[float, float]]:
        return [(float(a), float(b)) for a, b in zip(self.speeds, self.powers)]

    def summary(self) -> str:
        return (
            "机组「%s」：额定 %.0f kW，风轮直径 %.1f m，扫风面积 %.0f m²，"
            "比功率 %.0f W/m²，切入 %.1f / 额定 %.1f / 切出 %.0f m/s（来源：%s）"
            % (self.name, self.rated_power, self.diameter, self.rotor_area,
               self.specific_power, self.cut_in, float(self.rated_speed),
               self.cut_out, self.source)
        )


# ---------------------------------------------------------------- 推力系数


def default_ct_table(
    cut_in: float,
    rated_speed: float,
    cut_out: float,
    ct_region2: float = 0.82,
    ct_at_cutout: float = 0.10,
    step: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """构造**示例**推力系数 Ct(v) 表（不是实测数据）。

    形态依据：变速变桨机组在额定风速以下基本运行在最优叶尖速比，Ct 大致平稳
    （约 0.75~0.85）；达到额定后靠变桨限功率，推力系数随来流增大快速下降，
    工程上常近似为 Ct ∝ v^(-2.5) 衰减到切出风速处的约 0.1。
    """
    if not (0 < cut_in < rated_speed < cut_out):
        raise ValueError("必须满足 切入 < 额定 < 切出")
    v = np.arange(cut_in, cut_out + 1e-9, step)
    if v[-1] < cut_out:
        v = np.append(v, cut_out)
    v_ramp = np.clip((v - cut_in) / max(0.5, 0.15 * rated_speed), 0.0, 1.0)
    ct_r2 = ct_region2 * (0.75 + 0.25 * np.clip(v / max(rated_speed, 1e-6), 0.0, 1.0))
    ct = v_ramp * ct_r2
    above = v > rated_speed
    if np.any(above):
        decay = (rated_speed / v[above]) ** 2.5
        ct_above = ct_region2 * decay
        # 保证切出风速处不小于 ct_at_cutout，并单调衔接
        ct_above = np.maximum(ct_above, ct_at_cutout)
        ct[above] = np.minimum(ct_above, ct_region2)
    return v, np.clip(ct, 0.0, 0.95)


# ---------------------------------------------------------------- 内置示例曲线


def synthetic_curve(
    name: str,
    rated_power: float,
    diameter: float,
    cut_in: float = 3.0,
    cut_out: float = 25.0,
    cp_max: float = 0.45,
    air_density: float = DEFAULT_AIR_DENSITY,
    step: float = 0.05,
) -> PowerCurve:
    """按「恒定 Cp + 额定限功率」构造一条**示例**功率曲线（非厂商实测数据）。

    构造逻辑（物理上自洽）::

        额定风速以下:  P(v) = 0.5 * ρ * A * Cp * v^3      （最优叶尖速比运行）
        额定风速以上:  P(v) = P_rated                       （变桨限功率）
        额定风速由   v_rated = (P_rated / (0.5 ρ A Cp))^(1/3) 唯一确定

    切入风速处功率为 0，切入后 0.5 m/s 内快速爬升（真实厂商曲线的常见形态）。
    """
    area = math.pi * (diameter / 2.0) ** 2
    coef = 0.5 * air_density * area * cp_max          # W / (m/s)^3
    p_rated_w = rated_power * 1000.0
    v_rated = (p_rated_w / coef) ** (1.0 / 3.0)
    if not (cut_in < v_rated < cut_out):
        raise ValueError(
            "参数不自洽：由 Cp=%.2f 推出的额定风速 %.2f m/s 不在 (切入, 切出) 内" % (cp_max, v_rated)
        )

    def ideal(v: float) -> float:
        return min(coef * v ** 3 / 1000.0, rated_power)

    knots: List[float] = [0.0, max(cut_in - 0.5, 0.1), cut_in]
    vals: List[float] = [0.0, 0.0, 0.0]
    # 切入之后的爬升段与额定前的稳定段
    vv = cut_in + 0.5
    while vv < v_rated - 1e-9:
        knots.append(vv)
        vals.append(ideal(vv))
        vv = min(vv + 0.5, v_rated)
    knots.append(v_rated)
    vals.append(rated_power)
    knots.append(cut_out)
    vals.append(rated_power)

    kx = np.array(knots, dtype=np.float64)
    ky = np.array(vals, dtype=np.float64)
    # 去除重复节点（浮点相加可能产生相等值）
    keep = np.concatenate(([True], np.diff(kx) > 1e-9))
    kx, ky = kx[keep], ky[keep]

    dense = np.round(np.arange(0.0, cut_out + 1e-9, step), 6)
    if dense[-1] < cut_out:
        dense = np.append(dense, cut_out)
    # 把额定风速精确插入采样点，保证额定段是真正的常数（而不是插值斜坡）
    dense = np.unique(np.concatenate([dense, np.array([v_rated], dtype=np.float64)]))
    interp = pchip_eval(kx, ky, dense)
    interp = np.clip(interp, 0.0, rated_power)
    interp[dense < cut_in] = 0.0
    interp[dense >= v_rated] = rated_power

    return PowerCurve(
        name=name,
        speeds=dense,
        powers=interp,
        cut_in=cut_in,
        cut_out=cut_out,
        rated_power=rated_power,
        diameter=diameter,
        rated_speed=float(v_rated),
        source="示例曲线（按典型 %.1f MW 级机组形态构造，非厂商实测数据）" % (rated_power / 1000.0),
    )


def example_curve() -> PowerCurve:
    """内置示例曲线：5.0 MW / 风轮直径 132 m，切入 3、额定约 11、切出 25 m/s。

    ⚠️ **示例曲线**：按典型 5 MW 级机组形态构造，**非任何厂商实测数据**。
    """
    return synthetic_curve(
        name="SYN-5.0MW/132",
        rated_power=5000.0,
        diameter=132.0,
        cut_in=3.0,
        cut_out=25.0,
        cp_max=0.45,
    )


def builtin_curves() -> List[PowerCurve]:
    """三条内置**示例**曲线（均为构造数据，非厂商实测）：

    - ``SYN-4.0MW/170``：低比功率（176 W/m²）低风速机型，额定风速约 8.5 m/s
    - ``SYN-5.0MW/132``：中等比功率（365 W/m²），额定风速约 11 m/s
    - ``SYN-6.25MW/172``：中低比功率（269 W/m²），额定风速约 9.9 m/s
    """
    return [
        synthetic_curve("SYN-4.0MW/170", 4000.0, 170.0, cut_in=2.5, cut_out=25.0, cp_max=0.47),
        example_curve(),
        synthetic_curve("SYN-6.25MW/172", 6250.0, 172.0, cut_in=3.0, cut_out=25.0, cp_max=0.46),
    ]


# ---------------------------------------------------------------- CSV 导入导出


def load_curve_csv(
    path: str,
    name: Optional[str] = None,
    diameter: float = 132.0,
    rated_power: Optional[float] = None,
    cut_in: Optional[float] = None,
    cut_out: Optional[float] = None,
    source: str = "用户导入的 CSV 功率曲线",
) -> PowerCurve:
    """从 CSV 导入功率曲线。

    文件格式：两列 ``风速,功率(kW)``；允许 ``#`` 注释行、允许首行表头（非数字则跳过）；
    如果存在多余的列只取前两列。切出风速以上/切入以下的行会被忽略（由边界规则处理）。
    """
    speeds: List[float] = []
    powers: List[float] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = [x.strip() for x in line.replace(";", ",").split(",")]
            if len(parts) < 2:
                continue
            try:
                a, b = float(parts[0]), float(parts[1])
            except ValueError:
                continue        # 表头行
            speeds.append(a)
            powers.append(b)

    if len(speeds) < 2:
        raise ValueError("CSV 中未解析到有效的 (风速, 功率) 数据行：%s" % path)

    order = np.argsort(speeds)
    s = np.asarray(speeds, dtype=np.float64)[order]
    p = np.asarray(powers, dtype=np.float64)[order]
    # 去掉重复风速
    uniq = np.concatenate(([True], np.diff(s) > 1e-9))
    s, p = s[uniq], p[uniq]

    if rated_power is None:
        rated_power = float(np.max(p))
    if cut_in is None:
        nz = np.nonzero(p > 0)[0]
        cut_in = float(s[nz[0]]) if nz.size else float(s[0])
    if cut_out is None:
        rated = rated_power * 0.999
        above = np.nonzero(p >= rated)[0]
        if above.size and above[-1] + 1 < s.size:
            cut_out = float(s[above[-1] + 1])
        else:
            cut_out = float(s[-1]) + 1.0
    if name is None:
        name = "CSV-%s" % str(path).split("/")[-1].split("\\")[-1]

    return PowerCurve(
        name=name, speeds=s, powers=p,
        cut_in=float(cut_in), cut_out=float(cut_out),
        rated_power=float(rated_power), diameter=float(diameter), source=source,
    )


def save_curve_csv(curve: PowerCurve, path: str) -> str:
    """把功率曲线写成 CSV（表头为中文，注释行说明来源）。"""
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write("# %s\n" % curve.summary())
        writer = csv.writer(fh)
        writer.writerow(["风速_m_s", "功率_kW"])
        for v, p in curve.to_csv_rows():
            writer.writerow(["%.2f" % v, "%.3f" % p])
    return path
