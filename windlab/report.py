"""报告输出模块：结构化 Markdown 方案报告 + CSV 数据表 + PNG 图。

PNG 完全手写
------------
本模块自己实现了最小 PNG 编码（``zlib`` + ``struct``）与一个极简绘图引擎
（线段 / 矩形 / 标记 / 5x7 点阵字体），**不依赖 matplotlib / Pillow / imageio**。
图不追求美观，但每一个像素都来自真实计算结果。

坐标约定：画布像素原点在左上角，``x`` 向右、``y`` 向下。
"""

from __future__ import annotations

import csv
import math
import os
import struct
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .aep import AEPResult
from .compare import SchemeResult, comparison_table, competitive_assessment, markdown_table
from .layout import Site, check_layout
from .power_curve import PowerCurve
from .resource import N_SECTORS, WindResource, WindRose, weibull_pdf, sector_name

# ---------------------------------------------------------------- 配色与字体

RGB = Tuple[float, float, float]

C_BG: RGB = (1.00, 1.00, 1.00)
C_AXIS: RGB = (0.25, 0.25, 0.28)
C_GRID: RGB = (0.88, 0.89, 0.91)
C_TEXT: RGB = (0.10, 0.11, 0.13)
C_HIST: RGB = (0.62, 0.72, 0.85)
C_LINE1: RGB = (0.12, 0.36, 0.72)
C_LINE2: RGB = (0.83, 0.36, 0.16)
C_LINE3: RGB = (0.15, 0.58, 0.35)
C_LINE4: RGB = (0.55, 0.30, 0.68)
C_MARK: RGB = (0.10, 0.10, 0.10)
C_SITE: RGB = (0.55, 0.58, 0.62)
C_TURB: RGB = (0.13, 0.38, 0.70)
C_WAKE: RGB = (0.85, 0.60, 0.20)

PALETTE: Tuple[RGB, ...] = (C_LINE1, C_LINE2, C_LINE3, C_LINE4,
                            (0.20, 0.55, 0.60), (0.70, 0.25, 0.45))

#: 5x7 点阵字体（'#' 为点亮像素），覆盖 0-9 A-Z 与常用符号
_FONT: Dict[str, Tuple[str, ...]] = {
    "0": (".###.", "#...#", "#..##", "#.#.#", "##..#", "#...#", ".###."),
    "1": ("..#..", ".##..", "..#..", "..#..", "..#..", "..#..", ".###."),
    "2": (".###.", "#...#", "....#", "...#.", "..#..", ".#...", "#####"),
    "3": ("#####", "...#.", "..#..", "...#.", "....#", "#...#", ".###."),
    "4": ("...#.", "..##.", ".#.#.", "#..#.", "#####", "...#.", "...#."),
    "5": ("#####", "#....", "####.", "....#", "....#", "#...#", ".###."),
    "6": ("..##.", ".#...", "#....", "####.", "#...#", "#...#", ".###."),
    "7": ("#####", "....#", "...#.", "..#..", ".#...", ".#...", ".#..."),
    "8": (".###.", "#...#", "#...#", ".###.", "#...#", "#...#", ".###."),
    "9": (".###.", "#...#", "#...#", ".####", "....#", "...#.", ".##.."),
    "A": (".###.", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"),
    "B": ("####.", "#...#", "#...#", "####.", "#...#", "#...#", "####."),
    "C": (".###.", "#...#", "#....", "#....", "#....", "#...#", ".###."),
    "D": ("###..", "#..#.", "#...#", "#...#", "#...#", "#..#.", "###.."),
    "E": ("#####", "#....", "#....", "####.", "#....", "#....", "#####"),
    "F": ("#####", "#....", "#....", "####.", "#....", "#....", "#...."),
    "G": (".###.", "#...#", "#....", "#.###", "#...#", "#...#", ".###."),
    "H": ("#...#", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"),
    "I": (".###.", "..#..", "..#..", "..#..", "..#..", "..#..", ".###."),
    "J": ("..###", "...#.", "...#.", "...#.", "...#.", "#..#.", ".##.."),
    "K": ("#...#", "#..#.", "#.#..", "##...", "#.#..", "#..#.", "#...#"),
    "L": ("#....", "#....", "#....", "#....", "#....", "#....", "#####"),
    "M": ("#...#", "##.##", "#.#.#", "#.#.#", "#...#", "#...#", "#...#"),
    "N": ("#...#", "##..#", "#.#.#", "#..##", "#...#", "#...#", "#...#"),
    "O": (".###.", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."),
    "P": ("####.", "#...#", "#...#", "####.", "#....", "#....", "#...."),
    "Q": (".###.", "#...#", "#...#", "#...#", "#.#.#", "#..#.", ".##.#"),
    "R": ("####.", "#...#", "#...#", "####.", "#.#..", "#..#.", "#...#"),
    "S": (".####", "#....", "#....", ".###.", "....#", "....#", "####."),
    "T": ("#####", "..#..", "..#..", "..#..", "..#..", "..#..", "..#.."),
    "U": ("#...#", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."),
    "V": ("#...#", "#...#", "#...#", "#...#", "#...#", ".#.#.", "..#.."),
    "W": ("#...#", "#...#", "#...#", "#.#.#", "#.#.#", "##.##", "#...#"),
    "X": ("#...#", "#...#", ".#.#.", "..#..", ".#.#.", "#...#", "#...#"),
    "Y": ("#...#", "#...#", ".#.#.", "..#..", "..#..", "..#..", "..#.."),
    "Z": ("#####", "....#", "...#.", "..#..", ".#...", "#....", "#####"),
    ".": (".....", ".....", ".....", ".....", ".....", ".##..", ".##.."),
    ",": (".....", ".....", ".....", ".....", ".##..", ".##..", ".#..."),
    "-": (".....", ".....", ".....", "#####", ".....", ".....", "....."),
    "+": (".....", "..#..", "..#..", "#####", "..#..", "..#..", "....."),
    "=": (".....", ".....", "#####", ".....", "#####", ".....", "....."),
    ":": (".....", ".##..", ".##..", ".....", ".##..", ".##..", "....."),
    "/": ("....#", "....#", "...#.", "..#..", ".#...", "#....", "#...."),
    "%": ("##..#", "##.#.", "...#.", "..#..", ".#...", "#.##.", "#.##."),
    "(": ("..##.", ".#...", ".#...", ".#...", ".#...", ".#...", "..##."),
    ")": (".##..", "...#.", "...#.", "...#.", "...#.", "...#.", ".##.."),
    "_": (".....", ".....", ".....", ".....", ".....", ".....", "#####"),
    "*": (".....", "#.#.#", ".###.", "#####", ".###.", "#.#.#", "....."),
    " ": (".....", ".....", ".....", ".....", ".....", ".....", "....."),
}
_GLYPH_W = 5
_GLYPH_H = 7


# ---------------------------------------------------------------- 最小 PNG 写出


def write_png(arr: np.ndarray, path: str) -> str:
    """把 ``[H,W]`` 或 ``[H,W,3]``、取值 [0,1] 的数组写成 PNG。

    手写实现（zlib + struct），避免引入 Pillow / imageio / matplotlib 依赖。
    """
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 2:
        lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
        a = (a - lo) / (hi - lo) if hi - lo > 1e-12 else np.zeros_like(a)
        a = np.stack([a] * 3, axis=-1)
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError("write_png 只接受 [H,W] 或 [H,W,3] 数组")
    a = (np.clip(a, 0.0, 1.0) * 255.0).astype(np.uint8)
    h, w, _ = a.shape
    raw = b"".join(b"\x00" + a[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 6))
           + chunk(b"IEND", b""))
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(png)
    return str(p)


# ---------------------------------------------------------------- 画布


class Canvas:
    """极简光栅画布（float RGB，取值 [0,1]）。"""

    def __init__(self, width: int = 960, height: int = 620, bg: RGB = C_BG) -> None:
        self.w = int(width)
        self.h = int(height)
        self.img = np.empty((self.h, self.w, 3), dtype=np.float64)
        self.img[:, :] = np.asarray(bg, dtype=np.float64)[None, None, :]

    # -- 基本图元 -------------------------------------------------

    def fill_rect(self, x0: int, y0: int, x1: int, y1: int, color: RGB) -> None:
        xa, xb = sorted((int(x0), int(x1)))
        ya, yb = sorted((int(y0), int(y1)))
        xa, xb = max(0, xa), min(self.w, xb)
        ya, yb = max(0, ya), min(self.h, yb)
        if xa >= xb or ya >= yb:
            return
        self.img[ya:yb, xa:xb] = np.asarray(color, dtype=np.float64)[None, None, :]

    def rect(self, x0: int, y0: int, x1: int, y1: int, color: RGB) -> None:
        self.fill_rect(x0, y0, x1, y0 + 1, color)
        self.fill_rect(x0, y1 - 1, x1, y1, color)
        self.fill_rect(x0, y0, x0 + 1, y1, color)
        self.fill_rect(x1 - 1, y0, x1, y1, color)

    def hline(self, x0: int, x1: int, y: int, color: RGB) -> None:
        self.fill_rect(x0, y, x1, y + 1, color)

    def vline(self, x: int, y0: int, y1: int, color: RGB) -> None:
        self.fill_rect(x, y0, x + 1, y1, color)

    def set_pixel(self, x: int, y: int, color: RGB) -> None:
        if 0 <= x < self.w and 0 <= y < self.h:
            self.img[y, x] = np.asarray(color, dtype=np.float64)

    def line(self, x0: int, y0: int, x1: int, y1: int, color: RGB) -> None:
        """Bresenham 直线。"""
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        guard = 0
        while True:
            self.set_pixel(x0, y0, color)
            if x0 == x1 and y0 == y1:
                break
            guard += 1
            if guard > 4 * (self.w + self.h + abs(dx) + abs(dy)):
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    def polyline(self, points: Sequence[Tuple[float, float]], color: RGB) -> None:
        pts = [(int(round(x)), int(round(y))) for x, y in points]
        for i in range(len(pts) - 1):
            self.line(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1], color)

    def marker(self, x: float, y: float, size: int, color: RGB) -> None:
        """方形标记。"""
        xi, yi = int(round(x)), int(round(y))
        self.fill_rect(xi - size, yi - size, xi + size + 1, yi + size + 1, color)

    def circle(self, cx: float, cy: float, r: float, color: RGB, thickness: int = 1) -> None:
        """空心圆（逐角度画点，够用且简单）。"""
        n = max(24, int(2 * math.pi * r))
        for k in range(n):
            th = 2 * math.pi * k / n
            for t in range(thickness):
                self.set_pixel(int(round(cx + (r - t) * math.cos(th))),
                               int(round(cy + (r - t) * math.sin(th))), color)

    def text(self, x: int, y: int, s: str, color: RGB = C_TEXT, scale: int = 1) -> None:
        """5x7 点阵文字（只支持 ASCII，自动大写）。"""
        scale = max(1, int(scale))
        cx = int(x)
        for ch in str(s):
            glyph = _FONT.get(ch.upper())
            if glyph is None:
                glyph = _FONT[" "]
            for row in range(_GLYPH_H):
                for col in range(_GLYPH_W):
                    if glyph[row][col] == "#":
                        self.fill_rect(cx + col * scale, y + row * scale,
                                       cx + (col + 1) * scale, y + (row + 1) * scale, color)
            cx += (_GLYPH_W + 1) * scale

    @staticmethod
    def text_width(s: str, scale: int = 1) -> int:
        return max(0, len(str(s)) * (_GLYPH_W + 1) * max(1, int(scale)) - max(1, int(scale)))

    def to_array(self) -> np.ndarray:
        return np.clip(self.img, 0.0, 1.0)


# ---------------------------------------------------------------- 坐标轴


def nice_ticks(lo: float, hi: float, n: int = 5) -> List[float]:
    """生成好看的刻度值。"""
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return [lo]
    raw = (hi - lo) / max(1, int(n))
    mag = 10.0 ** math.floor(math.log10(raw))
    for m in (1.0, 2.0, 2.5, 5.0, 10.0):
        step = m * mag
        if step >= raw:
            break
    start = math.ceil(lo / step) * step
    ticks: List[float] = []
    v = start
    while v <= hi + 1e-9 and len(ticks) < 40:
        ticks.append(round(v, 10))
        v += step
    return ticks


class Plot:
    """像素区域 <-> 数据坐标映射，并负责画坐标轴 / 网格 / 刻度。"""

    def __init__(self, canvas: Canvas, box: Tuple[int, int, int, int],
                 xlim: Tuple[float, float], ylim: Tuple[float, float]) -> None:
        self.c = canvas
        self.x0, self.y0, self.x1, self.y1 = box
        self.xlim = (float(xlim[0]), float(xlim[1]))
        self.ylim = (float(ylim[0]), float(ylim[1]))

    def px(self, x: float) -> float:
        span = self.xlim[1] - self.xlim[0]
        if abs(span) < 1e-12:
            return 0.5 * (self.x0 + self.x1)
        return self.x0 + (float(x) - self.xlim[0]) / span * (self.x1 - self.x0)

    def py(self, y: float) -> float:
        span = self.ylim[1] - self.ylim[0]
        if abs(span) < 1e-12:
            return 0.5 * (self.y0 + self.y1)
        return self.y1 - (float(y) - self.ylim[0]) / span * (self.y1 - self.y0)

    def draw_frame(self, xlabel: str = "", ylabel: str = "", title: str = "",
                   xticks: Optional[Sequence[float]] = None, yticks: Optional[Sequence[float]] = None,
                   xfmt: str = "%.0f", yfmt: str = "%.0f", grid: bool = True,
                   xtick_labels: Optional[Sequence[str]] = None) -> None:
        c = self.c
        if title:
            c.text(8, 8, title, C_TEXT, 2)
        if ylabel:
            c.text(8, 30, ylabel, C_AXIS, 1)
        c.rect(self.x0, self.y0, self.x1, self.y1, C_AXIS)

        ticks_x = list(xticks) if xticks is not None else nice_ticks(*self.xlim)
        ticks_y = list(yticks) if yticks is not None else nice_ticks(*self.ylim)
        for i, tx in enumerate(ticks_x):
            px = self.px(tx)
            if px < self.x0 - 1 or px > self.x1 + 1:
                continue
            if grid:
                c.vline(int(round(px)), self.y0, self.y1, C_GRID)
            c.vline(int(round(px)), self.y1, self.y1 + 4, C_AXIS)
            label = xtick_labels[i] if xtick_labels is not None and i < len(xtick_labels) \
                else (xfmt % tx)
            c.text(int(round(px)) - c.text_width(label) // 2, self.y1 + 8, label, C_AXIS, 1)
        for ty in ticks_y:
            py = self.py(ty)
            if py < self.y0 - 1 or py > self.y1 + 1:
                continue
            if grid:
                c.hline(self.x0, self.x1, int(round(py)), C_GRID)
            c.hline(self.x0 - 4, self.x0, int(round(py)), C_AXIS)
            label = yfmt % ty
            c.text(self.x0 - 8 - c.text_width(label), int(round(py)) - 3, label, C_AXIS, 1)
        if xlabel:
            c.text(int(0.5 * (self.x0 + self.x1)) - c.text_width(xlabel) // 2,
                   self.y1 + 26, xlabel, C_AXIS, 1)

    def bars(self, centers: Sequence[float], heights: Sequence[float],
             width: float, color: RGB) -> None:
        for x, h in zip(centers, heights):
            if h <= 0:
                continue
            xa = self.px(float(x) - 0.5 * width)
            xb = self.px(float(x) + 0.5 * width)
            yb = self.py(0.0)
            ya = self.py(float(h))
            self.c.fill_rect(int(round(xa)), int(round(ya)), int(round(xb)), int(round(yb)), color)

    def curve(self, xs: Sequence[float], ys: Sequence[float], color: RGB) -> None:
        pts = [(self.px(x), self.py(y)) for x, y in zip(xs, ys)]
        self.c.polyline(pts, color)

    def scatter(self, xs: Sequence[float], ys: Sequence[float], color: RGB, size: int = 2) -> None:
        for x, y in zip(xs, ys):
            self.c.marker(self.px(x), self.py(y), size, color)


def _legend(canvas: Canvas, x: int, y: int, entries: Sequence[Tuple[str, RGB]],
            scale: int = 1) -> None:
    """右上角图例。"""
    line_h = _GLYPH_H * scale + 8
    w = max(canvas.text_width(t, scale) for t, _ in entries) + 30
    canvas.fill_rect(x - 6, y - 4, x + w, y + line_h * len(entries), (1.0, 1.0, 1.0))
    canvas.rect(x - 6, y - 4, x + w, y + line_h * len(entries), C_GRID)
    for i, (label, color) in enumerate(entries):
        yy = y + i * line_h
        canvas.fill_rect(x, yy + 1, x + 16, yy + 6, color)
        canvas.text(x + 22, yy, label, C_TEXT, scale)


# ---------------------------------------------------------------- 各类图


def figure_wind_frequency(
    centers: np.ndarray,
    freq: np.ndarray,
    k: float,
    c: float,
    bin_width: float = 1.0,
    title: str = "WIND SPEED FREQUENCY AND WEIBULL FIT",
    name: str = "",
) -> np.ndarray:
    """风频直方图 + Weibull 概率密度拟合曲线。y 轴为各风速仓的出现频率（%）。"""
    centers = np.asarray(centers, dtype=np.float64).ravel()
    freq = np.asarray(freq, dtype=np.float64).ravel()
    canvas = Canvas(960, 620)
    ymax = max(float(np.max(freq)) * 100.0 * 1.25, 1.0) if freq.size else 1.0
    xmax = float(np.max(centers)) + bin_width if centers.size else 25.0
    plot = Plot(canvas, (80, 60, 930, 545), (0.0, xmax), (0.0, ymax))
    plot.draw_frame(xlabel="WIND SPEED (M/S)", ylabel="FREQUENCY (%)",
                    title=title, xfmt="%.0f", yfmt="%.1f")
    plot.bars(centers, freq * 100.0, bin_width * 0.92, C_HIST)

    v = np.linspace(0.0, xmax, 400)
    pdf = weibull_pdf(v, k, c)
    plot.curve(v, pdf * bin_width * 100.0, C_LINE1)

    entries = [("HISTOGRAM", C_HIST), ("WEIBULL PDF", C_LINE1)]
    if name:
        entries.append((name, C_TEXT))
    _legend(canvas, 700, 74, entries)
    canvas.text(90, 575, "k=%.3f  c=%.3f m/s  bin=%.2f m/s" % (k, c, bin_width), C_TEXT, 1)
    return canvas.to_array()


def figure_power_curves(
    curves: Sequence[PowerCurve],
    title: str = "POWER CURVES (EXAMPLE / SYNTHETIC DATA)",
    highlight: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """多条功率曲线对比图（可叠加若干参考风速竖线）。"""
    canvas = Canvas(960, 620)
    v_max = max(cu.cut_out for cu in curves)
    p_max = max(cu.rated_power for cu in curves) * 1.15
    plot = Plot(canvas, (80, 60, 930, 545), (0.0, v_max), (0.0, p_max))
    plot.draw_frame(xlabel="WIND SPEED (M/S)", ylabel="POWER (KW)", title=title,
                    xfmt="%.0f", yfmt="%.0f")
    entries: List[Tuple[str, RGB]] = []
    for i, cu in enumerate(curves):
        color = PALETTE[i % len(PALETTE)]
        v = np.linspace(0.0, cu.cut_out, 600)
        plot.curve(v, cu.power(v), color)
        entries.append(("%s (%s)" % (cu.name, "%.1f MW" % (cu.rated_power / 1000.0)), color))
    if highlight:
        for v_ref in highlight:
            px = plot.px(v_ref)
            for yy in range(60, 546, 8):
                canvas.set_pixel(int(round(px)), yy, C_SITE)
    _legend(canvas, 620, 74, entries)
    canvas.text(90, 575, "MARKED LINES: TYPICAL WIND SPEEDS (5 / 8 / 11 M/S)", C_AXIS, 1)
    return canvas.to_array()


def _draw_rose(canvas: Canvas, cx: float, cy: float, radius: float,
               rose: WindRose, color: RGB = C_TURB) -> None:
    """在画布上叠加一个风玫瑰（气象习惯：0 度在上，顺时针）。"""
    canvas.circle(cx, cy, radius, C_SITE, 1)
    freqs = np.asarray(rose.freqs, dtype=np.float64)
    fmax = float(np.max(freqs)) if freqs.size else 1.0
    for i in range(N_SECTORS):
        theta = math.radians(i * 360.0 / N_SECTORS)
        ux, uy = math.sin(theta), -math.cos(theta)      # 像素坐标：北为上
        r = radius * 0.12 + radius * 0.88 * (freqs[i] / fmax if fmax > 0 else 0.0)
        canvas.line(int(round(cx + ux * radius * 0.10)), int(round(cy + uy * radius * 0.10)),
                    int(round(cx + ux * r)), int(round(cy + uy * r)), color)
    canvas.text(int(cx - 4), int(cy - radius - 12), "N", C_TEXT, 1)
    canvas.text(int(cx + radius + 6), int(cy - 4), "E", C_AXIS, 1)
    canvas.text(int(cx - radius - 14), int(cy - 4), "W", C_AXIS, 1)


def figure_layout(
    site: Site,
    positions: np.ndarray,
    diameter: float,
    rose: Optional[WindRose] = None,
    title: str = "TURBINE LAYOUT",
    main_dir_deg: float = 270.0,
    subtitle: str = "",
) -> np.ndarray:
    """机位排布图：场址边界 + 各机组风轮圆 + 主风向箭头 + 风玫瑰小图。"""
    canvas = Canvas(960, 620)
    pos = np.atleast_2d(np.asarray(positions, dtype=np.float64))
    x0, y0, x1, y1 = site.bbox
    pad = 0.10 * max(x1 - x0, y1 - y0)
    plot = Plot(canvas, (70, 60, 930, 560), (x0 - pad, x1 + pad), (y0 - pad, y1 + pad))
    plot.draw_frame(xlabel="EAST (M)", ylabel="NORTH (M)", title=title,
                    xfmt="%.0f", yfmt="%.0f")

    poly = site.polygon
    for i in range(poly.shape[0]):
        a = poly[i]
        b = poly[(i + 1) % poly.shape[0]]
        canvas.line(int(round(plot.px(a[0]))), int(round(plot.py(a[1]))),
                    int(round(plot.px(b[0]))), int(round(plot.py(b[1]))), C_SITE)

    # 主风向箭头（风来自 main_dir_deg，向下游吹）
    th = math.radians(main_dir_deg)
    ux, uy = -math.sin(th), -math.cos(th)
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    span = 0.18 * max(x1 - x0, y1 - y0)
    sx_px, sy_px = plot.px(cx - ux * span), plot.py(cy - uy * span)
    ex_px, ey_px = plot.px(cx + ux * span), plot.py(cy + uy * span)
    for k in range(3):
        canvas.line(int(round(sx_px)), int(round(sy_px)) + k,
                    int(round(ex_px)), int(round(ey_px)) + k, C_WAKE)
    canvas.text(int(ex_px) + 6, int(ey_px) - 3, "MAIN WIND %.0f DEG" % main_dir_deg, C_WAKE, 1)

    for i in range(pos.shape[0]):
        px, py = plot.px(pos[i, 0]), plot.py(pos[i, 1])
        r_px = abs(plot.px(pos[i, 0] + 0.5 * diameter) - px)
        canvas.circle(px, py, max(2.0, r_px), C_TURB, 1)
        canvas.marker(px, py, 1, C_TURB)
        canvas.text(int(px) + 3, int(py) + 4, str(i + 1), C_TEXT, 1)

    if rose is not None:
        _draw_rose(canvas, 840, 480, 70, rose)
        canvas.text(790, 570, "WIND ROSE", C_AXIS, 1)
    if subtitle:
        canvas.text(80, 582, subtitle, C_TEXT, 1)
    return canvas.to_array()


def figure_directional_wake(
    rows: Sequence[Dict[str, float]],
    title: str = "WAKE LOSS VS WIND DIRECTION",
    dominant_deg: Optional[float] = None,
) -> np.ndarray:
    """尾流功率损失随风向的变化曲线。"""
    canvas = Canvas(960, 620)
    x = [r["direction_deg"] for r in rows]
    y = [r["power_loss_pct"] for r in rows]
    ymax = max(max(y) * 1.2 if y else 1.0, 0.5)
    plot = Plot(canvas, (80, 60, 930, 545), (0.0, 360.0), (0.0, ymax))
    ticks = list(range(0, 361, 45))
    labels = ["%d" % t for t in ticks]
    plot.draw_frame(xlabel="WIND DIRECTION (DEG, FROM)", ylabel="POWER LOSS (%)",
                    title=title, xticks=ticks, yticks=nice_ticks(0.0, ymax),
                    xtick_labels=labels, yfmt="%.1f")
    if x:
        xx = list(x) + [x[-1] + 10.0]
        yy = list(y) + [y[0]]
        plot.curve(xx, yy, C_LINE1)
        plot.scatter(x, y, C_LINE1, 2)
    if dominant_deg is not None:
        canvas.vline(int(round(plot.px(dominant_deg))), 60, 545, C_LINE2)
        canvas.text(int(plot.px(dominant_deg)) + 4, 64, "DOMINANT", C_LINE2, 1)
    _legend(canvas, 700, 74, [("POWER LOSS", C_LINE1)])
    canvas.text(90, 575, "X AXIS: METEOROLOGICAL DIRECTION (WIND FROM)", C_AXIS, 1)
    return canvas.to_array()


def figure_pareto(
    front: Sequence[Dict[str, float]],
    title: str = "PARETO FRONT (NSGA-II)",
    x_key: str = "footprint_km2",
    y_key: str = "aep_gwh",
    xlabel: str = "FOOTPRINT (KM2)",
    ylabel: str = "NET AEP (GWH/Y)",
) -> np.ndarray:
    """Pareto 前沿散点图。"""
    canvas = Canvas(960, 620)
    xs = [float(d[x_key]) for d in front]
    ys = [float(d[y_key]) for d in front]
    if not xs:
        xs, ys = [0.0, 1.0], [0.0, 1.0]
    x_lo, x_hi = min(xs), max(xs)
    y_lo, y_hi = min(ys), max(ys)
    x_pad = max(0.1 * (x_hi - x_lo), 0.02)
    y_pad = max(0.1 * (y_hi - y_lo), 0.05)
    plot = Plot(canvas, (80, 60, 930, 545), (x_lo - x_pad, x_hi + x_pad),
                (y_lo - y_pad, y_hi + y_pad))
    plot.draw_frame(xlabel=xlabel, ylabel=ylabel, title=title, xfmt="%.1f", yfmt="%.1f")
    if len(xs) > 1:
        order = np.argsort(xs)
        plot.curve([xs[i] for i in order], [ys[i] for i in order], C_LINE2)
    plot.scatter(xs, ys, C_LINE1, 3)
    if xs:
        i_max = int(np.argmax(ys))
        canvas.marker(plot.px(xs[i_max]), plot.py(ys[i_max]), 5, C_LINE3)
        canvas.text(int(plot.px(xs[i_max])) + 8, int(plot.py(ys[i_max])) - 6,
                    "MAX AEP", C_LINE3, 1)
        i_min = int(np.argmin(xs))
        canvas.marker(plot.px(xs[i_min]), plot.py(ys[i_min]), 5, C_LINE4)
        canvas.text(int(plot.px(xs[i_min])) + 8, int(plot.py(ys[i_min])) + 6,
                    "MIN FOOTPRINT", C_LINE4, 1)
    _legend(canvas, 700, 74, [("PARETO", C_LINE1), ("FRONT LINE", C_LINE2)])
    canvas.text(90, 575, "NON-DOMINATED SOLUTIONS: %d" % len(xs), C_TEXT, 1)
    return canvas.to_array()


def figure_tornado(
    rows: Sequence[Dict[str, float]],
    title: str = "SENSITIVITY (TORNADO)",
    xlabel: str = "LCOE (YUAN/KWH)",
    value_key_base: str = "base",
    unit_fmt: str = "%.4f",
) -> np.ndarray:
    """龙卷风图：各因素在低值/高值情形下的指标变化。"""
    canvas = Canvas(960, 620)
    rows = list(rows)
    if not rows:
        rows = [{"factor": "N/A", "low_value": 0.0, "high_value": 1.0, "base": 0.5, "span": 1.0}]
    vals = [float(r["low_value"]) for r in rows] + [float(r["high_value"]) for r in rows] \
        + [float(r["base"]) for r in rows]
    lo, hi = min(vals), max(vals)
    pad = max(0.06 * (hi - lo), 1e-4)
    n = len(rows)
    top, bottom = 90, 545
    row_h = (bottom - top) / max(n, 1)
    plot = Plot(canvas, (300, top, 930, bottom), (lo - pad, hi + pad), (0.0, float(n)))
    plot.draw_frame(xlabel=xlabel, ylabel="", title=title, yfmt="%.0f",
                    yticks=list(range(0, n + 1)))
    base_x = plot.px(float(rows[0]["base"]))
    canvas.vline(int(round(base_x)), top, bottom, C_MARK)

    for i, r in enumerate(rows):
        y_center = bottom - (i + 0.5) * row_h
        y_top = int(round(y_center - 0.28 * row_h))
        y_bot = int(round(y_center + 0.28 * row_h))
        xa, xb = plot.px(float(r["low_value"])), plot.px(float(r["high_value"]))
        if xa > xb:
            xa, xb = xb, xa
        xa_i, xb_i = int(round(xa)), int(round(xb))
        # 低值 <= 基准 <= 高值 时的左右着色意义不同，这里统一用两段颜色体现跨度
        canvas.fill_rect(xa_i, y_top, base_x, y_bot, PALETTE[i % len(PALETTE)])
        canvas.fill_rect(base_x, y_top, xb_i, y_bot, C_WAKE)
        canvas.text(10, int(y_center) - 4, str(r["factor"])[:22], C_TEXT, 1)
        canvas.text(min(xa_i, xb_i), y_bot + 3, unit_fmt % float(r["low_value"]), C_AXIS, 1)
        canvas.text(int(round(xb)), y_bot + 3, unit_fmt % float(r["high_value"]), C_AXIS, 1)
    canvas.text(310, 566, "BASE = %s" % (unit_fmt % float(rows[0]["base"])), C_TEXT, 1)
    return canvas.to_array()


def figure_convergence(
    history: Sequence[Dict[str, float]],
    x_key: str = "iteration",
    title: str = "OPTIMIZATION CONVERGENCE",
    ylabel: str = "NET AEP (GWH/Y)",
) -> np.ndarray:
    """收敛曲线：每代最优 / 当代值 / 累计最优。"""
    canvas = Canvas(960, 620)
    hist = list(history)
    if not hist:
        hist = [{x_key: 0.0, "value": 0.0, "best_value": 0.0}]
    xs = [float(h.get(x_key, i)) for i, h in enumerate(hist)]
    ys = [float(h.get("value", 0.0)) for h in hist]
    best = [float(h.get("best_value", h.get("value", 0.0))) for h in hist]
    lo = min(min(ys), min(best))
    hi = max(max(ys), max(best))
    pad = max(0.08 * (hi - lo), 1e-3)
    plot = Plot(canvas, (80, 60, 930, 545), (min(xs), max(xs)), (lo - pad, hi + pad))
    plot.draw_frame(xlabel=x_key.upper(), ylabel=ylabel, title=title, xfmt="%.0f", yfmt="%.2f")
    plot.curve(xs, ys, C_SITE)
    plot.curve(xs, best, C_LINE1)
    plot.scatter(xs, best, C_LINE1, 2)
    _legend(canvas, 640, 74, [("CURRENT", C_SITE), ("BEST SO FAR", C_LINE1)])
    canvas.text(90, 575, "EVALUATIONS: %d" % len(hist), C_TEXT, 1)
    return canvas.to_array()


def figure_scheme_compare(
    results: Sequence[SchemeResult],
    title: str = "SCHEME COMPARISON",
) -> np.ndarray:
    """方案对比双面板：左 AEP 柱状、右 LCOE 柱状。"""
    canvas = Canvas(960, 620)
    names = [r.name for r in results]
    aeps = [r.aep_gwh for r in results]
    lcoes = [r.lcoe for r in results]

    left = Plot(canvas, (90, 70, 470, 520), (0.0, max(len(names), 1)), (0.0, max(aeps) * 1.2))
    left.draw_frame(ylabel="NET AEP (GWH/Y)", title=title[:28], yfmt="%.0f",
                    xticks=[], yticks=nice_ticks(0.0, max(aeps) * 1.2))
    for i, v in enumerate(aeps):
        left.bars([i + 0.5], [v], 0.6, PALETTE[i % len(PALETTE)])
        canvas.text(int(left.px(i + 0.5)) - 18, int(left.py(v)) - 12, "%.1f" % v, C_TEXT, 1)
    for i, nm in enumerate(names):
        canvas.text(int(left.px(i + 0.5)) - canvas.text_width(nm[:9], 1) // 2, 528,
                    nm[:9], C_AXIS, 1)

    right = Plot(canvas, (560, 70, 930, 520), (0.0, max(len(names), 1)),
                 (0.0, max(lcoes) * 1.3))
    right.draw_frame(ylabel="LCOE (YUAN/KWH)", title="LCOE BY SCHEME", yfmt="%.2f",
                     xticks=[], yticks=nice_ticks(0.0, max(lcoes) * 1.3))
    for i, v in enumerate(lcoes):
        right.bars([i + 0.5], [v], 0.6, PALETTE[i % len(PALETTE)])
        canvas.text(int(right.px(i + 0.5)) - 20, int(right.py(v)) - 12, "%.3f" % v, C_TEXT, 1)
    for i, nm in enumerate(names):
        canvas.text(int(right.px(i + 0.5)) - canvas.text_width(nm[:9], 1) // 2, 528,
                    nm[:9], C_AXIS, 1)
    canvas.text(90, 566, "BAR COLORS FOLLOW THE SCHEME ORDER IN THE TABLE", C_AXIS, 1)
    return canvas.to_array()


# ---------------------------------------------------------------- CSV 输出


def write_csv(path: str, rows: Sequence[Dict[str, object]],
              columns: Optional[Sequence[str]] = None) -> str:
    """把行字典列表写成 CSV（UTF-8 with BOM，方便 Excel 直接打开中文列名）。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text("", encoding="utf-8-sig")
        return str(p)
    cols = list(columns) if columns else list(rows[0].keys())
    with open(p, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return str(p)


def layout_csv_rows(
    positions: np.ndarray,
    diameter: float,
    per_turbine_gwh: Optional[np.ndarray] = None,
    u_eff_by_sector: Optional[np.ndarray] = None,
) -> List[Dict[str, object]]:
    """机位坐标表。"""
    pos = np.atleast_2d(np.asarray(positions, dtype=np.float64))
    rows: List[Dict[str, object]] = []
    for i in range(pos.shape[0]):
        row: Dict[str, object] = {
            "机组编号": i + 1,
            "X_东向_m": round(float(pos[i, 0]), 2),
            "Y_北向_m": round(float(pos[i, 1]), 2),
        }
        if per_turbine_gwh is not None and i < len(per_turbine_gwh):
            row["单机净发电量_GWh"] = round(float(per_turbine_gwh[i]), 4)
        rows.append(row)
    return rows


def sensitivity_csv_rows(rows: Sequence[Dict[str, float]], metric: str = "LCOE_元每kWh") -> List[Dict[str, object]]:
    """敏感性分析表。"""
    out: List[Dict[str, object]] = []
    for r in rows:
        out.append({
            "因素": r["factor"],
            "低值乘子": r["low_multiplier"],
            "高值乘子": r["high_multiplier"],
            metric + "_低值": round(float(r["low_value"]), 6),
            metric + "_基准": round(float(r["base"]), 6),
            metric + "_高值": round(float(r["high_value"]), 6),
            "跨度": round(float(r["span"]), 6),
        })
    return out


def pareto_csv_rows(front: Sequence[Dict[str, float]]) -> List[Dict[str, object]]:
    """Pareto 前沿表。"""
    rows: List[Dict[str, object]] = []
    for i, d in enumerate(front):
        rows.append({
            "解编号": i + 1,
            "净AEP_GWh": round(float(d["aep_gwh"]), 4),
            "尾流损失_%": round(float(d["wake_loss_pct"]), 4),
            "占地_km2": round(float(d["footprint_km2"]), 4),
            "LCOE_元每kWh": (round(float(d["lcoe"]), 6) if d.get("lcoe") == d.get("lcoe") else ""),
        })
    return rows


def history_csv_rows(history: Sequence[Dict[str, float]]) -> List[Dict[str, object]]:
    """收敛历史表。"""
    rows: List[Dict[str, object]] = []
    for h in history:
        rows.append({k: (round(float(v), 6) if isinstance(v, (int, float)) else v)
                     for k, v in h.items()})
    return rows


# ---------------------------------------------------------------- Markdown 报告


@dataclass
class ReportBundle:
    """报告产物收集器：统一管理输出目录与文件清单。"""

    out_dir: str = "reports"
    files: List[str] = field(default_factory=list)
    blocks: List[str] = field(default_factory=list)

    def path(self, name: str) -> str:
        p = Path(self.out_dir)
        p.mkdir(parents=True, exist_ok=True)
        return str(p / name)

    def add(self, path: str) -> str:
        self.files.append(str(path))
        return str(path)

    def save_png(self, name: str, arr: np.ndarray) -> str:
        return self.add(write_png(arr, self.path(name)))

    def save_csv(self, name: str, rows: Sequence[Dict[str, object]],
                 columns: Optional[Sequence[str]] = None) -> str:
        return self.add(write_csv(self.path(name), rows, columns))

    def section(self, heading: str, body: str, level: int = 2) -> None:
        self.blocks.append("%s %s\n\n%s\n" % ("#" * level, heading, body.strip()))

    def raw(self, text: str) -> None:
        self.blocks.append(text.strip() + "\n")

    def figure_section(self, heading: str, png_path: str, caption: str, level: int = 2) -> None:
        rel = os.path.basename(png_path)
        self.blocks.append("%s %s\n\n![%s](%s)\n\n%s\n" % (
            "#" * level, heading, caption, rel, caption))

    def write_markdown(self, name: str, title: str, intro: str = "",
                       footer: str = "") -> str:
        parts = ["# %s\n" % title]
        if intro:
            parts.append(intro.strip() + "\n")
        parts.extend(self.blocks)
        if footer:
            parts.append(footer.strip() + "\n")
        text = "\n".join(parts)
        p = self.path(name)
        Path(p).write_text(text, encoding="utf-8")
        return self.add(p)

    def manifest(self) -> str:
        """产物清单（中文）。"""
        lines = ["生成的报告产物："]
        for f in self.files:
            size = os.path.getsize(f) if os.path.exists(f) else 0
            lines.append("  %-46s %8d B" % (os.path.basename(f), size))
        return "\n".join(lines)


def resource_block(resource: WindResource, k_fit: float, c_fit: float,
                   fit_method: str = "矩估计") -> str:
    """风资源表（中文 Markdown）。"""
    rose = resource.rose
    lines = [
        "| 指标 | 数值 |",
        "|---|---|",
        "| 数据来源 | %s |" % rose.name,
        "| 年平均风速 | %.2f m/s |" % rose.mean_speed,
        "| 主风向 | %s（%.1f°） |" % (rose.dominant_sector_name, rose.dominant_direction_deg),
        "| 风向集中度 | %.3f（0=均匀，1=单扇区） |" % rose.directional_spread,
        "| 场地 Weibull k | %.3f |" % resource.k,
        "| 等效 Weibull c | %.3f m/s |" % resource.equivalent_c,
        "| %s拟合 (k, c) | %.3f, %.3f m/s |" % (fit_method, k_fit, c_fit),
        "| 风功率密度 | %.1f W/m² |" % resource.power_density,
        "| 空气密度 | %.3f kg/m³ |" % resource.air_density,
        "",
        "逐扇区分布：",
        "",
        "| 扇区 | 风向(°) | 频率(%) | 平均风速(m/s) | 扇区风功率密度(W/m²) |",
        "|---|---|---|---|---|",
    ]
    for i in range(N_SECTORS):
        lines.append("| %s | %.1f | %.2f | %.2f | %.1f |" % (
            sector_name(i), i * 22.5, 100.0 * rose.freqs[i],
            rose.mean_speeds[i], resource.sector_power_density(i)))
    return "\n".join(lines)


def scheme_block(results: Sequence[SchemeResult], baseline: Optional[str] = None) -> str:
    """方案对比表 + 竞争力判断（中文 Markdown）。"""
    rows = comparison_table(results)
    out = [markdown_table(rows), "", "相对基准方案的竞争力判断：", ""]
    out.append(markdown_table(competitive_assessment(results, baseline)))
    return "\n".join(out)


def aep_block(res: AEPResult, name: str = "方案") -> str:
    """单个方案的发电量明细表。"""
    lines = [
        "| 指标 | 数值 |",
        "|---|---|",
        "| 方案 | %s |" % name,
        "| 机组数 / 装机 | %d 台 / %.1f MW |" % (res.n_turbines, res.capacity_mw),
        "| 毛发电量 | %.2f GWh/年 |" % res.gross_gwh,
        "| 尾流后发电量 | %.2f GWh/年 |" % res.wake_gwh,
        "| 净发电量 | %.2f GWh/年 |" % res.net_gwh,
        "| 尾流损失 | %.2f %% |" % res.wake_loss_pct,
        "| 可用率 | %.3f |" % res.availability,
        "| 其他综合折减 | %.3f |" % res.other_loss_factor,
        "| 毛容量系数 | %.2f %% |" % (100.0 * res.gross_capacity_factor),
        "| 净容量系数 | %.2f %% |" % (100.0 * res.net_capacity_factor),
        "| 年满发小时 | %.0f h |" % res.full_load_hours,
        "| 总折减率 | %.2f %% |" % res.total_loss_pct,
    ]
    return "\n".join(lines)


def sector_block(res: AEPResult, top: int = 6) -> str:
    """逐扇区发电量明细（按频率降序取前 top 个）。"""
    rows = sorted(res.per_sector, key=lambda d: -float(d["freq"]))[:top]
    lines = [
        "| 扇区 | 风向(°) | 频率(%) | 平均风速(m/s) | 毛发电量(GWh) | 尾流后(GWh) | 尾流损失(%) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append("| %s | %.1f | %.2f | %.2f | %.3f | %.3f | %.2f |" % (
            r["name"], r["direction_deg"], 100.0 * r["freq"], r["mean_speed"],
            r["gross_gwh"], r["wake_gwh"], r["wake_loss_pct"]))
    return "\n".join(lines)


def summary_text(results: Sequence[SchemeResult]) -> str:
    """控制台用中文摘要表。"""
    lines = ["方案对比汇总："]
    for r in results:
        lines.append("  " + r.summary_line())
    return "\n".join(lines)
