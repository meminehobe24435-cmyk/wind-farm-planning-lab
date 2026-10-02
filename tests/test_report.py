"""报告模块测试：手写 PNG 编码、绘图函数、CSV 与 Markdown 输出。"""

from __future__ import annotations

import os
import struct
import zlib
from pathlib import Path

import numpy as np
import pytest

from windlab.aep import evaluate_aep
from windlab.compare import Scheme, evaluate_scheme
from windlab.economics import Economics, sensitivity_lcoe
from windlab.layout import Site, grid_layout
from windlab.power_curve import example_curve
from windlab.report import (
    C_LINE1,
    Canvas,
    Plot,
    ReportBundle,
    aep_block,
    figure_convergence,
    figure_directional_wake,
    figure_layout,
    figure_pareto,
    figure_power_curves,
    figure_scheme_compare,
    figure_tornado,
    figure_wind_frequency,
    history_csv_rows,
    layout_csv_rows,
    markdown_table,
    nice_ticks,
    pareto_csv_rows,
    resource_block,
    scheme_block,
    sector_block,
    sensitivity_csv_rows,
    summary_text,
    write_csv,
    write_png,
)
from windlab.resource import (
    example_wind_resource,
    sample_speeds,
    speed_histogram,
    fit_weibull_moments,
)
from windlab.wake import WakeModel

D = 132.0


def parse_png_chunks(path: str):
    """按 PNG 规范逐块解析，返回 [(chunk_tag, payload), ...]。"""
    data = Path(path).read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", "PNG 魔数错误"
    pos = 8
    chunks = []
    while pos + 12 <= len(data):
        length = struct.unpack(">I", data[pos:pos + 4])[0]
        tag = data[pos + 4:pos + 8]
        payload = data[pos + 8:pos + 8 + length]
        crc = struct.unpack(">I", data[pos + 8 + length:pos + 12 + length])[0]
        body = tag + payload
        assert crc == (zlib.crc32(body) & 0xFFFFFFFF), "CRC 校验失败：%r" % tag
        chunks.append((tag, payload))
        pos += 12 + length
    assert pos == len(data), "PNG 尾部有残留字节"
    return chunks


def read_png_header(path: str):
    """解析 PNG 头，返回 (width, height, bit_depth, color_type)。"""
    chunks = parse_png_chunks(path)
    assert chunks[0][0] == b"IHDR"
    assert chunks[-1][0] == b"IEND"
    w, h, depth, ctype = struct.unpack(">IIBB", chunks[0][1][:10])
    assert chunks[0][1][10:13] == b"\x00\x00\x00"
    return w, h, depth, ctype


# ---------------------------------------------------------------- PNG 编码


def test_write_png_roundtrip_gray(tmp_path) -> None:
    """灰度数组写出的 PNG 头部、CRC 与像素数据都可验证。"""
    arr = np.array([[0.0, 1.0], [0.5, 0.5]])
    p = str(tmp_path / "small.png")
    write_png(arr, p)
    w, h, depth, ctype = read_png_header(p)
    assert (w, h) == (2, 2)
    assert depth == 8 and ctype == 2
    chunks = parse_png_chunks(p)
    idat = b"".join(payload for tag, payload in chunks if tag == b"IDAT")
    raw = zlib.decompress(idat)
    assert len(raw) == h * (1 + w * 3)
    # 第 0 行：滤波字节 + 像素 (0,0,0) + 像素 (255,255,255)
    assert raw[0] == 0
    assert list(raw[1:4]) == [0, 0, 0]
    assert list(raw[4:7]) == [255, 255, 255]
    # 第 1 行：滤波字节 + 0.5 的灰度（*255 后取整为 127）
    assert raw[7] == 0
    assert list(raw[8:11]) == [127, 127, 127]
    assert os.path.getsize(p) > 0


def test_write_png_gray_normalization(tmp_path) -> None:
    """二维数组会按自身极值归一化到 [0,1] 再编码。"""
    arr = np.linspace(0.0, 1.0, 64 * 32).reshape(32, 64)
    p = str(tmp_path / "gray.png")
    write_png(arr, p)
    w, h, _, _ = read_png_header(p)
    assert (w, h) == (64, 32)
    chunks = parse_png_chunks(p)
    idat = b"".join(payload for tag, payload in chunks if tag == b"IDAT")
    raw = zlib.decompress(idat)
    assert raw[1] == 0                       # 第一行第一个像素最暗
    assert raw[1 + 63 * 3] == 7              # 第一行最后一个像素（局部最大值）


def test_write_png_rejects_bad_shape(tmp_path) -> None:
    with pytest.raises(ValueError):
        write_png(np.zeros((4, 4, 4)), str(tmp_path / "bad.png"))


def test_write_png_creates_parent_dir(tmp_path) -> None:
    """输出目录不存在时自动创建。"""
    p = write_png(np.zeros((8, 8)), str(tmp_path / "sub" / "dir" / "x.png"))
    assert Path(p).exists()


# ---------------------------------------------------------------- 画布与坐标轴


def test_canvas_basic_primitives() -> None:
    """画布图元不越界且写入生效。"""
    c = Canvas(100, 80)
    assert c.to_array().shape == (80, 100, 3)
    c.fill_rect(10, 10, 20, 20, C_LINE1)
    arr = c.to_array()
    assert np.allclose(arr[15, 15], C_LINE1)
    assert np.allclose(arr[0, 0], (1.0, 1.0, 1.0))
    c.line(0, 0, 99, 79, (0.0, 0.0, 0.0))
    c.circle(50, 40, 10, (0.0, 0.0, 0.0))
    c.marker(50, 40, 2, (0.0, 0.0, 0.0))
    c.hline(0, 99, 5, (0.0, 0.0, 0.0))
    c.vline(5, 0, 79, (0.0, 0.0, 0.0))
    c.rect(30, 30, 60, 60, (0.0, 0.0, 0.0))
    assert arr.shape == (80, 100, 3)
    # 越界绘制不应抛错
    c.set_pixel(-5, -5, (0.0, 0.0, 0.0))
    c.fill_rect(-100, -100, -50, -50, (0.0, 0.0, 0.0))
    c.text(0, 0, "中文会被忽略", (0.0, 0.0, 0.0))


def test_canvas_text_renders_pixels() -> None:
    """点阵文字真的画出了像素，且宽度随字串长度与缩放变化。"""
    c = Canvas(200, 40)
    c.text(5, 5, "A1", (0.0, 0.0, 0.0), 2)
    arr = c.to_array()
    assert float(np.min(arr)) < 0.5
    assert Canvas.text_width("ABC", 1) > Canvas.text_width("AB", 1)
    assert Canvas.text_width("A", 2) > Canvas.text_width("A", 1)
    assert Canvas.text_width("", 1) == 0


def test_nice_ticks() -> None:
    """刻度值落在范围内、间距均匀、数量合理。"""
    t = nice_ticks(0.0, 100.0, 5)
    assert t[0] >= 0.0 and t[-1] <= 100.0
    assert len(t) >= 3
    assert np.allclose(np.diff(t), np.diff(t)[0])
    assert nice_ticks(5.0, 5.0) == [5.0]


def test_plot_mapping() -> None:
    """数据坐标 -> 像素坐标映射（y 轴翻转）。"""
    c = Canvas(200, 200)
    p = Plot(c, (0, 0, 100, 100), (0.0, 10.0), (0.0, 10.0))
    assert abs(p.px(0.0) - 0.0) < 1e-9
    assert abs(p.px(10.0) - 100.0) < 1e-9
    assert abs(p.py(0.0) - 100.0) < 1e-9
    assert abs(p.py(10.0) - 0.0) < 1e-9
    p.draw_frame(xlabel="X", ylabel="Y", title="T")
    assert c.to_array().shape == (200, 200, 3)


# ---------------------------------------------------------------- 各类图


def test_figure_wind_frequency() -> None:
    """风频图：尺寸正确、有内容。"""
    s = sample_speeds(2.0, 8.0, n=50000, seed=1)
    centers, freq, _ = speed_histogram(s, bin_width=1.0)
    k, cc = fit_weibull_moments(s)
    arr = figure_wind_frequency(centers, freq, k, cc, bin_width=1.0, name="TEST")
    assert arr.shape == (620, 960, 3)
    assert float(arr.min()) < 0.5
    assert 0.0 <= float(arr.max()) <= 1.0


def test_figure_power_curves() -> None:
    """功率曲线图。"""
    from windlab.power_curve import builtin_curves
    arr = figure_power_curves(builtin_curves(), highlight=[5.0, 8.0, 11.0])
    assert arr.shape == (620, 960, 3)
    assert float(arr.min()) < 0.5


def test_figure_layout_with_rose() -> None:
    """机位排布图（含风玫瑰与主风向箭头）。"""
    curve = example_curve()
    res = example_wind_resource()
    site = Site.rectangle(3600.0, 2400.0)
    pos = grid_layout(site, 3, 4, 5 * curve.diameter, 3 * curve.diameter,
                      main_dir_deg=270.0, margin=100.0)
    arr = figure_layout(site, pos, curve.diameter, rose=res.rose,
                        main_dir_deg=270.0, subtitle="12 TURBINES")
    assert arr.shape == (620, 960, 3)
    assert float(arr.min()) < 0.5
    # 不带风玫瑰也能画
    arr2 = figure_layout(site, pos, curve.diameter, rose=None)
    assert arr2.shape == arr.shape


def test_figure_directional_wake() -> None:
    """尾流随风向变化图。"""
    from windlab.aep import directional_wake_loss
    curve = example_curve()
    pos = np.zeros((4, 2))
    pos[:, 0] = np.arange(4) * 5 * D
    model = WakeModel(diameter=D, k_wake=0.075, rotor_grid=8)
    rows = directional_wake_loss(pos, curve, model, 8.0, step_deg=15.0)
    arr = figure_directional_wake(rows, dominant_deg=270.0)
    assert arr.shape == (620, 960, 3)
    assert float(arr.min()) < 0.5


def test_figure_pareto_and_empty() -> None:
    """Pareto 前沿图（含空前沿的健壮性）。"""
    front = [
        {"aep_gwh": 100.0, "wake_loss_pct": 12.0, "footprint_km2": 3.0, "lcoe": 0.32},
        {"aep_gwh": 110.0, "wake_loss_pct": 10.0, "footprint_km2": 5.0, "lcoe": 0.30},
        {"aep_gwh": 115.0, "wake_loss_pct": 9.0, "footprint_km2": 9.0, "lcoe": 0.29},
    ]
    arr = figure_pareto(front)
    assert arr.shape == (620, 960, 3)
    assert figure_pareto([], ).shape == (620, 960, 3)


def test_figure_tornado() -> None:
    """龙卷风图。"""
    rows = sensitivity_lcoe(Economics(), 60000.0, 150.0)
    arr = figure_tornado(rows)
    assert arr.shape == (620, 960, 3)
    assert float(arr.min()) < 0.5
    assert figure_tornado([]).shape == arr.shape


def test_figure_convergence() -> None:
    """收敛曲线（含空历史）。"""
    hist = [{"iteration": float(i), "value": 100.0 - i, "best_value": 101.0 - i} for i in range(1, 11)]
    arr = figure_convergence(hist)
    assert arr.shape == (620, 960, 3)
    assert figure_convergence([]).shape == arr.shape


def test_figure_scheme_compare() -> None:
    """方案对比双面板柱状图。"""
    res = example_wind_resource()
    curve = example_curve()
    site = Site.rectangle(3600.0, 2400.0)
    econ = Economics()
    pos = grid_layout(site, 3, 4, 5 * curve.diameter, 3 * curve.diameter,
                      main_dir_deg=270.0, margin=100.0)
    results = [evaluate_scheme(Scheme(name="S%d" % i, curve=curve, layout=pos, site=site),
                               res, econ, bin_width=4.0) for i in range(3)]
    arr = figure_scheme_compare(results)
    assert arr.shape == (620, 960, 3)


# ---------------------------------------------------------------- CSV


def test_write_csv_and_rows(tmp_path) -> None:
    """CSV 写出（含中文表头）与内容检查。"""
    rows = [{"机组编号": 1, "X_东向_m": 100.0}, {"机组编号": 2, "X_东向_m": 200.0}]
    p = write_csv(str(tmp_path / "t.csv"), rows)
    assert Path(p).stat().st_size > 0
    text = Path(p).read_text(encoding="utf-8-sig")
    assert "机组编号" in text
    assert text.count("\n") == 3
    # 空表也能写出（0 字节文件）
    p2 = write_csv(str(tmp_path / "empty.csv"), [])
    assert Path(p2).exists()


def test_layout_csv_rows() -> None:
    pos = np.array([[0.0, 0.0], [500.0, 300.0]])
    rows = layout_csv_rows(pos, D, per_turbine_gwh=np.array([10.0, 9.0]))
    assert len(rows) == 2
    assert rows[0]["机组编号"] == 1
    assert "单机净发电量_GWh" in rows[0]
    assert rows[1]["Y_北向_m"] == 300.0


def test_sensitivity_and_pareto_and_history_csv_rows() -> None:
    """敏感性 / Pareto / 收敛历史的 CSV 行结构。"""
    srows = sensitivity_csv_rows(sensitivity_lcoe(Economics(), 60000.0, 150.0))
    assert len(srows) >= 4
    assert "LCOE_元每kWh_基准" in srows[0]
    prows = pareto_csv_rows([{"aep_gwh": 100.0, "wake_loss_pct": 5.0,
                              "footprint_km2": 3.0, "lcoe": 0.3}])
    assert prows[0]["解编号"] == 1
    hrows = history_csv_rows([{"iteration": 1.0, "value": 2.0}])
    assert hrows[0]["iteration"] == 1.0


# ---------------------------------------------------------------- ReportBundle 与 Markdown


@pytest.fixture()
def report_inputs():
    res = example_wind_resource()
    curve = example_curve()
    site = Site.rectangle(3600.0, 2400.0)
    econ = Economics()
    pos = grid_layout(site, 3, 4, 5 * curve.diameter, 3 * curve.diameter,
                      main_dir_deg=270.0, margin=100.0)
    model = WakeModel(diameter=curve.diameter, k_wake=0.075, rotor_grid=8)
    aep_res = evaluate_aep(pos, res, curve, model, bin_width=4.0)
    results = [
        evaluate_scheme(Scheme(name="A", curve=curve, layout=pos, site=site), res, econ, bin_width=4.0),
        evaluate_scheme(Scheme(name="B", curve=curve, layout=pos + 200.0, site=None), res, econ, bin_width=4.0),
    ]
    return res, curve, site, econ, pos, aep_res, results


def test_report_bundle_writes_all_files(tmp_path, report_inputs) -> None:
    """ReportBundle 能一次写出 PNG / CSV / Markdown，且全部非空。"""
    res, curve, site, econ, pos, aep_res, results = report_inputs
    bundle = ReportBundle(out_dir=str(tmp_path / "reports"))
    k, cc = fit_weibull_moments(sample_speeds(2.0, 8.0, n=20000, seed=2))
    centers, freq, _ = speed_histogram(sample_speeds(2.0, 8.0, n=20000, seed=2), bin_width=1.0)

    bundle.save_png("fig_freq.png", figure_wind_frequency(centers, freq, k, cc))
    bundle.save_png("fig_curve.png", figure_power_curves([curve]))
    bundle.save_png("fig_layout.png", figure_layout(site, pos, curve.diameter, rose=res.rose))
    bundle.save_png("fig_compare.png", figure_scheme_compare(results))
    bundle.save_csv("layout.csv", layout_csv_rows(pos, curve.diameter, aep_res.per_turbine_gwh))
    bundle.save_csv("sector.csv", aep_res.sector_table_rows())
    bundle.section("风资源", resource_block(res, k, cc))
    bundle.section("方案对比", scheme_block(results, baseline="A"))
    bundle.figure_section("风频图", bundle.files[0], "示例图注")
    path = bundle.write_markdown("report.md", "测试报告", intro="测试用报告")

    assert Path(path).exists()
    assert Path(path).stat().st_size > 0
    for f in bundle.files:
        assert Path(f).exists()
        assert Path(f).stat().st_size > 0, "空文件：%s" % f
    text = Path(path).read_text(encoding="utf-8")
    assert "# 测试报告" in text
    assert "风资源" in text
    assert "![示例图注]" in text
    assert "| 方案 |" in text
    manifest = bundle.manifest()
    assert "生成的报告产物" in manifest


def test_resource_block_content() -> None:
    """风资源表包含逐扇区行与来源说明。"""
    res = example_wind_resource()
    text = resource_block(res, 2.0, 8.4, fit_method="矩估计")
    assert "| 指标 | 数值 |" in text
    assert "W" in text
    assert text.count("|") > 60
    assert "合成数据" in text


def test_aep_and_sector_blocks(report_inputs) -> None:
    """发电量表与逐扇区表。"""
    res, curve, site, econ, pos, aep_res, results = report_inputs
    text = aep_block(aep_res, name="测试方案")
    assert "净发电量" in text
    assert "尾流损失" in text
    sec = sector_block(aep_res, top=3)
    assert sec.count("\n") == 4
    assert "扇区" in sec


def test_summary_text_and_markdown_table(report_inputs) -> None:
    res, curve, site, econ, pos, aep_res, results = report_inputs
    text = summary_text(results)
    assert "方案对比汇总" in text
    tbl = markdown_table([{"a": 1.5, "b": "x"}, {"a": 2, "b": "y"}])
    assert tbl.startswith("| a | b |")
