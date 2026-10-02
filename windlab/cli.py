"""命令行入口：``python -m windlab.cli <命令>``。

命令一览::

    resource    打印示例风资源与 Weibull 拟合自检结果
    aep         对一个示例排布算发电量（含/不含尾流对照）
    compare     多种机型 × 排布方案对比
    optimize    排布优化（随机搜索 / 贪心 / NSGA-II）
    demo        跑完整的端到端示例（等价于 python scripts/demo.py）

所有内置数据都是**示例/合成数据**，详见 README 的「能力边界」一节。
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import List, Optional

import numpy as np

from . import __version__
from .aep import evaluate_aep
from .compare import Scheme, compare_summary, evaluate_scheme
from .economics import Economics, economic_summary
from .layout import Site, check_layout, grid_layout, layout_summary, random_layout, staggered_layout
from .optimize import greedy_layout, nsga2_layout, random_search_layout
from .power_curve import builtin_curves, example_curve
from .resource import (
    N_SECTORS,
    example_wind_resource,
    fit_weibull_mle,
    fit_weibull_moments,
    sample_speeds,
    sector_name,
)
from .wake import WakeModel


def build_default_scenario(
    mean_speed: float = 7.5,
    k: float = 2.0,
    site_width: float = 3600.0,
    site_height: float = 2400.0,
    n_rows: int = 3,
    n_cols: int = 4,
):
    """构造默认的示例场景（合成数据）：示例场地 + 示例风玫瑰 + 示例机型 + 网格排布。"""
    resource = example_wind_resource(mean_speed=mean_speed, k=k)
    curve = example_curve()
    site = Site.rectangle(site_width, site_height, name="示例场地（合成数据）")
    main_dir = resource.dominant_direction_deg
    layout = grid_layout(site, n_rows, n_cols, 5.0 * curve.diameter, 3.0 * curve.diameter,
                         main_dir_deg=main_dir, margin=150.0)
    return resource, curve, site, layout


def cmd_resource(args: argparse.Namespace) -> int:
    """打印示例风资源与 Weibull 拟合能力自检。"""
    res = example_wind_resource(mean_speed=args.mean_speed)
    print(res.summary())
    print()
    print("逐扇区风玫瑰（示例数据）：")
    print("  %-4s %-8s %-10s %-12s %s" % ("扇区", "风向(°)", "频率(%)", "平均风速", "风功率密度(W/m²)"))
    for i in range(N_SECTORS):
        print("  %-4s %8.1f %10.3f %12.3f %14.1f" % (
            sector_name(i), i * 22.5, 100.0 * res.rose.freqs[i],
            res.rose.mean_speeds[i], res.sector_power_density(i)))

    print()
    print("Weibull 参数拟合自检（用合成样本检验能否还原真值）：")
    for k_true, c_true in ((1.8, 6.5), (2.0, 8.0), (2.6, 10.0)):
        s = sample_speeds(k_true, c_true, n=200000, seed=42)
        km, cm = fit_weibull_moments(s)
        kl, cl = fit_weibull_mle(s)
        print("  真值 k=%.2f c=%.2f -> 矩估计 k=%.3f c=%.3f（误差 %+.2f%% / %+.2f%%）"
              "  MLE k=%.3f c=%.3f（误差 %+.2f%% / %+.2f%%）"
              % (k_true, c_true, km, cm, 100 * (km - k_true) / k_true, 100 * (cm - c_true) / c_true,
                 kl, cl, 100 * (kl - k_true) / k_true, 100 * (cl - c_true) / c_true))
    return 0


def cmd_aep(args: argparse.Namespace) -> int:
    """对示例排布计算发电量，并给出无尾流对照。"""
    resource, curve, site, layout = build_default_scenario(mean_speed=args.mean_speed)
    model = WakeModel(diameter=curve.diameter, k_wake=args.wake_decay,
                      rotor_grid=args.rotor_grid)
    print(resource.summary())
    print(curve.summary())
    print(layout_summary(layout, curve.diameter))
    print()
    t0 = time.time()
    with_wake = evaluate_aep(layout, resource, curve, model, bin_width=args.bin_width,
                             availability=args.availability,
                             other_loss_factor=args.other_loss)
    no_wake = evaluate_aep(layout, resource, curve, None, include_wake=False,
                           bin_width=args.bin_width, availability=args.availability,
                           other_loss_factor=args.other_loss)
    print("含尾流：")
    print(with_wake.summary())
    print()
    print("无尾流对照：净发电量 %.2f GWh/年，净容量系数 %.2f%%，尾流损失 %.2f%%"
          % (no_wake.net_gwh, 100.0 * no_wake.net_capacity_factor, with_wake.wake_loss_pct))
    print("计算耗时 %.2f s（分仓 %.1f m/s，风轮离散 %d×%d）"
          % (time.time() - t0, args.bin_width, args.rotor_grid, args.rotor_grid))
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """多机型 × 多排布方案对比。"""
    resource, _, site, layout_grid = build_default_scenario(mean_speed=args.mean_speed)
    econ = Economics(capex_per_kw=args.capex, opex_per_kw_year=args.opex,
                     discount_rate=args.discount, electricity_price=args.price)
    rng = np.random.default_rng(args.seed)
    main_dir = resource.dominant_direction_deg

    schemes: List[Scheme] = []
    for curve in builtin_curves():
        g = grid_layout(site, 3, 4, 5.0 * curve.diameter, 3.0 * curve.diameter,
                        main_dir_deg=main_dir, margin=150.0)
        schemes.append(Scheme(name="%s-网格" % curve.name, curve=curve, layout=g, site=site,
                              description="规则网格 3×4"))
        try:
            st = staggered_layout(site, 3, 4, 5.0 * curve.diameter, 3.0 * curve.diameter,
                                  main_dir_deg=main_dir, stagger=0.5)
            if st.shape[0] >= 8:
                schemes.append(Scheme(name="%s-错列" % curve.name, curve=curve, layout=st,
                                      site=site, description="相邻迎风排横风向错开 0.5×3D"))
        except Exception as exc:      # pragma: no cover - 仅防御
            print("错列排布生成失败：%s" % exc, file=sys.stderr)
        try:
            rd = random_layout(site, 12, curve.diameter, np.random.default_rng(args.seed),
                               main_dir_deg=main_dir)
            schemes.append(Scheme(name="%s-随机" % curve.name, curve=curve, layout=rd, site=site,
                                  description="随机撒点（拒绝采样）"))
        except RuntimeError as exc:
            print("随机排布生成失败：%s" % exc, file=sys.stderr)

    results = []
    t0 = time.time()
    for s in schemes:
        results.append(evaluate_scheme(s, resource, econ, bin_width=args.bin_width))
    print(resource.summary())
    print()
    print(compare_summary(results))
    print()
    print("耗时 %.2f s" % (time.time() - t0))
    return 0


def cmd_optimize(args: argparse.Namespace) -> int:
    """排布优化：随机搜索 + 贪心 + NSGA-II。"""
    resource, curve, site, layout_grid = build_default_scenario(mean_speed=args.mean_speed)
    n = args.n_turbines
    print(resource.summary())
    print(curve.summary())
    print("场址 %s，机位数 %d，主风向 %.1f°"
          % (site.name, n, resource.dominant_direction_deg))
    print()

    t0 = time.time()
    rs = random_search_layout(site, n, resource, curve, iterations=args.iterations,
                              seed=args.seed, bin_width=args.bin_width, rotor_grid=args.rotor_grid)
    print(rs.summary())
    gr = greedy_layout(site, n, resource, curve, candidates_per_step=args.candidates,
                       seed=args.seed, bin_width=args.bin_width, rotor_grid=args.rotor_grid)
    print(gr.summary())
    pa = nsga2_layout(site, n, resource, curve, population_size=args.population,
                      generations=args.generations, seed=args.seed,
                      bin_width=args.bin_width, rotor_grid=args.rotor_grid)
    print(pa.summary())
    print("总耗时 %.1f s" % (time.time() - t0))
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """运行完整的端到端示例脚本。"""
    import runpy
    from pathlib import Path
    script = Path(__file__).resolve().parent.parent / "scripts" / "demo.py"
    if not script.exists():
        print("找不到 scripts/demo.py：%s" % script, file=sys.stderr)
        return 2
    runpy.run_path(str(script), run_name="__main__")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="windlab",
        description="风电场方案设计与发电量评估工具（示例数据，非实测数据）",
    )
    p.add_argument("--version", action="version", version="windlab %s" % __version__)
    sub = p.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--mean-speed", type=float, default=7.5, help="场地年平均风速 (m/s)")
    common.add_argument("--bin-width", type=float, default=1.0, help="风速分仓宽度 (m/s)")
    common.add_argument("--seed", type=int, default=0, help="随机种子")

    p_res = sub.add_parser("resource", parents=[common], help="打印示例风资源与拟合自检")
    p_res.set_defaults(func=cmd_resource)

    p_aep = sub.add_parser("aep", parents=[common], help="算一个示例排布的发电量")
    p_aep.add_argument("--wake-decay", type=float, default=0.075, help="尾流衰减系数 k")
    p_aep.add_argument("--rotor-grid", type=int, default=20, help="风轮离散网格")
    p_aep.add_argument("--availability", type=float, default=0.95, help="可用率")
    p_aep.add_argument("--other-loss", type=float, default=0.90, help="其他综合折减系数")
    p_aep.set_defaults(func=cmd_aep)

    p_cmp = sub.add_parser("compare", parents=[common], help="多方案对比")
    p_cmp.add_argument("--capex", type=float, default=5800.0, help="单位千瓦造价 (元/kW)")
    p_cmp.add_argument("--opex", type=float, default=200.0, help="年运营成本 (元/kW/年)")
    p_cmp.add_argument("--discount", type=float, default=0.08, help="折现率")
    p_cmp.add_argument("--price", type=float, default=0.35, help="上网电价 (元/kWh)")
    p_cmp.set_defaults(func=cmd_compare)

    p_opt = sub.add_parser("optimize", parents=[common], help="排布优化")
    p_opt.add_argument("--n-turbines", type=int, default=12, help="机位数")
    p_opt.add_argument("--iterations", type=int, default=60, help="随机搜索次数")
    p_opt.add_argument("--candidates", type=int, default=40, help="贪心每步候选位置数")
    p_opt.add_argument("--population", type=int, default=20, help="NSGA-II 种群规模")
    p_opt.add_argument("--generations", type=int, default=8, help="NSGA-II 代数")
    p_opt.add_argument("--rotor-grid", type=int, default=8, help="风轮离散网格（优化用粗网格提速）")
    p_opt.set_defaults(func=cmd_optimize)

    p_demo = sub.add_parser("demo", help="跑完整端到端示例")
    p_demo.set_defaults(func=cmd_demo)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
