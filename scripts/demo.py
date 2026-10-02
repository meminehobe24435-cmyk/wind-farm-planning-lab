"""端到端示例：从风资源到方案对比、排布优化、经济性与报告输出。

运行方式::

    python scripts/demo.py

⚠️ 诚实声明（重要）
    本脚本使用的**全部输入数据都是示例/合成数据**：
    - 风资源：由 :func:`windlab.resource.example_wind_rose` 按双峰 von Mises 分布构造，
      不是任何实测测风塔数据；
    - 功率曲线：由 :func:`windlab.power_curve.synthetic_curve` 按「恒定 Cp + 额定限功率」
      构造，**不是任何厂商的实测功率曲线**；
    - 经济性参数：陆上风电的典型取值范围，不是任何实际项目的报价或中标电价。
    所有**输出数字**都是这套输入在这个代码里跑出来的真实结果，可复现、可核对。
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from windlab import __version__                                    # noqa: E402
from windlab.aep import directional_wake_loss, evaluate_aep        # noqa: E402
from windlab.compare import (                                      # noqa: E402
    Scheme,
    SchemeResult,
    compare_summary,
    comparison_table,
    competitive_assessment,
    evaluate_scheme,
    markdown_table,
    rank_schemes,
)
from windlab.economics import (                                    # noqa: E402
    Economics,
    economic_summary,
    sensitivity_lcoe,
    sensitivity_npv,
    simple_metrics,
)
from windlab.layout import (                                       # noqa: E402
    Site,
    check_layout,
    grid_layout,
    footprint_area,
    random_layout,
    staggered_layout,
)
from windlab.optimize import (                                     # noqa: E402
    FAST_BIN_WIDTH,
    FAST_ROTOR_GRID,
    LayoutEvaluator,
    best_layout_from_pareto,
    greedy_layout,
    nsga2_layout,
    random_search_layout,
)
from windlab.power_curve import PowerCurve, builtin_curves, example_curve, save_curve_csv  # noqa: E402
from windlab.report import (                                       # noqa: E402
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
    pareto_csv_rows,
    resource_block,
    scheme_block,
    sector_block,
    sensitivity_csv_rows,
    summary_text,
)
from windlab.resource import (                                     # noqa: E402
    N_SECTORS,
    fit_weibull_mle,
    fit_weibull_moments,
    sample_speeds,
    sector_name,
    speed_histogram,
)
from windlab.wake import WakeModel                                 # noqa: E402

# ---------------------------------------------------------------- 场景参数

SITE_WIDTH = 3600.0          # m（东西）
SITE_HEIGHT = 2400.0         # m（南北）
N_TURBINES = 12
MEAN_SPEED = 7.5             # m/s（合成风玫瑰的场地平均风速）
WEIBULL_K = 2.0
AIR_DENSITY = 1.225
AVAILABILITY = 0.95          # 可用率
OTHER_LOSS = 0.90            # 其他综合折减系数
WAKE_DECAY = 0.075           # 陆上 Jensen 尾流衰减系数
ROTOR_GRID = 20              # 最终评估用的风轮离散网格
BIN_WIDTH = 1.0              # 最终评估用的风速分仓宽度
SEED = 20240501

#: 经济性假设（**典型取值范围，非任何实际项目报价**）
ECON = Economics(capex_per_kw=5800.0, opex_per_kw_year=200.0, discount_rate=0.08,
                 lifetime_years=20, electricity_price=0.35, annual_degradation=0.005)


def banner(title: str, char: str = "=") -> None:
    print()
    print(char * 78)
    print(title)
    print(char * 78)


def main() -> int:
    t_start = time.time()
    out_dir = ROOT / "reports"
    bundle = ReportBundle(out_dir=str(out_dir))

    banner("wind-farm-planning-lab 端到端示例（示例数据 / 可复现）")
    print("windlab 版本：%s" % __version__)
    print("Python %s | numpy %s" % (sys.version.split()[0], np.__version__))
    print("输出目录：%s" % out_dir)
    print()
    print("⚠️ 输入数据全部为示例/合成数据（非实测测风塔、非厂商功率曲线、非实际报价）；")
    print("   所有输出数字均由本代码实际计算得到，可用 pytest 复现。")

    # ============================================================ 1. 风资源
    from windlab.resource import example_wind_resource
    resource = example_wind_resource(mean_speed=MEAN_SPEED, k=WEIBULL_K,
                                     air_density=AIR_DENSITY)
    banner("1. 风资源（示例合成数据）")
    print(resource.summary())
    print()
    print("主风向扇区：%s（%.1f°），风向集中度 %.3f"
          % (resource.rose.dominant_sector_name, resource.dominant_direction_deg,
             resource.rose.directional_spread))
    print("风矢量合成方向：%.1f°" % resource.rose.resultant_direction_deg())
    print()
    print("逐扇区明细（前 6 个高频率扇区）：")
    order = np.argsort(-resource.rose.freqs)
    print("  %-5s %8s %10s %12s %14s" % ("扇区", "风向(°)", "频率(%)", "平均风速", "风功率密度"))
    print("  %-5s %8s %10s %12s %14s" % ("", "", "", "(m/s)", "(W/m²)"))
    for i in order[:6]:
        print("  %-5s %8.1f %10.2f %12.2f %14.1f"
              % (sector_name(int(i)), i * 22.5, 100.0 * resource.rose.freqs[i],
                 resource.rose.mean_speeds[i], resource.sector_power_density(int(i))))

    # Weibull 拟合自检（用合成样本验证能否还原真值）
    print()
    print("Weibull 参数拟合自检（合成样本，真值 k=%.1f c=%.2f）：" % (WEIBULL_K, 8.0))
    samples = sample_speeds(WEIBULL_K, 8.0, n=200000, seed=SEED)
    k_mom, c_mom = fit_weibull_moments(samples)
    k_mle, c_mle = fit_weibull_mle(samples)
    print("  矩估计  k=%.4f（误差 %+.2f%%）  c=%.4f（误差 %+.2f%%）"
          % (k_mom, 100 * (k_mom - WEIBULL_K) / WEIBULL_K, c_mom, 100 * (c_mom - 8.0) / 8.0))
    print("  MLE     k=%.4f（误差 %+.2f%%）  c=%.4f（误差 %+.2f%%）"
          % (k_mle, 100 * (k_mle - WEIBULL_K) / WEIBULL_K, c_mle, 100 * (c_mle - 8.0) / 8.0))

    hist_centers, hist_freq, _ = speed_histogram(samples, bin_width=1.0, v_max=26.0)

    # ============================================================ 2. 机型
    banner("2. 机型功率曲线（示例曲线，非厂商实测数据）")
    curves = builtin_curves()
    for c in curves:
        print(c.summary())
        print("    毛容量系数（k=%.1f, c=%.1f m/s）：%.4f"
              % (WEIBULL_K, resource.equivalent_c, c.capacity_factor(WEIBULL_K, resource.equivalent_c)))
    curve_main = example_curve()
    save_curve_csv(curve_main, bundle.path("curve_SYN-5.0MW-132.csv"))

    # ============================================================ 3. 场址与排布
    banner("3. 场址与机位排布")
    site = Site.rectangle(SITE_WIDTH, SITE_HEIGHT, name="示例场地（合成数据）")
    main_dir = resource.dominant_direction_deg
    D = curve_main.diameter
    print("场址：%s，面积 %.3f km²，形心 (%.0f, %.0f)，主风向 %.1f°"
          % (site.name, site.area_km2, site.centroid[0], site.centroid[1], main_dir))
    print("机型：%s，风轮直径 %.1f m，间距约束 主风向 >= 5D = %.0f m，横风向 >= 3D = %.0f m"
          % (curve_main.name, D, 5 * D, 3 * D))

    layout_grid = grid_layout(site, 3, 4, 5.0 * D, 3.0 * D,
                              main_dir_deg=main_dir, margin=150.0)
    layout_stagger = staggered_layout(site, 3, 4, 5.0 * D, 3.0 * D,
                                      main_dir_deg=main_dir, stagger=0.5)
    layout_random = random_layout(site, N_TURBINES, D, np.random.default_rng(SEED),
                                  main_dir_deg=main_dir)

    for name, pos in (("规则网格 3×4", layout_grid),
                      ("错列 3×4（奇数位机组横向偏移 0.5×3D）", layout_stagger),
                      ("随机撒点（拒绝采样）", layout_random)):
        rep = check_layout(site, pos, D, main_dir_deg=main_dir)
        print()
        print("  %s：%d 台" % (name, pos.shape[0]))
        print("    %s" % rep.summary())

    # ============================================================ 4. 发电量（基准方案）
    banner("4. 发电量评估（规则网格基准方案，逐扇区加权 + Jensen 尾流）")
    wake_model = WakeModel(diameter=D, k_wake=WAKE_DECAY, rotor_grid=ROTOR_GRID)
    t0 = time.time()
    aep_grid = evaluate_aep(layout_grid, resource, curve_main, wake_model,
                            availability=AVAILABILITY, other_loss_factor=OTHER_LOSS,
                            bin_width=BIN_WIDTH)
    aep_nowake = evaluate_aep(layout_grid, resource, curve_main, None, include_wake=False,
                              availability=AVAILABILITY, other_loss_factor=OTHER_LOSS,
                              bin_width=BIN_WIDTH)
    print("计算耗时 %.2f s（%d 扇区 × 风速分仓 %.1f m/s × 风轮离散 %d×%d）"
          % (time.time() - t0, N_SECTORS, BIN_WIDTH, ROTOR_GRID, ROTOR_GRID))
    print()
    print(aep_grid.summary())
    print()
    print("无尾流对照：净发电量 %.2f GWh/年，净容量系数 %.2f%%"
          % (aep_nowake.net_gwh, 100.0 * aep_nowake.net_capacity_factor))
    print("单机净发电量（GWh/年，按机位编号）：")
    print("  " + "  ".join("%.2f" % v for v in aep_grid.per_turbine_gwh))

    # ============================================================ 5. 风向影响
    banner("5. 风向对尾流的影响（同一排布）")
    dir_rows = directional_wake_loss(layout_grid, curve_main, wake_model, 8.0, step_deg=22.5)
    print("  %-10s %-14s %-14s" % ("风向(°)", "速度亏损(%)", "功率损失(%)"))
    for r in dir_rows:
        print("  %-10.1f %-14.2f %-14.2f"
              % (r["direction_deg"], r["speed_deficit_pct"], r["power_loss_pct"]))
    worst = max(dir_rows, key=lambda r: r["power_loss_pct"])
    best = min(dir_rows, key=lambda r: r["power_loss_pct"])
    print("尾流最严重风向：%.0f°（功率损失 %.2f%%）；最轻风向：%.0f°（%.2f%%）"
          % (worst["direction_deg"], worst["power_loss_pct"],
             best["direction_deg"], best["power_loss_pct"]))
    print("说明：主风向 %.0f° 与机位连线方向一致，尾流叠加最严重——这正是需要优化排布的原因。"
          % main_dir)

    # ============================================================ 6. 排布优化
    banner("6. 排布优化（随机搜索 / 贪心加点 / 简化 NSGA-II）")
    print("优化用快速评估参数：风速分仓 %.1f m/s、风轮离散 %d×%d（最终结果重新用 %.1f / %d×%d 精算）"
          % (FAST_BIN_WIDTH, FAST_ROTOR_GRID, FAST_ROTOR_GRID, BIN_WIDTH, ROTOR_GRID, ROTOR_GRID))
    evaluator = LayoutEvaluator(resource, curve_main, wake_decay=WAKE_DECAY,
                                bin_width=FAST_BIN_WIDTH, rotor_grid=FAST_ROTOR_GRID,
                                econ=ECON)
    t0 = time.time()
    opt_random = random_search_layout(site, N_TURBINES, resource, curve_main,
                                      iterations=80, seed=SEED, evaluator=evaluator)
    print()
    print(opt_random.summary())
    opt_greedy = greedy_layout(site, N_TURBINES, resource, curve_main,
                               candidates_per_step=50, seed=SEED, evaluator=evaluator)
    print(opt_greedy.summary())
    pareto = nsga2_layout(site, N_TURBINES, resource, curve_main, population_size=24,
                          generations=12, seed=SEED, objectives=("aep", "footprint"),
                          evaluator=evaluator)
    print(pareto.summary())
    knee = best_layout_from_pareto(pareto, "knee")
    print("Pareto 折中解（离理想点最近）：AEP %.3f GWh，占地 %.3f km²"
          % (knee["aep_gwh"], knee["footprint_km2"]))
    print("优化总耗时 %.1f s，累计评估 %d 次排布" % (time.time() - t0, evaluator.count))
    print()
    print("贪心加点收敛过程（快速评估口径）：")
    print("  %-6s %-12s %-14s" % ("台数", "整场AEP(GWh)", "边际增益(GWh)"))
    for h in opt_greedy.history:
        print("  %-6d %-12.3f %-14.3f" % (int(h["n_turbines"]), h["value"], h["marginal"]))

    # ============================================================ 7. 方案对比
    banner("7. 方案对比（多机型 × 多排布）")
    econ = ECON
    print("经济性假设（典型值，非实际报价）：CAPEX %.0f 元/kW，OPEX %.0f 元/kW/年，"
          "折现率 %.0f%%，寿命 %d 年，电价 %.3f 元/kWh，电量年衰减 %.1f%%"
          % (econ.capex_per_kw, econ.opex_per_kw_year, 100 * econ.discount_rate,
             econ.lifetime_years, econ.electricity_price, 100 * econ.annual_degradation))

    schemes: List[Scheme] = []
    schemes.append(Scheme(name="A-网格", curve=curve_main, layout=layout_grid, site=site,
                          description="5.0MW/132m 规则网格 3×4（基线）"))
    schemes.append(Scheme(name="B-错列", curve=curve_main, layout=layout_stagger, site=site,
                          description="5.0MW/132m 错列排布（相邻迎风排横风向错开 0.5×3D）"))
    schemes.append(Scheme(name="C-随机", curve=curve_main, layout=layout_random, site=site,
                          description="5.0MW/132m 随机撒点"))
    schemes.append(Scheme(name="D-贪心优化", curve=curve_main, layout=opt_greedy.positions,
                          site=site, description="5.0MW/132m 贪心加点优化"))
    schemes.append(Scheme(name="E-随机搜索优化", curve=curve_main, layout=opt_random.positions,
                          site=site, description="5.0MW/132m 随机搜索优化"))
    schemes.append(Scheme(name="F-NSGA-II折中", curve=curve_main, layout=knee["positions"],
                          site=site, description="5.0MW/132m NSGA-II 折中解"))
    for c in curves:
        if c.name == curve_main.name:
            continue
        pos = grid_layout(site, 3, 4, 5.0 * c.diameter, 3.0 * c.diameter,
                          main_dir_deg=main_dir, margin=150.0)
        schemes.append(Scheme(name="G-%s网格" % c.name.split("-")[1], curve=c, layout=pos,
                              site=site, description="%s 规则网格 3×4" % c.name))

    results: List[SchemeResult] = []
    t0 = time.time()
    for s in schemes:
        results.append(evaluate_scheme(s, resource, econ, bin_width=BIN_WIDTH,
                                       main_dir_deg=main_dir, wake_model=None))
    print("全部方案精算耗时 %.1f s（分仓 %.1f m/s，风轮离散 %d×%d）"
          % (time.time() - t0, BIN_WIDTH, ROTOR_GRID, ROTOR_GRID))
    print()
    print(compare_summary(results, baseline="A-网格"))

    print()
    print("方案对比表：")
    print(markdown_table(comparison_table(results)))
    print()
    print("多目标排序（AEP 最大 + LCOE 最小：非支配分层 + 拥挤度）：")
    print(markdown_table(rank_schemes(results)))

    # ============================================================ 8. 经济性与敏感性
    banner("8. 经济性与敏感性分析")
    baseline = results[0]
    capacity_kw = baseline.capacity_mw * 1000.0
    print(economic_summary(econ, capacity_kw, baseline.aep_gwh))
    print()
    print("LCOE 单因素敏感性（龙卷风图数据）：")
    sens_lcoe = sensitivity_lcoe(econ, capacity_kw, baseline.aep_gwh)
    print("  %-18s %12s %12s %12s %12s" % ("因素", "低值", "基准", "高值", "跨度"))
    for r in sens_lcoe:
        print("  %-18s %12.4f %12.4f %12.4f %12.4f"
              % (r["factor"], r["low_value"], r["base"], r["high_value"], r["span"]))
    print("  注：上网电价不进入 LCOE（成本口径指标），其影响体现在 NPV 敏感性中。")
    print()
    print("NPV 单因素敏感性（万元）：")
    sens_npv = sensitivity_npv(econ, capacity_kw, baseline.aep_gwh)
    print("  %-18s %14s %14s %14s" % ("因素", "低值", "基准", "高值"))
    for r in sens_npv:
        print("  %-18s %14.0f %14.0f %14.0f"
              % (r["factor"], r["low_value"] / 1e4, r["base"] / 1e4, r["high_value"] / 1e4))

    # ============================================================ 9. 输出产物
    banner("9. 生成报告产物（PNG / CSV / Markdown）")

    # -- 图 ------------------------------------------------------
    bundle.save_png("fig_wind_frequency.png",
                    figure_wind_frequency(hist_centers, hist_freq, k_mom, c_mom,
                                          bin_width=1.0,
                                          title="WIND SPEED FREQUENCY AND WEIBULL FIT (SYNTHETIC)",
                                          name="200000 SYNTHETIC SAMPLES"))
    bundle.save_png("fig_power_curves.png",
                    figure_power_curves(curves, highlight=[5.0, 8.0, 11.0]))
    bundle.save_png("fig_layout_baseline.png",
                    figure_layout(site, layout_grid, D, rose=resource.rose,
                                  main_dir_deg=main_dir,
                                  title="BASELINE LAYOUT: GRID 3X4",
                                  subtitle="12 TURBINES, 5D X 3D SPACING"))
    bundle.save_png("fig_layout_optimized.png",
                    figure_layout(site, opt_greedy.positions, D, rose=resource.rose,
                                  main_dir_deg=main_dir,
                                  title="OPTIMIZED LAYOUT: GREEDY",
                                  subtitle="12 TURBINES, CONSTRAINT-FEASIBLE"))
    bundle.save_png("fig_directional_wake.png",
                    figure_directional_wake(directional_wake_loss(layout_grid, curve_main,
                                                                  wake_model, 8.0, step_deg=10.0),
                                            dominant_deg=main_dir))
    bundle.save_png("fig_scheme_compare.png", figure_scheme_compare(results))
    bundle.save_png("fig_pareto.png", figure_pareto(pareto.front))
    bundle.save_png("fig_tornado_lcoe.png",
                    figure_tornado(sens_lcoe, title="LCOE SENSITIVITY (TORNADO)",
                                   xlabel="LCOE (YUAN/KWH)"))
    bundle.save_png("fig_tornado_npv.png",
                    figure_tornado([{"factor": r["factor"], "low_value": r["low_value"] / 1e4,
                                     "high_value": r["high_value"] / 1e4,
                                     "base": r["base"] / 1e4, "span": r["span"] / 1e4}
                                    for r in sens_npv],
                                   title="NPV SENSITIVITY (TORNADO)",
                                   xlabel="NPV (10K YUAN)", unit_fmt="%.0f"))
    bundle.save_png("fig_convergence.png",
                    figure_convergence(opt_random.history, x_key="iteration",
                                       title="RANDOM SEARCH CONVERGENCE"))

    # -- CSV -----------------------------------------------------
    bundle.save_csv("scheme_comparison.csv", comparison_table(results))
    bundle.save_csv("scheme_ranking.csv", rank_schemes(results))
    bundle.save_csv("competitive_assessment.csv", competitive_assessment(results, "A-网格"))
    bundle.save_csv("layout_baseline.csv",
                    layout_csv_rows(layout_grid, D, aep_grid.per_turbine_gwh))
    bundle.save_csv("layout_optimized_greedy.csv",
                    layout_csv_rows(opt_greedy.positions, D))
    bundle.save_csv("aep_by_sector_baseline.csv", aep_grid.sector_table_rows())
    bundle.save_csv("directional_wake.csv",
                    [{"风向_deg": r["direction_deg"], "速度亏损_%": round(r["speed_deficit_pct"], 4),
                      "功率损失_%": round(r["power_loss_pct"], 4)}
                     for r in directional_wake_loss(layout_grid, curve_main, wake_model, 8.0,
                                                    step_deg=10.0)])
    bundle.save_csv("sensitivity_lcoe.csv", sensitivity_csv_rows(sens_lcoe))
    bundle.save_csv("sensitivity_npv.csv", sensitivity_csv_rows(sens_npv, metric="NPV_元"))
    bundle.save_csv("pareto_front.csv", pareto_csv_rows(pareto.front))
    bundle.save_csv("convergence_greedy.csv", history_csv_rows(opt_greedy.history))
    bundle.save_csv("convergence_random_search.csv", history_csv_rows(opt_random.history))
    bundle.save_csv("convergence_nsga2.csv", history_csv_rows(pareto.history))

    # -- Markdown ------------------------------------------------
    bundle.section("1. 风资源（示例合成数据）",
                   resource_block(resource, k_mom, c_mom, fit_method="矩估计"), level=2)
    bundle.figure_section("1.1 风频分布与 Weibull 拟合", bundle.files[0],
                          "图 1：合成风速样本的直方图（200000 样本）与矩估计 Weibull 概率密度拟合。"
                          "拟合参数 k=%.4f、c=%.4f m/s，与样本真值 k=2.0、c=8.0 的误差分别为 "
                          "%+.2f%% 与 %+.2f%%。" % (k_mom, c_mom,
                                                    100 * (k_mom - 2.0) / 2.0,
                                                    100 * (c_mom - 8.0) / 8.0), level=3)

    bundle.section("2. 机型功率曲线", "\n".join(
        ["| 机型 | 额定功率(kW) | 风轮直径(m) | 扫风面积(m²) | 比功率(W/m²) | 切入/额定/切出(m/s) | 毛容量系数 |",
         "|---|---|---|---|---|---|---|"] +
        ["| %s | %.0f | %.1f | %.0f | %.1f | %.1f / %.2f / %.0f | %.4f |"
         % (c.name, c.rated_power, c.diameter, c.rotor_area, c.specific_power,
            c.cut_in, float(c.rated_speed), c.cut_out,
            c.capacity_factor(WEIBULL_K, resource.equivalent_c)) for c in curves]
        + ["", "⚠️ 三条曲线均为**按典型机组形态构造的示例曲线**（恒定 Cp + 额定限功率模型），"
               "**不是任何厂商的实测数据**；工具支持通过 `windlab.power_curve.load_curve_csv` "
               "导入真实功率曲线。"]))

    bundle.section("3. 机位排布与约束检查", "\n".join(
        ["| 排布 | 机位数 | 约束检查 | 最小间距(D) | 主风向最小间距(D) | 占地(km²) |",
         "|---|---|---|---|---|---|"] +
        ["| %s | %d | %s | %.2f | %.2f | %.3f |"
         % (name, pos.shape[0], "通过" if check_layout(site, pos, D, main_dir_deg=main_dir).ok
            else "不通过",
            check_layout(site, pos, D, main_dir_deg=main_dir).min_pair_distance_d,
            check_layout(site, pos, D, main_dir_deg=main_dir).min_main_separation_d,
            footprint_area(pos) / 1e6)
         for name, pos in (("规则网格 3×4", layout_grid), ("错列(0.5×3D 横向错开)", layout_stagger),
                           ("随机撒点", layout_random),
                           ("贪心优化", opt_greedy.positions),
                           ("随机搜索优化", opt_random.positions),
                           ("NSGA-II 折中解", knee["positions"]))]))

    bundle.section("4. 基准方案发电量", aep_block(aep_grid, name="A-网格（5.0MW/132m，规则网格 3×4）"))
    bundle.section("4.1 逐扇区发电量（频率最高的 6 个扇区）", sector_block(aep_grid, top=6), level=3)

    bundle.section("5. 方案对比", scheme_block(results, baseline="A-网格"))
    bundle.section("6. 多目标排序（非支配分层 + 拥挤度）",
                   markdown_table(rank_schemes(results)))
    bundle.section("7. 排布优化", "\n".join([
        "| 算法 | 机组数 | 净AEP(GWh) | 尾流损失(%) | 占地(km²) | 可行 | 评估次数 |",
        "|---|---|---|---|---|---|---|",
        "| %s | %d | %.3f | %.3f | %.3f | %s | %d |"
        % (opt_random.label, opt_random.positions.shape[0], opt_random.aep_gwh,
           opt_random.wake_loss_pct, opt_random.footprint_km2, opt_random.feasibility_ok,
           opt_random.evaluations),
        "| %s | %d | %.3f | %.3f | %.3f | %s | %d |"
        % (opt_greedy.label, opt_greedy.positions.shape[0], opt_greedy.aep_gwh,
           opt_greedy.wake_loss_pct, opt_greedy.footprint_km2, opt_greedy.feasibility_ok,
           opt_greedy.evaluations),
        "| NSGA-II（折中解） | %d | %.3f | %.3f | %.3f | 是 | %d |"
        % (knee["positions"].shape[0], knee["aep_gwh"], knee["wake_loss_pct"],
           knee["footprint_km2"], pareto.evaluations),
        "",
        "NSGA-II 目标对：(AEP 最大, 占地最小)，种群 %d，代数 %d，Pareto 前沿 %d 个非支配解。"
        % (pareto.population_size, pareto.generations, len(pareto.front)),
        "",
        "> 关于「最大化 AEP / 最小化尾流损失」这对目标：在**固定机位数与机型**时，"
        "毛发电量与机位无关，净发电量只是尾流损失的单调函数，两个目标完全相关、"
        "Pareto 前沿会退化为单点（见 docs/design.md 与 "
        "`tests/test_optimize.py::test_nsga2_aep_wake_loss_is_degenerate`）。"
        "因此实际优化改用真正互相冲突的「AEP 最大 + 占地最小」。",
    ]))

    bundle.section("8. 经济性", "\n".join([
        "| 参数 | 取值 |", "|---|---|",
        "| 单位千瓦造价 CAPEX | %.0f 元/kW |" % econ.capex_per_kw,
        "| 年运营成本 OPEX | %.0f 元/kW/年 |" % econ.opex_per_kw_year,
        "| 折现率 | %.1f%% |" % (100 * econ.discount_rate),
        "| 项目寿命 | %d 年 |" % econ.lifetime_years,
        "| 上网电价 | %.3f 元/kWh |" % econ.electricity_price,
        "| 电量年衰减 | %.2f%% |" % (100 * econ.annual_degradation),
        "",
        "基准方案（A-网格）财务指标：",
        "",
        "| 指标 | 数值 |", "|---|---|",
        "| 静态投资 | %.0f 万元 |" % (baseline.capex_yuan / 1e4),
        "| 年运营成本 | %.0f 万元/年 |" % (baseline.opex_year_yuan / 1e4),
        "| 年售电收入 | %.0f 万元 |" % (baseline.aep_gwh * 1e6 * econ.electricity_price / 1e4),
        "| LCOE | %.4f 元/kWh |" % baseline.lcoe,
        "| NPV | %.0f 万元 |" % (baseline.npv_yuan / 1e4),
        "| 投资回收期 | %.1f 年 |" % baseline.payback_years,
    ]))
    bundle.section("8.1 LCOE 敏感性与 NPV 敏感性", "\n".join([
        "LCOE 单因素敏感性（元/kWh）：", "",
        markdown_table([{"因素": r["factor"], "低值": round(r["low_value"], 4),
                         "基准": round(r["base"], 4), "高值": round(r["high_value"], 4),
                         "跨度": round(r["span"], 4)} for r in sens_lcoe]),
        "",
        "NPV 单因素敏感性（万元）：", "",
        markdown_table([{"因素": r["factor"], "低值": round(r["low_value"] / 1e4, 0),
                         "基准": round(r["base"] / 1e4, 0), "高值": round(r["high_value"] / 1e4, 0),
                         "跨度": round(r["span"] / 1e4, 0)} for r in sens_npv]),
        "",
        "> 注意：**上网电价不会改变 LCOE**（LCOE 是成本口径指标），"
        "因此 LCOE 敏感性表中不包含电价，电价的影响体现在 NPV/回收期上。",
    ]), level=3)
    bundle.section("9. 图表索引", "\n".join([
        "- 图 1 风频分布与 Weibull 拟合：`fig_wind_frequency.png`",
        "- 图 2 三条示例功率曲线：`fig_power_curves.png`",
        "- 图 3 基准排布（含风玫瑰与主风向）：`fig_layout_baseline.png`",
        "- 图 4 贪心优化排布：`fig_layout_optimized.png`",
        "- 图 5 尾流功率损失随风向变化：`fig_directional_wake.png`",
        "- 图 6 方案对比柱状图：`fig_scheme_compare.png`",
        "- 图 7 NSGA-II Pareto 前沿：`fig_pareto.png`",
        "- 图 8 LCOE 龙卷风图：`fig_tornado_lcoe.png`",
        "- 图 9 NPV 龙卷风图：`fig_tornado_npv.png`",
        "- 图 10 随机搜索收敛曲线：`fig_convergence.png`",
    ]))

    report_path = bundle.write_markdown(
        "report.md",
        "风电场方案设计与发电量评估报告（示例数据）",
        intro=(
            "本报告由 `python scripts/demo.py` 自动生成，全部数字来自本仓库代码的实际计算。\n\n"
            "**⚠️ 数据来源声明**：风资源为合成数据（双峰 von Mises 风玫瑰 + Weibull 分布）；"
            "功率曲线为按典型机组形态构造的示例曲线（非厂商实测）；经济性参数为典型取值范围"
            "（非实际项目报价）。本工具为自学性质的方案评估工具，**没有任何风电行业实际项目经验**。\n\n"
            "场景：%.1f km × %.1f km 示例场地，%d 台机组，场地平均风速 %.1f m/s，"
            "主风向 %.0f°，尾流模型 Jensen（k=%.3f），可用率 %.2f，其他综合折减 %.2f。"
            % (SITE_WIDTH / 1000.0, SITE_HEIGHT / 1000.0, N_TURBINES, MEAN_SPEED,
               main_dir, WAKE_DECAY, AVAILABILITY, OTHER_LOSS)
        ),
        footer=(
            "---\n\n生成时间：%s\n\n"
            "复现方式：`python -m pytest tests -q`（单元测试）与 `python scripts/demo.py`（本报告）。"
            % time.strftime("%Y-%m-%d %H:%M:%S")
        ),
    )

    # ============================================================ 10. 汇总
    banner("10. 产物清单")
    print(bundle.manifest())
    print()
    best = min(results, key=lambda r: r.lcoe)
    max_aep = max(results, key=lambda r: r.aep_gwh)
    print("结论摘要：")
    print("  LCOE 最低：%s（%.4f 元/kWh，AEP %.2f GWh/年，净容量系数 %.2f%%）"
          % (best.name, best.lcoe, best.aep_gwh, 100 * best.net_capacity_factor))
    print("  AEP 最大 ：%s（%.2f GWh/年，LCOE %.4f 元/kWh）"
          % (max_aep.name, max_aep.aep_gwh, max_aep.lcoe))
    gain_aep = 100.0 * (best.aep_gwh - results[0].aep_gwh) / results[0].aep_gwh
    gain_lcoe = 100.0 * (results[0].lcoe - best.lcoe) / results[0].lcoe
    print("  相对基线 A-网格：AEP %+.2f%%，LCOE %+.2f%%" % (gain_aep, -gain_lcoe))
    print()
    print("全部完成，总耗时 %.1f s。报告：%s" % (time.time() - t_start, report_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
