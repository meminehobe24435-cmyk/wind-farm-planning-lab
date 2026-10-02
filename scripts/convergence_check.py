"""数值收敛性自检（会把结果写进 docs/results.md）。

验证三件事对最终结论的影响：
1. 风速分仓宽度（能量等效代表风速）对 AEP 的影响；
2. 风轮离散网格（部分遮挡精度）对 AEP / 尾流损失的影响；
3. 优化用的快速评估参数与最终精算参数的偏差。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from windlab.aep import evaluate_aep            # noqa: E402
from windlab.layout import Site, grid_layout     # noqa: E402
from windlab.power_curve import example_curve    # noqa: E402
from windlab.resource import example_wind_resource  # noqa: E402
from windlab.wake import WakeModel               # noqa: E402

res = example_wind_resource()
curve = example_curve()
site = Site.rectangle(3600.0, 2400.0)
pos = grid_layout(site, 3, 4, 5 * curve.diameter, 3 * curve.diameter,
                  main_dir_deg=res.dominant_direction_deg, margin=150.0)

print("=" * 70)
print("风速分仓宽度收敛性（风轮离散固定 20x20）")
print("=" * 70)
ref = None
for bw in (0.25, 0.5, 1.0, 2.0, 4.0, 5.0):
    m = WakeModel(diameter=curve.diameter, k_wake=0.075, rotor_grid=20)
    t0 = time.time()
    r = evaluate_aep(pos, res, curve, m, bin_width=bw)
    if bw == 0.25:
        ref = r
    dev = 100.0 * (r.net_gwh - ref.net_gwh) / ref.net_gwh
    print("  分仓 %.2f m/s: 净AEP %.3f GWh  尾流损失 %.4f%%  与 0.25 偏差 %+.4f%%  耗时 %.2f s"
          % (bw, r.net_gwh, r.wake_loss_pct, dev, time.time() - t0))

print()
print("=" * 70)
print("风轮离散网格收敛性（分仓固定 1.0 m/s）")
print("=" * 70)
ref2 = None
for g in (6, 8, 12, 16, 20, 28):
    m = WakeModel(diameter=curve.diameter, k_wake=0.075, rotor_grid=g)
    t0 = time.time()
    r = evaluate_aep(pos, res, curve, m, bin_width=1.0)
    if g == 28:
        pass
    if ref2 is None:
        pass
    dev = 100.0 * (r.net_gwh - 174.70) / 174.70
    print("  网格 %2dx%-2d (%3d 点): 净AEP %.3f GWh  尾流损失 %.4f%%  耗时 %.2f s"
          % (g, g, m.rotor_point_count(), r.net_gwh, r.wake_loss_pct, time.time() - t0))

print()
print("=" * 70)
print("快速评估参数（分仓 4.0 / 网格 8x8）与精算参数（分仓 1.0 / 网格 20x20）的偏差")
print("=" * 70)
fast = WakeModel(diameter=curve.diameter, k_wake=0.075, rotor_grid=8)
fine = WakeModel(diameter=curve.diameter, k_wake=0.075, rotor_grid=20)
a = evaluate_aep(pos, res, curve, fast, bin_width=4.0)
b = evaluate_aep(pos, res, curve, fine, bin_width=1.0)
print("  快速：净AEP %.4f GWh  尾流损失 %.4f%%" % (a.net_gwh, a.wake_loss_pct))
print("  精算：净AEP %.4f GWh  尾流损失 %.4f%%" % (b.net_gwh, b.wake_loss_pct))
print("  偏差：净AEP %+.4f%%  尾流损失 %+.4f 个百分点"
      % (100 * (a.net_gwh - b.net_gwh) / b.net_gwh, a.wake_loss_pct - b.wake_loss_pct))
