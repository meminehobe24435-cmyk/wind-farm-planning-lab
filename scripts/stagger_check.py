"""错列 vs 网格 的逐扇区尾流损失对照（结果写入 docs/results.md 第 6 节）。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from windlab.aep import evaluate_aep                       # noqa: E402
from windlab.layout import Site, grid_layout, staggered_layout  # noqa: E402
from windlab.power_curve import example_curve               # noqa: E402
from windlab.resource import example_wind_resource          # noqa: E402
from windlab.wake import WakeModel                          # noqa: E402

res = example_wind_resource()
curve = example_curve()
site = Site.rectangle(3600.0, 2400.0)
D = curve.diameter
main_dir = res.dominant_direction_deg
grid = grid_layout(site, 3, 4, 5 * D, 3 * D, main_dir_deg=main_dir, margin=150.0)
stag = staggered_layout(site, 3, 4, 5 * D, 3 * D, main_dir_deg=main_dir, stagger=0.5)
model = WakeModel(diameter=D, k_wake=0.075, rotor_grid=20)

a = evaluate_aep(grid, res, curve, model, bin_width=1.0)
b = evaluate_aep(stag, res, curve, model, bin_width=1.0)

print("%-6s %8s %12s %12s %10s" % ("扇区", "频率(%)", "网格损失(%)", "错列损失(%)", "差值"))
for i in range(16):
    fa = a.per_sector[i]
    fb = b.per_sector[i]
    print("%-6s %8.2f %12.2f %12.2f %+10.2f"
          % (fa["name"], 100 * fa["freq"], fa["wake_loss_pct"], fb["wake_loss_pct"],
             fb["wake_loss_pct"] - fa["wake_loss_pct"]))
print()
print("全场尾流损失：网格 %.4f%%  错列 %.4f%%  差 %+.4f 个百分点"
      % (a.wake_loss_pct, b.wake_loss_pct, b.wake_loss_pct - a.wake_loss_pct))
print("净发电量：网格 %.3f GWh  错列 %.3f GWh  差 %+.3f GWh"
      % (a.net_gwh, b.net_gwh, b.net_gwh - a.net_gwh))
