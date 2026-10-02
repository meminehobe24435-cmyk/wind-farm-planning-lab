"""README 里的快速开始示例是否真的能跑（自检脚本）。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from windlab import example_wind_resource, example_curve, WakeModel, evaluate_aep
from windlab.layout import Site, grid_layout
from windlab.economics import Economics, lcoe

resource = example_wind_resource()          # 示例风资源（合成数据）
curve = example_curve()                     # 示例功率曲线（构造数据）
site = Site.rectangle(3600.0, 2400.0)       # 3.6 km × 2.4 km 场址
layout = grid_layout(site, 3, 4, 5 * curve.diameter, 3 * curve.diameter,
                     main_dir_deg=resource.dominant_direction_deg, margin=150.0)

wake = WakeModel(diameter=curve.diameter, k_wake=0.075, rotor_grid=20)
result = evaluate_aep(layout, resource, curve, wake,
                      availability=0.95, other_loss_factor=0.90, bin_width=1.0)
print(result.summary())

econ = Economics(capex_per_kw=5800.0, opex_per_kw_year=200.0,
                 discount_rate=0.08, lifetime_years=20, electricity_price=0.35)
print("LCOE = %.4f 元/kWh" % lcoe(econ, result.capacity_mw * 1000, result.net_gwh))
print()
print("README 示例运行正常。")
