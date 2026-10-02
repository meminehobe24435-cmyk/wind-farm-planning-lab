"""wind-farm-planning-lab —— 风电场方案设计与发电量评估工具。

公开 API 一览（子模块可单独使用）::

    from windlab import example_wind_resource, example_curve, WakeModel, evaluate_aep
"""

from __future__ import annotations

__version__ = "0.1.0"

from .resource import (
    N_SECTORS,
    SECTOR_NAMES,
    DEFAULT_AIR_DENSITY,
    WindResource,
    WindRose,
    c_from_mean,
    example_wind_resource,
    example_wind_rose,
    fit_weibull_mle,
    fit_weibull_moments,
    sample_speeds,
    sector_direction_deg,
    sector_name,
    speed_histogram,
    weibull_cdf,
    weibull_mean,
    weibull_moment,
    weibull_pdf,
    weibull_std,
    wind_power_density,
)
from .power_curve import (
    PowerCurve,
    builtin_curves,
    example_curve,
    load_curve_csv,
    save_curve_csv,
    synthetic_curve,
)
from .wake import WakeModel, effective_speeds, loss_matrix, wind_unit_vector
from .layout import (
    LayoutReport,
    Site,
    check_layout,
    footprint_area,
    grid_layout,
    random_layout,
    repair_layout,
    staggered_layout,
)
from .aep import AEPResult, evaluate_aep, weibull_energy_bins
from .economics import (
    Economics,
    lcoe,
    npv,
    payback_period,
    sensitivity_lcoe,
    sensitivity_npv,
)
from .compare import (
    Scheme,
    SchemeResult,
    comparison_table,
    crowding_distance,
    evaluate_scheme,
    non_dominated_sort,
)
from .optimize import (
    OptimizeResult,
    greedy_layout,
    nsga2_layout,
    random_search_layout,
)

__all__ = [
    "__version__",
    "N_SECTORS", "SECTOR_NAMES", "DEFAULT_AIR_DENSITY",
    "WindResource", "WindRose", "c_from_mean", "example_wind_resource",
    "example_wind_rose", "fit_weibull_mle", "fit_weibull_moments", "sample_speeds",
    "sector_direction_deg", "sector_name", "speed_histogram", "weibull_cdf",
    "weibull_mean", "weibull_moment", "weibull_pdf", "weibull_std", "wind_power_density",
    "PowerCurve", "builtin_curves", "example_curve", "load_curve_csv",
    "save_curve_csv", "synthetic_curve",
    "WakeModel", "effective_speeds", "loss_matrix", "wind_unit_vector",
    "LayoutReport", "Site", "check_layout", "footprint_area", "grid_layout",
    "random_layout", "repair_layout", "staggered_layout",
    "AEPResult", "evaluate_aep", "weibull_energy_bins",
    "Economics", "lcoe", "npv", "payback_period", "sensitivity_lcoe", "sensitivity_npv",
    "Scheme", "SchemeResult", "comparison_table", "crowding_distance",
    "evaluate_scheme", "non_dominated_sort",
    "OptimizeResult", "greedy_layout", "nsga2_layout", "random_search_layout",
]
