"""风资源模块测试：Weibull 分布的解析性质、参数拟合可还原性、风频与风玫瑰。"""

from __future__ import annotations

import math

import numpy as np
import pytest

from windlab.resource import (
    DEFAULT_AIR_DENSITY,
    N_SECTORS,
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


def trapz(y: np.ndarray, x: np.ndarray) -> float:
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


# ---------------------------------------------------------------- 分布基本性质


@pytest.mark.parametrize("k,c", [(1.0, 7.0), (1.5, 6.0), (2.0, 8.0), (2.5, 10.0), (3.2, 12.0)])
def test_weibull_pdf_normalized(k: float, c: float) -> None:
    """锚点 2：Weibull 概率密度在 [0, ∞) 上积分为 1。"""
    v = np.linspace(0.0, 100.0, 400001)
    integral = trapz(weibull_pdf(v, k, c), v)
    assert abs(integral - 1.0) < 1e-5, "Weibull 归一化失败: %.8f" % integral


@pytest.mark.parametrize("k,c", [(1.2, 5.5), (2.0, 8.0), (2.8, 11.0)])
def test_weibull_cdf_matches_pdf_integral(k: float, c: float) -> None:
    """CDF 应等于 PDF 的数值积分（k<2 时低风速段曲线变化快，容差放到 1e-4）。"""
    v = np.linspace(0.0, 30.0, 30001)
    f = weibull_pdf(v, k, c)
    pdf_int = np.concatenate(([0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(v))))
    assert np.max(np.abs(pdf_int - weibull_cdf(v, k, c))) < 1e-4


def test_weibull_pdf_negative_speed_is_zero() -> None:
    """负风速概率密度为 0（物理上无意义）。"""
    assert weibull_pdf(-1.0, 2.0, 8.0) == 0.0
    out = weibull_pdf(np.array([-5.0, -0.1, 0.0, 1.0]), 2.0, 8.0)
    assert out[0] == 0.0 and out[1] == 0.0
    assert out[2] == 0.0            # k>1 时 v=0 的密度为 0
    assert out[3] > 0.0


def test_weibull_cdf_monotone_and_bounds() -> None:
    """CDF 单调不减，取值在 [0, 1]。"""
    v = np.linspace(0.0, 40.0, 500)
    f = weibull_cdf(v, 2.0, 8.5)
    assert np.all(np.diff(f) >= -1e-12)
    assert f[0] == 0.0
    assert 0.0 <= f[-1] <= 1.0


def test_weibull_moment_analytic_values() -> None:
    """E[v] = c·Γ(1+1/k)、E[v²]、E[v³] 的解析值与数值积分一致。"""
    k, c = 2.0, 8.0
    m1 = weibull_moment(k, c, 1.0)
    assert abs(m1 - c * math.gamma(1.5)) < 1e-9
    v = np.linspace(0.0, 60.0, 600001)
    f = weibull_pdf(v, k, c)
    assert abs(trapz(v * f, v) - weibull_moment(k, c, 1.0)) < 2e-4
    assert abs(trapz(v ** 2 * f, v) - weibull_moment(k, c, 2.0)) < 2e-2
    assert abs(trapz(v ** 3 * f, v) - weibull_moment(k, c, 3.0)) < 1.0


def test_weibull_mean_std_consistency() -> None:
    """均值/标准差与数值积分一致。"""
    k, c = 2.3, 9.0
    v = np.linspace(0.0, 70.0, 700001)
    f = weibull_pdf(v, k, c)
    m = trapz(v * f, v)
    s = math.sqrt(trapz((v - m) ** 2 * f, v))
    assert abs(m - weibull_mean(k, c)) < 1e-3
    assert abs(s - weibull_std(k, c)) < 1e-3


def test_wind_power_density_known_value() -> None:
    """k=2, c=8 m/s 时风功率密度 = 0.5·ρ·c³·Γ(2.5) ≈ 417 W/m²。"""
    k, c = 2.0, 8.0
    expected = 0.5 * DEFAULT_AIR_DENSITY * c ** 3 * math.gamma(2.5)
    assert abs(wind_power_density(k, c) - expected) < 1e-9
    assert abs(wind_power_density(k, c) - 416.9) < 0.5


def test_wind_power_density_scales_with_air_density() -> None:
    """风功率密度与空气密度成正比。"""
    a = wind_power_density(2.0, 8.0, air_density=1.225)
    b = wind_power_density(2.0, 8.0, air_density=1.0)
    assert abs(b / a - 1.0 / 1.225) < 1e-12


def test_c_from_mean_inverts_weibull_mean() -> None:
    """由 k 与平均风速反解的 c，代回后应还原平均风速。"""
    for k in (1.5, 2.0, 2.7):
        c = c_from_mean(k, 7.5)
        assert abs(weibull_mean(k, c) - 7.5) < 1e-10


def test_invalid_parameters_raise() -> None:
    """k、c 非法时抛错。"""
    with pytest.raises(ValueError):
        weibull_pdf(5.0, 0.0, 8.0)
    with pytest.raises(ValueError):
        weibull_pdf(5.0, 2.0, -1.0)
    with pytest.raises(ValueError):
        c_from_mean(2.0, 0.0)


# ---------------------------------------------------------------- 参数拟合


@pytest.mark.parametrize("k_true,c_true", [(1.8, 6.5), (2.0, 8.0), (2.6, 10.0)])
def test_fit_moments_recovers_parameters(k_true: float, c_true: float) -> None:
    """锚点 1：矩估计能从大样本还原 k、c（k 相对误差 < 10%，c 相对误差 < 5%）。"""
    s = sample_speeds(k_true, c_true, n=200000, seed=7)
    k_hat, c_hat = fit_weibull_moments(s)
    assert abs(k_hat - k_true) / k_true < 0.10, "k 误差过大: %.3f vs %.3f" % (k_hat, k_true)
    assert abs(c_hat - c_true) / c_true < 0.05, "c 误差过大: %.3f vs %.3f" % (c_hat, c_true)


@pytest.mark.parametrize("k_true,c_true", [(1.8, 6.5), (2.0, 8.0), (2.6, 10.0)])
def test_fit_mle_recovers_parameters(k_true: float, c_true: float) -> None:
    """锚点 1：最大似然估计能从大样本还原 k、c（容差同矩估计）。"""
    s = sample_speeds(k_true, c_true, n=200000, seed=11)
    k_hat, c_hat = fit_weibull_mle(s)
    assert abs(k_hat - k_true) / k_true < 0.10, "k 误差过大: %.3f vs %.3f" % (k_hat, k_true)
    assert abs(c_hat - c_true) / c_true < 0.05, "c 误差过大: %.3f vs %.3f" % (c_hat, c_true)


def test_fit_mle_and_moments_agree_on_large_sample() -> None:
    """大样本下矩估计与 MLE 应非常接近（差异 < 3%）。"""
    s = sample_speeds(2.1, 7.7, n=200000, seed=3)
    km, cm = fit_weibull_moments(s)
    kl, cl = fit_weibull_mle(s)
    assert abs(km - kl) / kl < 0.03
    assert abs(cm - cl) / cl < 0.01


def test_fit_ignores_calm_and_invalid_samples() -> None:
    """静风（0）与非法值（NaN/负）不参与拟合，且不改变结论。"""
    s = sample_speeds(2.0, 8.0, n=50000, seed=5)
    k1, c1 = fit_weibull_mle(s)
    s2 = np.concatenate([s, np.zeros(500), np.full(50, np.nan), -np.ones(30)])
    k2, c2 = fit_weibull_mle(s2)
    assert abs(k1 - k2) < 0.02
    assert abs(c1 - c2) < 0.05


def test_fit_requires_minimum_samples() -> None:
    """样本太少时报错。"""
    with pytest.raises(ValueError):
        fit_weibull_mle([1.0, 2.0])
    with pytest.raises(ValueError):
        fit_weibull_moments([1.0, 2.0])


def test_fit_handles_nearly_constant_samples() -> None:
    """风速几乎恒定时 k 取上界而不崩溃。"""
    k, c = fit_weibull_moments(np.full(100, 7.0))
    assert k > 20.0
    assert abs(c - 7.0) < 1e-6


def test_sample_speeds_statistics() -> None:
    """合成样本的均值/标准差应符合 Weibull 理论值。"""
    k, c = 2.0, 8.0
    s = sample_speeds(k, c, n=200000, seed=13)
    assert abs(np.mean(s) - weibull_mean(k, c)) < 0.05
    assert abs(np.std(s) - weibull_std(k, c)) < 0.05
    assert np.all(s >= 0.0)


def test_sample_speeds_reproducible() -> None:
    """同种子结果完全一致，不同种子不同。"""
    a = sample_speeds(2.0, 8.0, n=1000, seed=1)
    b = sample_speeds(2.0, 8.0, n=1000, seed=1)
    c = sample_speeds(2.0, 8.0, n=1000, seed=2)
    assert np.allclose(a, b)
    assert not np.allclose(a, c)


# ---------------------------------------------------------------- 风频直方图


def test_speed_histogram_sums_to_one() -> None:
    """风频直方图的频率之和为 1，区间中点取 0.5、1.5、...。"""
    s = sample_speeds(2.0, 8.0, n=100000, seed=17)
    centers, freq, edges = speed_histogram(s, bin_width=1.0)
    assert abs(float(np.sum(freq)) - 1.0) < 1e-12
    assert abs(centers[0] - 0.5) < 1e-12
    assert abs(centers[1] - 1.5) < 1e-12
    assert freq.shape[0] == edges.shape[0] - 1
    assert np.all(freq >= 0.0)


def test_speed_histogram_recovers_mean() -> None:
    """用直方图还原的平均风速与样本均值一致（误差 < 0.02 m/s）。"""
    s = sample_speeds(2.0, 8.0, n=200000, seed=19)
    centers, freq, _ = speed_histogram(s, bin_width=0.5)
    est = float(np.sum(centers * freq))
    assert abs(est - float(np.mean(s))) < 0.02


def test_speed_histogram_rejects_bad_width() -> None:
    with pytest.raises(ValueError):
        speed_histogram([1.0, 2.0], bin_width=0.0)
    with pytest.raises(ValueError):
        speed_histogram([], bin_width=1.0)


# ---------------------------------------------------------------- 风玫瑰与风资源


def test_sector_names_and_directions() -> None:
    """16 扇区命名与中心风向。"""
    assert N_SECTORS == 16
    assert sector_name(0) == "N"
    assert sector_name(4) == "E"
    assert sector_name(8) == "S"
    assert sector_name(12) == "W"
    assert sector_name(16) == "N"
    assert sector_direction_deg(12) == 270.0
    assert abs(sector_direction_deg(1) - 22.5) < 1e-12


def test_rose_normalizes_frequencies() -> None:
    """风玫瑰频率自动归一化，且能识别主风向。"""
    f = np.ones(N_SECTORS) * 3.0
    f[12] = 9.0
    rose = WindRose(freqs=f, mean_speeds=np.full(N_SECTORS, 7.0))
    assert abs(float(np.sum(rose.freqs)) - 1.0) < 1e-12
    assert rose.dominant_sector_index == 12
    assert rose.dominant_sector_name == "W"
    assert rose.dominant_direction_deg == 270.0
    assert abs(rose.mean_speed - 7.0) < 1e-12


def test_rose_directional_spread_limits() -> None:
    """集中度：均匀风向 -> 0；单一扇区 -> 1。"""
    uniform = WindRose(freqs=np.ones(N_SECTORS), mean_speeds=np.full(N_SECTORS, 7.0))
    assert abs(uniform.directional_spread) < 1e-12
    single = np.zeros(N_SECTORS)
    single[5] = 1.0
    spike = WindRose(freqs=single, mean_speeds=np.full(N_SECTORS, 7.0))
    assert abs(spike.directional_spread - 1.0) < 1e-9
    assert spike.dominant_sector_index == 5


def test_rose_resultant_direction() -> None:
    """风矢量合成方向：全部集中在西扇区时约为 270°。"""
    f = np.zeros(N_SECTORS)
    f[12] = 1.0
    rose = WindRose(freqs=f, mean_speeds=np.full(N_SECTORS, 7.0))
    assert abs(rose.resultant_direction_deg() - 270.0) < 1e-6


def test_rose_validation() -> None:
    """长度不对、含负频率、平均风速非正都会报错。"""
    with pytest.raises(ValueError):
        WindRose(freqs=np.ones(8), mean_speeds=np.ones(8))
    with pytest.raises(ValueError):
        WindRose(freqs=-np.ones(N_SECTORS), mean_speeds=np.ones(N_SECTORS))
    with pytest.raises(ValueError):
        WindRose(freqs=np.ones(N_SECTORS), mean_speeds=np.zeros(N_SECTORS))
    with pytest.raises(ValueError):
        WindRose(freqs=np.zeros(N_SECTORS), mean_speeds=np.ones(N_SECTORS))


def test_example_rose_is_dominated_by_west() -> None:
    """示例风玫瑰的主风向为西风（合成数据的构造约定）。"""
    rose = example_wind_rose()
    assert rose.dominant_sector_name == "W"
    assert abs(float(np.sum(rose.freqs)) - 1.0) < 1e-12
    assert 0.10 < rose.freqs[12] < 0.35
    assert rose.freqs[12] > np.max(rose.freqs[np.arange(N_SECTORS) != 12])
    assert 6.0 < rose.mean_speed < 9.0


def test_resource_sector_weibull_consistency() -> None:
    """扇区 Weibull 的均值应等于该扇区给定的平均风速。"""
    res = example_wind_resource()
    for i in (0, 6, 12, 15):
        k_s, c_s = res.sector_weibull(i)
        assert abs(k_s - res.k) < 1e-12
        assert abs(weibull_mean(k_s, c_s) - res.rose.mean_speeds[i]) < 1e-9


def test_resource_power_density_is_frequency_weighted() -> None:
    """场地风功率密度 = 各扇区按频率加权的风功率密度。"""
    res = example_wind_resource()
    manual = sum(float(res.rose.freqs[i]) * res.sector_power_density(i) for i in range(N_SECTORS))
    assert abs(res.power_density - manual) < 1e-9
    assert res.power_density > 0.0


def test_resource_summary_contains_key_numbers() -> None:
    """摘要里应出现平均风速、k、c、主风向等信息（中文输出）。"""
    res = example_wind_resource(mean_speed=7.5)
    text = res.summary()
    assert "平均风速" in text
    assert "风功率密度" in text
    assert "主风向" in text
    assert abs(res.mean_speed - 7.5) < 0.05


def test_resource_validation() -> None:
    """非法空气密度报错。"""
    rose = example_wind_rose()
    with pytest.raises(ValueError):
        WindResource(rose=rose, k=2.0, air_density=0.0)
    with pytest.raises(ValueError):
        WindResource(rose=rose, k=-1.0)
