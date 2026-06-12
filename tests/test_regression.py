import sys
import os
import json
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.algorithms.modal.wavelet_denoise import (
    WaveletThresholdDenoiser,
    preprocess_vibration_signal,
)
from backend.algorithms.ssi_modal import (
    StochasticSubspaceIdentification,
    detect_delamination_regions,
    BondStrengthAssessor,
)
from backend.algorithms.grouting_diffusion import (
    NewtonianSphericalDiffusion,
    assess_reinforcement_effectiveness,
    PressureFlowPolynomialRegressor,
    generate_simulated_pressure_flow_data,
)
from backend.algorithms.drying_shrinkage import (
    AHTDryingShrinkageModel,
    get_formulation_by_id,
    compare_formulations,
    optimize_curing_schedule,
)
from backend.algorithms.priority_ranking import (
    NSGAIISolver,
    CaveProtectionPriorityRanking,
    CaveConditionData,
    generate_simulated_visitor_flow,
)
from shared.config import SharedSettings
from shared.schemas import VibrationDataBatch, ThermalImageData


class TestWaveletDenoise:
    def test_denoise_removes_high_freq_noise(self):
        np.random.seed(42)
        fs = 2000.0
        t = np.arange(5000) / fs
        clean = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 25 * t)
        noise = 0.3 * np.random.randn(len(t))
        noisy = clean + noise

        denoiser = WaveletThresholdDenoiser(wavelet="db8", level=5, mode="soft", threshold_method="rigrsure")
        denoised = denoiser.denoise(noisy)

        err_noisy = np.mean((noisy - clean) ** 2)
        err_denoised = np.mean((denoised - clean) ** 2)
        assert err_denoised < err_noisy, f"去噪后误差应小于原始误差: {err_denoised:.4f} >= {err_noisy:.4f}"

    def test_denoise_multichannel(self):
        np.random.seed(0)
        n_ch, n_s = 6, 3000
        data = np.random.randn(n_ch, n_s) * 0.1
        for ch in range(n_ch):
            data[ch] += np.sin(2 * np.pi * (5 + ch * 3) * np.arange(n_s) / 2000)

        denoiser = WaveletThresholdDenoiser()
        result = denoiser.denoise_multichannel(data)
        assert result.shape == data.shape

    def test_preprocess_pipeline(self):
        np.random.seed(7)
        signal = np.random.randn(2000) * 0.5 + np.sin(2 * np.pi * 15 * np.arange(2000) / 2000)
        processed = preprocess_vibration_signal(signal, fs=2000.0, detrend=True)
        assert len(processed) == len(signal)

    def test_soft_vs_hard(self):
        np.random.seed(1)
        sig = np.sin(2 * np.pi * 10 * np.arange(1000) / 2000) + 0.2 * np.random.randn(1000)
        soft = WaveletThresholdDenoiser(mode="soft").denoise(sig)
        hard = WaveletThresholdDenoiser(mode="hard").denoise(sig)
        assert len(soft) == len(hard) == len(sig)

    def test_threshold_methods(self):
        np.random.seed(2)
        sig = np.sin(2 * np.pi * 8 * np.arange(2000) / 2000) + 0.3 * np.random.randn(2000)
        for method in ["universal", "rigrsure", "sqtwolog"]:
            denoiser = WaveletThresholdDenoiser(threshold_method=method)
            result = denoiser.denoise(sig)
            assert len(result) == len(sig)


class TestSSIModal:
    def _make_vibration_data(self, n_sensors=5, n_samples=2000, fs=2000.0):
        np.random.seed(100)
        data = {}
        freqs_true = [5.0, 12.0, 25.0]
        t = np.arange(n_samples) / fs
        for i in range(n_sensors):
            axes = {}
            for axis in ["x", "y", "z"]:
                sig = np.zeros(n_samples)
                for f in freqs_true:
                    sig += (0.5 + np.random.rand() * 0.5) * np.sin(2 * np.pi * f * t + np.random.rand() * np.pi)
                sig += 0.05 * np.random.randn(n_samples)
                axes[axis] = sig.tolist()
            data[f"S-{i:03d}"] = axes
        return data

    def test_identify_returns_frequencies(self):
        data = self._make_vibration_data()
        ssi = StochasticSubspaceIdentification(
            fs=2000.0, order_min=10, order_max=30,
            wavelet_denoise=True,
        )
        result = ssi.identify(data)
        assert len(result.frequencies) > 0
        assert len(result.damping_ratios) > 0
        assert len(result.frequencies) == len(result.damping_ratios)
        assert np.all(result.frequencies > 0)
        assert np.all(result.damping_ratios >= 0)

    def test_ssi_with_denoise_vs_without(self):
        data = self._make_vibration_data()
        ssi_dn = StochasticSubspaceIdentification(fs=2000.0, order_min=10, order_max=25, wavelet_denoise=True)
        ssi_nd = StochasticSubspaceIdentification(fs=2000.0, order_min=10, order_max=25, wavelet_denoise=False)
        r_dn = ssi_dn.identify(data)
        r_nd = ssi_nd.identify(data)
        assert len(r_dn.frequencies) > 0
        assert len(r_nd.frequencies) > 0

    def test_detect_delamination(self):
        data = self._make_vibration_data()
        ssi = StochasticSubspaceIdentification(fs=2000.0, order_min=10, order_max=25, wavelet_denoise=True)
        result = ssi.identify(data)
        baseline = result.frequencies * 1.08
        positions = [{"x": i, "y": 0, "z": 0} for i in range(5)]
        regions = detect_delamination_regions(result, baseline, positions)
        assert isinstance(regions, list)


class TestGroutDiffusion:
    def test_single_point_prediction(self):
        model = NewtonianSphericalDiffusion(
            viscosity_pa_s=0.25,
            porosity=0.35,
            permeability_m2=1e-12,
            wall_thickness_mm=50.0,
        )
        ip = {"id": "IP-01", "x": 5.0, "y": 10.0, "z": 8.0}
        result = model.predict_single_point(ip, elapsed_seconds=3600, pressure_kpa=50.0)
        assert result.predicted_radius_mm > 0
        assert result.penetration_depth_mm > 0
        assert result.flow_rate_mls >= 0
        assert len(result.diffusion_front) > 0
        assert len(result.particle_pathlines) > 0

    def test_multi_point_prediction(self):
        model = NewtonianSphericalDiffusion()
        ips = [
            {"id": f"IP-{i:02d}", "x": float(i * 2), "y": 5.0, "z": 5.0}
            for i in range(3)
        ]
        results = model.predict_multi_point(ips, elapsed_seconds=1800, pressure_kpa=60.0)
        assert len(results) == 3
        for r in results:
            assert r.predicted_radius_mm > 0

    def test_diffusion_increases_with_time(self):
        model = NewtonianSphericalDiffusion()
        ip = {"id": "IP-T", "x": 1.0, "y": 1.0, "z": 1.0}
        r1 = model.predict_single_point(ip, elapsed_seconds=600, pressure_kpa=50.0)
        r2 = model.predict_single_point(ip, elapsed_seconds=3600, pressure_kpa=50.0)
        assert r2.predicted_radius_mm > r1.predicted_radius_mm

    def test_assess_effectiveness(self):
        pre_freqs = [8.5, 22.0, 41.5, 60.2, 85.0]
        post_freqs = [9.5, 24.0, 44.0, 63.5, 89.0]
        assessment = assess_reinforcement_effectiveness(pre_freqs, post_freqs, 0.85, 0.15)
        assert "overall_score" in assessment
        assert "grade" in assessment
        assert assessment["overall_score"] > 0


class TestSharedConfig:
    def test_settings_defaults(self):
        s = SharedSettings()
        assert s.SSI_MODEL_ORDER_MIN == 10
        assert s.SSI_MODEL_ORDER_MAX == 50
        assert s.WAVELET_NAME == "db8"
        assert s.REDIS_URL.startswith("redis://")
        assert s.ALERT_AREA_INCREASE_PCT == 10.0
        assert s.ALERT_FREQ_DROP_PCT == 5.0

    def test_yaml_load(self):
        s = SharedSettings()
        yaml_path = os.path.join(os.path.dirname(__file__), "..", "params", "default.yaml")
        s.load_params_from_yaml(yaml_path)
        assert s.SSI_MODEL_ORDER_MIN >= 1


class TestSchemas:
    def test_vibration_batch(self):
        batch = VibrationDataBatch(
            timestamp="2026-01-01T00:00:00Z",
            sensors={"S-001": {"x": [0.1, 0.2], "y": [0.3, 0.4], "z": [0.5, 0.6]}},
        )
        assert batch.sensors["S-001"]["x"] == [0.1, 0.2]

    def test_thermal_image(self):
        img = ThermalImageData(
            camera_id="TH-001",
            max_temp=25.5,
            min_temp=18.2,
            avg_temp=21.3,
        )
        assert img.camera_id == "TH-001"
        assert img.max_temp == 25.5


class TestRedisStreamFormat:
    def test_message_serialization(self):
        data = {
            "type": "vibration_raw",
            "surface_id": "C096-N",
            "timestamp": "2026-01-01T00:00:00Z",
            "sensors": json.dumps({"S-001": {"x": [0.1]}}),
        }
        for k, v in data.items():
            assert isinstance(v, str), f"Redis Stream value must be str: {k}={type(v)}"

    def test_delamination_event_format(self):
        event = {
            "type": "delamination_detected",
            "surface_id": "C096-N",
            "timestamp": "2026-01-01T00:00:00Z",
            "n_regions": "3",
            "regions": json.dumps([{"region_id": "DEL-1", "area_sqm": 0.5}]),
        }
        parsed = json.loads(event["regions"])
        assert len(parsed) == 1
        assert parsed[0]["region_id"] == "DEL-1"


class TestEndToEndPipeline:
    def test_vibration_to_modal_pipeline(self):
        np.random.seed(42)
        fs = 2000.0
        n_samples = 2000
        t = np.arange(n_samples) / fs
        freqs_true = [8.0, 20.0]
        sensors = {}
        for i in range(4):
            axes = {}
            for axis in ["x", "y", "z"]:
                sig = sum(
                    (0.5 + np.random.rand() * 0.3) * np.sin(2 * np.pi * f * t + np.random.rand() * 2 * np.pi)
                    for f in freqs_true
                )
                sig += 0.05 * np.random.randn(n_samples)
                axes[axis] = sig.tolist()
            sensors[f"VB-{i:03d}"] = axes

        denoiser = WaveletThresholdDenoiser(wavelet="db8", level=4, mode="soft", threshold_method="rigrsure")
        for sid, axes in sensors.items():
            for axis in ["x", "y", "z"]:
                raw = np.array(axes[axis])
                denoised = denoiser.denoise(raw - np.mean(raw))
                sensors[sid][axis] = denoised.tolist()

        ssi = StochasticSubspaceIdentification(fs=fs, order_min=6, order_max=20, wavelet_denoise=False)
        result = ssi.identify(sensors)
        assert len(result.frequencies) > 0

    def test_grout_pipeline(self):
        model = NewtonianSphericalDiffusion(
            viscosity_pa_s=0.25, porosity=0.35, permeability_m2=1e-12, wall_thickness_mm=50.0,
        )
        ips = [{"id": "IP-01", "x": 5.0, "y": 10.0, "z": 8.0}]
        results = model.predict_multi_point(ips, elapsed_seconds=3600, pressure_kpa=50.0)
        assert len(results) == 1

        pre_freqs = [8.0, 20.0, 40.0]
        post_freqs = [f * 1.1 for f in pre_freqs]
        assessment = assess_reinforcement_effectiveness(pre_freqs, post_freqs, 0.8, 0.2)
        assert assessment["overall_score"] > 50


class TestPressureFlowPolynomialRegression:
    def test_simulated_data_generation(self):
        data = generate_simulated_pressure_flow_data(n_points=20, pressure_range=(100.0, 450.0))
        assert len(data) == 20
        for dp in data:
            assert 100.0 <= dp.pressure_kpa <= 450.0
            assert dp.flow_rate_mls > 0

    def test_polynomial_fit(self):
        np.random.seed(42)
        data = generate_simulated_pressure_flow_data(n_points=30)
        regressor = PressureFlowPolynomialRegressor(max_degree=5, use_cross_validation=True)
        regressor.fit(data)
        assert regressor.best_degree >= 2
        assert regressor.best_degree <= 5
        assert regressor.r_squared > 0.7

    def test_optimize_injection_rate(self):
        np.random.seed(42)
        data = generate_simulated_pressure_flow_data(n_points=30)
        regressor = PressureFlowPolynomialRegressor(max_degree=4)
        regressor.fit(data)
        result = regressor.optimize_injection_rate(
            pressure_range=(150.0, 400.0),
            max_allowable_risk_pct=30.0,
        )
        assert result.optimal_pressure_kpa > 0
        assert result.optimal_flow_rate_mls > 0
        assert 0 <= result.secondary_delamination_risk_pct <= 100
        assert len(result.pressure_flow_curve) > 0

    def test_flow_rate_increases_with_pressure(self):
        np.random.seed(42)
        data = generate_simulated_pressure_flow_data(n_points=25)
        regressor = PressureFlowPolynomialRegressor(max_degree=3)
        regressor.fit(data)
        result = regressor.optimize_injection_rate()
        q_low = regressor.predict_flow_rate(100.0)
        q_high = regressor.predict_flow_rate(400.0)
        assert q_high > q_low


class TestBondStrengthAssessment:
    def test_bond_strength_assessor_init(self):
        assessor = BondStrengthAssessor(
            reference_damping=0.02,
            alpha=2.5,
            min_delta_damp=0.001,
        )
        assert assessor.reference_damping == 0.02
        assert assessor.alpha == 2.5

    def test_assess_bond_strength_degradation(self):
        np.random.seed(42)
        baseline_damping = [0.018, 0.022, 0.025, 0.030]
        current_damping = [0.028, 0.035, 0.042, 0.050]
        frequencies = [8.5, 22.0, 41.5, 65.0]

        assessor = BondStrengthAssessor()
        result = assessor.assess_bond_strength(
            surface_id="C096-N",
            baseline_damping_ratios=baseline_damping,
            current_damping_ratios=current_damping,
            frequencies=frequencies,
        )

        assert result["bond_strength_degradation_pct"] > 0
        assert result["bond_strength_degradation_pct"] <= 100
        assert result["remaining_bond_strength_mpa"] > 0
        assert result["risk_level"] in ["无", "低", "中", "高", "极高"]
        assert len(result["recommendations"]) > 0

    def test_more_degradation_higher_risk(self):
        baseline = [0.02, 0.025, 0.03]
        slight_damage = [0.025, 0.030, 0.035]
        severe_damage = [0.05, 0.06, 0.07]

        assessor = BondStrengthAssessor()
        r_slight = assessor.assess_bond_strength("test", baseline, slight_damage)
        r_severe = assessor.assess_bond_strength("test", baseline, severe_damage)

        assert r_severe["bond_strength_degradation_pct"] > r_slight["bond_strength_degradation_pct"]
        assert r_severe["remaining_bond_strength_mpa"] < r_slight["remaining_bond_strength_mpa"]


class TestDryingShrinkagePrediction:
    def test_get_formulation_by_id(self):
        formulation = get_formulation_by_id("sintered_stone_powder_ps_basic")
        assert formulation is not None
        assert formulation.id == "sintered_stone_powder_ps_basic"
        assert "烧结石粉" in formulation.name

    def test_invalid_formulation_id(self):
        formulation = get_formulation_by_id("invalid_id")
        assert formulation is None

    def test_shrinkage_prediction_basic(self):
        np.random.seed(42)
        model = AHTDryingShrinkageModel()
        formulation = get_formulation_by_id("sintered_stone_powder_ps_basic")
        model.set_formulation(formulation)

        result = model.predict(
            ambient_temperature_c=20.0,
            ambient_humidity_pct=60.0,
            constraint_factor=0.7,
            wall_thickness_mm=50.0,
        )

        assert result["final_shrinkage_strain"] > 0
        assert result["final_shrinkage_strain"] < 0.01
        assert 0 <= result["crack_risk_index"] <= 1.0
        assert result["crack_risk_level"] in ["无", "低", "中", "高", "极高"]
        assert len(result["time_curve_days"]) > 0
        assert len(result["time_curve_days"]) == len(result["time_curve_strain"])

    def test_humidity_affects_shrinkage(self):
        model = AHTDryingShrinkageModel()
        formulation = get_formulation_by_id("sintered_stone_powder_ps_basic")
        model.set_formulation(formulation)

        r_dry = model.predict(ambient_temperature_c=20.0, ambient_humidity_pct=30.0)
        r_humid = model.predict(ambient_temperature_c=20.0, ambient_humidity_pct=80.0)

        assert r_dry["final_shrinkage_strain"] > r_humid["final_shrinkage_strain"]
        assert r_dry["crack_risk_index"] > r_humid["crack_risk_index"]

    def test_formulation_comparison(self):
        results = compare_formulations(
            ambient_temperature_c=25.0,
            ambient_humidity_pct=55.0,
        )
        assert len(results) >= 4
        for r in results:
            assert "formulation_id" in r
            assert "crack_risk_level" in r

    def test_curing_schedule_optimization(self):
        result = optimize_curing_schedule(
            formulation_id="sintered_stone_powder_ps_basic",
            ambient_temperature_c=20.0,
            ambient_humidity_pct=40.0,
            target_risk_index=0.5,
        )
        assert "optimization_needed" in result
        assert "recommendation" in result


class TestPriorityRanking:
    def test_simulated_visitor_flow(self):
        result = generate_simulated_visitor_flow(
            cave_id="C096",
            days=30,
            base_daily=500.0,
        )
        assert result["cave_id"] == "C096"
        assert len(result["daily_visitors"]) == 30
        assert all(v >= 0 for v in result["daily_visitors"])
        assert result["avg_daily"] > 0

    def test_nsga_ii_solver_basic(self):
        np.random.seed(42)

        def objective_function(x):
            f1 = x[0] ** 2 + x[1] ** 2
            f2 = (x[0] - 2) ** 2 + x[1] ** 2
            return [f1, f2]

        solver = NSGAIISolver(
            population_size=50,
            max_generations=30,
            crossover_prob=0.9,
            mutation_prob=0.1,
        )
        result = solver.solve(
            n_variables=2,
            n_objectives=2,
            lower_bounds=[-5.0, -5.0],
            upper_bounds=[5.0, 5.0],
            objective_function=objective_function,
            maximize=[False, False],
        )

        assert "pareto_front" in result
        assert len(result["pareto_front"]) > 0
        assert result["generations"] == 30
        assert result["population_size"] == 50

    def test_cave_priority_ranking(self):
        caves_data = [
            CaveConditionData(
                cave_id="C096",
                cave_name="第96窟",
                total_delamination_area_sqm=2.5,
                max_severity_score=0.85,
                historical_repair_count=3,
                last_repair_years_ago=5.0,
                avg_visitor_flow_daily=1500.0,
                peak_visitor_flow_daily=3000.0,
                cultural_significance_score=1.0,
                structural_importance=0.95,
                avg_bond_strength_remaining_mpa=0.65,
                active_alerts_count=2,
                wall_surfaces_count=5,
            ),
            CaveConditionData(
                cave_id="C257",
                cave_name="第257窟",
                total_delamination_area_sqm=0.8,
                max_severity_score=0.5,
                historical_repair_count=1,
                last_repair_years_ago=2.0,
                avg_visitor_flow_daily=800.0,
                peak_visitor_flow_daily=1500.0,
                cultural_significance_score=0.95,
                structural_importance=0.9,
                avg_bond_strength_remaining_mpa=0.9,
                active_alerts_count=0,
                wall_surfaces_count=5,
            ),
        ]

        ranker = CaveProtectionPriorityRanking()
        results = ranker.rank_caves(caves_data, use_nsga_ii=False)
        assert len(results) == 2
        assert results[0]["priority_rank"] == 1
        assert results[1]["priority_rank"] == 2
        assert results[0]["urgency_score"] >= results[1]["urgency_score"]

    def test_nsga_ii_priority_ranking(self):
        np.random.seed(42)
        caves_data = [
            CaveConditionData(
                cave_id=f"C{i:03d}",
                cave_name=f"第{i}窟",
                total_delamination_area_sqm=np.random.uniform(0.1, 3.0),
                max_severity_score=np.random.uniform(0.3, 0.9),
                historical_repair_count=np.random.randint(0, 5),
                last_repair_years_ago=np.random.uniform(1.0, 15.0),
                avg_visitor_flow_daily=np.random.uniform(200, 2000),
                peak_visitor_flow_daily=np.random.uniform(500, 4000),
                cultural_significance_score=np.random.uniform(0.7, 1.0),
                structural_importance=np.random.uniform(0.6, 1.0),
                avg_bond_strength_remaining_mpa=np.random.uniform(0.3, 1.0),
                active_alerts_count=np.random.randint(0, 5),
                wall_surfaces_count=5,
            )
            for i in range(5)
        ]

        ranker = CaveProtectionPriorityRanking()
        results = ranker.rank_caves(caves_data, use_nsga_ii=True)
        assert len(results) == 5
        for r in results:
            assert "urgency_score" in r
            assert "cost_efficiency_score" in r
            assert "cultural_impact_score" in r
            assert r["method"] == "nsga_ii_multi_objective"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
