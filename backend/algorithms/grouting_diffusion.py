import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum


class CrackRiskLevel(str, Enum):
    NONE = "无"
    LOW = "低"
    MEDIUM = "中"
    HIGH = "高"
    CRITICAL = "极高"


@dataclass
class PressureFlowDataPoint:
    pressure_kpa: float
    flow_rate_mls: float
    elapsed_seconds: float
    temperature_c: Optional[float] = None
    is_reliable: bool = True


@dataclass
class InjectionRateOptimizationResult:
    optimal_pressure_kpa: float
    optimal_flow_rate_mls: float
    polynomial_degree: int
    polynomial_coefficients: List[float]
    r_squared: float
    secondary_delamination_risk_pct: float
    recommended_max_flow_mls: float
    pressure_flow_curve: List[Dict]
    warning_message: Optional[str] = None


@dataclass
class BondStrengthAssessmentResult:
    surface_id: str
    timestamp: str
    baseline_damping_ratios: List[float]
    current_damping_ratios: List[float]
    bond_strength_degradation_pct: float
    remaining_bond_strength_mpa: float
    critical_mode_index: Optional[int]
    assessment_confidence: float
    risk_level: str
    recommendations: List[str]


@dataclass
class DryingShrinkagePredictionResult:
    grout_formulation: str
    ambient_temperature_c: float
    ambient_humidity_pct: float
    predicted_shrinkage_strain: float
    predicted_crack_width_mm: float
    crack_risk_level: str
    shrinkage_time_curve_days: List[float]
    shrinkage_time_curve_strain: List[float]
    critical_period_days: float
    recommendations: List[str]


@dataclass
class DiffusionPoint:
    x: float
    y: float
    z: float
    time_sec: float
    pressure: float
    velocity_mag: float


@dataclass
class GroutDiffusionResult:
    injection_point_id: str
    elapsed_seconds: int
    predicted_radius_mm: float
    penetration_depth_mm: float
    pressure_kpa: float
    viscosity_pa_s: float
    porosity: float
    flow_rate_mls: float
    diffusion_front: List[Dict] = field(default_factory=list)
    particle_pathlines: List[Dict] = field(default_factory=list)
    volume_ml: float = 0.0
    concentration_distribution: Optional[np.ndarray] = None


class NewtonianSphericalDiffusion:
    def __init__(
        self,
        viscosity_pa_s: float = 0.25,
        porosity: float = 0.35,
        permeability_m2: float = 1e-12,
        wall_thickness_mm: float = 50.0,
    ):
        self.viscosity_pa_s = viscosity_pa_s
        self.porosity = porosity
        self.permeability_m2 = permeability_m2
        self.wall_thickness_m = wall_thickness_mm / 1000.0

    def _hagen_poiseuille_radius(
        self,
        delta_p_pa: float,
        time_s: float,
        radius_initial_m: float = 0.002,
    ) -> float:
        t = max(time_s, 1e-6)
        dp = max(delta_p_pa, 1e3)
        k = self.permeability_m2
        phi = self.porosity
        mu = self.viscosity_pa_s

        coefficient = (2.0 * k * dp) / (phi * mu)
        radius_squared = radius_initial_m ** 2 + coefficient * t
        radius_m = np.sqrt(max(radius_squared, radius_initial_m ** 2))

        radius_m = min(radius_m, self.wall_thickness_m * 2.5)

        return radius_m

    def _penetration_depth(
        self,
        delta_p_pa: float,
        time_s: float,
    ) -> float:
        t = max(time_s, 1e-6)
        dp = max(delta_p_pa, 1e3)
        k = self.permeability_m2
        phi = self.porosity
        mu = self.viscosity_pa_s

        depth_m = np.sqrt((2.0 * k * dp * t) / (phi * mu * 3.0))
        depth_m = min(depth_m, self.wall_thickness_m)

        return depth_m

    def _flow_rate(
        self,
        delta_p_pa: float,
        radius_m: float,
    ) -> float:
        dp = max(delta_p_pa, 1e3)
        k = self.permeability_m2
        mu = self.viscosity_pa_s
        thickness = self.wall_thickness_m

        q_m3s = (k * dp * 2.0 * np.pi * radius_m * thickness) / (mu * max(radius_m, 0.001))
        q_mls = q_m3s * 1e6

        return q_mls

    def _build_diffusion_front(
        self,
        center: Tuple[float, float, float],
        radius_m: float,
        n_points: int = 64,
    ) -> List[Dict]:
        cx, cy, cz = center
        points = []

        for i in range(n_points):
            theta = 2.0 * np.pi * i / n_points
            phi_angle = np.pi / 2.0
            jitter = 0.92 + 0.16 * np.random.rand()

            r = radius_m * jitter
            x = cx + r * np.sin(phi_angle) * np.cos(theta)
            y = cy + r * np.cos(phi_angle)
            z = cz + r * np.sin(phi_angle) * np.sin(theta)

            points.append({
                "x": round(float(x), 4),
                "y": round(float(y), 4),
                "z": round(float(z), 4),
                "concentration": round(float(jitter), 3),
            })

        return points

    def _build_particle_pathlines(
        self,
        center: Tuple[float, float, float],
        radius_m: float,
        depth_m: float,
        time_s: float,
        n_streamlines: int = 12,
        n_steps_per_line: int = 30,
    ) -> List[Dict]:
        cx, cy, cz = center
        streamlines = []

        for sl in range(n_streamlines):
            theta_0 = 2.0 * np.pi * sl / n_streamlines
            phi_0 = np.pi / 4.0 + np.random.uniform(-np.pi / 8, np.pi / 8)

            points = []
            for step in range(n_steps_per_line):
                t_frac = (step + 1) / n_steps_per_line
                r_current = radius_m * t_frac * (0.85 + 0.3 * np.random.rand())
                d_current = depth_m * t_frac

                theta = theta_0 + 0.15 * np.sin(t_frac * np.pi * 3)
                phi_a = phi_0 + 0.1 * np.cos(t_frac * np.pi * 2)

                x = cx + r_current * np.sin(phi_a) * np.cos(theta)
                y = cy + d_current * np.random.uniform(0.7, 1.0)
                z = cz + r_current * np.sin(phi_a) * np.sin(theta)

                vel = (r_current / max(time_s * t_frac, 0.001)) * 1000.0

                points.append({
                    "x": round(float(x), 4),
                    "y": round(float(y), 4),
                    "z": round(float(z), 4),
                    "t": round(float(time_s * t_frac), 1),
                    "v": round(float(vel), 3),
                    "c": round(float(1.0 - 0.5 * t_frac), 3),
                })

            streamlines.append({
                "streamline_id": f"SL-{sl:03d}",
                "start_theta": round(float(theta_0), 3),
                "points": points,
            })

        return streamlines

    def _compute_total_volume(
        self,
        radius_m: float,
        penetration_depth_m: float,
    ) -> float:
        r = radius_m
        h = penetration_depth_m
        volume_m3 = (2.0 / 3.0) * np.pi * (r ** 2) * h * self.porosity
        volume_ml = volume_m3 * 1e6
        return float(volume_ml)

    def predict_single_point(
        self,
        injection_point: Dict,
        elapsed_seconds: int,
        pressure_kpa: float,
    ) -> GroutDiffusionResult:
        center = (
            float(injection_point.get("x", 0.0)),
            float(injection_point.get("y", 0.0)),
            float(injection_point.get("z", 0.0)),
        )
        ip_id = injection_point.get("id", "IP-001")

        delta_p_pa = pressure_kpa * 1000.0
        t = float(max(elapsed_seconds, 1))

        radius_m = self._hagen_poiseuille_radius(delta_p_pa, t)
        penetration_m = self._penetration_depth(delta_p_pa, t)
        radius_mm = radius_m * 1000.0
        penetration_mm = penetration_m * 1000.0

        flow_rate = self._flow_rate(delta_p_pa, radius_m)
        volume_ml = self._compute_total_volume(radius_m, penetration_m)

        diffusion_front = self._build_diffusion_front(center, radius_m)
        pathlines = self._build_particle_pathlines(center, radius_m, penetration_m, t)

        return GroutDiffusionResult(
            injection_point_id=ip_id,
            elapsed_seconds=elapsed_seconds,
            predicted_radius_mm=round(float(radius_mm), 4),
            penetration_depth_mm=round(float(penetration_mm), 4),
            pressure_kpa=float(pressure_kpa),
            viscosity_pa_s=float(self.viscosity_pa_s),
            porosity=float(self.porosity),
            flow_rate_mls=round(float(flow_rate), 4),
            diffusion_front=diffusion_front,
            particle_pathlines=pathlines,
            volume_ml=round(float(volume_ml), 4),
        )

    def predict_multi_point(
        self,
        injection_points: List[Dict],
        elapsed_seconds: int,
        pressure_kpa: float,
        combine: bool = True,
    ) -> List[GroutDiffusionResult]:
        results = []
        for ip in injection_points:
            res = self.predict_single_point(ip, elapsed_seconds, pressure_kpa)
            results.append(res)

        return results


def assess_reinforcement_effectiveness(
    pre_freqs: List[float],
    post_freqs: List[float],
    pre_delamination_area: float,
    post_delamination_area: float,
    target_strength_mpa: float = 0.8,
) -> Dict:
    pre = np.array(pre_freqs, dtype=np.float64)
    post = np.array(post_freqs, dtype=np.float64)

    min_len = min(len(pre), len(post))
    pre = pre[:min_len]
    post = post[:min_len]

    freq_recovery_pcts = []
    for i in range(min_len):
        if pre[i] > 0:
            recovery = (post[i] - pre[i]) / pre[i] * 100.0
            freq_recovery_pcts.append(float(recovery))

    avg_freq_recovery = float(np.mean(freq_recovery_pcts)) if freq_recovery_pcts else 0.0

    if pre_delamination_area > 0:
        area_reduction_pct = (pre_delamination_area - post_delamination_area) / pre_delamination_area * 100.0
    else:
        area_reduction_pct = 100.0 if post_delamination_area == 0 else 0.0

    normalized_freq_recovery = min(max(avg_freq_recovery, 0.0), 30.0) / 30.0 * 100.0
    normalized_area_reduction = min(max(area_reduction_pct, 0.0), 100.0)

    strength_factor = min(target_strength_mpa, target_strength_mpa * normalized_freq_recovery / 100.0)
    bonding_strength = strength_factor * (0.8 + 0.2 * normalized_area_reduction / 100.0)

    area_recovery_bonus = min((1.0 - post_delamination_area / max(pre_delamination_area, 0.001)) * 50.0, 50.0)
    strength_bonus = min(bonding_strength / target_strength_mpa * 50.0, 50.0)
    overall_score = normalized_freq_recovery * 0.5 + area_recovery_bonus + strength_bonus
    overall_score = min(max(overall_score, 0.0), 100.0)

    if overall_score >= 85:
        grade = "优秀"
        notes = "灌浆加固效果优异，壁画地仗层力学性能显著恢复"
    elif overall_score >= 70:
        grade = "良好"
        notes = "灌浆加固效果良好，大部分空鼓区域已得到有效填充"
    elif overall_score >= 55:
        grade = "合格"
        notes = "灌浆加固效果基本达标，建议对残余空鼓区域进行二次注浆"
    else:
        grade = "不合格"
        notes = "灌浆加固效果未达预期，需要重新评估注浆方案"

    return {
        "frequency_recovery_pct": round(avg_freq_recovery, 4),
        "delamination_area_reduction_pct": round(area_reduction_pct, 4),
        "bonding_strength_mpa": round(float(bonding_strength), 6),
        "overall_score": round(float(overall_score), 2),
        "grade": grade,
        "assessment_notes": notes,
        "per_mode_recovery_pct": [round(x, 4) for x in freq_recovery_pcts],
    }


def generate_simulated_pressure_flow_data(
    n_points: int = 30,
    pressure_range_kpa: Tuple[float, float] = (50.0, 600.0),
    pressure_range: Tuple[float, float] = None,
    viscosity_pa_s: float = 0.25,
    noise_std_pct: float = 0.08,
    max_flow_rate_mls: float = 500.0,
) -> List[PressureFlowDataPoint]:
    if pressure_range is not None:
        pressure_range_kpa = pressure_range
    points = []
    pressures = np.linspace(pressure_range_kpa[0], pressure_range_kpa[1], n_points)
    p_min, p_max = pressure_range_kpa
    q_at_max = max_flow_rate_mls * 0.95

    for i, p in enumerate(pressures):
        t_sec = i * 30.0 + np.random.uniform(0, 10.0)

        p_norm = (p - p_min) / max(p_max - p_min, 1.0)
        theoretical_flow = q_at_max * (
            0.15 * p_norm
            + 0.55 * (p_norm ** 1.5)
            + 0.30 * (p_norm ** 2.5)
        )

        if p > 400:
            superlinear = 1.0 + 0.0025 * (p - 400)
            theoretical_flow *= min(superlinear, 1.5)

        theoretical_flow = min(theoretical_flow, max_flow_rate_mls)
        noise = np.random.normal(0, noise_std_pct * max(theoretical_flow, 10.0))
        measured_flow = max(theoretical_flow + noise, 5.0)

        is_reliable = True
        if np.random.rand() < 0.08:
            measured_flow *= np.random.uniform(0.55, 0.80)
            is_reliable = False

        temp_c = 18.0 + np.random.normal(0, 2.0)

        points.append(PressureFlowDataPoint(
            pressure_kpa=float(p),
            flow_rate_mls=float(measured_flow),
            elapsed_seconds=float(t_sec),
            temperature_c=float(temp_c),
            is_reliable=is_reliable,
        ))

    return points


class PressureFlowPolynomialRegressor:
    def __init__(
        self,
        max_degree: int = 5,
        use_cross_validation: bool = True,
        delamination_threshold_pressure_kpa: float = 450.0,
        max_flow_rate_mls: float = 500.0,
        ridge_alpha: float = 1e-3,
    ):
        self.max_degree = max_degree
        self.use_cross_validation = use_cross_validation
        self.delamination_threshold_pressure_kpa = delamination_threshold_pressure_kpa
        self.max_flow_rate_mls = max_flow_rate_mls
        self.ridge_alpha = ridge_alpha
        self.optimal_degree = None
        self.coefficients = None
        self.r_squared = 0.0

    @property
    def best_degree(self):
        return self.optimal_degree

    @best_degree.setter
    def best_degree(self, value):
        self.optimal_degree = value

    def predict_flow_rate(self, pressure_kpa):
        return self.predict(pressure_kpa)

    def _polynomial_features(self, x: np.ndarray, degree: int) -> np.ndarray:
        return np.vander(x, degree + 1, increasing=True)

    def _standardize_features(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mu = np.mean(X, axis=0)
        sigma = np.std(X, axis=0)
        sigma[sigma < 1e-12] = 1.0
        X_std = (X - mu) / sigma
        X_std[:, 0] = X[:, 0]
        mu[0] = 0.0
        sigma[0] = 1.0
        return X_std, mu, sigma

    def _fit_ridge(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X_std, mu, sigma = self._standardize_features(X)
        n_features = X_std.shape[1]
        A = X_std.T @ X_std
        diag = np.eye(n_features)
        diag[0, 0] = 0.0
        A += self.ridge_alpha * diag
        b = X_std.T @ y
        try:
            beta_std = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            beta_std = np.linalg.lstsq(A, b, rcond=None)[0]
        beta = beta_std / sigma
        beta[0] = beta[0] - np.sum(mu * beta_std / sigma)
        return beta, mu, sigma

    def _compute_r_squared(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def _cross_validation_score(
        self,
        X: np.ndarray,
        y: np.ndarray,
        degree: int,
        n_folds: int = 5,
    ) -> float:
        n_samples = len(y)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        fold_size = n_samples // n_folds

        scores = []
        for fold in range(n_folds):
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < n_folds - 1 else n_samples

            val_indices = indices[val_start:val_end]
            train_indices = np.concatenate([indices[:val_start], indices[val_end:]])

            X_train, X_val = X[train_indices], X[val_indices]
            y_train, y_val = y[train_indices], y[val_indices]

            coeffs, _, _ = self._fit_ridge(X_train, y_train)
            y_pred = X_val @ coeffs

            scores.append(self._compute_r_squared(y_val, y_pred))

        return float(np.mean(scores))

    def fit(
        self,
        data_points: List[PressureFlowDataPoint],
    ) -> "PressureFlowPolynomialRegressor":
        reliable_points = [p for p in data_points if p.is_reliable]
        if len(reliable_points) < 10:
            reliable_points = data_points

        pressures = np.array([p.pressure_kpa for p in reliable_points], dtype=np.float64)
        flows = np.array([p.flow_rate_mls for p in reliable_points], dtype=np.float64)

        press_norm = pressures / 1000.0

        best_score = -np.inf
        best_degree = 2

        for degree in range(2, self.max_degree + 1):
            X = self._polynomial_features(press_norm, degree)

            if self.use_cross_validation and len(reliable_points) >= 15:
                score = self._cross_validation_score(X, flows, degree)
            else:
                coeffs, _, _ = self._fit_ridge(X, flows)
                y_pred = X @ coeffs
                score = self._compute_r_squared(flows, y_pred)

            if score > best_score:
                best_score = score
                best_degree = degree

        X_best = self._polynomial_features(press_norm, best_degree)
        best_coeffs, _, _ = self._fit_ridge(X_best, flows)
        y_pred = X_best @ best_coeffs
        self.r_squared = self._compute_r_squared(flows, y_pred)

        self.optimal_degree = best_degree
        self.coefficients = best_coeffs

        return self

    def predict(self, pressure_kpa: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        if self.coefficients is None or self.optimal_degree is None:
            raise ValueError("Model not fitted. Call fit() first.")

        scalar_input = np.isscalar(pressure_kpa)
        p_norm = np.atleast_1d(np.array(pressure_kpa, dtype=np.float64) / 1000.0)
        X = self._polynomial_features(p_norm, self.optimal_degree)

        result = X @ self.coefficients
        if scalar_input:
            return float(max(result[0], 0.0))
        return np.maximum(result, 0.0)

    def _compute_delamination_risk(self, pressure_kpa: float, flow_rate_mls: float) -> float:
        p_factor = min(max((pressure_kpa - 300.0) / (self.delamination_threshold_pressure_kpa - 300.0), 0.0), 1.0)
        q_factor = min(max((flow_rate_mls - 200.0) / (self.max_flow_rate_mls - 200.0), 0.0), 1.0)
        risk = (p_factor * 0.6 + q_factor * 0.4) * 100.0
        return float(risk)

    def optimize_injection_rate(
        self,
        pressure_range: Tuple[float, float] = (100.0, 500.0),
        target_radius_mm: Optional[float] = None,
        max_allowable_risk_pct: float = 30.0,
    ) -> InjectionRateOptimizationResult:
        if self.coefficients is None or self.optimal_degree is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        test_pressures = np.linspace(pressure_range[0], pressure_range[1], 200)
        predicted_flows = self.predict(test_pressures)
        
        curve_points = []
        for p, q in zip(test_pressures, predicted_flows):
            risk = self._compute_delamination_risk(float(p), float(q))
            curve_points.append({
                "pressure_kpa": round(float(p), 2),
                "flow_rate_mls": round(float(q), 4),
                "delamination_risk_pct": round(risk, 2),
            })
        
        safe_mask = np.array([
            self._compute_delamination_risk(float(p), float(q)) <= max_allowable_risk_pct
            for p, q in zip(test_pressures, predicted_flows)
        ])
        
        if not np.any(safe_mask):
            safe_p_idx = np.argmin([
                self._compute_delamination_risk(float(p), float(q))
                for p, q in zip(test_pressures, predicted_flows)
            ])
            warning_msg = f"所有工况风险均超过{max_allowable_risk_pct}%，建议降低注浆压力"
        else:
            safe_pressures = test_pressures[safe_mask]
            safe_flows = predicted_flows[safe_mask]
            
            if target_radius_mm is not None:
                target_flow = target_radius_mm * 8.0
                flow_diffs = np.abs(safe_flows - target_flow)
                safe_p_idx = np.argmin(flow_diffs)
                warning_msg = None
            else:
                safe_p_idx = np.argmax(safe_flows)
                warning_msg = None
            
            test_pressures = safe_pressures
            predicted_flows = safe_flows
        
        optimal_p = float(test_pressures[safe_p_idx])
        optimal_q = float(predicted_flows[safe_p_idx])
        risk_pct = self._compute_delamination_risk(optimal_p, optimal_q)
        
        max_safe_flow_idx = np.argmax(predicted_flows)
        max_safe_flow = float(predicted_flows[max_safe_flow_idx])
        
        return InjectionRateOptimizationResult(
            optimal_pressure_kpa=round(optimal_p, 2),
            optimal_flow_rate_mls=round(optimal_q, 4),
            polynomial_degree=int(self.optimal_degree),
            polynomial_coefficients=[round(float(c), 6) for c in self.coefficients],
            r_squared=round(self.r_squared, 4),
            secondary_delamination_risk_pct=round(risk_pct, 2),
            recommended_max_flow_mls=round(max_safe_flow, 4),
            pressure_flow_curve=curve_points,
            warning_message=warning_msg,
        )
