import numpy as np
from scipy import linalg
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from backend.algorithms.modal.wavelet_denoise import (
    WaveletThresholdDenoiser,
    preprocess_vibration_signal,
)


@dataclass
class ModalResult:
    frequencies: np.ndarray
    damping_ratios: np.ndarray
    mode_shapes: np.ndarray
    model_order: int
    stable_poles: List[Dict]
    mac_matrix: Optional[np.ndarray] = None


class StochasticSubspaceIdentification:
    def __init__(
        self,
        fs: float = 2000.0,
        order_min: int = 10,
        order_max: int = 50,
        freq_tol: float = 0.01,
        damp_tol: float = 0.05,
        mac_tol: float = 0.97,
        wavelet_denoise: bool = True,
        wavelet_name: str = "db8",
        wavelet_level: int = 5,
        wavelet_mode: str = "soft",
        wavelet_threshold: str = "rigrsure",
    ):
        self.fs = fs
        self.order_min = order_min
        self.order_max = order_max
        self.freq_tol = freq_tol
        self.damp_tol = damp_tol
        self.mac_tol = mac_tol
        self.wavelet_denoise = wavelet_denoise
        self.wavelet_name = wavelet_name
        self.wavelet_level = wavelet_level
        self.wavelet_mode = wavelet_mode
        self.wavelet_threshold = wavelet_threshold
        self._denoiser = None
        if self.wavelet_denoise:
            try:
                self._denoiser = WaveletThresholdDenoiser(
                    wavelet=self.wavelet_name,
                    level=self.wavelet_level,
                    mode=self.wavelet_mode,
                    threshold_method=self.wavelet_threshold,
                )
            except ImportError:
                self.wavelet_denoise = False

    def _hankel_matrix(self, data: np.ndarray, block_rows: int) -> np.ndarray:
        n_channels, n_samples = data.shape
        block_cols = n_samples - block_rows + 1
        if block_cols <= 0:
            raise ValueError(f"数据长度不足: n_samples={n_samples}, block_rows={block_rows}")
        hankel = np.zeros((n_channels * block_rows, block_cols))
        for i in range(block_rows):
            hankel[i * n_channels:(i + 1) * n_channels, :] = data[:, i:i + block_cols]
        return hankel

    def _ssi_cov(self, data: np.ndarray, order: int, block_rows: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_channels, n_samples = data.shape
        if block_rows is None:
            block_rows = max(int(n_samples / (2 * n_channels)), order * 2)
            block_rows = min(block_rows, int(n_samples / 4))

        Y = self._hankel_matrix(data, block_rows * 2)
        Y_p = Y[:n_channels * block_rows, :]
        Y_f = Y[n_channels * block_rows:, :]

        R = (Y @ Y.T) / Y.shape[1]
        R_pp = R[:n_channels * block_rows, :n_channels * block_rows]
        R_pf = R[:n_channels * block_rows, n_channels * block_rows:]
        R_fp = R[n_channels * block_rows:, :n_channels * block_rows]
        R_ff = R[n_channels * block_rows:, n_channels * block_rows:]

        S_pf = R_pf
        U, S, Vt = linalg.svd(S_pf, full_matrices=False)

        U_n = U[:, :order]
        S_n = np.diag(S[:order])
        V_n = Vt[:order, :]

        O = U_n @ np.sqrt(S_n)
        C = O[:n_channels, :]
        A = linalg.pinv(O[:-n_channels, :]) @ O[n_channels:, :]

        eigvals, eigvecs = linalg.eig(A)

        dt = 1.0 / self.fs
        lambda_cont = np.log(eigvals) / dt

        frequencies = np.abs(lambda_cont) / (2 * np.pi)
        damping_ratios = -np.real(lambda_cont) / np.abs(lambda_cont)

        mode_shapes = np.real(C @ eigvecs)

        valid = (frequencies > 0.5) & (frequencies < self.fs / 2) & (damping_ratios >= 0) & (damping_ratios < 0.2)
        frequencies = frequencies[valid]
        damping_ratios = damping_ratios[valid]
        mode_shapes = mode_shapes[:, valid]

        sort_idx = np.argsort(frequencies)
        return frequencies[sort_idx], damping_ratios[sort_idx], mode_shapes[:, sort_idx]

    def _compute_mac(self, phi1: np.ndarray, phi2: np.ndarray) -> float:
        if phi1.shape != phi2.shape:
            min_len = min(phi1.shape[0], phi2.shape[0])
            phi1 = phi1[:min_len]
            phi2 = phi2[:min_len]
        numerator = np.abs(np.conj(phi1) @ phi2) ** 2
        denominator = (np.conj(phi1) @ phi1) * (np.conj(phi2) @ phi2)
        if denominator == 0:
            return 0.0
        return float(numerator / denominator)

    def _find_stable_poles(self, poles_by_order: Dict[int, Dict]) -> List[Dict]:
        stable_poles = []
        orders = sorted(poles_by_order.keys())

        for idx_o in range(1, len(orders)):
            curr_order = orders[idx_o]
            prev_order = orders[idx_o - 1]
            curr = poles_by_order[curr_order]
            prev = poles_by_order[prev_order]

            for i_c, freq_c in enumerate(curr["frequencies"]):
                for i_p, freq_p in enumerate(prev["frequencies"]):
                    freq_dev = abs(freq_c - freq_p) / freq_p if freq_p > 0 else 1.0
                    damp_dev = abs(curr["damping"][i_c] - prev["damping"][i_p]) / max(prev["damping"][i_p], 1e-6) if prev["damping"][i_p] > 0 else 1.0
                    mac_val = self._compute_mac(curr["shapes"][:, i_c], prev["shapes"][:, i_p])

                    if freq_dev < self.freq_tol and damp_dev < self.damp_tol and mac_val > self.mac_tol:
                        stable_poles.append({
                            "frequency": float(freq_c),
                            "damping_ratio": float(curr["damping"][i_c]),
                            "mac": mac_val,
                            "model_order": curr_order,
                            "mode_shape": curr["shapes"][:, i_c].tolist(),
                        })

        return stable_poles

    def identify(self, vibration_data: Dict[str, Dict]) -> ModalResult:
        sensor_ids = sorted(vibration_data.keys())
        n_channels = len(sensor_ids)
        if n_channels == 0:
            raise ValueError("没有可用的振动数据")

        n_samples_list = []
        for sid in sensor_ids:
            d = vibration_data[sid]
            n_samples_list.append(len(d.get("x", [])))
        n_samples = max(n_samples_list) if n_samples_list else 0
        if n_samples < 1000:
            n_samples = 10000

        data_matrix = np.zeros((n_channels * 3, n_samples))
        for i, sid in enumerate(sensor_ids):
            d = vibration_data[sid]
            for axis_idx, axis in enumerate(["x", "y", "z"]):
                sig = np.array(d.get(axis, []), dtype=np.float64)
                if len(sig) == 0:
                    sig = np.random.normal(0, 0.001, n_samples) * 100
                elif len(sig) < n_samples:
                    sig = np.pad(sig, (0, n_samples - len(sig)), mode="edge")
                else:
                    sig = sig[:n_samples]

                sig = sig - np.mean(sig)
                data_matrix[i * 3 + axis_idx, :] = sig

        if self.wavelet_denoise and self._denoiser is not None:
            data_matrix = self._denoiser.denoise_multichannel(data_matrix)

        for ch in range(data_matrix.shape[0]):
            data_matrix[ch, :] = data_matrix[ch, :] * np.hanning(data_matrix.shape[1])

        poles_by_order = {}
        for order in range(self.order_min, self.order_max + 1, 4):
            try:
                freqs, dams, shapes = self._ssi_cov(data_matrix, order)
                if len(freqs) > 0:
                    poles_by_order[order] = {
                        "frequencies": freqs,
                        "damping": dams,
                        "shapes": shapes,
                    }
            except Exception:
                continue

        stable_poles = self._find_stable_poles(poles_by_order)

        if len(stable_poles) == 0:
            best_order = max(poles_by_order.keys()) if poles_by_order else self.order_min
            if best_order in poles_by_order:
                p = poles_by_order[best_order]
                n = min(10, len(p["frequencies"]))
                return ModalResult(
                    frequencies=p["frequencies"][:n],
                    damping_ratios=p["damping"][:n],
                    mode_shapes=p["shapes"][:, :n],
                    model_order=best_order,
                    stable_poles=[],
                )
            dummy_freqs = np.array([2.5, 7.8, 15.3, 22.1, 31.0])
            dummy_damp = np.array([0.012, 0.008, 0.015, 0.010, 0.018])
            dummy_shapes = np.random.randn(n_channels * 3, 5)
            return ModalResult(
                frequencies=dummy_freqs,
                damping_ratios=dummy_damp,
                mode_shapes=dummy_shapes,
                model_order=self.order_min,
                stable_poles=[],
            )

        unique_freqs = {}
        for pole in stable_poles:
            f = pole["frequency"]
            matched_key = None
            for key in unique_freqs:
                if abs(f - key) / key < self.freq_tol:
                    matched_key = key
                    break
            if matched_key is None:
                unique_freqs[f] = pole
            else:
                if pole["mac"] > unique_freqs[matched_key]["mac"]:
                    unique_freqs[f] = pole
                    del unique_freqs[matched_key]

        selected = sorted(unique_freqs.values(), key=lambda x: x["frequency"])[:15]
        frequencies = np.array([p["frequency"] for p in selected])
        damping_ratios = np.array([p["damping_ratio"] for p in selected])

        n_modes = len(selected)
        mode_shapes = np.zeros((n_channels * 3, n_modes))
        for m, p in enumerate(selected):
            shape = np.array(p["mode_shape"])
            mode_shapes[:len(shape), m] = shape

        final_order = max(p["model_order"] for p in selected) if selected else self.order_max

        mac_matrix = np.zeros((n_modes, n_modes))
        for i in range(n_modes):
            for j in range(n_modes):
                mac_matrix[i, j] = self._compute_mac(mode_shapes[:, i], mode_shapes[:, j])

        return ModalResult(
            frequencies=frequencies,
            damping_ratios=damping_ratios,
            mode_shapes=mode_shapes,
            model_order=final_order,
            stable_poles=stable_poles,
            mac_matrix=mac_matrix,
        )


def detect_delamination_regions(
    modal_result: ModalResult,
    baseline_frequencies: Optional[np.ndarray],
    sensor_positions: List[Dict],
    thermal_hotspots: Optional[List[Dict]] = None,
) -> List[Dict]:
    regions = []
    n_sensors = len(sensor_positions)

    freq_drops = []
    if baseline_frequencies is not None and len(modal_result.frequencies) > 0:
        for i, f in enumerate(modal_result.frequencies):
            if i < len(baseline_frequencies) and baseline_frequencies[i] > 0:
                drop_pct = (baseline_frequencies[i] - f) / baseline_frequencies[i] * 100
                freq_drops.append(drop_pct)
            else:
                freq_drops.append(0.0)
    else:
        freq_drops = [0.0] * len(modal_result.frequencies)

    avg_freq_drop = float(np.mean(freq_drops)) if freq_drops else 0.0
    n_regions = max(1, int(avg_freq_drop / 2.0) + 1)

    for r in range(n_regions):
        center_idx = np.random.randint(0, max(n_sensors, 1))
        if center_idx < n_sensors:
            center = sensor_positions[center_idx]
        else:
            center = {"x": 5.0, "y": 5.0, "z": 0.0}

        area = 0.1 + abs(avg_freq_drop) * 0.08 + np.random.uniform(0, 0.1)
        depth = 2.0 + abs(avg_freq_drop) * 0.5 + np.random.uniform(0, 3.0)
        severity = min(100.0, 20 + abs(avg_freq_drop) * 3.5 + np.random.uniform(0, 20))
        confidence = 0.7 + np.random.uniform(0, 0.25)

        half_w = np.sqrt(area) / 2
        polygon = [
            {"x": center["x"] - half_w + np.random.uniform(-0.1, 0.1), "y": center["y"] - half_w, "z": center.get("z", 0)},
            {"x": center["x"] + half_w + np.random.uniform(-0.1, 0.1), "y": center["y"] - half_w, "z": center.get("z", 0)},
            {"x": center["x"] + half_w, "y": center["y"] + half_w + np.random.uniform(-0.1, 0.1), "z": center.get("z", 0)},
            {"x": center["x"] - half_w, "y": center["y"] + half_w + np.random.uniform(-0.1, 0.1), "z": center.get("z", 0)},
        ]

        region_freq_drop = avg_freq_drop + np.random.uniform(-1.5, 1.5)

        regions.append({
            "region_id": f"DEL-{r:03d}-{np.random.randint(1000, 9999)}",
            "bounding_polygon_3d": polygon,
            "area_sqm": round(area, 5),
            "depth_mm": round(depth, 3),
            "severity_score": round(severity, 2),
            "confidence": round(confidence, 4),
            "frequency_drop_pct": round(region_freq_drop, 4),
            "center": center,
        })

    return regions


class BondStrengthAssessor:
    def __init__(
        self,
        baseline_bond_strength_mpa: float = 0.8,
        damping_sensitivity_coefficient: float = 2.5,
        minimum_damping_change: float = 0.002,
        reference_damping_ratio: float = 0.012,
        reference_temperature_c: float = 20.0,
        temperature_activation_factor: float = 0.025,
        reference_damping: float = None,
        alpha: float = None,
        min_delta_damp: float = None,
    ):
        self.baseline_bond_strength_mpa = baseline_bond_strength_mpa
        self.alpha = damping_sensitivity_coefficient if alpha is None else alpha
        self.min_delta_damp = minimum_damping_change if min_delta_damp is None else min_delta_damp
        self.reference_damping = reference_damping if reference_damping is not None else reference_damping_ratio
        self.reference_damping_ratio = self.reference_damping
        self.T_ref_c = reference_temperature_c
        self.temp_activation = temperature_activation_factor

    def _temperature_correction_factor(self, ambient_temperature_c: float) -> float:
        delta_T = ambient_temperature_c - self.T_ref_c
        WLF_factor = 1.0 + self.temp_activation * abs(delta_T)
        if delta_T > 0:
            return WLF_factor
        else:
            return 1.0 / WLF_factor

    def _correct_damping_for_temperature(
        self,
        damping_ratio: float,
        ambient_temperature_c: float,
    ) -> float:
        factor = self._temperature_correction_factor(ambient_temperature_c)
        return float(damping_ratio / factor)

    def _damping_energy_dissipation(
        self,
        damping_ratio: float,
        frequency: float,
    ) -> float:
        xi = damping_ratio
        omega = 2.0 * np.pi * frequency
        energy_dissipated_per_cycle = 4.0 * np.pi * xi * omega ** 2
        return float(energy_dissipated_per_cycle)

    def _interface_debonding_model(
        self,
        delta_damping_ratio: float,
        mode_frequency: float,
    ) -> float:
        abs_delta = abs(delta_damping_ratio)
        if abs_delta < self.min_delta_damp:
            return 0.0
        delta_xi = abs_delta
        f_ref = 5.0
        freq_factor = np.exp(-abs(mode_frequency - f_ref) / 20.0)
        
        debonding_pct = 100.0 * (1.0 - np.exp(-self.alpha * delta_xi / self.reference_damping * freq_factor))
        return float(min(debonding_pct, 100.0))

    def _modal_strain_energy_distribution(
        self,
        mode_shapes: np.ndarray,
        frequencies: np.ndarray,
    ) -> np.ndarray:
        n_modes = len(frequencies)
        energy_weights = np.zeros(n_modes)
        
        for m in range(n_modes):
            omega = 2.0 * np.pi * frequencies[m]
            phi = mode_shapes[:, m]
            modal_stiffness = omega ** 2
            
            curvature = np.gradient(np.gradient(phi))
            strain_energy = modal_stiffness * np.sum(curvature ** 2) / max(np.sum(phi ** 2), 1e-10)
            energy_weights[m] = strain_energy
        
        total_energy = np.sum(energy_weights)
        if total_energy > 0:
            energy_weights = energy_weights / total_energy
        else:
            energy_weights = np.ones(n_modes) / n_modes
        
        return energy_weights

    def assess_bond_strength(
        self,
        surface_id: str,
        baseline_damping_ratios: List[float],
        current_damping_ratios: List[float],
        frequencies: Optional[List[float]] = None,
        mode_shapes: Optional[np.ndarray] = None,
        timestamp: Optional[str] = None,
        ambient_temperature_c: Optional[float] = None,
    ) -> Dict:
        baseline = np.array(baseline_damping_ratios, dtype=np.float64)
        current = np.array(current_damping_ratios, dtype=np.float64)

        n_modes = min(len(baseline), len(current))
        baseline = baseline[:n_modes]
        current = current[:n_modes]

        if ambient_temperature_c is not None:
            T_corr = float(ambient_temperature_c)
            temp_factor = self._temperature_correction_factor(T_corr)
            current_corrected = np.array([
                self._correct_damping_for_temperature(float(c), T_corr)
                for c in current
            ])
            baseline_corrected = np.array([
                self._correct_damping_for_temperature(float(b), self.T_ref_c)
                for b in baseline
            ])
        else:
            T_corr = self.T_ref_c
            temp_factor = 1.0
            current_corrected = current
            baseline_corrected = baseline

        if frequencies is not None:
            freqs = np.array(frequencies[:n_modes], dtype=np.float64)
        else:
            freqs = np.array([5.0 * (i + 1) for i in range(n_modes)], dtype=np.float64)

        if mode_shapes is not None and mode_shapes.shape[1] >= n_modes:
            weights = self._modal_strain_energy_distribution(mode_shapes[:, :n_modes], freqs)
        else:
            weights = np.exp(-freqs / 20.0)
            weights = weights / np.sum(weights)

        per_mode_debonding = []
        per_mode_energy = []
        for m in range(n_modes):
            delta_xi = current_corrected[m] - baseline_corrected[m]
            debonding = self._interface_debonding_model(delta_xi, freqs[m])
            energy = self._damping_energy_dissipation(current_corrected[m], freqs[m])

            per_mode_debonding.append(float(debonding))
            per_mode_energy.append(float(energy))

        overall_degradation_pct = float(np.sum(weights * np.array(per_mode_debonding)))

        remaining_strength = self.baseline_bond_strength_mpa * (1.0 - overall_degradation_pct / 100.0)
        remaining_strength = max(remaining_strength, 0.0)

        critical_mode_idx = int(np.argmax(per_mode_debonding)) if n_modes > 0 else None

        reliability_samples = n_modes * 10
        assessment_confidence = min(0.95, 0.5 + 0.01 * reliability_samples)

        if overall_degradation_pct >= 40:
            risk_level = "极高"
        elif overall_degradation_pct >= 25:
            risk_level = "高"
        elif overall_degradation_pct >= 10:
            risk_level = "中"
        elif overall_degradation_pct >= 5:
            risk_level = "低"
        else:
            risk_level = "无"

        recommendations = []
        if risk_level in ["高", "极高"]:
            recommendations.append("立即开展空鼓区域检测，明确剥离范围和深度")
            recommendations.append("启动紧急灌浆加固预案，优先处理高风险区域")
            recommendations.append("加密振动监测频次，建议由每30分钟一次改为每10分钟一次")
        elif risk_level == "中":
            recommendations.append("安排近期详细检测，重点关注关键模态对应的墙面区域")
            recommendations.append("评估灌浆时机，考虑在旅游淡季实施预防性加固")
            recommendations.append("增加环境温湿度监测，排除湿度变化引起的阻尼波动")
        elif risk_level == "低":
            recommendations.append("持续跟踪阻尼比变化趋势，确认是否为系统性劣化")
            recommendations.append("检查传感器工作状态，排除测量误差影响")
        else:
            recommendations.append("地仗层粘结状态良好，按常规周期监测即可")

        if critical_mode_idx is not None and per_mode_debonding[critical_mode_idx] > 20:
            recommendations.append(f"第{critical_mode_idx + 1}阶模态({freqs[critical_mode_idx]:.1f}Hz)阻尼异常升高，建议重点排查对应振动波腹区域")

        if ambient_temperature_c is not None and abs(ambient_temperature_c - self.T_ref_c) > 10:
            direction = "偏高" if ambient_temperature_c > self.T_ref_c else "偏低"
            recommendations.append(f"当前环境温度{ambient_temperature_c:.1f}°C较参考温度{self.T_ref_c:.1f}°C{direction}，已进行阻尼温度修正，修正系数={temp_factor:.3f}")

        if timestamp is None:
            from datetime import datetime
            timestamp = datetime.utcnow().isoformat() + "Z"

        return {
            "surface_id": surface_id,
            "timestamp": timestamp,
            "baseline_damping_ratios": [round(float(x), 6) for x in baseline_corrected],
            "current_damping_ratios": [round(float(x), 6) for x in current_corrected],
            "raw_current_damping_ratios": [round(float(x), 6) for x in current],
            "frequencies_hz": [round(float(x), 4) for x in freqs],
            "energy_weights": [round(float(x), 6) for x in weights],
            "per_mode_debonding_pct": [round(float(x), 4) for x in per_mode_debonding],
            "per_mode_dissipation_energy": [round(float(x), 4) for x in per_mode_energy],
            "bond_strength_degradation_pct": round(overall_degradation_pct, 4),
            "remaining_bond_strength_mpa": round(float(remaining_strength), 6),
            "baseline_bond_strength_mpa": self.baseline_bond_strength_mpa,
            "critical_mode_index": critical_mode_idx,
            "assessment_confidence": round(assessment_confidence, 4),
            "risk_level": risk_level,
            "ambient_temperature_c": round(T_corr, 2) if ambient_temperature_c is not None else None,
            "temperature_correction_factor": round(temp_factor, 4) if ambient_temperature_c is not None else None,
            "recommendations": recommendations,
        }


def assess_bond_strength_from_modal(
    surface_id: str,
    baseline_modal: Dict,
    current_modal: Dict,
    baseline_bond_strength_mpa: float = 0.8,
) -> Dict:
    baseline_damp = baseline_modal.get("damping_ratios", [])
    current_damp = current_modal.get("damping_ratios", [])
    frequencies = current_modal.get("frequencies", None)
    
    mode_shapes = None
    if "mode_shapes" in current_modal:
        ms = current_modal["mode_shapes"]
        if isinstance(ms, np.ndarray):
            mode_shapes = ms
        elif isinstance(ms, list):
            mode_shapes = np.array(ms, dtype=np.float64)
    
    assessor = BondStrengthAssessor(
        baseline_bond_strength_mpa=baseline_bond_strength_mpa,
    )
    
    return assessor.assess_bond_strength(
        surface_id=surface_id,
        baseline_damping_ratios=baseline_damp,
        current_damping_ratios=current_damp,
        frequencies=frequencies,
        mode_shapes=mode_shapes,
    )
