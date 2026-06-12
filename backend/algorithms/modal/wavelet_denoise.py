import numpy as np
from typing import Optional


class WaveletThresholdDenoiser:
    def __init__(
        self,
        wavelet: str = "db8",
        level: int = 5,
        mode: str = "soft",
        threshold_method: str = "rigrsure",
    ):
        self.wavelet = wavelet
        self.level = level
        self.mode = mode
        self.threshold_method = threshold_method
        self._pywt = None

    def _ensure_pywt(self):
        if self._pywt is None:
            try:
                import pywt
                self._pywt = pywt
            except ImportError:
                raise ImportError(
                    "pywt未安装，请执行: pip install PyWavelets"
                )

    def _compute_threshold(self, detail_coeffs: np.ndarray) -> float:
        n = len(detail_coeffs)
        if n == 0:
            return 0.0

        sigma = np.median(np.abs(detail_coeffs)) / 0.6745

        if self.threshold_method == "universal":
            return sigma * np.sqrt(2 * np.log(n))
        elif self.threshold_method == "rigrsure":
            sorted_coeffs = np.sort(np.abs(detail_coeffs) ** 2)
            risks = np.zeros(n)
            for k in range(n):
                risk = (n - 2 * (k + 1) + np.sum(sorted_coeffs[: k + 1]) +
                        (n - k - 1) * sorted_coeffs[k]) / n
                risks[k] = risk
            best_k = np.argmin(risks)
            threshold = np.sqrt(sorted_coeffs[best_k])
            return float(threshold)
        elif self.threshold_method == "heursure":
            universal = sigma * np.sqrt(2 * np.log(n))
            eta = (np.sum(detail_coeffs ** 2) - n) / n
            mu = (np.log2(n) ** 1.5) / np.sqrt(n)
            if eta < mu:
                return universal
            else:
                rigrsure_thr = self._compute_threshold_rigrsure(detail_coeffs, sigma, n)
                return min(universal, rigrsure_thr)
        elif self.threshold_method == "sqtwolog":
            return sigma * np.sqrt(2 * np.log(n))
        else:
            return sigma * np.sqrt(2 * np.log(n))

    def _compute_threshold_rigrsure(self, detail_coeffs, sigma, n):
        sorted_coeffs = np.sort(np.abs(detail_coeffs) ** 2)
        risks = np.zeros(n)
        for k in range(n):
            risk = (n - 2 * (k + 1) + np.sum(sorted_coeffs[: k + 1]) +
                    (n - k - 1) * sorted_coeffs[k]) / n
            risks[k] = risk
        best_k = np.argmin(risks)
        return float(np.sqrt(sorted_coeffs[best_k]))

    def _apply_threshold(
        self, coeffs: list, threshold: float
    ) -> list:
        denoised = [coeffs[0]]
        for i in range(1, len(coeffs)):
            detail = coeffs[i]
            if self.mode == "soft":
                denoised_detail = np.sign(detail) * np.maximum(
                    np.abs(detail) - threshold, 0
                )
            else:
                denoised_detail = np.where(
                    np.abs(detail) > threshold, detail, 0
                )
            denoised.append(denoised_detail)
        return denoised

    def denoise(self, signal: np.ndarray) -> np.ndarray:
        self._ensure_pywt()
        pywt = self._pywt

        if len(signal) < 2 ** self.level:
            self.level = max(1, int(np.log2(len(signal))) - 1)

        coeffs = pywt.wavedec(signal, self.wavelet, level=self.level)

        detail_coeffs_all = np.concatenate(coeffs[1:])
        threshold = self._compute_threshold(detail_coeffs_all)

        denoised_coeffs = self._apply_threshold(coeffs, threshold)

        reconstructed = pywt.waverec(denoised_coeffs, self.wavelet)

        if len(reconstructed) > len(signal):
            reconstructed = reconstructed[: len(signal)]
        elif len(reconstructed) < len(signal):
            reconstructed = np.pad(
                reconstructed, (0, len(signal) - len(reconstructed)), mode="edge"
            )

        return reconstructed

    def denoise_multichannel(self, data_matrix: np.ndarray) -> np.ndarray:
        n_channels, n_samples = data_matrix.shape
        denoised = np.zeros_like(data_matrix)
        for ch in range(n_channels):
            sig = data_matrix[ch, :]
            if np.std(sig) < 1e-12:
                denoised[ch, :] = sig
                continue
            denoised[ch, :] = self.denoise(sig)
        return denoised


def preprocess_vibration_signal(
    signal: np.ndarray,
    fs: float = 2000.0,
    wavelet: str = "db8",
    level: int = 5,
    mode: str = "soft",
    threshold_method: str = "rigrsure",
    detrend: bool = True,
    bandpass: Optional[tuple] = None,
) -> np.ndarray:
    processed = signal.copy()

    if detrend:
        processed = processed - np.mean(processed)
        t = np.arange(len(processed))
        if len(processed) > 10:
            coeffs = np.polyfit(t, processed, 1)
            processed = processed - np.polyval(coeffs, t)

    if bandpass is not None:
        try:
            from scipy.signal import butter, filtfilt
            low, high = bandpass
            nyq = fs / 2.0
            low_n = max(low / nyq, 0.001)
            high_n = min(high / nyq, 0.999)
            if low_n < high_n:
                b, a = butter(4, [low_n, high_n], btype="band")
                processed = filtfilt(b, a, processed)
        except Exception:
            pass

    denoiser = WaveletThresholdDenoiser(
        wavelet=wavelet,
        level=level,
        mode=mode,
        threshold_method=threshold_method,
    )
    processed = denoiser.denoise(processed)

    return processed
