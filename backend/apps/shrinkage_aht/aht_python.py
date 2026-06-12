import numpy as np
from math import exp, sqrt


AHT_BETA = 0.15
AHT_GAMMA = 0.6
HUMIDITY_SENSITIVITY = 0.8
TEMPERATURE_ACTIVATION_ENERGY = 35000.0
R_GAS_CONSTANT = 8.314
T_REF_K = 293.15
RH_REF = 0.6


def arrhenius_temperature_factor(T_c, T_ref, Ea, R):
    T = T_c + 273.15
    return exp(Ea / R * (1.0 / T_ref - 1.0 / T))


def humidity_diffusion_factor(RH_pct, sensitivity):
    RH = RH_pct / 100.0
    if RH >= RH_REF:
        factor = 1.0 - sensitivity * (RH - RH_REF)
    else:
        factor = 1.0 + 2.0 * sensitivity * (RH_REF - RH)
    return max(factor, 0.1)


def aht_final_shrinkage(alpha, beta, gamma, time_days, T_c, RH_pct, constraint_factor):
    t_eff = time_days
    T_factor = arrhenius_temperature_factor(T_c, T_REF_K, TEMPERATURE_ACTIVATION_ENERGY, R_GAS_CONSTANT)
    RH_factor = humidity_diffusion_factor(RH_pct, HUMIDITY_SENSITIVITY)
    
    t_eff = t_eff * T_factor * RH_factor
    
    if t_eff <= 0:
        return 0.0
    
    a = 1.0 - exp(-beta * (t_eff ** gamma))
    return a * constraint_factor


def compute_alpha_curve(time_days, T_c, RH_pct, initial_curing_days, drying_rate_factor):
    n = len(time_days)
    alpha = np.empty(n, dtype=np.float64)
    
    T_factor = arrhenius_temperature_factor(T_c, T_REF_K, TEMPERATURE_ACTIVATION_ENERGY, R_GAS_CONSTANT)
    RH_factor = humidity_diffusion_factor(RH_pct, HUMIDITY_SENSITIVITY)
    
    for i in range(n):
        t_eq = max(time_days[i] - initial_curing_days, 0.0)
        t_eff = t_eq * T_factor * RH_factor * drying_rate_factor
        if t_eff <= 0:
            alpha[i] = 0.0
        else:
            alpha[i] = 1.0 - exp(-AHT_BETA * (t_eff ** AHT_GAMMA))
    
    return alpha
