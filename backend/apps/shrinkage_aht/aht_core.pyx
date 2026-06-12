# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as np
from libc.math cimport exp, sqrt


cdef double AHT_BETA = 0.15
cdef double AHT_GAMMA = 0.6
cdef double HUMIDITY_SENSITIVITY = 0.8
cdef double TEMPERATURE_ACTIVATION_ENERGY = 35000.0
cdef double R_GAS_CONSTANT = 8.314
cdef double T_REF_K = 293.15
cdef double RH_REF = 0.6


cdef inline double dmax(double a, double b):
    return a if a > b else b


cpdef double arrhenius_temperature_factor(double T_c, double T_ref, double Ea, double R):
    cdef double T = T_c + 273.15
    return exp(Ea / R * (1.0 / T_ref - 1.0 / T))


cpdef double humidity_diffusion_factor(double RH_pct, double sensitivity):
    cdef double RH = RH_pct / 100.0
    cdef double factor
    if RH >= RH_REF:
        factor = 1.0 - sensitivity * (RH - RH_REF)
    else:
        factor = 1.0 + 2.0 * sensitivity * (RH_REF - RH)
    return dmax(factor, 0.1)


cpdef double aht_final_shrinkage(double alpha, double beta, double gamma, double time_days, double T_c, double RH_pct, double constraint_factor):
    cdef double t_eff = time_days
    cdef double T_factor = arrhenius_temperature_factor(T_c, T_REF_K, TEMPERATURE_ACTIVATION_ENERGY, R_GAS_CONSTANT)
    cdef double RH_factor = humidity_diffusion_factor(RH_pct, HUMIDITY_SENSITIVITY)

    t_eff = t_eff * T_factor * RH_factor

    if t_eff <= 0:
        return 0.0

    cdef double a = 1.0 - exp(-beta * (t_eff ** gamma))
    return a * constraint_factor


cpdef np.ndarray[np.float64_t, ndim=1] compute_alpha_curve(np.ndarray[np.float64_t, ndim=1] time_days,
                                                          double T_c, double RH_pct,
                                                          double initial_curing_days,
                                                          double drying_rate_factor):
    cdef int n = time_days.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] alpha = np.empty(n, dtype=np.float64)
    cdef int i
    cdef double t_eq, T_factor, RH_factor, t_eff

    T_factor = arrhenius_temperature_factor(T_c, T_REF_K, TEMPERATURE_ACTIVATION_ENERGY, R_GAS_CONSTANT)
    RH_factor = humidity_diffusion_factor(RH_pct, HUMIDITY_SENSITIVITY)

    for i in range(n):
        t_eq = dmax(time_days[i] - initial_curing_days, 0.0)
        t_eff = t_eq * T_factor * RH_factor * drying_rate_factor
        if t_eff <= 0:
            alpha[i] = 0.0
        else:
            alpha[i] = 1.0 - exp(-AHT_BETA * (t_eff ** AHT_GAMMA))

    return alpha
