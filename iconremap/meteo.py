"""Meteorological constants and physical extrapolation helpers."""
import numpy as np

# Standard meteorological constants
G       = 9.80665      # m/s² gravitational acceleration
R_D     = 287.04       # J/(kg·K) gas constant for dry air
R_V     = 461.50       # J/(kg·K) gas constant for water vapour
CP_D    = 1004.64      # J/(kg·K) specific heat at constant pressure
KAPPA   = R_D / CP_D   # ≈ 0.2857
P0      = 100000.0     # Pa reference pressure for theta
LAPSE_STD = 6.5e-3     # K/m standard atmosphere lapse rate
EPS_VAP = R_D / R_V    # ≈ 0.622


def extrapolate_temperature(t_bottom: np.ndarray, dz: np.ndarray) -> np.ndarray:
    """Extrapolate temperature using standard lapse rate.

    dz > 0 means we want T at altitude *higher* than t_bottom by dz metres
    (going up the atmosphere → temperature decreases).
    dz < 0 means going below source surface (hypothetical) → temperature increases.
    """
    return t_bottom - LAPSE_STD * dz


def extrapolate_pressure(p_bottom: np.ndarray, t_mean: np.ndarray, dz: np.ndarray) -> np.ndarray:
    """Hydrostatic pressure extrapolation.

    p_target = p_bottom * exp(-g * dz / (R_d * T_mean))
    dz > 0: going up → pressure decreases.
    """
    return p_bottom * np.exp(-G * dz / (R_D * t_mean))


def extrapolate_qv(qv_bottom: np.ndarray, p_bottom: np.ndarray, p_target: np.ndarray) -> np.ndarray:
    """Specific humidity extrapolation by constant relative humidity assumption.

    Approximation: qv scales with pressure ratio (preserves mixing ratio).
    Floor at zero.
    """
    return np.maximum(qv_bottom * (p_target / p_bottom), 0.0)


def saturation_vapor_pressure(t: np.ndarray) -> np.ndarray:
    """Tetens formula, hPa."""
    return 6.112 * np.exp(17.67 * (t - 273.15) / (t - 29.65))
