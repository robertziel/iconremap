"""Topography-aware vertical interpolation: source profile → target profile.

Algorithm (per cell):
  1. Source has values f_src[level] at heights z_src[level] (decreasing top→bottom).
  2. Target needs values at heights z_tgt[level].
  3. For each target level, find bracketing source levels and linear-interp in z.
  4. If target_z below source surface (z_tgt < z_src[bottom]): extrapolate down
     using physical scheme per variable type.
  5. If target_z above source top: extrapolate up via constant gradient (rare).

Vector operations over (time, lev, cell). All in numpy.
"""
import numpy as np
from . import meteo


# Variable categories drive extrapolation strategy below source surface.
VAR_CATEGORY = {
    "T":  "temperature",
    "P":  "pressure",
    "QV": "qv",
    "U":  "wind",
    "V":  "wind",
    "W":  "wind",
    "QC": "zero_below",
    "QI": "zero_below",
    "QR": "zero_below",
    "QS": "zero_below",
}


def _full_levels_from_half(hhl: np.ndarray) -> np.ndarray:
    """hhl shape (time, nlev_intf, cell) → full levels shape (time, nlev, cell)."""
    return 0.5 * (hhl[:, :-1, :] + hhl[:, 1:, :])


def interpolate_field(
    field_src: np.ndarray,
    z_src_full: np.ndarray,
    z_tgt_full: np.ndarray,
    var_name: str,
    *,
    p_src_full: np.ndarray | None = None,
    t_src_full: np.ndarray | None = None,
) -> np.ndarray:
    """Interpolate one (time, lev, cell) field from source to target heights.

    p_src_full, t_src_full needed only for pressure/qv extrapolation paths.
    """
    nt, nlev_src, ncell = field_src.shape
    nlev_tgt = z_tgt_full.shape[1]
    out = np.empty((nt, nlev_tgt, ncell), dtype=field_src.dtype)
    category = VAR_CATEGORY.get(var_name, "wind")

    # ICON convention: level 0 = top, level nlev-1 = bottom (heights descending)
    # np.interp requires xp ascending; we work with reversed arrays per-cell.

    for t_i in range(nt):
        for c in range(ncell):
            zs = z_src_full[t_i, ::-1, c]   # ascending (bottom → top)
            fs = field_src[t_i, ::-1, c]
            zt = z_tgt_full[t_i, ::-1, c]   # ascending

            zs_top, zs_bot = zs[-1], zs[0]
            in_range = (zt >= zs_bot) & (zt <= zs_top)

            # In-range interpolation
            interp_vals = np.interp(zt, zs, fs)

            # Below source surface — physical extrapolation
            below = zt < zs_bot
            if below.any():
                f_b = fs[0]                 # field at source bottom
                z_b = zs[0]                 # source bottom altitude
                dz_below = z_b - zt[below]  # positive distances below source surf

                if category == "temperature":
                    interp_vals[below] = meteo.extrapolate_temperature(f_b, -dz_below)
                elif category == "pressure":
                    if t_src_full is None:
                        raise ValueError("t_src_full required for pressure extrap")
                    t_b = t_src_full[t_i, -1, c]
                    interp_vals[below] = meteo.extrapolate_pressure(f_b, t_b, -dz_below)
                elif category == "qv":
                    if p_src_full is None:
                        raise ValueError("p_src_full required for qv extrap")
                    p_b = p_src_full[t_i, -1, c]
                    # use hydrostatic to get p_target locally
                    if t_src_full is not None:
                        t_b = t_src_full[t_i, -1, c]
                        p_t = meteo.extrapolate_pressure(p_b, t_b, -dz_below)
                    else:
                        p_t = p_b * np.exp(dz_below * meteo.G / (meteo.R_D * 280.0))
                    interp_vals[below] = meteo.extrapolate_qv(f_b, p_b, p_t)
                elif category == "zero_below":
                    interp_vals[below] = 0.0
                else:  # wind
                    # Log-law surface-layer decay below source bottom. Without this,
                    # ICON-EU's lowest model level (~1700 m AGL) propagates its
                    # free-troposphere wind down to the LAM PBL, producing a
                    # spurious 70+ km/h surface wind at h=0 and a 175 km/h gust10
                    # spike that the max-over-window accumulator carries forward.
                    # u(z) = u(z_b) · ln((z+z0)/z0) / ln((z_b+z0)/z0), z0 = 0.1 m.
                    # Hard-cap at ±25 m/s per component as defense against extreme
                    # jet-stream values in the source bottom level that the log-law
                    # alone doesn't fully tame.
                    z0 = 0.1
                    z_abs = np.maximum(z_b - dz_below, 0.0)
                    log_ratio = np.log((z_abs + z0) / z0) / np.log((z_b + z0) / z0)
                    val = f_b * np.clip(log_ratio, 0.0, 1.0)
                    interp_vals[below] = np.sign(val) * np.minimum(np.abs(val), 25.0)

            # Above source top — physical extrapolation per variable category.
            # Constant copy (legacy) leaves P unchanged from source top to LAM
            # top, producing inconsistent (T, P) pairs that crash Sundqvist's
            # qv/qsat. Use hydrostatic-aware decay so cells stay inside ICON's
            # e_s lookup range. DWD opendata cuts off at ~17 km; LAM top is at
            # 23 km, so the extrapolated zone is real and ~6 km thick.
            above = zt > zs_top
            if above.any():
                f_t = fs[-1]
                z_above = zt[above] - zs_top   # positive distance above source top
                if category == "pressure":
                    if t_src_full is not None:
                        t_top = t_src_full[t_i, -1, c]
                    else:
                        t_top = 215.0   # typical lower stratosphere
                    # Hydrostatic exponential decay; H = R_d·T/g ≈ 6.3 km @ T=215K
                    interp_vals[above] = meteo.extrapolate_pressure(f_t, t_top, z_above)
                elif category == "qv":
                    # Stratospheric water vapor decays with H_qv ≈ 2 km (Brewer-Dobson)
                    interp_vals[above] = f_t * np.exp(-z_above / 2000.0)
                elif category == "zero_below":
                    # QC/QI/QR/QS: vanish in stratosphere
                    interp_vals[above] = 0.0
                elif category == "temperature":
                    # Isothermal stratosphere (good approx for 17-23 km)
                    interp_vals[above] = f_t
                else:  # wind
                    # Winds in lower stratosphere ~similar to tropopause winds
                    interp_vals[above] = f_t

            # write back in descending-z (ICON-standard) order
            out[t_i, :, c] = interp_vals[::-1]

    return out


def shift_hhl_to_target_topo(
    hhl_src: np.ndarray, hsurf_src: np.ndarray, hsurf_tgt: np.ndarray
) -> np.ndarray:
    """Compute target HHL by terrain-following shift.

    hhl_src:   (time, nlev_intf, cell)  source half-level heights
    hsurf_src: (cell,)                  source surface = hhl_src[:, -1, :].mean(0)
    hsurf_tgt: (cell,)                  target surface from EXTPAR

    Returns: hhl_tgt with same shape as hhl_src, shifted so that
             hhl_tgt[:, -1, :] == hsurf_tgt
    """
    dhsurf = hsurf_tgt - hsurf_src                    # (cell,)
    return hhl_src + dhsurf[np.newaxis, np.newaxis, :]
