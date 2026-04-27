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
                    interp_vals[below] = f_b

            # Above source top — constant gradient (rare; near 23 km top of model)
            above = zt > zs_top
            if above.any():
                f_t = fs[-1]
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
