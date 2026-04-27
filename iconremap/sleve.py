"""SLEVE vertical coordinate computation, transcribed from ICON's mo_init_vgrid.f90.

We bypass the layout-algorithm step (init_sleve_coord) by reading vct_a values
that ICON itself computed and printed to its run log. This guarantees an
exact match with what ICON's runtime computes — eliminating the source-target
height mismatch that crashed vert_interp_atm.

Then we apply the terrain-following transformation (init_vert_coord SLEVE branch)
to get per-cell z_ifc.
"""
import numpy as np


# vct_a values for our specific NAMELIST_lam.in config — extracted verbatim
# from icon.log "Nominal heights of coordinate half levels" output.
# 61 entries for nlev=60. jk=1 is model top, jk=61 is surface.
VCT_A = np.array([
    23000.000, 17484.248, 14268.446, 13868.446, 13468.446, 13068.446,
    12668.446, 12268.446, 11868.446, 11468.446, 11068.446, 10668.446,
    10268.446,  9868.446,  9468.446,  9068.446,  8668.446,  8268.446,
     7868.446,  7479.708,  7107.087,  6749.570,  6406.256,  6076.342,
     5759.108,  5453.904,  5160.146,  4877.301,  4604.888,  4342.467,
     4089.638,  3846.034,  3611.320,  3385.191,  3167.364,  2957.585,
     2755.618,  2561.249,  2374.283,  2194.544,  2021.874,  1856.129,
     1697.186,  1544.934,  1399.282,  1260.153,  1127.491,  1001.255,
      881.428,   768.012,   661.039,   560.568,   466.698,   379.575,
      299.405,   226.482,   161.226,   104.259,    56.575,    20.000,
        0.000,
], dtype=np.float64)

# SLEVE namelist parameters from forecast/NAMELIST_lam.in
TOP_HEIGHT     = 23000.0
DECAY_SCALE_1  =  4000.0
DECAY_SCALE_2  =  2500.0
DECAY_EXP      =     1.2
FLAT_HEIGHT    = 16000.0


def find_nflat(vct_a: np.ndarray = VCT_A, flat_height: float = FLAT_HEIGHT) -> int:
    """Return nflat: largest jk where vct_a[jk] >= flat_height.

    Levels jk=1..nflat are flat (no terrain influence); jk=nflat+1..nlev are SLEVE.
    Index returned is 1-based (matches ICON convention).
    """
    # vct_a indexed 0..nlev (Python). ICON jk starts at 1.
    above = np.where(vct_a >= flat_height)[0]
    if len(above) == 0:
        return 0
    return int(above[-1] + 1)   # convert 0-based index to 1-based


def compute_z_ifc(
    topo: np.ndarray,
    topo_smt: np.ndarray | None = None,
    vct_a: np.ndarray = VCT_A,
    *,
    top_height: float = TOP_HEIGHT,
    decay_scale_1: float = DECAY_SCALE_1,
    decay_scale_2: float = DECAY_SCALE_2,
    decay_exp: float = DECAY_EXP,
    flat_height: float = FLAT_HEIGHT,
) -> np.ndarray:
    """Compute SLEVE z_ifc for one or many cells.

    topo:     (ncell,) full surface elevation [m]
    topo_smt: (ncell,) smoothed surface elevation. If None, set = topo.
    vct_a:    (nlev+1,) reference vertical coordinate (model-top-down)

    Returns z_ifc shape (nlev+1, ncell), top-down ordered (index 0 = model top).
    """
    if topo_smt is None:
        topo_smt = topo

    nlevp1 = vct_a.shape[0]
    nlev = nlevp1 - 1
    ncell = topo.shape[0]
    nflat = find_nflat(vct_a, flat_height)
    z_ifc = np.empty((nlevp1, ncell), dtype=np.float64)

    # Flat region: jk=1..nflat (1-based) → indices 0..nflat-1 (0-based)
    for jk in range(nflat):
        z_ifc[jk, :] = vct_a[jk]

    # Terrain following: jk=nflat+1..nlev (1-based) → indices nflat..nlev-1 (0-based)
    H = top_height
    p = decay_exp
    sinh_d1 = np.sinh((H / decay_scale_1) ** p)
    sinh_d2 = np.sinh((H / decay_scale_2) ** p)
    z_topo_dev = topo - topo_smt          # small-scale topography

    for jk in range(nflat, nlev):  # 0-based; covers ICON jk = nflat+1..nlev
        a = vct_a[jk]
        z_fac1 = (np.sinh((H / decay_scale_1) ** p - (a / decay_scale_1) ** p)) / sinh_d1
        z_fac2 = (np.sinh((H / decay_scale_2) ** p - (a / decay_scale_2) ** p)) / sinh_d2
        z_ifc[jk, :] = a + topo_smt * z_fac1 + z_topo_dev * z_fac2

    # Surface (jk = nlev+1, 0-based index nlev)
    z_ifc[nlev, :] = topo

    # Layer-thickness limiter — minimum thickness preventing instabilities.
    # Transcribed from mo_init_vgrid.f90 lines 489-516.
    dvct1, dvct2 = 100.0, 500.0
    minrat1 = 1.0 / 3.0
    minrat2 = 0.5
    min_lay_thckn_param = 20.0  # from sleve_nml min_lay_thckn

    for jk in range(nlev - 1, -1, -1):     # nlev-1 down to 0
        dvct = vct_a[jk] - vct_a[jk + 1]
        if dvct < dvct1:
            min_lay_spacing = minrat1 * dvct
        elif dvct < dvct2:
            wfac = ((dvct2 - dvct) / (dvct2 - dvct1)) ** 2
            min_lay_spacing = (minrat1 * wfac + minrat2 * (1.0 - wfac)) * dvct
        else:
            min_lay_spacing = minrat2 * dvct2 * (dvct / dvct2) ** (1.0 / 3.0)
        min_lay_spacing = max(min_lay_spacing, min(50.0, min_lay_thckn_param))
        # For each cell, if spacing too small, push up
        too_thin = z_ifc[jk + 1, :] + min_lay_spacing > z_ifc[jk, :]
        z_ifc[jk, too_thin] = z_ifc[jk + 1, too_thin] + min_lay_spacing

    return z_ifc
