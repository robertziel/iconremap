"""End-to-end IC remap pipeline.

Input:  CDO-remapped IC NetCDF (atm fields on source HHL) + LAM EXTPAR
Output: IC NetCDF with atm fields vertically interpolated onto target z_ifc
        (where target z_ifc derives from terrain-following shift of source HHL
         to LAM topography from EXTPAR).
"""
import sys
import time as _time
import numpy as np
import xarray as xr

from . import io as ic_io
from . import vertical as vert
from . import meteo


def remap_ic(input_nc: str, extpar_nc: str, output_nc: str, *, verbose: bool = True) -> None:
    t0 = _time.time()
    if verbose:
        print(f"[iconremap] reading IC: {input_nc}")
    ds = ic_io.read_ic(input_nc)

    if verbose:
        print(f"[iconremap] reading EXTPAR HSURF: {extpar_nc}")
    hsurf_tgt = ic_io.read_extpar_hsurf(extpar_nc)

    # HHL is (time, lev_intf, cell) — cell count must match EXTPAR HSURF length.
    hhl_src = ds["HHL"].values
    if hhl_src.ndim == 2:
        hhl_src = hhl_src[np.newaxis, ...]    # add time dim
    nt, nlev_intf, ncell = hhl_src.shape

    if hsurf_tgt.shape[0] != ncell:
        raise ValueError(
            f"HSURF cell count ({hsurf_tgt.shape[0]}) does not match HHL cell count ({ncell})"
        )

    # KEEP source HHL as the ACTUAL source heights (DWD ICON-EU's published HHL).
    # Compute TARGET z_ifc independently via SLEVE for our LAM grid. Source and
    # target may have DIFFERENT level counts (DWD now publishes 65 atm layers / 66
    # half-levels; LAM uses 60 / 61) — interpolation is real cross-grid mapping.
    from . import sleve
    hhl_orig = hhl_src.copy()                          # (time, src_nlev_intf, cell) — DWD HHL
    z_ifc_target = sleve.compute_z_ifc(topo=hsurf_tgt) # (tgt_nlev_intf, cell) — LAM SLEVE
    if verbose:
        zsm = hhl_orig[:, -1, :].mean()
        ztm = z_ifc_target[-1, :].mean()
        print(f"[iconremap] source HHL: {hhl_orig.shape[1]} half-levels, "
              f"top={hhl_orig[:, 0, :].mean():.1f}m, bottom={zsm:.1f}m")
        print(f"[iconremap] target z_ifc: {z_ifc_target.shape[0]} half-levels, "
              f"top={z_ifc_target[0, :].mean():.1f}m, bottom={ztm:.1f}m (≡ HSURF)")
        if abs(zsm - ztm) > 50.0:
            print(f"[iconremap]   bottom mismatch {zsm - ztm:.1f} m → some cells need extrap")
    # Terrain-following shift on source HHL: anchor it to LAM HSURF so source
    # and target are in the SAME AGL coordinate system. Otherwise interpolation
    # is in absolute height and over mountains (where LAM HSURF >> DWD HSURF)
    # asks "what's the wind at 1879 m absolute (= 10 m AGL above LAM Tatra)?"
    # while source data only goes up to ~1307 m absolute (= 10 m AGL above DWD's
    # smoothed Tatra) — extrapolates upward into faster free-atmosphere wind,
    # producing physically wrong "10 m AGL" values 2× too high over orography.
    nt = hhl_orig.shape[0]
    ncell = hhl_orig.shape[2]
    tgt_nlev_intf = z_ifc_target.shape[0]
    hsurf_src = hhl_orig[:, -1, :].mean(axis=0)
    hhl_src = vert.shift_hhl_to_target_topo(hhl_orig, hsurf_src, hsurf_tgt)
    hhl_tgt = np.broadcast_to(z_ifc_target[None, :, :], (nt, tgt_nlev_intf, ncell)).copy()

    if verbose:
        dh = hsurf_tgt - hsurf_src
        print(f"[iconremap] HSURF shift LAM-DWD: min/mean/max = {dh.min():+.1f} / {dh.mean():+.1f} / {dh.max():+.1f} m"
              f"  → source HHL terrain-shifted onto LAM topography")

    # Full-level heights (cell-centered) — now both in absolute height above
    # the SAME (LAM) terrain, so interpolation effectively works in AGL.
    z_src_full = vert._full_levels_from_half(hhl_src)
    z_tgt_full = vert._full_levels_from_half(hhl_tgt)

    # Pre-interpolate T and P first — needed by qv/pressure extrapolation
    if verbose:
        print(f"[iconremap] interpolating T...")
    t_src = ds["T"].values
    if t_src.ndim == 2:
        t_src = t_src[np.newaxis, ...]
    t_tgt = vert.interpolate_field(t_src, z_src_full, z_tgt_full, "T")

    if verbose:
        print(f"[iconremap] interpolating P...")
    p_src = ds["P"].values
    if p_src.ndim == 2:
        p_src = p_src[np.newaxis, ...]
    p_tgt = vert.interpolate_field(
        p_src, z_src_full, z_tgt_full, "P", t_src_full=t_src
    )

    # Now interpolate the rest of the atm vars
    interp_results = {"T": t_tgt, "P": p_tgt}
    for var in vert.VAR_CATEGORY:
        if var in interp_results:
            continue
        if var not in ds.data_vars:
            if verbose:
                print(f"[iconremap]   skip {var} (not in input)")
            continue
        if verbose:
            print(f"[iconremap] interpolating {var}...")
        src = ds[var].values
        if src.ndim == 2:
            src = src[np.newaxis, ...]
        kwargs = {}
        if var == "QV":
            kwargs["p_src_full"] = p_src
            kwargs["t_src_full"] = t_src
        interp_results[var] = vert.interpolate_field(
            src, z_src_full, z_tgt_full, var, **kwargs
        )

    # Build output dataset. Source and target may have different level counts;
    # use fresh dim names ("lev" for atm full levels, "height_4" for HHL
    # half-levels) so we don't collide with the source's "height" / "height_bnds"
    # which xarray pins via coords.
    if verbose:
        print(f"[iconremap] assembling output dataset (src nlev={hhl_orig.shape[1]-1} → tgt nlev={tgt_nlev_intf-1})...")
    ds_out = ds.copy()
    drop_list = list(interp_results.keys()) + ["HHL"]
    ds_out = ds_out.drop_vars([v for v in drop_list if v in ds_out.data_vars])

    # Re-add atm vars on a fresh "lev" dim sized to target nlev
    for var, arr in interp_results.items():
        src_dims = ds[var].dims
        new_dims = tuple(
            d if d in ("time", "cell") else "lev"
            for d in src_dims
        )
        if "time" in src_dims:
            ds_out[var] = (new_dims, arr.astype(np.float32))
        else:
            ds_out[var] = (new_dims, arr.squeeze(0).astype(np.float32))

    # HHL on its own half-level dim ("height_4") — same as the prep convention
    if "time" in ds["HHL"].dims:
        ds_out["HHL"] = (("time", "height_4", "cell"), hhl_tgt.astype(np.float32))
    else:
        ds_out["HHL"] = (("height_4", "cell"), hhl_tgt.squeeze(0).astype(np.float32))

    # GEOSP — surface geopotential — required by check_variables in async_latbc.
    # Compute as HSURF * g.
    from . import meteo as _meteo
    geosp = (hsurf_tgt * _meteo.G).astype(np.float32)
    ds_out["GEOSP"] = (("time", "cell"), geosp[None, :])
    if verbose:
        print(f"[iconremap] added GEOSP = HSURF × g, range {geosp.min():.0f}..{geosp.max():.0f} m²/s²")

    # ICON expects W (vertical wind) on HALF-levels (nlev+1=61), not full levels (60).
    # Our CDO-remapped W is on full levels — pad to half-levels by averaging adjacent
    # full levels and duplicating end values for top/bottom boundaries.
    if "W" in ds_out.data_vars:
        w_full = ds_out["W"].values   # shape (time, 60, cell)
        nt, nlev, ncell = w_full.shape
        w_half = np.zeros((nt, nlev + 1, ncell), dtype=np.float32)
        w_half[:, 0,    :] = w_full[:, 0, :]                    # top half-level = top full
        w_half[:, 1:nlev, :] = 0.5 * (w_full[:, :-1, :] + w_full[:, 1:, :])  # interior averages
        w_half[:, nlev, :] = w_full[:, -1, :]                   # surface half-level = bottom full
        ds_out = ds_out.drop_vars("W")
        ds_out["W"] = (("time", "height_4", "cell"), w_half)
        if verbose:
            print(f"[iconremap] padded W from {nlev} full-levels to {nlev+1} half-levels")

    if verbose:
        print(f"[iconremap] writing {output_nc}")
    ic_io.write_ic(ds_out, output_nc)

    elapsed = _time.time() - t0
    if verbose:
        print(f"[iconremap] DONE in {elapsed:.1f}s")
