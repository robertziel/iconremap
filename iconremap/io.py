"""NetCDF read/write helpers tailored to ICON's IFS2ICON-style input expectations."""
import numpy as np
import xarray as xr


# Variables we vertically interpolate. All are 3D (time, lev, cell).
ATM_3D_VARS = ("T", "U", "V", "W", "P", "QV", "QC", "QI")
ATM_3D_OPT  = ("QR", "QS")
# Surface fields just pass through unchanged (already on cell grid, single level).
SFC_PASSTHROUGH = (
    "T_G", "T_SNOW", "qv_s", "t_sk", "w_snow", "freshsnow", "w_i",
    "T_SO", "W_SO",
    "sde", "sd", "rsn", "sit", "sithick",  # ECMWF-named snow/ice that may persist
)


def read_ic(path: str) -> xr.Dataset:
    """Read a CDO-remapped IC NetCDF (atm + surface + HHL on LAM cell grid)."""
    ds = xr.open_dataset(path)
    # Sanity: must have HHL (height of half levels) for vertical interpolation.
    if "HHL" not in ds.data_vars:
        raise KeyError(f"input {path} missing HHL — required for vertical interp")
    return ds


def read_extpar_hsurf(path: str) -> np.ndarray:
    """Extract HSURF (surface elevation, m) from EXTPAR file.

    Returns shape (cell,).
    """
    ds = xr.open_dataset(path)
    for name in ("topography_c", "HSURF", "hsurf", "z_topo", "Z0"):
        if name in ds.data_vars:
            arr = ds[name].values
            # may have a leading time/level dim of size 1
            arr = arr.squeeze()
            ds.close()
            return arr.astype(np.float32)
    ds.close()
    raise KeyError(f"no HSURF-like variable in {path}")


def write_ic(ds_out: xr.Dataset, path: str) -> None:
    """Write IC with appropriate compression."""
    encoding = {
        v: {"zlib": True, "complevel": 4}
        for v in ds_out.data_vars
        if ds_out[v].dtype != object
    }
    ds_out.to_netcdf(path, encoding=encoding)
