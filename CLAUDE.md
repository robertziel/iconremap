# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project purpose

Minimal Python replacement for DWD's `iconremap` Fortran tool, scoped to a single use case: producing ICON-LAM initial conditions by **vertically** interpolating CDO-horizontally-remapped ICON-EU fields onto a target LAM's terrain-following SLEVE coordinate. Horizontal remapping is assumed to have already been done with CDO; this tool only does the vertical step plus the surface-fitting that ICON's official `iconremap` would otherwise handle.

## Commands

Run the CLI (in-tree, no install step):

```bash
python -m iconremap --input IC.nc --extpar EXTPAR.nc --output OUT.nc [--quiet]
```

Run the test suite (no pytest dependency — tests are plain `assert` functions invoked from `__main__`):

```bash
python tests/test_basic.py
```

Run a single test:

```bash
python -c "from tests.test_basic import test_lapse_extrapolation_below; test_lapse_extrapolation_below()"
```

Build / run the Docker image (bundles CDO, eccodes, NCO alongside the Python tool):

```bash
docker build -t iconremap .
docker run --rm -v "$PWD:/data" iconremap --input /data/IC.nc --extpar /data/EXTPAR.nc --output /data/OUT.nc
```

## Architecture

The pipeline is a single linear pass orchestrated in `iconremap/pipeline.py::remap_ic`. The key insight is the split between **source heights** and **target heights**:

- **Source heights** = the `HHL` field from DWD's CDO-remapped ICON-EU IC NetCDF, kept verbatim. ICON-EU's lowest atmospheric level sits ~1700 m above the actual surface, so most LAM cells require lapse-rate **extrapolation downward**, not interpolation. Treating source HHL as identity-target was the original failure mode and must not be reintroduced.
- **Target heights** = SLEVE `z_ifc` recomputed for the LAM grid by `iconremap/sleve.py::compute_z_ifc` from EXTPAR `HSURF`.

`sleve.py` ports the SLEVE branch of ICON's `mo_init_vgrid.f90` (`init_vert_coord` plus the layer-thickness limiter at lines 489–516). Crucially, **`VCT_A` is hardcoded** from values that ICON itself printed to its run log ("Nominal heights of coordinate half levels") for the specific LAM namelist in use. This bypasses the layout algorithm in `init_sleve_coord` and guarantees byte-exact agreement with what ICON's runtime expects — this exact match is what previously prevented `vert_interp_atm` crashes. If the LAM namelist changes (different `nlev`, top height, decay scales, flat height), `VCT_A` and the constants at the top of `sleve.py` must be re-extracted from a fresh ICON log.

`vertical.py::interpolate_field` is the per-cell vertical interpolator. ICON convention is level 0 = top, level nlev-1 = bottom (descending heights), but `np.interp` needs ascending `xp`, so each cell column is reversed for the interp call and reversed back on write. Variable category drives the below-surface extrapolation path:

- `T` → standard lapse (`meteo.extrapolate_temperature`)
- `P` → hydrostatic with source-bottom `T` (`meteo.extrapolate_pressure`); requires `t_src_full`, so **T must be interpolated before P** in the pipeline.
- `QV` → constant-RH approximation scaling with the hydrostatically-extrapolated pressure ratio; requires `p_src_full` and `t_src_full`, so **P must be interpolated before QV**.
- `QC`, `QI`, `QR`, `QS` → zero below source surface
- `U`, `V`, `W` → constant (= source-bottom value)

The `pipeline.py` ordering reflects these dependencies: T → P → everything else (with QV passing P/T explicitly). Don't reorder without re-checking.

Two ICON-specific output fix-ups happen at the end of `pipeline.py`:

1. **`GEOSP` is synthesized** as `HSURF * G` and added to the output, because ICON's `check_variables` in `async_latbc` requires it.
2. **`W` is repadded from full-levels to half-levels** (nlev → nlev+1) by averaging adjacent full levels and duplicating endpoints. CDO horizontal remapping leaves W on full levels, but ICON expects half-level W.

`io.py` lists the variables the pipeline knows about (`ATM_3D_VARS`, `ATM_3D_OPT`, `SFC_PASSTHROUGH`). Surface fields pass through unchanged. `read_extpar_hsurf` tolerates several common HSURF variable names since EXTPAR files differ between sources.

## Conventions and gotchas

- All shape comments use `(time, lev, cell)` for 3D atm fields and `(time, lev_intf, cell)` for HHL — the `time` dim is added if missing on read.
- The `data/` and `examples/` directories are intentionally empty placeholders in the repo; populate locally with input NetCDFs when running.
- No package metadata (`pyproject.toml`/`setup.py`) — the tool is run in-tree via `python -m iconremap` or via the Docker image, which sets `PYTHONPATH=/opt/iconremap`.
