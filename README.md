# iconremap (Python)

A small Python patch that makes it easier to spin up a local
[ICON-LAM](https://docs.icon-model.org/) run when you only have access to
**publicly available** input data.

## Why it exists

Setting up an ICON limited-area run normally requires initial/boundary
conditions interpolated onto your LAM grid by DWD's official `iconremap`
(part of [dwd-icon-tools](https://gitlab.dkrz.de/dwd-sw/dwd_icon_tools)).
That tool expects high-resolution IFS analyses or full ICON analysis fields
that are not on the open server — getting them requires a DWD/ECMWF account
or institutional access.

The freely-available alternative is the
[DWD Open Data Server](https://opendata.dwd.de/weather/) ICON-EU forecast
output, which can be horizontally remapped onto an arbitrary LAM grid with
[CDO](https://code.mpimet.mpg.de/projects/cdo) alone. What's still missing
after CDO is the **vertical** step: interpolating the atmospheric fields
from ICON-EU's vertical levels onto the target LAM's terrain-following
SLEVE coordinate, with physically sensible extrapolation where the LAM
surface lies below ICON-EU's lowest level (~1700 m above ground).

This tool fills exactly that gap, and adds the few output fix-ups
(`GEOSP`, half-level `W`) that ICON's runtime `check_variables` requires.
The result is an IC NetCDF that ICON-LAM can ingest directly, produced
entirely from open data.

## Scope

- Input: a CDO-horizontally-remapped ICON-EU IC NetCDF on your LAM grid + the LAM `EXTPAR` file.
- Output: an IC NetCDF with all atmospheric fields on the LAM's SLEVE z\_ifc.
- **Not** a general replacement for `iconremap` — horizontal remapping, GRIB I/O, and arbitrary vertical-coordinate layouts are out of scope.

## Usage

```bash
python -m iconremap --input IC.nc --extpar EXTPAR.nc --output OUT.nc
```

A Dockerfile is provided that bundles CDO, eccodes, and NCO alongside the Python tool for a self-contained workflow.

Sources:
- [DWD Open Data Server](https://dwd.de/EN/ourservices/opendata/opendata.html)
- [ICON Initial and Boundary Data from IFS](https://docs.icon-model.org/buildrun/buildrun_ini_lbc_data.html)
- [ICON in Climate Limited-area Mode (GMD preprint)](https://gmd.copernicus.org/preprints/gmd-2020-20/gmd-2020-20.pdf)
