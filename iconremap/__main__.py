"""CLI: python -m iconremap --input IC.nc --extpar EXTPAR.nc --output OUT.nc"""
import argparse
import sys

from .pipeline import remap_ic


def main():
    p = argparse.ArgumentParser(prog="iconremap", description=__doc__)
    p.add_argument("--input",   required=True, help="CDO-remapped IC NetCDF (with HHL)")
    p.add_argument("--extpar",  required=True, help="LAM EXTPAR file (with HSURF)")
    p.add_argument("--output",  required=True, help="output IC NetCDF")
    p.add_argument("--quiet",   action="store_true")
    args = p.parse_args()
    remap_ic(args.input, args.extpar, args.output, verbose=not args.quiet)


if __name__ == "__main__":
    main()
