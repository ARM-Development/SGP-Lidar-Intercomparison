"""
===============================================================
Israel Silber
===============================================================
Wrapper / test script for the lidar profile intercomparison
(REAL) data pipeline.
Each instrument is loaded, corrected (if required), and
interpolated onto a common uniform time x range grid.
Instruments with missing data files are skipped with a warning.
===============================================================
"""

from pathlib import Path
import warnings
import argparse
from datetime import datetime

import numpy as np
import xarray as xr

from real_prof_io import load_and_process_arm_data, export_dataset, extract_geographic_metadata
from real_prof_init import get_supplementary_facilities

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SITE       = "sgp"
FACILITY   = "C1"
CONFIG_DIR = str(Path(__file__).parent / "configs")

TIME_MIN = np.datetime64("2026-03-10T20:00:00", "ns")
TIME_MAX = np.datetime64("2026-03-10T21:00:00", "ns")
#TIME_MIN = np.datetime64("2025-11-29T11:00:00", "ns")
#TIME_MAX = np.datetime64("2025-11-29T13:00:00", "ns")

# ARM nested archive: /data/archive/{site}/{site}{instrument_class}{facility}.{level}/
DATA_PATH_TEMPLATE = "/data/archive/{site}/{site}{instrument_class}{facility}.{level}"

# All instruments for data loading (verify a corresponding JSON config exists)
ALL_INSTRUMENTS = [
    "ceil",
    "ceilpol",
    "hsrl",
    "rl",
    "minimpl",
    "mpl",
    "dl",
    "dlwindstat",
    "dlwind",
    "interpolatedsonde",
    "pblhtbeml",
]

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    time_min: "np.datetime64 | None" = None,
    time_max: "np.datetime64 | None" = None,
    site: "str | None" = None,
    facility: "str | None" = None,
    data_path_template: "str | None" = None,
) -> "xr.Dataset":
    # Use module-level defaults if not provided
    if time_min is None:
        time_min = TIME_MIN
    if time_max is None:
        time_max = TIME_MAX
    if site is None:
        site = SITE
    if facility is None:
        facility = FACILITY
    if data_path_template is None:
        data_path_template = DATA_PATH_TEMPLATE

    print("=" * 70)
    print(f"Site: {site}  Facility: {facility}")
    print(f"Period: {time_min} → {time_max}")
    print(f"Path template: {data_path_template}")
    print("=" * 70)

    # Extract geographic metadata (lat/lon/alt) from first available data file
    geo_data = extract_geographic_metadata(
        ALL_INSTRUMENTS,
        site,
        facility,
        None,
        time_min,
        time_max,
        config_dir=CONFIG_DIR,
        data_path_template=data_path_template,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ds = load_and_process_arm_data(
            instrument_type    = ALL_INSTRUMENTS,
            site               = site,
            facility           = facility,
            data_path          = None,
            time_min           = time_min,
            time_max           = time_max,
            config_dir         = CONFIG_DIR,
            data_path_template = data_path_template,
            interpolate        = True,
        )

        # ------------------------------------------------------------------
        # Load supplementary facility data if configured
        # ------------------------------------------------------------------
        supp_facilities = get_supplementary_facilities(SITE, FACILITY, CONFIG_DIR)
        contributing_supp_facilities = []

        for supp_facility in supp_facilities:
            print(f"\nLoading supplementary facility '{supp_facility}'...")
            try:
                ds_supp = load_and_process_arm_data(
                    instrument_type    = ALL_INSTRUMENTS,
                    site               = site,
                    facility           = supp_facility,
                    data_path          = None,
                    time_min           = time_min,
                    time_max           = time_max,
                    config_dir         = CONFIG_DIR,
                    data_path_template = data_path_template,
                    interpolate        = True,
                )
                if ds_supp.data_vars:
                    # Build rename map: for each variable, find its instrument prefix
                    # and rename {instr}_{rest} -> {instr}_supp_{rest}
                    rename_map = {}
                    for var in ds_supp.data_vars:
                        # Find which instrument this variable belongs to
                        for instr in ALL_INSTRUMENTS:
                            prefix = f"{instr}_"
                            if var.startswith(prefix):
                                rest = var[len(prefix):]
                                rename_map[var] = f"{instr}_supp_{rest}"
                                break
                        # If no instrument prefix found, keep the variable as-is
                        if var not in rename_map:
                            rename_map[var] = var

                    ds_supp = ds_supp.rename(rename_map)
                    ds = xr.merge([ds, ds_supp])
                    contributing_supp_facilities.append(supp_facility)
            except FileNotFoundError as exc:
                warnings.warn(
                    f"[{supp_facility}] skipped - {exc}", stacklevel=2
                )

        # Attach supplementary facility info (set to "N/A" if none)
        ds.attrs["supplementary_facility"] = (
            ", ".join(contributing_supp_facilities)
            if contributing_supp_facilities
            else "N/A"
        )

    skipped = [str(w.message) for w in caught]
    if skipped:
        print("\nSkipped instruments/facilities:")
        for msg in skipped:
            print(f"  {msg}")

    print(f"\nMerged dataset: {dict(ds.sizes)}  vars={len(ds.data_vars)}")
    if "time" in ds.coords:
        print(f"  time: {str(ds.time.values[0])[:19]} → {str(ds.time.values[-1])[:19]}")
    if "range" in ds.coords:
        print(f"  range: [{float(ds.coords['range'].min()):.3f},"
              f" {float(ds.coords['range'].max()):.3f}] km")

    # ------------------------------------------------------------------
    # Group variables by instrument: primary then supplementary
    # ------------------------------------------------------------------
    def _instr_group_key(var):
        """Sort key: (instr_index, 0_for_primary_1_for_supp, var_name)."""
        for idx, instr in enumerate(ALL_INSTRUMENTS):
            if var.startswith(f"{instr}_supp_"):
                return (idx, 1, var)
            if var.startswith(f"{instr}_"):
                return (idx, 0, var)
        return (len(ALL_INSTRUMENTS), 0, var)
    
    ds = ds[sorted(ds.data_vars, key=_instr_group_key)]

    # ------------------------------------------------------------------
    # Add lat/lon/alt as the last variables in the dataset
    # ------------------------------------------------------------------
    for field in ["lat", "lon", "alt"]:
        if field in geo_data:
            ds[field] = geo_data[field]

    return ds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process lidar profile intercomparison data."
    )
    parser.add_argument(
        "--period-start",
        type=str,
        default=None,
        help="Start time in yyyymmddTHHMMSS format (default: {TIME_MIN})",
    )
    parser.add_argument(
        "--period-end",
        type=str,
        default=None,
        help="End time in yyyymmddTHHMMSS format (default: {TIME_MAX})",
    )
    parser.add_argument(
        "--site",
        type=str,
        default=None,
        help=f"Site code (default: {SITE})",
    )
    parser.add_argument(
        "--facility",
        type=str,
        default=None,
        help=f"Facility code (default: {FACILITY})",
    )
    parser.add_argument(
        "--data-path-template",
        type=str,
        default=None,
        help=f"Data path template (default: {DATA_PATH_TEMPLATE})",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="./",
        help="Output directory for exported files (default: ./)",
    )
    args = parser.parse_args()
    
    # Parse period_start and period_end from CLI or use defaults
    if args.period_start:
        period_start = np.datetime64(
            datetime.strptime(args.period_start, "%Y%m%dT%H%M%S").isoformat(), "ns"
        )
    else:
        period_start = TIME_MIN
    
    if args.period_end:
        period_end = np.datetime64(
            datetime.strptime(args.period_end, "%Y%m%dT%H%M%S").isoformat(), "ns"
        )
    else:
        period_end = TIME_MAX
    
    # Process multiple hours of data, exporting 1-hour files
    _hour = np.timedelta64(1, "h")
    t = period_start
    
    while t < period_end:
        t_next = t + _hour
        result = main(
            time_min=t,
            time_max=t_next,
            site=args.site,
            facility=args.facility,
            data_path_template=args.data_path_template,
        )
        if result.data_vars:
            # Use resolved site and facility values
            resolved_site = args.site if args.site else SITE
            resolved_facility = args.facility if args.facility else FACILITY
            out = export_dataset(
                result,
                site=resolved_site,
                facility=resolved_facility,
                output_path=args.output_path,
            )
            print(f"\nExported: {out}")
        t = t_next
