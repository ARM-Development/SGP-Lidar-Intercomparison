"""
===============================================================
Israel Silber
===============================================================
Lidar profile intercomparison utility functions
===============================================================
"""

from typing import Dict, List, Optional
import numpy as np
import xarray as xr
import real_prof_init as pi


def get_common_variables(instrument_types: List[str], config_dir: str = "./configs") -> List[str]:
    """
    Get variables that are common across multiple instrument types.
    
    Parameters
    ----------
    instrument_types : List[str]
        List of instrument types to compare
    config_dir : str, optional
        Directory containing the JSON configuration files, by default "./configs"
        
    Returns
    -------
    List[str]
        List of common variable names (standard names)
    """
    if not instrument_types:
        return []
        
    # Load all configs and get variable sets
    variable_sets = []
    for inst_type in instrument_types:
        config = pi.load_config(inst_type, config_dir)
        variable_sets.append(set(config["variables"].keys()))
    
    # Find intersection
    common_vars = variable_sets[0]
    for var_set in variable_sets[1:]:
        common_vars &= var_set
        
    return list(common_vars)


# Default output grids
_DEFAULT_RANGE_KM = np.arange(0.0, 20.0 + 0.005, 0.010)   # 0–20 km, 10 m steps (IS Update 6/16/2026 per REAL meeting feedback)


def interpolate_data(
    ds: xr.Dataset,
    time_min: np.datetime64,
    time_max: np.datetime64,
    range_km: Optional[np.ndarray] = None,
    time_step: np.timedelta64 = np.timedelta64(15, "s"),
    range_units: str = "m",
) -> xr.Dataset:
    """
    Interpolate all variables in a dataset onto uniform time and range grids.

    Interpolation method depends on variable name:

    * Variables whose name ends with ``_mask`` or starts with ``qc_``:
      **nearest-neighbour** in all applicable dimensions.
    * All other variables: **linear** interpolation.

    The interpolation is applied per dimension type:

    * **(time, range) 2-D variables**: bilinear (or nearest) gridded interpolation.
    * **time-only 1-D variables**: 1-D linear (or nearest) interpolation along time.
    * **(time, layer) 2-D variables**: linear (or nearest for ``qc_``/``_mask``
      variables) interpolation along time, applied independently for each layer index.
    * Variables with no time dimension are passed through unchanged.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset.
    time_min : np.datetime64
        Start of the output time grid (inclusive).
    time_max : np.datetime64
        End of the output time grid (inclusive).
    range_km : np.ndarray, optional
        1-D array of output range gate centres in **km**.  Defaults to
        ``np.arange(0, 20.015, 0.015)`` (0-20 km in 15 m increments).
        Ignored for variables that have no range dimension.
    time_step : np.timedelta64, optional
        Spacing between output time steps.  Default is 15 s.
    range_units : str, optional
        Units of the ``range`` coordinate in *ds* (``"m"`` or ``"km"``).
        If ``"m"``, the coordinate is divided by 1000 before interpolation.
        Default is ``"m"``.

    Returns
    -------
    xr.Dataset
        New dataset on the uniform (time, range_km) grid.  Variables that
        could not be interpolated (e.g. scalar coordinates) are dropped
        silently.
    """
    if range_km is None:
        range_km = _DEFAULT_RANGE_KM

    if "time" in ds.coords and ds.sizes.get("time", 0) == 0:
        raise ValueError(
            "interpolate_data received a dataset with an empty time dimension."
        )

    # ------------------------------------------------------------------
    # Build output time coordinate
    # ------------------------------------------------------------------
    step_ns = int(time_step / np.timedelta64(1, "ns"))
    t_min_ns = time_min.astype("datetime64[ns]").astype(np.int64)
    t_max_ns = time_max.astype("datetime64[ns]").astype(np.int64)
    out_time = (np.arange(t_min_ns, t_max_ns, step_ns)
                  .astype("datetime64[ns]"))

    # ------------------------------------------------------------------
    # Standardise range coordinate to km
    # ------------------------------------------------------------------
    if "range" in ds.coords:
        range_coord = ds["range"].values
        if range_units.lower() == "m":
            range_coord = range_coord / 1000.0
        # Rebuild dataset with km-valued range
        ds = ds.assign_coords(range=range_coord)

    # ------------------------------------------------------------------
    # Ensure missing values are NaN (replace common fill values)
    # ------------------------------------------------------------------
    for var in ds.data_vars:
        da = ds[var]
        if np.issubdtype(da.dtype, np.floating):
            fill = da.attrs.get("missing_value", da.attrs.get("_FillValue", None))
            if fill is not None:
                ds[var] = da.where(da != fill)

    # ------------------------------------------------------------------
    # Helper: decide method for a given variable name
    # ------------------------------------------------------------------
    def _method(name: str) -> str:
        if name.endswith("_mask") or name.startswith("qc_"):
            return "nearest"
        return "linear"

    # ------------------------------------------------------------------
    # Interpolate each variable
    # ------------------------------------------------------------------
    out_vars: dict[str, xr.DataArray] = {}

    for var in ds.data_vars:
        da = ds[var]
        dims = da.dims
        method = _method(var)

        # Skip variables whose time dimension is empty
        if "time" in dims and da.sizes["time"] == 0:
            continue

        # Skip range-only variables (no time dim); they are meaningless on
        # the interpolated output grid and are excluded unconditionally.
        if "time" not in dims:
            continue

        # Pre-compute input coordinate bounds for explicit extrapolation masking
        t_in_min = da.time.values[0]   if "time"  in dims else None
        t_in_max = da.time.values[-1]  if "time"  in dims else None
        r_in_min = float(da["range"].min()) if "range" in dims else None
        r_in_max = float(da["range"].max()) if "range" in dims else None

        # Boolean masks over the *output* grids marking in-bounds points
        t_valid = (
            xr.DataArray(
                (out_time >= t_in_min) & (out_time <= t_in_max),
                dims=["time"], coords={"time": out_time},
            )
            if t_in_min is not None else None
        )
        r_valid = (
            xr.DataArray(
                (range_km >= r_in_min) & (range_km <= r_in_max),
                dims=["range"], coords={"range": range_km},
            )
            if r_in_min is not None else None
        )

        if "time" in dims and "range" in dims:
            # 2-D (time × range) — gridded interpolation
            try:
                result = da.interp(
                    time=out_time,
                    range=range_km,
                    method=method,
                    kwargs={"fill_value": np.nan, "bounds_error": False},
                )
                # Explicitly zero out any extrapolated cells (handles nearest)
                result = result.where(t_valid & r_valid)
                out_vars[var] = result
            except Exception:
                pass

        elif "time" in dims and "layer" in dims:
            # 2-D (time × layer) — interpolate along time, layer unchanged
            n_layers = da.sizes["layer"]
            layers = []
            for li in range(n_layers):
                sl = da.isel(layer=li)
                try:
                    interped = sl.interp(
                        time=out_time,
                        method=method,
                        kwargs={"fill_value": np.nan, "bounds_error": False},
                    )
                    # Mask time extrapolation explicitly
                    interped = interped.where(t_valid)
                    layers.append(interped)
                except Exception:
                    layers.append(xr.full_like(
                        sl.isel(time=0).expand_dims(time=out_time), np.nan
                    ))
            out_vars[var] = xr.concat(layers, dim="layer").transpose("time", "layer")

        elif "time" in dims:
            # 1-D (time only)
            try:
                result = da.interp(
                    time=out_time,
                    method=method,
                    kwargs={"fill_value": np.nan, "bounds_error": False},
                )
                # Mask time extrapolation explicitly
                result = result.where(t_valid)
                out_vars[var] = result
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Assemble output dataset, carrying over variable attributes
    # ------------------------------------------------------------------
    out_ds = xr.Dataset(out_vars)
    for var in out_ds.data_vars:
        if var in ds.data_vars:
            out_ds[var].attrs.update(ds[var].attrs)

    return out_ds