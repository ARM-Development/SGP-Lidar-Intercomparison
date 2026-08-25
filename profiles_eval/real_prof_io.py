"""
===============================================================
Israel Silber
===============================================================
Lidar profile intercomparison real-data I/O module
===============================================================
"""
import warnings
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import xarray as xr

from real_prof_init import load_config
from real_prof_utils import interpolate_data
from real_lidar_corrections import compute_nrb_dataset

# Number of cloud/precipitation layers used throughout the pipeline
N_LAYERS = 10

# Canonical variable ordering: variables are grouped by keyword presence in
# the order listed below.  qc_ variants are inserted after their parent.
_PRIORITY_VARS = [
    "particulate_backscatter",
    "extinction",
    "attenuated_backscatter",
    "depol",
    "cloud_base",
    "cloud_top"
]


def _native_time_res_str(time_arr: np.ndarray) -> str:
    """
    Compute median native time resolution from a time array.
    
    Parameters
    ----------
    time_arr : np.ndarray
        1D array of datetime64 values.
        
    Returns
    -------
    str
        Median time step formatted as e.g. "30 s".
    """
    if len(time_arr) < 2:
        return "undetermined (only one time point)"
    diffs = np.diff(time_arr)
    # Convert to timedelta64 and extract median in seconds
    diffs_s = diffs / np.timedelta64(1, "s")
    median_s = float(np.median(diffs_s))
    
    # Format: prefer integer or fraction seconds if < 60, otherwise round to nearest second.
    if median_s < 60:    
        return f"{int(median_s)} s" if median_s == int(median_s) else f"{median_s:.1f} s"
    else:
        return f"{int(median_s)} s"


def _native_range_res_str(range_da: xr.DataArray, units_str: str) -> str:
    """
    Compute median native range resolution from a range array.
    
    Always returns result in km.
    
    Parameters
    ----------
    range_da : xr.DataArray
        1D array of range/height values.
    units_str : str
        Unit string (e.g. "m", "km").
        
    Returns
    -------
    str
        Median range spacing in km, formatted as e.g. "0.015 km", "0.5 km".
    """
    if len(range_da) < 2:
        return "undetermined (only one range gate)"
    diffs = np.diff(range_da.values)
    median_diff = float(np.median(np.abs(diffs)))
    
    # Convert to km if needed
    scale_to_km = 1.0 / 1000.0 if units_str.lower() == "m" else 1.0
    median_km = median_diff * scale_to_km
    
    return f"{median_km:.4g} km"


def _ordered_vars(var_names: list[str]) -> list[str]:
    """Return *var_names* in canonical order with qc_ vars after their parent.

    Variables whose names contain a keyword from ``_PRIORITY_VARS`` are placed
    first (in keyword order), each immediately followed by its ``qc_`` twin.
    Remaining variables follow in their original relative order, likewise each
    followed by their ``qc_`` twin.  Orphaned ``qc_`` vars (whose parent was
    not present) appear last.
    """
    remaining = list(var_names)
    ordered: list[str] = []

    def _pop(name: str) -> bool:
        if name in remaining:
            remaining.remove(name)
            ordered.append(name)
            return True
        return False

    # Keyword groups: collect non-qc vars whose name contains the keyword,
    # preserving their relative order; each is followed by its qc_ twin.
    for keyword in _PRIORITY_VARS:
        group = [v for v in remaining if keyword in v and not v.startswith("qc_")]
        for base in group:
            _pop(base)
            _pop(f"qc_{base}")

    # Remaining non-qc variables (in their original relative order),
    # each followed by their qc_ twin
    for var in [v for v in remaining if not v.startswith("qc_")]:
        _pop(var)
        _pop(f"qc_{var}")

    # Any orphaned qc_ vars whose parent was not present
    for var in list(remaining):
        _pop(var)

    return ordered


def find_arm_files(
    instrument_type: str,
    site: str,
    facility: str,
    data_path: str | Path | None,
    time_min: np.datetime64,
    time_max: np.datetime64,
    safety_delta: np.timedelta64 = np.timedelta64(5, "m"),
    config_dir: str = "./configs",
    data_path_template: str | None = None,
) -> list[Path]:
    """
    Find ARM instrument data files within a given time range.

    Scans ``data_path`` for files whose name matches the ARM convention
    ``{site}{instrument_class}{facility}.{level}.YYYYMMDD.HHMMSS.*`` and
    whose start timestamp falls within
    ``[time_min - safety_delta, time_max + safety_delta]`` plus the single
    file whose timestamp is the largest one strictly before
    ``t_search_min``.

    The ``instrument_class`` and ``level`` are read from the instrument's
    JSON config file.

    Parameters
    ----------
    instrument_type : str
        Instrument identifier matching a JSON config file, e.g.
        ``"minimpl"`` or ``"ceil"``.
    site : str
        ARM site code, e.g. ``"sgp"``.
    facility : str
        ARM facility code, e.g. ``"C1"``.
    data_path : str, Path, or None
        Directory to search for data files.  Mutually exclusive with
        ``data_path_template``; pass ``None`` when using the template.
    time_min : np.datetime64
        Start of the desired time range.
    time_max : np.datetime64
        End of the desired time range.
    safety_delta : np.timedelta64, optional
        Widening margin applied symmetrically around ``[time_min, time_max]``
        when searching.  Default is 5 minutes.
    config_dir : str, optional
        Directory containing JSON configuration files.  Default is
        ``"./configs"``.
    data_path_template : str, optional
        Format string for the ARM nested archive layout, e.g.
        ``"/data/archive/{site}/{site}{instrument_class}{facility}.{level}"``.
        Used only when ``data_path`` is ``None``.

    Returns
    -------
    list[Path]
        Sorted list of matching file paths.
    """
    cfg = load_config(instrument_type, config_dir)
    info = cfg["instrument_info"]
    instrument_class = info["instrument_class"]
    level = info["level"]

    if data_path is None:
        if data_path_template is None:
            raise ValueError(
                "Either 'data_path' or 'data_path_template' must be provided."
            )
        data_path = Path(
            data_path_template.format(
                site=site,
                instrument_class=instrument_class,
                facility=facility,
                level=level,
            )
        )
    else:
        data_path = Path(data_path)
    stem_prefix = f"{site}{instrument_class}{facility}.{level}."

    t_search_min = time_min - safety_delta
    t_search_max = time_max + safety_delta

    # Two-pass scan:
    # Pass 1 – collect all candidate files and their parsed timestamps.
    # Pass 2 – keep files within [t_search_min, t_search_max] plus the single
    #           file whose timestamp is the largest one strictly before
    #           t_search_min (the "covering" file — e.g. a daily file that
    #           started at 00:00 but contains data through the whole day).
    candidates: list[tuple[np.datetime64, Path]] = []
    for fp in sorted(data_path.iterdir()):
        if not fp.name.startswith(stem_prefix):
            continue
        parts = fp.stem.split(".")
        if len(parts) < 4:
            continue
        try:
            file_dt = np.datetime64(
                datetime.strptime(f"{parts[2]}{parts[3]}", "%Y%m%d%H%M%S"), "s"
            )
        except ValueError:
            continue
        candidates.append((file_dt, fp))

    selected: list[Path] = []
    covering: tuple[np.datetime64, Path] | None = None  # latest file before window
    for file_dt, fp in candidates:
        if t_search_min <= file_dt <= t_search_max:
            selected.append(fp)
        elif file_dt < t_search_min:
            if covering is None or file_dt > covering[0]:
                covering = (file_dt, fp)

    if covering is not None:
        selected.insert(0, covering[1])

    if not selected:
        raise FileNotFoundError(
            f"No '{instrument_type}' files found in '{data_path}' for the period "
            f"{time_min} to {time_max} (safety_delta={safety_delta})."
        )

    return selected


def load_arm_data(
    instrument_type: str,
    site: str,
    facility: str,
    data_path: str | Path | None,
    time_min: np.datetime64,
    time_max: np.datetime64,
    safety_delta: np.timedelta64 = np.timedelta64(5, "m"),
    config_dir: str = "./configs",
    data_path_template: str | None = None,
) -> xr.Dataset:
    """
    Load ARM instrument data files into a single xarray Dataset.

    Delegates file discovery to :func:`find_arm_files`, then opens and
    concatenates the matching files along the time dimension.  Only
    variables (and correction fields, if defined) listed in the instrument's
    JSON config are retained.

    The returned dataset is trimmed so that:

    * The earliest sample is at or after ``time_min - safety_delta``.
    * The latest sample is the **first sample beyond** ``time_max``
      (inclusive), giving one look-ahead point for downstream interpolation
      or boundary handling.  If no such sample exists the dataset ends at
      the last available sample within the window.

    Parameters
    ----------
    instrument_type : str
        Instrument identifier matching a JSON config file, e.g.
        ``"minimpl"`` or ``"ceil"``.
    site : str
        ARM site code.
    facility : str
        ARM facility code.
    data_path : str, Path, or None
        Directory containing the instrument data files.  Pass ``None``
        when ``data_path_template`` is used instead.
    time_min : np.datetime64
        Start of the desired time range (inclusive).
    time_max : np.datetime64
        End of the desired time range.  One sample beyond this time will
        be included in the output if available.
    safety_delta : np.timedelta64, optional
        Look-back/look-ahead margin used both for file discovery and for
        trimming the output.  Default is 5 minutes.
    config_dir : str, optional
        Directory containing JSON configuration files.  Default is
        ``"./configs"``.
    data_path_template : str, optional
        Format string for the ARM nested archive layout (see
        :func:`find_arm_files` for placeholder details).  Used only when
        ``data_path`` is ``None``.

    Returns
    -------
    xr.Dataset
        Merged dataset containing only the fields defined in the config,
        trimmed to the requested time window with one sample of look-ahead
        beyond ``time_max``.
    """
    # ------------------------------------------------------------------
    # 1. Load config and resolve the set of NetCDF field names to keep
    # ------------------------------------------------------------------
    cfg = load_config(instrument_type, config_dir)
    info = cfg["instrument_info"]

    v = cfg["variables"]
    # nif_to_key: name_in_file -> canonical JSON key (for renaming after load)
    nif_to_key: dict[str, str] = {
        (entry["name_in_file"] if isinstance(entry, dict) else entry): key
        for key, entry in v.items()
    }
    field_names: set[str] = set(nif_to_key.keys())
    if "corrections" in cfg:
        field_names |= set(cfg["corrections"].values())

    # ------------------------------------------------------------------
    # 2. Discover files
    # ------------------------------------------------------------------
    files = find_arm_files(
        instrument_type, site, facility, data_path,
        time_min, time_max, safety_delta, config_dir,
        data_path_template=data_path_template,
    )

    # ------------------------------------------------------------------
    # 3. Open and concatenate, keeping only config-defined fields
    # ------------------------------------------------------------------
    # Open the first file to grab global provenance attrs AND per-variable
    # attrs. While xarray preserves variable attrs via open_mfdataset, we
    # collect them explicitly here so they can be applied unconditionally
    # in step 5, regardless of any merging edge-cases.
    with xr.open_dataset(str(files[0])) as _ds0:
        source_datastream    = _ds0.attrs.get("datastream", "")
        source_process_version = _ds0.attrs.get("process_version", "")
        file_var_attrs: dict[str, dict] = {
            fv: dict(_ds0[fv].attrs)
            for fv in _ds0.data_vars
            if fv in field_names
        }

    def _preprocess(ds: xr.Dataset) -> xr.Dataset:
        keep = [f for f in field_names if f in ds]
        return ds[keep]

    merged = xr.open_mfdataset(
        [str(fp) for fp in files],
        combine="nested",
        concat_dim="time",
        # "minimal": only concatenate variables that have a time dimension;
        # static variables (correction LUTs, range) come from the first file.
        data_vars="minimal",
        coords="minimal",
        preprocess=_preprocess,
        engine="netcdf4",
    )

    merged = merged.sortby("time")

    # ------------------------------------------------------------------
    # 4. Trim to [time_min - safety_delta, first sample > time_max]
    # ------------------------------------------------------------------
    t_arr = merged.time.values
    t_search_min_ns = (time_min - safety_delta).astype("datetime64[ns]")
    time_max_ns = time_max.astype("datetime64[ns]")

    mask_start = t_arr >= t_search_min_ns
    after_end = t_arr[t_arr > time_max_ns]
    mask_end = t_arr <= (after_end[0] if len(after_end) > 0 else time_max_ns)

    ds = merged.isel(time=mask_start & mask_end)

    if ds.sizes.get("time", 0) == 0:
        raise FileNotFoundError(
            f"No '{instrument_type}' data in '{data_path}' falls within the period "
            f"{time_min} to {time_max} (safety_delta={safety_delta})."
        )

    # ------------------------------------------------------------------
    # 5. Attach field attributes to every data variable
    # ------------------------------------------------------------------
    # Apply original file variable attrs first (fills in long_name, units,
    # valid_min/max, missing_value, etc.), then overwrite / add the three
    # provenance attrs.  Using setdefault for file attrs preserves any attrs
    # already set by xarray (they will be identical in practice).
    for var in ds.data_vars:
        for k, val in file_var_attrs.get(str(var), {}).items():
            ds[var].attrs.setdefault(k, val)
        ds[var].attrs["source_fieldname"]       = str(var)
        ds[var].attrs["source_datastream"]      = source_datastream
        ds[var].attrs["source_process_version"] = source_process_version

    return ds


# ---------------------------------------------------------------------------
# High-level loader + processor
# ---------------------------------------------------------------------------

def load_and_process_arm_data(
    instrument_type: str | list[str],
    site: str,
    facility: str,
    data_path: str | Path | None,
    time_min: np.datetime64,
    time_max: np.datetime64,
    safety_delta: np.timedelta64 = np.timedelta64(5, "m"),
    config_dir: str = "./configs",
    interpolate: bool = True,
    range_km: "np.ndarray | None" = None,
    time_step: np.timedelta64 = np.timedelta64(15, "s"),
    data_path_template: str | None = None,
    deadtime_poly_degree: int = 1,
    deadtime_n_extrap_samples: int = 3,
) -> xr.Dataset:
    """
    Load, optionally correct, and optionally interpolate ARM data.

    This function chains :func:`load_arm_data`, optional NRB corrections
    (when ``instrument_info.requires_corrections`` is ``true`` and
    ``instrument_info.type`` is ``"lidar"``), and optional uniform-grid
    interpolation via :func:`real_prof_utils.interpolate_data`.

    If the instrument config contains an ``instrument_class_highres`` key,
    a second higher-resolution dataset is loaded from the same directory.
    After interpolation, the high-resolution product fills the range bins
    up to its maximum available range; the standard-resolution product
    fills the remaining (higher) range bins.

    Cloud-base and cloud-top fields numbered with a suffix (``cloud_base``,
    ``cloud_base2``, ``cloud_base3``, …) are concatenated along a ``layer``
    dimension of size 10 (first layer = ``cloud_base``, subsequent layers
    from numbered fields; remaining layers are filled with NaN).

    Parameters
    ----------
    instrument_type : str or list of str
        Instrument identifier (class) matching a JSON config file, e.g.
        ``"ceil"`` or ``["ceil", "hsrl", "minimpl"]``.  When a list is
        provided, each instrument is processed independently and the
        resulting datasets are merged; every data variable is prefixed
        with ``"<instrument_type>_"`` to avoid name collisions.
    site : str
        ARM site code
    facility : str
        ARM facility code
    data_path : str, Path, or None
        Directory containing the instrument data files.  Pass ``None``
        when ``data_path_template`` is used instead.
    time_min : np.datetime64
        Start of the desired output time grid.
    time_max : np.datetime64
        End of the desired output time grid.
    safety_delta : np.timedelta64, optional
        Lookback/ahead margin for file discovery.  Default is 5 minutes.
    config_dir : str, optional
        Directory containing JSON configuration files.  Default is
        ``"./configs"``.
    interpolate : bool, optional
        Whether to interpolate onto a uniform grid after loading and
        correcting.  Default is ``True``.
    range_km : np.ndarray, optional
        Output range grid in km.  Passed through to
        :func:`real_prof_utils.interpolate_data`.  If ``None`` the default
        0-20 km / 15 m grid is used.
    time_step : np.timedelta64, optional
        Output time step.  Default is 15 s.
    data_path_template : str, optional
        Format string for the ARM nested archive layout (see
        :func:`find_arm_files`).  When provided, ``data_path`` may be
        ``None`` and the concrete path is derived per-instrument.  This is
        especially convenient when ``instrument_type`` is a list.
    deadtime_poly_degree : int, optional
        Polynomial degree for deadtime correction log-log extrapolation above
        the LUT maximum. Default is 1 (power-law). Set to 0 to clamp to the
        last LUT factor (nearest-neighbour behaviour).
    deadtime_n_extrap_samples : int, optional
        Number of trailing LUT points used to fit the deadtime extrapolation
        polynomial. Ignored when ``deadtime_poly_degree=0``. Default is 3.

    Returns
    -------
    xr.Dataset
        Processed (and optionally interpolated) dataset.  When multiple
        instruments are requested (list), variables from each are prefixed with
        ``"<instrument_type>_"`` and the datasets are merged.
    """
    # ------------------------------------------------------------------
    # Multi-instrument calls: recurse per instrument, prefix, merge
    # ------------------------------------------------------------------
    if isinstance(instrument_type, list):
        datasets = []
        for instr in instrument_type:
            print(f"Loading '{instr}' ...")
            try:
                ds_instr = load_and_process_arm_data(
                    instr, site, facility, data_path,
                    time_min, time_max, safety_delta, config_dir,
                    interpolate, range_km, time_step,
                    data_path_template=data_path_template,
                    deadtime_poly_degree=deadtime_poly_degree,
                    deadtime_n_extrap_samples=deadtime_n_extrap_samples,
                )
            except FileNotFoundError as exc:
                warnings.warn(f"[{instr}] skipped — {exc}", stacklevel=2)
                continue
            rename_map = {var: f"{instr}_{var}" for var in ds_instr.data_vars}
            # Rename any dimension that is not a shared axis (time, range, layer)
            # to avoid conflicts when instruments have different sizes for
            # the same dimension name.
            for dim in ds_instr.dims:
                if dim not in ("time", "range", "layer"):
                    rename_map[dim] = f"{instr}_{dim}"
            ds_instr = ds_instr.rename(rename_map)

            # Pad or trim 'layer' dimension to N_LAYERS so all instruments
            # share a single compatible layer coordinate when merged.
            if "layer" in ds_instr.dims and ds_instr.sizes["layer"] != N_LAYERS:
                cur = ds_instr.sizes["layer"]
                layer_vars = [v for v in ds_instr.data_vars if "layer" in ds_instr[v].dims]
                new_das = {}
                for lv in layer_vars:
                    da = ds_instr[lv]
                    arr = da.values
                    ax = list(da.dims).index("layer")
                    if cur < N_LAYERS:
                        pad_shape = list(arr.shape)
                        pad_shape[ax] = N_LAYERS - cur
                        arr = np.concatenate([arr, np.full(pad_shape, np.nan)], axis=ax)
                    else:
                        idx = [slice(None)] * arr.ndim
                        idx[ax] = slice(None, N_LAYERS)
                        arr = arr[tuple(idx)]
                    new_coords = {d: da.coords[d] for d in da.dims if d in da.coords and d != "layer"}
                    new_coords["layer"] = np.arange(N_LAYERS)
                    new_das[lv] = xr.DataArray(arr, dims=da.dims, coords=new_coords, attrs=da.attrs)
                # Drop all layer-dim vars + the stale layer coord, then re-add
                drop = layer_vars + (["layer"] if "layer" in ds_instr.coords else [])
                ds_instr = ds_instr.drop_vars(drop)
                for lv, new_da in new_das.items():
                    ds_instr = ds_instr.assign({lv: new_da})

            datasets.append(ds_instr)
        if not datasets:
            warnings.warn(
                f"No data found for any of the requested instruments "
                f"for the period {time_min} to {time_max}.",
                stacklevel=2,
            )
            return xr.Dataset()
        return xr.merge(datasets)

    cfg = load_config(instrument_type, config_dir)
    info = cfg["instrument_info"]
    is_lidar = info.get("type", "lidar") == "lidar"
    needs_corrections = is_lidar and info.get("requires_corrections", False)

    # key_to_nif: canonical JSON key -> name_in_file (used by step 4)
    _cfg_vars = cfg["variables"]
    key_to_nif: dict[str, str] = {
        k: (e["name_in_file"] if isinstance(e, dict) else e)
        for k, e in _cfg_vars.items()
    }

    # ------------------------------------------------------------------
    # 1. Load primary dataset
    # ------------------------------------------------------------------
    ds = load_arm_data(
        instrument_type, site, facility, data_path,
        time_min, time_max, safety_delta, config_dir,
        data_path_template=data_path_template,
    )

    # Derive range units from the config, treated as the source of truth
    # for the units of the range coordinate values. This ensures correct
    # handling regardless of whether NetCDF file attributes are missing.
    range_cfg_entry = _cfg_vars.get("range", {})
    range_units = (
        range_cfg_entry.get("units", "m")
        if isinstance(range_cfg_entry, dict)
        else "m"
    )

    # Get the actual field name for range in the loaded dataset
    range_nif = key_to_nif.get("range", "range")

    # ------------------------------------------------------------------
    # Capture native resolutions from primary dataset
    # ------------------------------------------------------------------
    native_time_res = _native_time_res_str(ds.time.values)
    native_range_res = (
        _native_range_res_str(ds["range"], range_units)
        if "range" in ds.coords else "unknown"
    )

    # ------------------------------------------------------------------
    # 2. Load high-resolution dataset if specified
    # ------------------------------------------------------------------
    highres_class = info.get("instrument_class_highres")
    ds_highres = None
    if highres_class is not None:
        # Build a temporary config-like instrument_type by scanning for a
        # matching config file; if not found, attempt to load by class name
        try:
            ds_highres = load_arm_data(
                highres_class, site, facility, data_path,
                time_min, time_max, safety_delta, config_dir,
                data_path_template=data_path_template,
            )
        except FileNotFoundError:
            ds_highres = None
    
    # Capture native resolutions from high-res dataset if available
    native_time_res2 = None
    native_range_res2 = None
    highres_range_units = None
    if ds_highres is not None:
        # Derive range units from highres dataset, not primary dataset
        _hr_range_da = ds_highres.get(range_nif, ds_highres.coords.get(range_nif))
        highres_range_units = (
            _hr_range_da.attrs.get("units", "m") if _hr_range_da is not None else "m"
        )
        native_time_res2 = _native_time_res_str(ds_highres.time.values)
        native_range_res2 = _native_range_res_str(ds_highres["range"], highres_range_units)

    # ------------------------------------------------------------------
    # 3. Apply NRB corrections if required
    # ------------------------------------------------------------------
    correction_field_names: list[str] = []
    if needs_corrections:

        c = cfg.get("corrections", {})
        correction_field_names = list(c.values())

        # Preserve provenance attributes from raw fields that will be replaced
        v = cfg["variables"]
        nrb_co_nif  = (v.get("attenuated_backscatter", {}).get("name_in_file", "")
                       if isinstance(v.get("attenuated_backscatter"), dict)
                       else v.get("attenuated_backscatter", ""))
        nrb_x_nif   = (v.get("attenuated_backscatter_cross_pol", {}).get("name_in_file", "")
                       if isinstance(v.get("attenuated_backscatter_cross_pol"), dict)
                       else v.get("attenuated_backscatter_cross_pol", ""))
        ldr_nif     = (v.get("linear_depol_ratio", {}).get("name_in_file", "")
                       if isinstance(v.get("linear_depol_ratio"), dict)
                       else v.get("linear_depol_ratio", ""))

        raw_co_nif  = (v.get("raw_signal_co_pol", {}).get("name_in_file", "")
                       if isinstance(v.get("raw_signal_co_pol"), dict)
                       else v.get("raw_signal_co_pol", ""))
        raw_x_nif   = (v.get("raw_signal_cross_pol", {}).get("name_in_file", "")
                       if isinstance(v.get("raw_signal_cross_pol"), dict)
                       else v.get("raw_signal_cross_pol", ""))

        ds_corrected = compute_nrb_dataset(
            ds, instrument_type=instrument_type, config_dir=config_dir,
            deadtime_poly_degree=deadtime_poly_degree,
            deadtime_n_extrap_samples=deadtime_n_extrap_samples,
        )

        # Propagate source attributes; read from any variable in ds since
        # load_arm_data stores them uniformly on every variable.
        _ref_var = next(iter(ds.data_vars), None)
        src_ds = ds[_ref_var].attrs.get("source_datastream", "") if _ref_var else ""
        src_pv = ds[_ref_var].attrs.get("source_process_version", "") if _ref_var else ""
        for nif, raw_fields in [
            (nrb_co_nif,  raw_co_nif),
            (nrb_x_nif,   raw_x_nif),
            (ldr_nif,     f"{raw_co_nif},{raw_x_nif}"),
        ]:
            if nif and nif in ds_corrected:
                ds_corrected[nif].attrs["source_fieldname"]       = raw_fields
                ds_corrected[nif].attrs["source_datastream"]      = src_ds
                ds_corrected[nif].attrs["source_process_version"] = src_pv

        # Remove raw input and correction lookup fields from output
        raw_fields_to_drop = [
            raw_co_nif, raw_x_nif,
            v.get("background_co_pol", {}).get("name_in_file", "")
            if isinstance(v.get("background_co_pol"), dict)
            else v.get("background_co_pol", ""),
            v.get("background_cross_pol", {}).get("name_in_file", "")
            if isinstance(v.get("background_cross_pol"), dict)
            else v.get("background_cross_pol", ""),
        ] + correction_field_names
        # Only drop data_vars, not coordinates (e.g., 'range' used in both data and corrections)
        drop = [f for f in raw_fields_to_drop if f and f in ds_corrected.data_vars]
        ds = ds_corrected.drop_vars(drop, errors="ignore")

    # ------------------------------------------------------------------
    # 4. Consolidate cloud_base / cloud_top fields into layer dimension
    # ------------------------------------------------------------------
    for base_key in ("cloud_base", "cloud_top"):
        # Resolve the actual variable names present in ds using nif names
        def _nif(key: str) -> str:
            return key_to_nif.get(key, key)

        base_nif = _nif(base_key)
        if base_nif not in ds:
            continue

        # Case A: base variable already has a layer dimension (e.g. ceilpol's
        # cloud_base_heights is (time, layer)).  Pad or trim to N_LAYERS.
        if "layer" in ds[base_nif].dims:
            da = ds[base_nif]
            cur_layers = da.sizes["layer"]
            if cur_layers != N_LAYERS:
                arr = da.values  # (time, cur_layers)
                if cur_layers < N_LAYERS:
                    pad = np.full((ds.sizes["time"], N_LAYERS - cur_layers), np.nan)
                    arr = np.concatenate([arr, pad], axis=-1)
                else:
                    arr = arr[:, :N_LAYERS]
                da = xr.DataArray(
                    arr,
                    dims=["time", "layer"],
                    coords={"time": da.time, "layer": np.arange(N_LAYERS)},
                    attrs=ds[base_nif].attrs,
                )
                # Drop the old variable and its stale layer coordinate
                # before assigning the resized DataArray; otherwise xarray
                # refuses to align the old size-N index with the new one.
                drop = [base_nif] + (["layer"] if "layer" in ds.coords else [])
                ds = ds.drop_vars(drop)
                ds[base_nif] = da
            continue

        # Case B: separate numbered variables (first_cbh, second_cbh, …)
        # that need to be stacked along a new layer dimension.
        # Look up nif names for each numbered canonical key.
        numbered_nifs = [base_nif] + [
            _nif(f"{base_key}{i}") for i in range(2, 20)
            if _nif(f"{base_key}{i}") in ds
        ]
        present = [n for n in numbered_nifs if n in ds]

        time_vals = ds.time
        layers = []
        for i in range(N_LAYERS):
            if i < len(present):
                layers.append(ds[present[i]].values)
            else:
                layers.append(np.full(ds.sizes["time"], np.nan))

        stacked = np.stack(layers, axis=-1)   # (time, 10)
        merged_da = xr.DataArray(
            stacked,
            dims=["time", "layer"],
            coords={"time": time_vals, "layer": np.arange(N_LAYERS)},
            attrs=ds[present[0]].attrs,
        )
        # source_fieldname lists all input field names
        merged_da.attrs["source_fieldname"] = ",".join(
            ds[n].attrs.get("source_fieldname", n) for n in present
        )
        ds = ds.drop_vars(present)
        ds[base_nif] = merged_da

    # ------------------------------------------------------------------
    # 5. Interpolate onto uniform grid
    # ------------------------------------------------------------------
    if interpolate:
        # Rename range dimension to canonical 'range' if the file uses a
        # different name (e.g. 'height' for interpolatedsonde); required by
        # interpolate_data which always looks for a 'range' dimension.
        range_nif = key_to_nif.get("range", "range")
        if range_nif != "range" and range_nif in ds.dims:
            ds = ds.rename({range_nif: "range"})
        # Identify range-only data variables (no time dimension) before
        # interpolation; they will be absent from the interpolated output
        # and are explicitly dropped so they do not linger.
        range_only_vars = [
            var for var in ds.data_vars
            if "time" not in ds[var].dims
        ]

        ds = interpolate_data(
            ds,
            time_min=time_min,
            time_max=time_max,
            range_km=range_km,
            time_step=time_step,
            range_units=range_units,
        )

        # Drop any range-only variables that may have survived (interpolate_data
        # already excludes them from out_vars, but be explicit for safety).
        ds = ds.drop_vars([v for v in range_only_vars if v in ds], errors="ignore")

        # If high-resolution data available, blend: highres up to its max range,
        # standard data above that
        if ds_highres is not None:
            ds_highres_interp = interpolate_data(
                ds_highres,
                time_min=time_min,
                time_max=time_max,
                range_km=range_km,
                time_step=time_step,
                range_units=highres_range_units,
            )
            highres_max_range = float(
                ds_highres["range"].max()
                / (1000.0 if highres_range_units.lower() == "m" else 1.0)
            )
            out_range = ds["range"].values if "range" in ds.coords else range_km
            for var in ds.data_vars:
                if var not in ds_highres_interp:
                    continue
                if "range" not in ds[var].dims:
                    continue
                blended = xr.where(
                    ds["range"] <= highres_max_range,
                    ds_highres_interp[var],
                    ds[var],
                )
                blended.attrs = ds[var].attrs
                # Add _2 suffixed source attributes from highres data
                hr_field = ds_highres_interp[var].attrs.get("source_fieldname", "")
                if hr_field:
                    blended.attrs["source_fieldname_2"] = hr_field
                hr_src = ds_highres_interp[var].attrs.get("source_datastream", "")
                if hr_src:
                    blended.attrs["source_datastream_2"] = hr_src
                hr_pv = ds_highres_interp[var].attrs.get("source_process_version", "")
                if hr_pv:
                    blended.attrs["source_process_version_2"] = hr_pv
                ds[var] = blended

    # ------------------------------------------------------------------
    # Attach temporal and vertical resolution attributes to all variables
    # ------------------------------------------------------------------
    for var in ds.data_vars:
        # All variables get temporal_resolution
        ds[var].attrs["temporal_resolution"] = native_time_res
        
        # Range-dimensioned variables get vertical_resolution (in km)
        if "range" in ds[var].dims:
            ds[var].attrs["vertical_resolution"] = native_range_res
        
        # Layer-dimensioned variables also get vertical_resolution (1 layer unit in km)
        if "layer" in ds[var].dims:
            ds[var].attrs["vertical_resolution"] = native_range_res
        
        # If HSRL high-res blending occurred, add _2 suffixed attrs
        if ds_highres is not None:
            # Temporal resolution applies to all variables with time dimension
            if "time" in ds[var].dims:
                ds[var].attrs["temporal_resolution_2"] = native_time_res2
            # Vertical resolution applies to variables with range dimension
            if "range" in ds[var].dims:
                ds[var].attrs["vertical_resolution_2"] = native_range_res2

    # ------------------------------------------------------------------
    # Final step: rename variables from name_in_file to canonical JSON key
    # ------------------------------------------------------------------
    # Drop variables with dimensions outside the canonical set
    # ------------------------------------------------------------------
    canonical_dims = {"time", "range", "layer"}
    drop_extra = [
        var for var in ds.data_vars
        if not set(ds[var].dims).issubset(canonical_dims)
    ]
    if drop_extra:
        ds = ds.drop_vars(drop_extra)
    # Drop any orphaned dimensions (and their coordinates)
    orphan_dims = [d for d in ds.dims if d not in canonical_dims]
    if orphan_dims:
        ds = ds.drop_dims(orphan_dims)

    # ------------------------------------------------------------------
    cfg_final = load_config(instrument_type, config_dir)
    v_final = cfg_final["variables"]
    nif_to_key: dict[str, str] = {
        (entry["name_in_file"] if isinstance(entry, dict) else entry): key
        for key, entry in v_final.items()
    }
    rename_map = {
        str(var): nif_to_key[str(var)]
        for var in ds.data_vars
        if str(var) in nif_to_key and nif_to_key[str(var)] != str(var)
    }
    if rename_map:
        ds = ds.rename(rename_map)

    # ------------------------------------------------------------------
    # Unit conversions — applied after rename so canonical names are used
    # ------------------------------------------------------------------
    # cloud_base / cloud_top: m -> km (only meaningful on the km range grid)
    if interpolate:
        for var in ds.data_vars:
            if "cloud_base" in var or "cloud_top" in var:
                if ds[var].attrs.get("units", "").lower() == "m":
                    attrs = dict(ds[var].attrs)
                    ds[var] = ds[var] / 1000.0
                    attrs["units"] = "km"
                    ds[var].attrs = attrs

    # pressure: Pa or kPa -> hPa (applied unconditionally)
    _PRES_SCALE = {"pa": 1e-2, "kpa": 10.0}
    for var in ds.data_vars:
        if "pres" in var:
            u = ds[var].attrs.get("units", "")
            scale = _PRES_SCALE.get(u.lower())
            if scale is not None:
                attrs = dict(ds[var].attrs)
                ds[var] = ds[var] * scale
                attrs["units"] = "hPa"
                ds[var].attrs = attrs

    # Attach site/facility/datastream to global attributes
    info_final = cfg_final["instrument_info"]
    instrument_class = info_final["instrument_class"]
    level = info_final["level"]
    ds.attrs["site"]       = site
    ds.attrs["facility"]   = facility
    ds.attrs["datastream"] = f"{site}{instrument_class}{facility}.{level}"

    # Order variables: backscatter → extinction → depol → other, qc_ after parent
    ds = ds[_ordered_vars(list(ds.data_vars))]

    return ds


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_dataset(
    ds: xr.Dataset,
    site: str,
    facility: str,
    output_path: str | Path = "./",
    level: str = "c0",
    instrument_class: str = "realbundle",
) -> Path:
    """
    Export a processed dataset to a NetCDF4-Classic file.

    The output filename follows the ARM convention::

        {site}{instrument_class}{facility}.{level}.YYYYMMDD.HHMMSS.nc

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to export.
    site : str
        ARM site code.
    facility : str
        ARM facility code.
    output_path : str or Path, optional
        Directory where the file will be written.  Default is ``"./"``.
    level : str, optional
        Data level string used in the filename.  Default is ``"c0"``.
    instrument_class : str, optional
        Instrument class string used in the filename.  Default is
        ``"realbundle"``.

    Returns
    -------
    Path
        Absolute path of the written file.
    """

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Derive timestamp from first time sample
    t0 = ds.time.values[0]
    dt_str = str(t0)[:19].replace("-", "").replace("T", ".").replace(":", "")
    # dt_str is now "YYYYMMDD.HHMMSS"

    filename = f"{site}{instrument_class}{facility}.{level}.{dt_str}.nc"
    out_file = output_path / filename

    # Stamp creation time in UTC and set output-level global attributes
    ds = ds.copy()
    ds.attrs["creation_date"] = (
        datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    ds.attrs["site"]        = site
    ds.attrs["facility"]    = facility
    ds.attrs["datastream"]  = f"{site}{instrument_class}{facility}.{level}"

    # Rebuild the dataset so that dimension coordinates (time, range, layer)
    # are written first.
    _DIM_COORD_ORDER = ("time", "range", "layer")
    ordered_coords = {k: ds.coords[k] for k in _DIM_COORD_ORDER if k in ds.coords}
    other_coords   = {k: ds.coords[k] for k in ds.coords if k not in ordered_coords}
    ds_out = xr.Dataset(coords={**ordered_coords, **other_coords}, attrs=ds.attrs)
    for v in ds.data_vars:
        ds_out[v] = ds[v]
    
    # Set range coordinate attributes if range exists
    if "range" in ds_out.coords:
        ds_out["range"].attrs["long_name"] = "Range to measurement volume"
        ds_out["range"].attrs["units"] = "km"

    # Materialise any dask-backed arrays before writing.  open_mfdataset
    # returns lazy arrays; driving them through to_netcdf via the dask
    # threaded scheduler together with zlib compression can cause a deadlock.
    # Loading into memory first avoids that entirely.
    ds_out = ds_out.load()

    # Build per-variable encoding: zlib compression level 9 for all variables
    encoding = {
        var: {"zlib": True, "complevel": 9}
        for var in ds_out.data_vars
    }

    ds_out.to_netcdf(
        str(out_file),
        format="NETCDF4_CLASSIC",
        encoding=encoding,
    )

    return out_file.resolve()


# ---------------------------------------------------------------------------
# Geographic metadata extraction
# ---------------------------------------------------------------------------

def extract_geographic_metadata(
    instrument_types: list[str],
    site: str,
    facility: str,
    data_path: str | Path | None,
    time_min: np.datetime64,
    time_max: np.datetime64,
    safety_delta: np.timedelta64 = np.timedelta64(5, "m"),
    config_dir: str = "./configs",
    data_path_template: str | None = None,
) -> dict[str, xr.DataArray]:
    """
    Extract geographic metadata (lat, lon, alt) from the first available data file.

    Attempts to locate and open data files for each instrument in order, extracting
    the 0-D geographic coordinate fields (lat, lon, alt) from the first successful
    file found. Returns a dictionary mapping field names to 0-D DataArrays with
    full attributes preserved.

    Parameters
    ----------
    instrument_types : list[str]
        List of instrument identifiers to search, in priority order.
    site : str
        ARM site code.
    facility : str
        ARM facility code.
    data_path : str, Path, or None
        Directory containing instrument data files. Pass ``None``
        when ``data_path_template`` is used instead.
    time_min : np.datetime64
        Start of time window for file discovery.
    time_max : np.datetime64
        End of time window for file discovery.
    safety_delta : np.timedelta64, optional
        Lookback/ahead margin for file discovery. Default is 5 minutes.
    config_dir : str, optional
        Directory containing JSON configuration files. Default is ``"./configs"``.
    data_path_template : str, optional
        Format string for the ARM nested archive layout. Used when ``data_path``
        is ``None``.

    Returns
    -------
    dict[str, xr.DataArray]
        Dictionary mapping field names ("lat", "lon", "alt") to 0-D DataArrays
        with preserved attributes. Returns only fields that were found; missing
        fields are omitted from the dictionary.
    """
    geo_data: dict[str, xr.DataArray] = {}

    for instr in instrument_types:
        try:
            files = find_arm_files(
                instr, site, facility, data_path,
                time_min, time_max, safety_delta, config_dir,
                data_path_template=data_path_template,
            )
            if files:
                with xr.open_dataset(str(files[0])) as raw_ds:
                    for field in ["lat", "lon", "alt"]:
                        if field in raw_ds:
                            # Keep the full 0-D DataArray with attributes
                            geo_data[field] = raw_ds[field]
                    if geo_data:
                        break
        except Exception:
            continue

    return geo_data
