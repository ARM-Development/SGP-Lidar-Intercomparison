"""
===============================================================
Israel Silber
===============================================================
Elastic lidar profile correction functions for the REAL intercomparison.
Includes:
- Deadtime correction via count-rate LUT
- Afterpulse correction via range LUTs
- Background subtraction using a per-profile background time series
- Range-squared correction
- Overlap correction via range LUT
===============================================================
"""
import numpy as np
import xarray as xr

from real_prof_init import load_config


def apply_deadtime_correction(
    raw_signal: np.ndarray,
    correction_counts: np.ndarray,
    correction_factors: np.ndarray,
    poly_degree: int = 1,
    n_extrap_samples: int = 3,
) -> np.ndarray:
    """
    Apply deadtime correction to raw photon-counting lidar signals using
    a precomputed lookup table (LUT).

    Values within the LUT range are linearly interpolated.  Values above the
    maximum LUT count rate are extrapolated via a polynomial fit in log-log
    space fitted to the last ``n_extrap_samples`` LUT points, or clamped
    if ``poly_degree=0``.  Values below the minimum LUT count rate are
    clamped to the first LUT factor.

    Parameters
    ----------
    raw_signal : np.ndarray
        2D array of shape (n_profiles, n_range_bins) containing raw
        photon-counting lidar signal in counts/us. Must be the raw detector
        output before any other correction.
    correction_counts : np.ndarray
        1D array of shape (n_lut,) containing the count-rate axis of the
        deadtime correction LUT in counts/us. Must be monotonically
        increasing.
    correction_factors : np.ndarray
        1D array of shape (n_lut,) containing the multiplicative correction
        factor at each count rate. A value of 1.0 means no correction;
        values > 1.0 indicate the true signal exceeds the measured signal.
    poly_degree : int, optional
        Polynomial degree for log-log extrapolation above the LUT maximum.
        Default is 1 (power-law). Set to 0 to clamp to the last LUT factor.

    n_extrap_samples : int, optional
        Number of trailing LUT points used to fit the extrapolation polynomial.
        Ignored when ``poly_degree=0``. Default is 3.

    Returns
    -------
    corrected : np.ndarray
        2D array of shape (n_profiles, n_range_bins) with deadtime correction
        applied, in counts/us.
    """
    if not isinstance(poly_degree, int) or poly_degree < 0:
        raise ValueError(f"poly_degree must be a non-negative integer, got {poly_degree}")

    flat = raw_signal.ravel()
    interp_factors = np.interp(flat, correction_counts, correction_factors)

    if poly_degree > 0:
        upper_mask = flat > correction_counts[-1]
        if upper_mask.any():
            n = min(n_extrap_samples, len(correction_counts))
            log_x_fit = np.log(correction_counts[-n:])
            log_y_fit = np.log(correction_factors[-n:])
            coeffs = np.polyfit(log_x_fit, log_y_fit, poly_degree)
            interp_factors[upper_mask] = np.exp(
                np.polyval(coeffs, np.log(flat[upper_mask]))
            )

    return raw_signal * interp_factors.reshape(raw_signal.shape)


def apply_afterpulse_correction(
    signal: np.ndarray,
    range_km: np.ndarray,
    afterpulse_range: np.ndarray,
    afterpulse_profile: np.ndarray,
    darkcounts_range: np.ndarray,
    darkcounts_profile: np.ndarray,
) -> np.ndarray:
    """
    Apply afterpulse correction to a deadtime-corrected lidar signal.

    Parameters
    ----------
    signal : np.ndarray
        2D array of shape (n_profiles, n_range_bins) containing the
        deadtime-corrected lidar signal in counts/us.
    range_km : np.ndarray
        1D array of shape (n_range_bins,) with range gate centres in km
        onto which the LUT profiles are interpolated.
    afterpulse_range : np.ndarray
        1D array of shape (n_lut_ap,) containing the range axis of the
        afterpulse LUT in km.
    afterpulse_profile : np.ndarray
        1D array of shape (n_lut_ap,) containing the afterpulse signal
        in counts/us.
    darkcounts_range : np.ndarray
        1D array of shape (n_lut_dc,) containing the range axis of the
        dark-count correction LUT in km.
    darkcounts_profile : np.ndarray
        1D array of shape (n_lut_dc,) containing the dark-count correction
        profile in counts/us.

    Returns
    -------
    corrected : np.ndarray
        2D array of shape (n_profiles, n_range_bins) with afterpulse
        subtracted, in counts/us.
    """
    # Interpolate both profiles onto the signal range grid
    ap_interp = np.interp(range_km, afterpulse_range, afterpulse_profile)
    dc_interp = np.interp(range_km, darkcounts_range, darkcounts_profile)

    return signal - (ap_interp - dc_interp)


def apply_background_subtraction(
    signal: np.ndarray,
    background: np.ndarray,
) -> np.ndarray:
    """
    Subtract a per-profile background signal from the lidar signal.

    Parameters
    ----------
    signal : np.ndarray
        2D array of shape (n_profiles, n_range_bins) containing the
        afterpulse-corrected lidar signal in counts/us.
    background : np.ndarray
        1D array of shape (n_profiles,) containing the background signal
        level for each profile in counts/us (e.g. derived from far-range
        bins or a co-located dark measurement).

    Returns
    -------
    corrected : np.ndarray
        2D array of shape (n_profiles, n_range_bins) with background
        subtracted, in counts/us.
    """
    if background.shape[0] != signal.shape[0]:
        raise ValueError(
            f"background length ({background.shape[0]}) must match "
            f"n_profiles ({signal.shape[0]})."
        )

    # background[:, np.newaxis] : (n_profiles, 1) broadcasts over all bins
    return signal - background[:, np.newaxis]


def apply_range_correction(
    signal: np.ndarray,
    range_km: np.ndarray,
) -> np.ndarray:
    """
    Apply range-squared correction.

    Parameters
    ----------
    signal : np.ndarray
        2D array of shape (n_profiles, n_range_bins) in counts/us.
    range_km : np.ndarray
        1D array of shape (n_range_bins,) with range gate centres in km.

    Returns
    -------
    corrected : np.ndarray
        2D array of shape (n_profiles, n_range_bins) in counts/us * km^2.
    """
    return signal * range_km ** 2


def apply_overlap_correction(
    signal: np.ndarray,
    overlap_range: np.ndarray,
    overlap_factors: np.ndarray,
    range_km: np.ndarray,
) -> np.ndarray:
    """
    Apply overlap correction by interpolating an overlap LUT onto the
    range grid of the signal and dividing.
    Bins where the interpolated overlap is zero are set to NaN to avoid
    division by zero.

    Parameters
    ----------
    signal : np.ndarray
        2D array of shape (n_profiles, n_range_bins) in counts/us * km^2.
    overlap_range : np.ndarray
        1D array of shape (n_lut,) containing the range axis of the overlap
        LUT in km. Must be monotonically increasing.
    overlap_factors : np.ndarray
        1D array of shape (n_lut,) containing the overlap correction factors
        in [0, 1] at each range in overlap_range.
    range_km : np.ndarray
        1D array of shape (n_range_bins,) with the range gate centres in km
        onto which the LUT is interpolated.

    Returns
    -------
    corrected : np.ndarray
        2D array of shape (n_profiles, n_range_bins) with overlap correction
        applied. Bins with zero overlap are NaN.
    """
    # Interpolate overlap LUT onto the signal range grid
    # Out-of-range bins are extrapolated using the nearest boundary value
    overlap_interp = np.interp(range_km, overlap_range, overlap_factors)

    # Zero-overlap bins cannot be corrected
    overlap_safe = np.where(overlap_interp == 0.0, np.nan, overlap_interp)

    return signal / overlap_safe   # broadcasts (n_profiles, n_bins) / (n_bins,)


def compute_nrb(
    raw_signal: np.ndarray,
    range_km: np.ndarray,
    background: np.ndarray,
    deadtime_correction_counts: np.ndarray,
    deadtime_correction_factors: np.ndarray,
    afterpulse_range: np.ndarray,
    afterpulse_profile: np.ndarray,
    darkcounts_range: np.ndarray,
    darkcounts_profile: np.ndarray,
    overlap_range: np.ndarray,
    overlap_factors: np.ndarray,
    calibration_constant: float = 1.0,
    deadtime_poly_degree: int = 1,
    deadtime_n_extrap_samples: int = 3,
) -> np.ndarray:
    """
    Compute the Normalized Relative Backscatter (NRB) from raw lidar data.

    Processing steps applied in order:

    1. **Deadtime correction** -- corrects photon pile-up via a count-rate LUT.
    2. **Afterpulse correction** -- removes detector afterpulse after
       subtracting its dark-count component.
    3. **Background subtraction** -- removes per-profile solar background and
       dark current provided as a time series.
    4. **Range-squared correction** -- compensates geometric signal decay.
    5. **Overlap correction** -- corrects near-field beam/FOV mismatch via a
       range LUT.
    6. **Calibration scaling** -- divides by the instrument calibration constant.

    Parameters
    ----------
    raw_signal : np.ndarray
        2D array of shape (n_profiles, n_range_bins) with raw lidar signal
        in counts/us.
    range_km : np.ndarray
        1D array of shape (n_range_bins,) with range gate centres in km.
    background : np.ndarray
        1D array of shape (n_profiles,) with the background signal level per
        profile in counts/us.
    deadtime_correction_counts : np.ndarray
        1D array of shape (n_lut_dt,) with count-rate axis of deadtime LUT
        in counts/us. Must be monotonically increasing.
    deadtime_correction_factors : np.ndarray
        1D array of shape (n_lut_dt,) with multiplicative deadtime correction
        factors.
    afterpulse_range : np.ndarray
        1D array of shape (n_lut_ap,) with range axis of the afterpulse LUT
        in km.
    afterpulse_profile : np.ndarray
        1D array of shape (n_lut_ap,) with afterpulse signal in counts/us.
    darkcounts_range : np.ndarray
        1D array of shape (n_lut_dc,) with range axis of the dark-count LUT
        in km.
    darkcounts_profile : np.ndarray
        1D array of shape (n_lut_dc,) with dark-count correction profile in
        counts/us.
    overlap_range : np.ndarray
        1D array of shape (n_lut_ol,) with range axis of overlap LUT in km.
        Must be monotonically increasing.
    overlap_factors : np.ndarray
        1D array of shape (n_lut_ol,) with overlap correction factors in [0,1].
    calibration_constant : float, optional
        Instrument calibration constant C. Use 1.0 for relative NRB.
        Default is 1.0.
    deadtime_poly_degree : int, optional
        Polynomial degree for deadtime correction log-log extrapolation above
        the LUT maximum. Default is 1 (power-law). Set to 0 to clamp to the
        last LUT factor (nearest-neighbour behaviour).
    deadtime_n_extrap_samples : int, optional
        Number of trailing LUT points used to fit the deadtime extrapolation
        polynomial. Ignored when ``deadtime_poly_degree=0``. Default is 3.

    Returns
    -------
    nrb : np.ndarray
        2D array of shape (n_profiles, n_range_bins) with NRB values in
        counts/us * km^2. Bins with zero overlap are NaN.
    """
    # Step 1 — deadtime correction
    signal = apply_deadtime_correction(
        raw_signal,
        deadtime_correction_counts,
        deadtime_correction_factors,
        poly_degree=deadtime_poly_degree,
        n_extrap_samples=deadtime_n_extrap_samples,
    )

    # Step 2 — afterpulse correction
    signal = apply_afterpulse_correction(
        signal,
        range_km,
        afterpulse_range,
        afterpulse_profile,
        darkcounts_range,
        darkcounts_profile,
    )

    # Step 3 — background subtraction
    signal = apply_background_subtraction(signal, background)

    # Step 4 — range-squared correction
    signal = apply_range_correction(signal, range_km)

    # Step 5 — overlap correction
    signal = apply_overlap_correction(
        signal,
        overlap_range,
        overlap_factors,
        range_km,
    )

    # Step 6 — calibration scaling
    return signal / calibration_constant

def compute_nrb_dataset(
    ds: "xr.Dataset",
    instrument_type: str = None,
    *,
    variables: dict = None,
    corrections: dict = None,
    calibration_constant: float = 1.0,
    cross_pol: bool = True,
    config_dir: str = "./configs",
    deadtime_poly_degree: int = 1,
    deadtime_n_extrap_samples: int = 3,
) -> "xr.Dataset":
    """
    Compute co-pol NRB and, optionally, cross-pol NRB and linear depolarization
    ratio (LDR) from an elastic lidar xarray Dataset.

    Field names can be provided in two mutually exclusive ways:

    1. **Config file** (recommended): pass ``instrument_type`` and the function
       loads ``<config_dir>/<instrument_type>.json`` via
       :func:`prof_init.load_config`.

    2. **Manual dicts**: pass ``variables`` and ``corrections`` directly,
       using the same key structure as the JSON file.

    The LDR is defined as:

        LDR = NRB_cross / (NRB_co + NRB_cross)

    Parameters
    ----------
    ds : xr.Dataset
        Dataset loaded from a lidar NetCDF file.
    instrument_type : str, optional
        Instrument identifier used to locate the JSON config file, e.g.
        ``"minimpl"`` or ``"mpl"``. 
        Mutually exclusive with ``variables`` and ``corrections``.
    variables : dict, optional
        Manual mapping of standard variable names to file field names.
    corrections : dict, optional
        Manual mapping of standard correction keys to file field names
        (flat strings).
    calibration_constant : float, optional
        Instrument calibration constant. Default is 1.0.
    cross_pol : bool, optional
        Whether cross-pol data are present. If True (default), ``nrb_cross``
        and ``ldr`` are computed and included in the output. If False, only
        ``nrb_co`` is returned.
    config_dir : str, optional
        Directory containing the JSON configuration files. Only used when
        ``instrument_type`` is provided. Default is ``"./configs"``.
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
        Dataset with variable ``nrb_co`` and, when ``cross_pol=True``,
        also ``nrb_cross`` and ``ldr``.
    """
    # Resolve field-name mappings
    if instrument_type is not None and (variables is not None or corrections is not None):
        raise ValueError(
            "Provide either 'instrument_type' or 'variables'/'corrections', not both."
        )
    if instrument_type is not None:
        _cfg = load_config(instrument_type, config_dir)
        v = _cfg["variables"]
        c = _cfg["corrections"]
    elif variables is not None and corrections is not None:
        v = variables
        c = corrections
    else:
        raise ValueError(
            "Provide either 'instrument_type' or both 'variables' and 'corrections'."
        )

    def _name(entry):
        """Accept a plain field-name string or a {name_in_file, ...} dict."""
        return entry if isinstance(entry, str) else entry["name_in_file"]

    def _units(entry):
        """Return units string, or empty string if not specified."""
        return "" if isinstance(entry, str) else entry.get("units", "")

    # Extract all arrays from the dataset upfront
    range_km         = ds[_name(v["range"])].values
    
    # Extract raw co-pol signal and ensure (time, range_or_spatial_dim) layout
    raw_co_da = ds[_name(v["raw_signal_co_pol"])]
    # Find the spatial dimension name (could be 'range', 'range_bins', 'height', etc.)
    dims_list = list(raw_co_da.dims)
    time_dim = next((d for d in dims_list if d == "time"), None)
    spatial_dim = next((d for d in dims_list if d != "time"), None)
    
    if time_dim and spatial_dim and [time_dim, spatial_dim] != dims_list:
        raw_co_da = raw_co_da.transpose(time_dim, spatial_dim)
    raw_co = raw_co_da.values
    
    background_co    = ds[_name(v["background_co_pol"])].values
    dt_counts        = ds[c["deadtime_counts"]].values
    dt_factors       = ds[c["deadtime_factors"]].values
    ap_range         = ds[c["afterpulse_range"]].values
    ap_profile_co    = ds[c["afterpulse_profile_co_pol"]].values
    dc_range         = ds[c["darkcounts_range"]].values
    dc_profile_co    = ds[c["darkcounts_profile_co_pol"]].values
    overlap_range    = ds[c["overlap_range"]].values
    overlap_factors  = ds[c["overlap_factors"]].values

    nrb_co = compute_nrb(
        raw_signal                  = raw_co,
        range_km                    = range_km,
        background                  = background_co,
        deadtime_correction_counts  = dt_counts,
        deadtime_correction_factors = dt_factors,
        afterpulse_range            = ap_range,
        afterpulse_profile          = ap_profile_co,
        darkcounts_range            = dc_range,
        darkcounts_profile          = dc_profile_co,
        overlap_range               = overlap_range,
        overlap_factors             = overlap_factors,
        calibration_constant        = calibration_constant,
        deadtime_poly_degree        = deadtime_poly_degree,
        deadtime_n_extrap_samples   = deadtime_n_extrap_samples,
    )

    dims = ("time", "range")
    data_vars = {
        _name(v["attenuated_backscatter"]): (
            dims, nrb_co,
            {"units": _units(v["attenuated_backscatter"]), "long_name": "NRB co-pol"},
        ),
    }

    if cross_pol:
        # Extract raw cross-pol signal and ensure (time, range_or_spatial_dim) layout
        raw_cross_da = ds[_name(v["raw_signal_cross_pol"])]
        dims_list = list(raw_cross_da.dims)
        time_dim = next((d for d in dims_list if d == "time"), None)
        spatial_dim = next((d for d in dims_list if d != "time"), None)
        
        if time_dim and spatial_dim and [time_dim, spatial_dim] != dims_list:
            raw_cross_da = raw_cross_da.transpose(time_dim, spatial_dim)
        raw_cross = raw_cross_da.values
        
        background_cross = ds[_name(v["background_cross_pol"])].values
        ap_profile_cross = ds[c["afterpulse_profile_cross_pol"]].values
        dc_profile_cross = ds[c["darkcounts_profile_cross_pol"]].values

        nrb_cross = compute_nrb(
            raw_signal                  = raw_cross,
            range_km                    = range_km,
            background                  = background_cross,
            deadtime_correction_counts  = dt_counts,
            deadtime_correction_factors = dt_factors,
            afterpulse_range            = ap_range,
            afterpulse_profile          = ap_profile_cross,
            darkcounts_range            = dc_range,
            darkcounts_profile          = dc_profile_cross,
            overlap_range               = overlap_range,
            overlap_factors             = overlap_factors,
            calibration_constant        = calibration_constant,
            deadtime_poly_degree        = deadtime_poly_degree,
            deadtime_n_extrap_samples   = deadtime_n_extrap_samples,
        )
        ldr = nrb_cross / (nrb_co + nrb_cross)
        data_vars[_name(v["attenuated_backscatter_cross_pol"])] = (
            dims, nrb_cross,
            {"units": _units(v["attenuated_backscatter_cross_pol"]), "long_name": "NRB cross-pol"},
        )
        data_vars[_name(v["linear_depol_ratio"])] = (
            dims, ldr,
            {"units": _units(v["linear_depol_ratio"]),
             "long_name": "Linear depolarization ratio"},
        )

    return xr.Dataset(
        data_vars,
        coords={
            "time":  ds[_name(v["time"])],
            "range": range_km,
        },
    )


# -----------------------------------------------------------------------
# Demo
# -----------------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    
    # ------------------------------------------------------------------
    # Load data and compute NRB dataset
    # ------------------------------------------------------------------
    FILE = "/data/archive/sgp/sgpminimplC1.b1/sgpminimplC1.b1.20260214.000009.nc"
    FILE = "/data/archive/sgp/sgpminimplC1.b1/sgpminimplC1.b1.20260612.000004.nc"  # test case for Donna
    FILE = "/data/archive/kcg/kcgmplpolfsM1.b1/kcgmplpolfsM1.b1.20240601.000009.nc"
    FILE = "/data/archive/kcg/kcgminimplS1.b1/kcgminimplS1.b1.20240601.000000.nc"  # test case from Damao
    ds     = xr.open_dataset(FILE)
    result = compute_nrb_dataset(ds, instrument_type="minimpl", config_dir="./configs",
                                 deadtime_poly_degree=1, deadtime_n_extrap_samples=3)

    has_cross    = "ldr" in result
    max_range_km = 5.0                         # maximum y-axis range (km)

    # Build plot dataset — keep only up to max_range_km
    range_sel = result.range <= max_range_km
    plot_ds = result.sel(range=range_sel)

    # Add log10-transformed fields (raw signal and NRB co); LDR stays linear
    range_mask = ds["range"].values <= max_range_km
    raw_vals   = ds["signal_return_co_pol"].values[:, range_mask]
    plot_ds["raw"] = xr.DataArray(
        np.log10(np.where(raw_vals > 0, raw_vals, np.nan)),
        dims=["time", "range"],
        coords={"time": plot_ds.time, "range": plot_ds.range},
    )
    plot_ds["log_nrb_co"] = np.log10(plot_ds["nrb_co"].where(plot_ds["nrb_co"] > 0))

    # (var, colorbar label, colormap, profile colour, title, fixed vmin, fixed vmax)
    # None means derive vmin/vmax from 5th/95th percentile
    rows = [
        ("raw",        "log₁₀ counts/µs",       "jet",   "steelblue",  "Raw Signal",   None, None),
        ("log_nrb_co", "log₁₀ counts/µs · km²", "jet",   "darkorange", "NRB (co-pol)", None, None),
    ]
    if has_cross:
        rows.append(("ldr", "LDR", "jet", "firebrick", "LDR", 0.0, 0.6))

    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, 1, figsize=(18, 5 * n_rows),
                             constrained_layout=True)
    if n_rows == 1:
        axes = [axes]

    for ax_c, (var, cb_label, cmap, _, title, fixed_vmin, fixed_vmax) in zip(axes, rows):
        da   = plot_ds[var]
        vmin = fixed_vmin if fixed_vmin is not None else float(da.quantile(0.01))
        vmax = fixed_vmax if fixed_vmax is not None else float(da.quantile(0.99))

        da.plot.pcolormesh(
            ax=ax_c, x="time", y="range",
            cmap=cmap, vmin=vmin, vmax=vmax,
            #cmap=cmap, vmin=0.0, vmax=0.08,
            cbar_kwargs={"label": cb_label},
        )
        ax_c.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        t_base = plot_ds.time.values[0].astype('datetime64[D]')
        #ax_c.set_xlim(t_base + np.timedelta64(14, 'h'), t_base + np.timedelta64(17, 'h'))
        ax_c.set_ylim(float(plot_ds.range[0]), max_range_km)
        #ax_c.set_ylim((0.5, 1.5))
        ax_c.set_title(f"{title} Curtain")
        ax_c.set_xlabel("Time (UTC)")
        ax_c.set_ylabel("Range (km)")

    fig.autofmt_xdate(rotation=30)
    plt.savefig("nrb_full_pipeline.png", dpi=150)
    plt.show()