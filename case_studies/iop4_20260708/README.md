# IOP4, 8 July 2026: EarthCARE ATLID, SGP sonde and two-way transmission T²

`iop4_20260708_atlid_sonde_T2.ipynb` rebuilds the IOP4 figure step by step:

1. fetch the SGP C1 sondes (ARM Live) and the ATLID L1B granule `ATL_NOM_1B` 11995D (ESA MAAP),
2. locate the closest approach (19.8 km east of C1, 20:47:45 UTC),
3. process the 20:30 UTC sonde (25 m layers, MetPy parcel) and map ATLID heights to pressure,
4. retrieve the two-way particle transmission T² from the Rayleigh channel, and show how to use it
   (AOD above height, extinction, attenuation-corrected backscatter, layer lidar ratio),
5. plot skew-T, Mie, Rayleigh and T² on a common pressure axis (`figures/`).

Credentials come from environment variables (`ARM_USERNAME`, `ARM_TOKEN`, `MAAP_OFFLINE_TOKEN`);
the notebook falls back to `"enter key here"` placeholders. Data are cached in `data/` (git-ignored).

Data: ARM `sgpsondewnpnC1.b1` (doi:10.5439/1595321); ESA EarthCARE ATLID L1B (doi:10.57780/esa-98de183).
