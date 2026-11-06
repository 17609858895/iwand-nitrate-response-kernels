# Signed rain-response kernels for sparse nitrate monitoring

Core analysis code for *Signed rain-response kernels reveal contrasting nitrate dynamics across sparsely monitored US catchments*.

The code estimates signed, attribute-regionalized rainfall-response shapes and updates local amplitudes using empirical Bayes partial pooling. It includes the sampling, sensor-reference, temporal-sensitivity and annual nitrogen-history comparisons described in the manuscript.

## Data and environment

This repository contains essential analysis scripts and a shared helper. The manuscript's Supplementary Code additionally contains compact fitted outputs and publication-figure sources. Raw observations, climate arrays, fitted outputs, manuscripts and graphics are not distributed here.

Obtain IWAND-Nitrogen version 1.0 from <https://doi.org/10.5281/zenodo.17677638>. Place `IWAND-Nitrogen-CSV.zip` at the repository root. Preparation creates `data_processed/`; analyses create `results/`. Allow sufficient space for the daily forcing arrays. Required packages appear in `requirements.txt`; `environment-recorded.txt` lists the local environment versions.

## Main analysis

Run from the repository root. These commands perform fitting and can require substantial memory and computation.

```sh
python scripts/01_build_features.py data_processed
python scripts/21_rain_history_bins.py data_processed
python scripts/22_rain_skdml.py real all
python scripts/22_rain_skdml.py placebo all
python scripts/29_rain_skdml_checks.py all shapes,boot,state,eb data_processed
```

Synthetic tasks replace the outcome and must each run alone:

```sh
python scripts/29_rain_skdml_checks.py all synth data_processed
python scripts/29_rain_skdml_checks.py all ebsynth data_processed
```

Repeat response and timing fits for `1980-1999`, `2000-2011` and `2012-2023`; script 30 combines temporal summaries. `PLACEBO=1` selects mirrored future histories in script 29. `QUICK=1` reduces fitting for smoke runs and does not reproduce the reported results.

## Supporting comparisons

| Comparison | Scripts |
|---|---|
| Lag, seasonal-adjustment and wetness diagnostics | 17–20, 25–26 |
| Sampling-calendar contrasts | 23, 25 |
| Discharge acquisition and diagnostics | 31–33 |
| Dense nitrate references and period-matched timing | 35–37 |
| Sensor-site local amplitude updates | 29 (`eb`), 39 |
| Annual nitrogen histories | 06, 13–14 |
| Auxiliary signature and sequence prediction | 03, 05 |

The acquisition script uses the USGS National Water Information System daily-values service:

```sh
python scripts/31_download_nwis_dv.py 00060 nwis_dv
python -I scripts/32_parse_nwis_dv.py data_external/nwis_dv 00060 nwis_dv Q_cfs
python scripts/31_download_nwis_dv.py 99133,99137 nwis_sensor_no3
python -I scripts/32_parse_nwis_dv.py data_external/nwis_sensor_no3 99133,99137 nwis_sensor_no3 NO3_mgL
python scripts/35_sensor_daily_panel.py data_processed
```

After preparing the sensor panel, set `HOLDOUT` to `data_processed/sensor_sites.csv` and run script 29's `holdout` task before script 36. Run `eb` with the same exclusion file before script 39. Use your shell's environment syntax, e.g. `$env:HOLDOUT='data_processed/sensor_sites.csv'` in PowerShell.

National and sensor-exclusion fits use some of the same output filenames. Run them in separate working copies or retain results separately before switching cases. Keep national, attribute-only sensor and locally updated sensor comparisons distinct.

Annual analyses additionally need the TREND-Nitrogen county inputs and watershed polygons described in Supporting Information. Script 13 specifies the expected input layout and accepts the watershed GeoPackage as its argument. These external files are not distributed here.

## Interpretation and verification

Response scores compare structural models conditional on common panel transformations and nuisance residuals. The response code does not implement fully nested nuisance fitting or spatially independent outer validation. Odd/even-year checks concern monitored catchments. Sensor curves are fitted references, and local updates use local grab samples. Monitoring contrasts concern the geometric-mean weather component.

The release was checked for source syntax, local dependencies and synchronization with the analysis sources. Full fitting was not rerun during packaging. Scientific settings and comparison definitions appear in the manuscript and Supporting Information.
