# Real-data applications

This directory implements the analyses in the **Real data application** section (`sec:real`) of the main manuscript *Is It Worth Collecting Additional Covariates?* and the additional decision maps (`sec:further-decision-map`) in its Supplementary Material. The manuscript source files are not included in this repository.

The target is the reduction in prediction-interval half-width from adding covariates `Z` to baseline covariates `X`, with `W = (X, Z)`:

$$
\Delta(s_0) = \zeta_X(s_0) - \zeta_W(s_0).
$$

Positive values indicate narrower prediction intervals after collecting `Z`. The analyses compare semiparametric (`semi`, plotted as **Semi**) and conformal (`conf`, plotted as **Conf**) estimators at prediction coverage `1 - alpha = 0.90`. Inference uses pointwise, one-sided tests of `H0: Delta(s0) <= tau` at level 0.05. This testing level is distinct from the prediction miscoverage `alpha = 0.10`.

## Datasets and manuscript outputs

The sample sizes below describe the analysis datasets used for the stored results. The public analysis CSVs are included; the restricted LUCAS observation-level data and derived spectral archive are not. Aggregate estimates, holdout coverage results, and figures are supplied for all four applications. Full feature lists, categorical-feature flags, and evaluation grids are defined in [config.py](config.py).

| Dataset / CLI name | Sample size | Outcome `Y` | Additional covariates `Z` | Conditioning variable `S` and grid |
| --- | ---: | --- | --- | --- |
| NHANES 2011–2016 / `nhanes` | 5,287 | Two-hour OGTT glucose (mg/dL) | Fasting glucose; HbA1c; or insulin, triglycerides, LDL, and HDL | Age: 25, 30, ..., 75 years; BMI: 20, 22.5, ..., 37.5 kg/m² |
| SonReb / `sonreb` | 3,226 | Concrete compressive strength, normalized to a 150 × 300 mm reference cylinder (MPa) | Ultrasonic pulse velocity | Rebound number: 20, 24, ..., 44 |
| LUCAS 2015 / `lucas` | 20,383 | Soil organic carbon (g/kg) | First 30 principal components of the vis–NIR spectrum | Mean annual temperature: 2, 4, ..., 16 °C |
| IWPC subset from `simsl` / `warfarin` | 1,780 | Stable weekly warfarin dose (mg/week) | VKORC1 or CYP2C9 genotype, analyzed separately | Standardized weight: −1.5, −1, ..., 1.5 |

Baseline `X` contains 13 questionnaire/examination variables for NHANES; 10 rebound-test, study, and specimen variables for SonReb; 29 geographic, terrain, climatic, and mapped environmental variables for LUCAS; and eight routine clinical/demographic variables for warfarin. Warfarin weight, height, and age are already standardized in the source subset.

Figure references use LaTeX labels because numbering can change between manuscript versions.

| Generated PDF | Manuscript location and label |
| --- | --- |
| `ogtt_delta_s0.pdf` | Main manuscript, NHANES estimates: `fig:ogtt-delta` |
| `ogtt_decision_map.pdf` | Main manuscript, NHANES decision maps: `fig:ogtt-map` |
| `further_delta_s0.pdf` | Main manuscript, concrete, soil carbon, and both warfarin genotype comparisons: `fig:further-delta` |
| `further_decision_map.pdf` | Supplementary Material, concrete, soil carbon, and VKORC1 decision maps: `fig:further-map` |

The estimate plots show pointwise two-sided 90% confidence intervals. Their lower edge is the one-sided 95% lower bound `L = delta - 1.6448536269514722 * se`. A decision-map cell rejects the null when `L > tau`. The CYP2C9 decision map is omitted in the manuscript because neither method rejects at `tau = 0` on its grid.

## Directory contents

| Path | Purpose |
| --- | --- |
| [config.py](config.py) | Dataset paths, `X`, `Z`, `S` grids, fold construction, bandwidth constants, and default split count |
| [estimation.py](estimation.py) | Nuisance fitting, local and marginal estimation, repeated cross-fitting, and holdout coverage checks |
| [run_estimation.py](run_estimation.py) | Command-line entry point for one dataset |
| [make_figures.py](make_figures.py) | Generate all four PDFs from the stored `alpha = 0.1` final results |
| [summarize.py](summarize.py) | Print estimates, standard errors, lower bounds, rejection grids, and NHANES comparison counts |
| [data/](data/) | Public analysis CSVs, raw-data download script, and preprocessing scripts |
| [data_restricted/](data_restricted/README.md) | LUCAS data-access and redistribution notice |
| [results/](results/) | Stored results for all four datasets at `alpha = 0.1` |
| [figures/](figures/) | Stored real-data figures |

Root solving, the spacing estimate of the residual boundary density, and kernel/bandwidth utilities are shared with [conditional_mc/estimators.py](../conditional/conditional_mc/estimators.py). The real-data wrapper implements the corresponding variance calculations using the fitted real-data nuisances.

## Estimation settings

- **Repeated cross-fitting:** two folds, repeated 20 times by default, with split seeds `20261019 + split` for zero-based split indices. The point estimate is the median across splits.
- **Regression:** `HistGradientBoostingRegressor` with `max_iter=500`, `learning_rate=0.03`, `min_samples_leaf=40`, `l2_regularization=1.0`, and early stopping using a 15% validation fraction.
- **Residual density:** a location-scale model `epsilon = sigma(V) U`. Inner two-fold out-of-fold residuals within each training fold are used to fit `sigma(V)` by boosting absolute residuals. A Gaussian KDE with a Silverman bandwidth estimates the density of standardized residuals.
- **Localization:** an Epanechnikov kernel with `h = c * sqrt(3) * sd(S) * n^(-1/4)` for Semi and `h = c * sqrt(3) * sd(S) * n^(-2/5)` for Conf. The analysis evaluates `c` in `{0.75, 1, 1.5, 2, 3}`; the main comparisons use `c = 2` for Semi and `c = 3` for Conf.
- **Additional comparisons:** `conf_hsemi` uses the conformal estimator with the Semi bandwidth (`c = 2` in the final results). Marginal estimates are also retained. Neither is included in the current figure plots.

Outer cross-fitting folds are random for NHANES and warfarin, stratified by study for SonReb, and assigned by 100 km spatial blocks for LUCAS. SonReb studies appear in both folds; its analysis is not leave-one-study-out.

The separate holdout check estimates intervals using `cycle == 0` and evaluates kernel-weighted coverage on `cycle == 1`. NHANES uses the 2011–2014 cycles for training and 2015–2016 for testing. The other builders assign approximately half the observations at random (warfarin), within each study (SonReb), or by spatial block (LUCAS). The `cycle` field is a holdout indicator, not the repeated cross-fitting fold index. Selected constants are fixed in `CHOSEN` in `config.py`; the runner does not automatically optimize them from the holdout results.

## Read results and regenerate figures

Run the following from the repository root with a working Python environment:

```bash
cd realdata
python -m pip install -r ../requirements.txt scikit-learn
python summarize.py
python make_figures.py figures_rebuilt
```

`scikit-learn` is needed for estimation and LUCAS preprocessing, but is not listed in the shared `requirements.txt`. The summary and figure commands only read stored result CSVs; they do not require the original datasets or rerun model fitting. Both scripts currently read `results/*_a0.1_final.csv` explicitly, regardless of other alpha values present in `results/`.

The example writes figures into `figures_rebuilt/`. Omitting the argument writes into this directory's `figures/`. If you maintain a separate manuscript checkout, pass its actual figure directory, for example `python make_figures.py /absolute/path/to/manuscript/figures`; existing PDFs with the same names will be overwritten. The manuscript checkout and its figure directory are not part of this repository.

## Rerun estimation

Run these commands from the repository's `realdata/` directory, after installing the dependencies above:

```bash
python run_estimation.py nhanes --alpha 0.1 --splits 20 --jobs 4
python run_estimation.py sonreb --alpha 0.1 --splits 20 --jobs 4
python run_estimation.py warfarin --alpha 0.1 --splits 20 --jobs 4
python run_estimation.py lucas --alpha 0.1 --splits 20 --jobs 4
```

The LUCAS command requires the authorized local analysis file at `data_restricted/lucas/lucas.csv`. Choose `--jobs` for the available CPU and memory; the script default is 10. `--splits` defaults to 20, and the holdout check runs in addition to the requested splits.

**Each run overwrites the four matching files in `results/`.** There is no output-directory option, and changing `--splits` does not change the output filenames. Preserve the stored results before rerunning, or use a separate repository copy for exploratory runs.

When using multiple worker processes, thread limits can prevent numerical libraries from starting additional threads inside each worker. Set these before the estimation commands if needed:

```bash
# Bash
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
```

```powershell
# PowerShell
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"
```

The real-data scripts set their own import paths. They do not require `../env.sh`, which contains paths specific to the original computing environment.

### Output CSVs

For dataset `d` and alpha `a`, the runner writes `results/d_a<alpha>_<kind>.csv` (for example, `nhanes_a0.1_final.csv`).

| Kind | Contents |
| --- | --- |
| `raw` | One row per split, conditioning variable/grid point, added covariate group, method, and bandwidth constant; also includes marginal rows |
| `summary` | Median estimates across splits, with `se² = median(se_b² + (delta_b - median(delta_b))²)` and one-sided normal-approximation `p0` for `H0: Delta <= 0` |
| `final` | Selected bandwidth rows: `semi` at `c = 2`, `conf` at `c = 3`, `conf_hsemi` at `c = 2`, plus all marginal rows |
| `holdout` | Local test-set coverage (`cover_test`) by target, grid point, method, and bandwidth constant |

`zeta_x` and `zeta_w` are half-width estimates, `delta` is the gain estimate, `se` is its standard error, and `h` is the localization bandwidth. `S = marginal` identifies unconditioned estimates, with blank `s0` and `c` fields. Summary columns are aggregated separately, so the reported median `delta` need not equal the difference between the two reported median half-widths. The main-analysis `n_eff` is a kernel-weight summary; holdout `n_test_eff` counts test observations with positive kernel weights.

## Rebuild the analysis datasets

The public analysis CSVs are already in `data/`. Rebuilding them requires internet access, **Bash 4 or later**, `curl`, and `tar`; extracting the warfarin subset also requires R with `Rscript`. On Windows, run the download script in a compatible Bash environment such as WSL.

From the repository's `realdata/` directory:

```bash
cd data
bash download.sh raw
python build_nhanes.py raw/nhanes .
Rscript simsl_to_csv.R raw/simsl/simsl/data/warfarin.RData raw/warfarin_raw.csv
python build_warfarin.py raw/warfarin_raw.csv .
python build_sonreb.py raw/sonreb .
```

These commands overwrite the public analysis CSVs. The download script requests NHANES cycles G/H/I, `simsl` version 0.2.1, and SonReb Zenodo record 15392443.

Preprocessing follows the supplied builders:

- **NHANES:** retain adults aged at least 20 with observed two-hour OGTT glucose, fasting glucose, and BMI; exclude affirmative reports of diagnosed diabetes, insulin use, or diabetes-pill use. Other missing predictors can remain in the analysis file.
- **SonReb:** require rebound number, ultrasonic velocity, and strength; remove duplicate `(study, RN, Vp, Y)` records; encode categorical variables and log-transform specimen age.
- **Warfarin:** extract the cleaned `simsl` subset, replace periods in column names with underscores, and add the holdout indicator. The builder uses the standardized covariates supplied by `simsl`.
- **LUCAS:** match topsoil, ancillary, and spectral records; require observed organic carbon at most 120 g/kg and observed `BIO1`; derive spatial/terrain variables and 30 spectral principal components.

For LUCAS, obtain the source files through ESDAC as described in [the restricted-data notice](data_restricted/README.md). The builder expects this layout under the download directory:

```text
<lucas_dir>/
    topsoil/LUCAS_Topsoil_2015_20200323.csv
    LUCAS2015_AncillaryData_20201007.csv
    spectra/LUCAS2015_Soil_Spectra_EU28.zip
```

From the repository's `realdata/data/` directory, replace `path/to/lucas_download` below with that directory:

```bash
python -c "from pathlib import Path; Path('../data_restricted/lucas').mkdir(parents=True, exist_ok=True)"
python lucas_spectra.py "path/to/lucas_download/spectra/LUCAS2015_Soil_Spectra_EU28.zip" ../data_restricted/lucas/spectra10nm.npz
python build_lucas.py "path/to/lucas_download" ../data_restricted/lucas/spectra10nm.npz ../data_restricted/lucas
```

These steps write or replace the derived LUCAS files. Spectra are reduced to 10 nm bins before PCA. `build_lucas.py` fits PCA on the retained analysis sample before the estimation script constructs its folds.

## Data sources and access

- **NHANES:** [CDC National Health and Nutrition Examination Survey](https://wwwn.cdc.gov/nchs/nhanes/), 2011–2012, 2013–2014, and 2015–2016 cycles.
- **SonReb:** [Zenodo record 15392443](https://doi.org/10.5281/zenodo.15392443), the concrete-strength database cited in the manuscript.
- **Warfarin:** the cleaned IWPC subset distributed in the R package [simsl](https://CRAN.R-project.org/package=simsl), version 0.2.1; the original IWPC study is cited in the manuscript.
- **LUCAS 2015:** access is subject to the conditions recorded in [data_restricted/README.md](data_restricted/README.md). That notice prohibits sharing the derived CSV and spectral archive and provides the required publication acknowledgment. Request the source data through ESDAC and follow that notice when using or distributing this project.
