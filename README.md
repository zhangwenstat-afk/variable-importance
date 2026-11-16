# Code for *Is It Worth Collecting Additional Covariates?*

This repository contains the simulation and real-data code accompanying the main manuscript *Is It Worth Collecting Additional Covariates?* and its Supplementary Material. The manuscript source files are not included. The experiments compare semiparametric (`semi`, or Semi) and conformal (`conf`, or Conf) estimators of prediction-interval half-widths and their difference.

For baseline covariates `X`, additional covariates `Z`, and `W = (X, Z)`, the target is

```text
Delta(s0) = zeta_x(s0) - zeta_w(s0).
```

Here, `zeta_x` and `zeta_w` are prediction-interval half-widths at coverage `1 - alpha`, optionally conditional on a baseline profile `S = s0`. Thus, `Delta` measures a reduction in **half-width**; the reduction in full interval length is `2 * Delta`. The manuscript tests `H0: Delta(s0) <= tau(s0)`. Simulation configurations call the threshold `delta0`.

The simulation drivers, configurations, and estimators are included. Original Monte Carlo batch results are **not included in this repository**. The real-data directory includes prepared public analysis files, saved aggregate results for all four applications, and figures. LUCAS observation-level data and the derived spectral archive are **not included**; obtain authorized access and rebuild them as described in the [restricted-data notice](realdata/data_restricted/README.md).

## Directory structure

```text
variable-importance/
  marginal_mc/           Shared data generation, nuisance fitting, and marginal estimators
  marginal/              Marginal simulation drivers, configurations, and plot exporters
  conditional/
    conditional_mc/      Conditional estimators, simulation engine, and summaries
    configs/             Current conditional configurations and earlier variants
    configs/historical/  Configurations from earlier manuscript runs
  realdata/              Four applications, data preparation, saved results, and figures
  tests/                 Numerical and implementation checks
  bootstrap.py           Local import-path setup used by the drivers
  requirements.txt       Simulation dependency ranges
  requirements-tested.txt  Recorded versions for earlier numerical checks
  env.sh                 Environment settings for the original Linux server
```

See the [real-data README](realdata/README.md) for NHANES, concrete strength, soil organic carbon, and warfarin analyses.

## Installation

Clone the repository and enter its root directory, using a Python environment of your choice:

```sh
git clone https://github.com/zhangwenstat-afk/variable-importance.git
cd variable-importance
```

Run the commands below from this repository root. The recorded numerical-check environment used Python 3.13.5; its package versions are listed in [requirements-tested.txt](requirements-tested.txt).

```sh
python -m pip install -r requirements.txt
```

Real-data estimation and LUCAS preprocessing additionally require scikit-learn, which is not listed in either requirements file:

```sh
python -m pip install scikit-learn
```

To use the recorded simulation package versions, install `requirements-tested.txt` in place of `requirements.txt`. It does not pin the real-data scikit-learn version. Rebuilding warfarin data from its R source also requires R.

The drivers configure local imports through `bootstrap.py`, so sourcing `env.sh` is not required. That script contains machine-specific defaults for `LD_PRELOAD`, temporary directories, and `MC_PROJECT`; adapt them before using it on another machine.

For parallel runs, keep numerical-library threads at one per worker. In Bash:

```bash
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

In PowerShell:

```powershell
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"
```

## Running simulations

Always pass an explicit configuration: the marginal driver's default `configs/smoke.json` is not included. These single-line commands work in Bash or PowerShell after activating the intended Python environment.

Inspect a run without executing Monte Carlo replications or writing output files:

```sh
python marginal/run_mc.py --config marginal/configs/learned_residual_cv_paper_a5_density_wide2_estimation_b02.json --out runs/marginal_estimation --dry-run
python conditional/run_conditional.py --config conditional/configs/conditional_est_y20_w_hfhalf_cn075_spacing_seed20261006_s00_wcv05_0625_r16_32_r1000.json --out runs/conditional_estimation_s00 --dry-run
```

Dry runs still construct the experiment design and may compute oracle calibration quantities. For a small execution check, restrict the design and use separate output directories:

```sh
python marginal/run_mc.py --config marginal/configs/learned_residual_cv_paper_a5_density_wide2_estimation_b02.json --out runs/check_marginal --reps 2 --ns 1000 --errors left --jobs 1
python conditional/run_conditional.py --config conditional/configs/conditional_est_y20_w_hfhalf_cn075_spacing_seed20261006_s00_wcv05_0625_r16_32_r1000.json --out runs/check_conditional --reps 2 --ns 1000 --errors left --locations 0 --jobs 1
```

These checks do not reproduce the paper's Monte Carlo results. Full runs use the configuration's sample sizes, errors, and 1,000 replications:

```sh
python marginal/run_mc.py --config marginal/configs/learned_residual_cv_paper_a5_density_wide2_estimation_b02.json --out runs/marginal_estimation --jobs 4
python conditional/run_conditional.py --config conditional/configs/conditional_est_y20_w_hfhalf_cn075_spacing_seed20261006_s00_wcv05_0625_r16_32_r1000.json --out runs/conditional_estimation_s00 --jobs 4
```

Choose `--jobs` to fit available CPU and memory. Use a fresh output directory for each configuration. To continue an interrupted run, repeat its command with `--resume` and unchanged settings. The marginal driver checks configuration equality; the conditional driver skips existing batch filenames without checking that their settings match. Do not reuse a small-check directory for a full run.

Both drivers write `config_resolved.json`, `run_manifest.json`, and batch CSV files under `raw/`. The marginal driver also writes `truth_cells.csv`. Summaries are generated automatically unless `--no-summary` is supplied:

| Workflow | Main summaries | Figures |
| --- | --- | --- |
| Marginal | `summaries/estimation_summary.csv`, `summaries/inference_summary.csv`, plus calibration and diagnostic summaries | Add `--plots`, or use `marginal/plot_mc.py` |
| Conditional | `raw_all.csv`, `summaries/summary.csv` | Generic RMSE and power plots accompany the summary |

To rebuild marginal summaries from saved batches:

```sh
python marginal/summarize_mc.py runs/marginal_estimation
```

For an unchanged conditional run, rerunning its original command with `--resume` also regenerates the summary. The marginal summarizer does not process conditional output.

## Simulation design and manuscript correspondence

The principal design uses independent `X ~ Uniform(-1, 1)` and `Z ~ Uniform(-sqrt(3), sqrt(3))`, with

```text
Y = m0(X) + beta * Z + sigma(X) * epsilon0
m0(x) = x + 2*x^2 + sin(pi*x)/2
sigma(x) = 1 + 0.4*x.
```

The main experiments use standardized left- and right-skewed normal-mixture errors; the supplement adds symmetric and heavy-tailed designs. Prediction coverage is `1 - alpha = 0.9`, the one-sided test level is `0.05`, and confidence intervals for the simulation targets have level `0.95`. Estimation studies fix `beta = 0.2`. Inference studies calibrate `beta` to `Delta = delta0 + gap` (or its conditional counterpart), with `delta0 = 0.20`; `gap = 0` is the null boundary.

The tables use LaTeX labels to remain identifiable when numbering changes. “Main” refers to the main manuscript; “Supplement” refers to its Supplementary Material.

### Marginal experiments

All configurations below are in [marginal/configs](marginal/configs/) and use seed `20260906`.

| Manuscript location and label | Configuration filename |
| --- | --- |
| Main: `tab:learned-cv-zeta`, `fig:learned-cv-zeta-rmse` | [learned_residual_cv_paper_a5_density_wide2_estimation_b02.json](marginal/configs/learned_residual_cv_paper_a5_density_wide2_estimation_b02.json) |
| Main: `tab:delta_calib` | [learned_residual_cv_paper_a5_density_wide2_inference_d020.json](marginal/configs/learned_residual_cv_paper_a5_density_wide2_inference_d020.json) |
| Supplement: `fig:delta_power` | [learned_residual_cv_paper_a5_density_wide2_inference_d020_power.json](marginal/configs/learned_residual_cv_paper_a5_density_wide2_inference_d020_power.json), combined with the preceding null-boundary run |
| Supplement: `tab:learned-cv-zeta-sym` | [learned_residual_cv_paper_a5_density_wide2_estimation_b02_sym.json](marginal/configs/learned_residual_cv_paper_a5_density_wide2_estimation_b02_sym.json) |
| Supplement: `tab:delta_calib-sym` | [learned_residual_cv_paper_a5_density_wide2_inference_d020_sym.json](marginal/configs/learned_residual_cv_paper_a5_density_wide2_inference_d020_sym.json) |
| Supplement: `tab:t4-learned-cv-zeta`, `fig:t4-learned-cv-zeta-rmse` | [learned_residual_cv_paper_a5_t4_density_wide2_estimation_b02.json](marginal/configs/learned_residual_cv_paper_a5_t4_density_wide2_estimation_b02.json) |
| Supplement: `tab:t4-delta-calib`, `fig:t4-delta-power` | [learned_residual_cv_paper_a5_t4_density_wide2_inference_d020.json](marginal/configs/learned_residual_cv_paper_a5_t4_density_wide2_inference_d020.json) |
| Supplement: `tab:sym-t4-learned-cv-zeta` | [learned_residual_cv_paper_a5_t4_density_wide2_estimation_b02_sym.json](marginal/configs/learned_residual_cv_paper_a5_t4_density_wide2_estimation_b02_sym.json) |
| Supplement: `tab:sym-t4-delta-calib` | [learned_residual_cv_paper_a5_t4_density_wide2_inference_d020_sym.json](marginal/configs/learned_residual_cv_paper_a5_t4_density_wide2_inference_d020_sym.json) |

The normal-mixture power configuration contains gaps `{-0.05, 0.05, 0.10, 0.20}`; obtain gap zero from the separate null-boundary run. The heavy-tailed inference configuration already includes all five gaps.

After running the estimation, null-boundary, and power configurations into `runs/marginal_estimation`, `runs/marginal_null`, and `runs/marginal_power`, respectively:

```sh
python marginal/paper_artifacts.py --fig1-from runs/marginal_estimation/summaries/estimation_summary.csv --fig1-out runs/figures/zeta_sqrt_n_rmse_by_error.png
python marginal/paper_artifacts.py --fig2-from runs/marginal_null/summaries/inference_summary.csv runs/marginal_power/summaries/inference_summary.csv --fig2-out runs/figures/delta_power_nominal.png
```

The `fig1` and `fig2` option names are historical. The second command now corresponds to the supplement's `fig:delta_power`. Combine only disjoint summaries, so the same gap is not included twice.

`marginal/export_paper_refresh.py <run-directory>` exports table snippets and plots. Its estimation table omits coverage and its RMSE plot is unscaled, so these are intermediate artifacts rather than exact replacements for the current main-text table and scaled RMSE figure.

### Conditional experiments

The configurations below use seed `20261006`, nuisance layer `learned_density_y20_w_hfhalf`, `w_hf_post_multiplier = 0.5`, and the spacing estimator of `g`. Localization bandwidths are `h_semi = 1.5*n^(-1/4)` and `h_conf = 1.5*n^(-2/5)`.

| Manuscript location and label | Configuration in `conditional/configs/` |
| --- | --- |
| Main: `tab:cond-zeta-0` | [conditional_est_y20_w_hfhalf_cn075_spacing_seed20261006_s00_wcv05_0625_r16_32_r1000.json](conditional/configs/conditional_est_y20_w_hfhalf_cn075_spacing_seed20261006_s00_wcv05_0625_r16_32_r1000.json) |
| Main: `tab:cond-delta-0` | [conditional_infer_y20_w_hfhalf_cn075_midroot_seed20261006_s00_wcv05_0625_r16_32_r1000.json](conditional/configs/conditional_infer_y20_w_hfhalf_cn075_midroot_seed20261006_s00_wcv05_0625_r16_32_r1000.json) |
| Main: `fig:acquire-n4000` | [conditional_acquire_y20_w_hfhalf_cn075_spacing_seed20261006_n4000_11pt_r1000.json](conditional/configs/conditional_acquire_y20_w_hfhalf_cn075_spacing_seed20261006_n4000_11pt_r1000.json) |
| Supplement: `tab:cond-zeta-m05` | [conditional_est_y20_w_hfhalf_cn075_spacing_seed20261006_s0m05_0_r1000.json](conditional/configs/conditional_est_y20_w_hfhalf_cn075_spacing_seed20261006_s0m05_0_r1000.json), selecting `s0 = -0.5` |
| Supplement: `tab:cond-zeta-p05` | [conditional_est_y20_w_hfhalf_cn075_spacing_seed20261006_s05_r1000.json](conditional/configs/conditional_est_y20_w_hfhalf_cn075_spacing_seed20261006_s05_r1000.json) |
| Supplement: `tab:cond-delta-m05` | [conditional_infer_y20_w_hfhalf_cn075_spacing_seed20261006_s0m05_r1000.json](conditional/configs/conditional_infer_y20_w_hfhalf_cn075_spacing_seed20261006_s0m05_r1000.json) |
| Supplement: `tab:cond-delta-p05` | [conditional_infer_y20_w_hfhalf_cn075_spacing_seed20261006_s05_r1000.json](conditional/configs/conditional_infer_y20_w_hfhalf_cn075_spacing_seed20261006_s05_r1000.json) |
| Supplement: `fig:cond-delta-power`, `fig:cond-delta-power-n1000`, `fig:cond-delta-power-n4000` | [conditional_infer_y20_w_hfhalf_cn075_spacing_seed20261006_gaps_r1000.json](conditional/configs/conditional_infer_y20_w_hfhalf_cn075_spacing_seed20261006_gaps_r1000.json) |
| Supplement: `fig:cond-zeta-rmse`, `fig:cond-zeta-w-rmse` | Estimation summaries at all three locations; see archive requirements below |

The acquisition configuration fixes `beta = 0.6779790806045591` (rounded to `0.68` in the paper), with `n = 4000` and an 11-point grid from `-0.5` to `0.5`. The power configuration uses `n` in `{1000, 4000, 16000}`, all three locations, and gaps `{-0.05, 0, 0.05, 0.10, 0.20}`. Despite its `midroot` filename, the `s0 = 0` inference configuration specifies `g_estimator = "spacing"` and `semi_root = "midpoint_positive"`.

Mean-regression bandwidth menus are partly set in [conditional_mc/estimators.py](conditional/conditional_mc/estimators.py). For `m_w`, the effective menus are:

| Location | Mean multiplier `c` | Anisotropy parameter `r` |
| --- | --- | --- |
| `s0 = -0.5` | 0.1875, 0.25 | 32, 64 |
| `s0 = 0` | 0.5, 0.625 | 16, 32 |
| `s0 = 0.5` | 1, 1.5, 2, 2.5, 3 | 2, 4, 8, 12, 16 |

The first two rows are target-specific overrides; `m_x` retains its own location-dependent menu. The alternative `kernel_g.py` and `kernel_g_profile.py` estimators are not selected by these configurations.

The spacing parameter is `c_n = min{0.05, 0.15*c_n_factor*(n*h/2)^(-1/5)}`, with `c_n_factor = 0.75` in the supplied configurations.

After running the power and acquisition configurations into `runs/conditional_power` and `runs/conditional_acquisition`, redraw their figures using absolute run paths:

```sh
python conditional/plot_power_gaps.py --run /absolute/path/to/runs/conditional_power --outdir runs/conditional_figures
python conditional/plot_acquire_n4000.py --run /absolute/path/to/runs/conditional_acquisition --outdir runs/conditional_figures
```

Replace the example paths with actual paths, such as `D:/project/variable-importance/runs/conditional_power` on Windows. Absolute `--run` paths avoid server-specific defaults. The power plot reads raw batches or `raw_all.csv`; the acquisition plot reads `summaries/summary.csv`.

## Archived results and reproduction limits

Earlier simulation results were stored outside this archive under `code_new/marginal_mc_project/results/`. This repository does not contain them. Historical result names and script defaults provide provenance, but do not establish exact reproduction of every manuscript value.

For marginal runs, the historical result basename generally equals the configuration stem plus `_r1000`. The normal-mixture null-boundary run instead ends in `_inference_d020_size_r1000`.

For conditional estimation, [plot_zeta_rmse.py](conditional/plot_zeta_rmse.py) reads three fixed directories under `MC_PROJECT/results/`:

```text
conditional_est_y20_w_hfhalf_cn075_spacing_seed20261006_s0m05_0_r1000
conditional_est_y20_w_hfhalf_cn075_spacing_seed20261006_s00_wcv05_0625_r16_32_r1000
conditional_est_y20_w_hfhalf_cn075_spacing_seonly_seed20261006_s05_r1000
```

It selects one location from each and expects 1,000 valid estimation replications per summary row. Earlier archive notes identify the `s0 = 0` rows of the combined `s0m05_0` run as using an older `m_w` menu; use the dedicated `s00_wcv05_0625_r16_32` run for the current zero-location table.

The positive-location tables were assembled using earlier point estimates with recomputed spacing-based SEs. These utilities require the external raw archive:

| Script | Required runs under `MC_PROJECT/results/` |
| --- | --- |
| [recompute_est_spacing_se.py](conditional/recompute_est_spacing_se.py) | `conditional_est_y20_w_hfhalf_cn05_midroot_gslope_seed20261006_r1000` and its `_n16000_r1000` counterpart |
| [recompute_spacing_cn_se.py](conditional/recompute_spacing_cn_se.py) | `conditional_infer_y20_w_hfhalf_cn05_midroot_seed20261006_n1k4k_r1000`, plus the corresponding `n8000_r1000` and `n16000_r1000` runs |

The estimation utility also reads the `n16000` run's `config_resolved.json`. The inference utility requires `MC_PROJECT/configs/conditional_infer_y20_w_hfhalf_cn05_midroot_seed20261006_n1k4k_r1000.json`. With those files available, set `MC_PROJECT` to the archived project root and run:

```sh
python conditional/recompute_est_spacing_se.py --out runs/conditional_estimation_s05_se --jobs 4
python conditional/recompute_spacing_cn_se.py --out runs/conditional_inference_s05_se --jobs 4
python conditional/plot_zeta_rmse.py --outdir runs/archived_conditional_figures
```

The recomputed outputs go to the specified new directories; the plotting command still reads its fixed archive names above. For a new simulation, use the current `s05` configurations instead. Earlier archive notes also record that the corrected negative-location inference summary was not retained; its `cn05` run used a different spacing factor. Use the current `cn075_spacing` configuration for a new run.

Files in `conditional/configs/historical/` use seed `20260906` and earlier nuisance settings. Current power and acquisition plotters default to refreshed `20261006` run names. Rerunning a historical JSON with today's bandwidth overrides does not reconstruct the earlier implementation.

`MC_PROJECT` has different meanings across utilities: the RMSE plotter and SE-recomputation scripts append `results/`, while the power and acquisition plotters append the run name directly. For the latter two, absolute `--run` paths avoid this ambiguity.

## Reading the numerical summaries

CSV values use the original scale. Manuscript estimation tables multiply bias, SD, SE, and RMSE by 100 and display coverage as a percentage. Inference tables also multiply rejection rates, contrast bias, coverage, and interval length by 100; `SD(T)` remains unscaled. The statistic is `(Delta_hat - delta0)/se_delta`, with the standard-normal 0.95 quantile for a one-sided 5% test and the 0.975 quantile for two-sided 95% intervals.

Validity rules differ across routines. Marginal estimation summaries use valid point estimates and each target's finite, positive SE for SE/coverage; marginal inference summaries use `inference_valid`. Conditional replication output records intervals and tests only when joint inference is valid. The generic conditional summarizer selects `estimator_valid` rows and averages available values column by column. The SE-recomputation scripts and conditional power plotter explicitly filter inference-valid rows for inferential summaries. Check the masks and valid-replication counts when rebuilding paper tables.

The conditional RMSE plotting script draws left-skewed errors above right-skewed errors; the current supplementary captions describe the reverse order. Use the plotted row labels when interpreting these figures. This manuscript/code detail needs reconciliation.

## Implementation checks

From the repository root, run the main numerical checks:

```sh
python -m unittest tests.test_core
```

A focused check verifies that the conditional mean-bandwidth overrides do not mutate shared settings:

```sh
python -m unittest tests.test_core.NuisanceTests.test_s0_mean_bandwidth_grid_is_local_and_does_not_mutate
```

The two alternative kernel-density test modules import `recompute_kernel_g_se`, which is absent from this archive, so unrestricted test discovery is not currently self-contained. These checks do not replace the full Monte Carlo experiments or establish exact reproduction of the manuscript tables.
