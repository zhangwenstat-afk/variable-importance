#!/usr/bin/env bash
# Source this before running the archived drivers.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${ROOT}:${ROOT}/marginal:${ROOT}/conditional${PYTHONPATH:+:$PYTHONPATH}"
export LD_PRELOAD="${LD_PRELOAD:-/home/eason/anaconda3/lib/libstdc++.so.6.0.34}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TMPDIR="${TMPDIR:-/data/eason/mc_tmp}"
export JOBLIB_TEMP_FOLDER="${JOBLIB_TEMP_FOLDER:-/data/eason/mc_tmp}"
# Finished raw results stay in the working project. Do not point a new run at those directories.
export MC_PROJECT="${MC_PROJECT:-/home/eason/variable importance/code_new/marginal_mc_project}"
