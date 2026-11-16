"""Reproducible experiments, independent size calibration, and atomic checkpoints."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import traceback
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config

from .dgp import ErrorLaw, generate, stable_rng, truth, MEAN_FNS
from .estimators import estimate_layer


def short_hash(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    c = deepcopy(config)
    defaults = {"seed": 20260906, "alpha": 0.10, "test_level": 0.05,
                "ci_level": 0.95, "oracle_nodes": 160, "mixture_p": 0.2,
                "mixture_a": 5.0, "error_base": "normal", "emg_lambda": 1.5,
                "mean_fn": "paper",
                "g_estimator": "spacing", "variance": "empirical",
                "methods": ["semi", "conf"],
                "batch_size": 10, "diagnostics": {"enabled": True, "every": 10},
                "size_adjustment": {"enabled": False, "calibration_reps": 0},
                "fixed_beta": None}
    for key, value in defaults.items():
        c.setdefault(key, value)
    for key in ("alpha", "test_level", "ci_level"):
        if not 0 < float(c[key]) < 1:
            raise ValueError(f"{key} must lie in (0,1)")
    if c["variance"] not in {"empirical", "manuscript"}:
        raise ValueError("variance must be empirical or manuscript")
    if c["g_estimator"] not in {"spacing", "oracle"}:
        raise ValueError("g_estimator must be spacing or oracle")
    if not c["methods"] or any(method not in {"semi", "conf"} for method in c["methods"]):
        raise ValueError("methods must be a nonempty subset of semi and conf")
    if len(set(c["methods"])) != len(c["methods"]):
        raise ValueError("methods must be unique")
    for key in ("repetitions", "batch_size", "oracle_nodes"):
        if int(c[key]) < 1:
            raise ValueError(f"{key} must be positive")
    if int(c["oracle_nodes"]) < 16:
        raise ValueError("oracle_nodes must be at least 16")
    if int(c["diagnostics"].get("every", 1)) < 1:
        raise ValueError("diagnostics.every must be positive")
    if c["size_adjustment"].get("enabled") and int(c["size_adjustment"].get("calibration_reps", 0)) < 1:
        raise ValueError("independent size calibration requires calibration_reps > 0")
    if not c.get("sample_sizes") or any(int(n) < 40 or int(n) % 2 for n in c["sample_sizes"]):
        raise ValueError("all sample_sizes must be even and >=40")
    if len(set(c["sample_sizes"])) != len(c["sample_sizes"]):
        raise ValueError("sample_sizes must be unique")
    if c.get("error_base", "normal") not in {"normal", "t4", "exponnorm"}:
        raise ValueError("error_base must be normal, t4, or exponnorm")
    if c.get("error_base") == "exponnorm":
        lam = float(c.get("emg_lambda", 1.5))
        if not math.isfinite(lam) or lam <= 0.0:
            raise ValueError("emg_lambda must be finite and positive")
        c["emg_lambda"] = lam
        if "symmetric" in set(c.get("errors", [])):
            raise ValueError("exponnorm error_base cannot be combined with symmetric errors")
    if c.get("mean_fn", "paper") not in MEAN_FNS:
        raise ValueError("mean_fn must be paper, quadratic, quad_half, or sine_quad")
    if int(c["diagnostics"].get("b_mc_size", 0)) < 0:
        raise ValueError("diagnostics.b_mc_size cannot be negative")
    if c.get("fixed_beta") is not None:
        beta = float(c["fixed_beta"])
        if not math.isfinite(beta) or beta <= 0.0:
            raise ValueError("fixed_beta must be finite and positive")
        c["fixed_beta"] = beta
        if c["size_adjustment"].get("enabled"):
            raise ValueError("fixed_beta cannot be combined with size_adjustment")
    if not c.get("errors") or not set(c["errors"]) <= {"symmetric", "right", "left"}:
        raise ValueError("invalid errors")
    if len(set(c["errors"])) != len(c["errors"]):
        raise ValueError("errors must be unique")
    ids = [layer["id"] for layer in c["layers"]]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("layer IDs must be nonempty and unique")
    for layer in c["layers"]:
        if layer["kind"] not in {"oracle", "controlled", "learned",
                                 "learned_mean_oracle_density",
                                 "fake_mean_oracle_density",
                                 "fake_mean_fake_A"}:
            raise ValueError(f"invalid layer: {layer}")
        if layer["kind"] in {"fake_mean_oracle_density", "fake_mean_fake_A"}:
            for key in ("gamma_m_x", "gamma_m_w"):
                if float(layer.get(key, 0)) <= 0:
                    raise ValueError(f"{key} must be positive")
            if float(layer.get("c_m", 1)) < 0:
                raise ValueError("c_m cannot be negative")
            if layer.get("shift", "positive") not in {"positive", "negative", "rademacher"}:
                raise ValueError("shift must be positive, negative, or rademacher")
            if layer["kind"] == "fake_mean_fake_A":
                if float(layer.get("gamma_a", 0)) <= 0:
                    raise ValueError("gamma_a must be positive")
                if float(layer.get("c_a", 1)) < 0:
                    raise ValueError("c_a cannot be negative")
        if layer["kind"] == "controlled":
            if float(layer["gamma_m"]) <= 0 or float(layer["gamma_f"]) <= 0:
                raise ValueError("controlled exponents must be positive")
            if float(layer.get("c_m", 1)) < 0 or float(layer.get("c_f", 1)) < 0:
                raise ValueError("controlled magnitudes cannot be negative")
            if layer.get("shift", "positive") not in {"positive", "negative", "rademacher"}:
                raise ValueError("shift must be positive, negative, or rademacher")
            if float(layer.get("dagger_sd", 2)) <= 0:
                raise ValueError("dagger_sd must be positive")
            smallest_train = min(c["sample_sizes"])/2
            if float(layer.get("c_f", 1))*smallest_train**(-float(layer["gamma_f"])) > 1:
                raise ValueError("controlled mixture weight exceeds one")
        if layer["kind"] in {"learned", "learned_mean_oracle_density"}:
            for key in ("m_bw_mult", "f_bw_mult", "y_bw_mult"):
                if float(layer.get(key, 1)) <= 0:
                    raise ValueError(f"{key} must be positive")
            if layer.get("mean_bandwidth", "isotropic") not in {"isotropic", "anisotropic"}:
                raise ValueError("mean_bandwidth must be isotropic or anisotropic")
            if layer.get("mean_model", "local_linear") not in {"local_linear", "additive"}:
                raise ValueError("mean_model must be local_linear or additive")
            if int(layer.get("mean_degree", 1)) not in {1, 2, 3}:
                raise ValueError("mean_degree must be 1, 2, or 3")
            if float(layer.get("undersmooth", 1.0)) <= 0:
                raise ValueError("undersmooth must be positive")
            if float(layer.get("z_scale", 3.0)) <= 0:
                raise ValueError("z_scale must be positive")
            selection = layer.get("bandwidth_selection", {})
            if not isinstance(selection, dict):
                raise ValueError("learned bandwidth_selection must be an object")
            method = selection.get("method", "rule")
            if method not in {"rule", "inner_cv"}:
                raise ValueError("learned bandwidth_selection.method must be rule or inner_cv")
            if method == "inner_cv":
                if int(selection.get("folds", 3)) < 2:
                    raise ValueError("learned inner CV requires at least two folds")
                if int(selection.get("max_points", 512)) < 1:
                    raise ValueError("learned inner CV max_points must be positive")
                for key in ("m_multipliers", "f_multipliers", "y_multipliers", "z_scales"):
                    values = selection.get(key)
                    if values is None:
                        if key == "z_scales":
                            continue
                        raise ValueError(f"learned {key} must contain positive values")
                    if not values or any(float(value) <= 0 for value in values):
                        raise ValueError(f"learned {key} must contain positive values")
            if layer["kind"] == "learned":
                density_estimator = layer.get("density_estimator", "response")
                if density_estimator not in {"response", "residual_loo"}:
                    raise ValueError(
                        "learned density_estimator must be response or residual_loo"
                    )
                density_mode = layer.get("density_feature_mode", "all")
                if density_mode not in {"all", "x_only"}:
                    raise ValueError(
                        "learned density_feature_mode must be all or x_only"
                    )
                feature_selection = layer.get(
                    "density_feature_selection", "fixed"
                )
                if feature_selection not in {"fixed", "inner_cv"}:
                    raise ValueError(
                        "learned density_feature_selection must be fixed or inner_cv"
                    )
                if density_estimator == "response" and density_mode != "all":
                    raise ValueError(
                        "response density estimation requires density_feature_mode=all"
                    )
                if feature_selection == "inner_cv":
                    if density_estimator != "residual_loo":
                        raise ValueError(
                            "density feature selection requires residual_loo"
                        )
                    if selection.get("method", "rule") != "inner_cv":
                        raise ValueError(
                            "density feature selection requires inner_cv bandwidth selection"
                        )
                    candidates = layer.get(
                        "density_feature_candidates", ["all", "x_only"]
                    )
                    if (not candidates
                            or any(mode not in {"all", "x_only"}
                                   for mode in candidates)):
                        raise ValueError(
                            "density_feature_candidates must contain all or x_only"
                        )
    study_names = [s["name"] for s in c["studies"]]
    if len(study_names) != len(set(study_names)):
        raise ValueError("study names must be unique")
    for study in c["studies"]:
        if float(study["delta0"]) <= 0:
            raise ValueError("positive delta0 required to avoid the degenerate beta=0 boundary")
        if ("gaps" in study) == ("local_cs" in study):
            raise ValueError("each study must specify exactly one of gaps or local_cs")
        if study.get("type", "estimation") not in {"estimation", "inference", "local_power"}:
            raise ValueError("invalid study type")
        values = study.get("gaps", study.get("local_cs", []))
        if not values or len(values) != len(set(values)):
            raise ValueError("gap/local-c values must be nonempty and unique")
        if c.get("fixed_beta") is not None:
            if "local_cs" in study:
                raise ValueError("fixed_beta cannot be combined with local_cs")
            if any(float(gap) != 0.0 for gap in study["gaps"]):
                raise ValueError("fixed_beta requires all gaps to be 0")
    return c


def make_cells(config: dict[str, Any]) -> list[dict[str, Any]]:
    cells, seen_calibration = [], set()
    for error in config["errors"]:
        for n in config["sample_sizes"]:
            for study in config["studies"]:
                delta0 = float(study["delta0"])
                calibration_key = short_hash({"error": error, "n": n, "delta0": delta0,
                    "alpha": config["alpha"], "mixture_p": config["mixture_p"],
                    "mixture_a": config["mixture_a"],
                    "error_base": config["error_base"],
                    "emg_lambda": config.get("emg_lambda", 1.5),
                    "fixed_beta": config.get("fixed_beta")})
                base = {"error": error, "n": int(n), "delta0": delta0,
                        "error_base": config["error_base"],
                        "calibration_key": calibration_key}
                local = "local_cs" in study
                values = study["local_cs"] if local else study["gaps"]
                for value in values:
                    gap = float(value)/math.sqrt(n) if local else float(value)
                    target_delta = float(f"{delta0+gap:.12g}")
                    if target_delta <= 0:
                        raise ValueError("all simulated Delta values must be positive in this regular-inference design")
                    cell = {**base, "study": study["name"],
                            "study_type": study.get("type", "estimation"),
                            "role": "evaluation", "gap_type": "local" if local else "fixed",
                            "gap_param": float(value), "gap": gap, "delta_target": target_delta,
                            "repetitions": int(config["repetitions"])}
                    cell["cell_id"] = short_hash(cell)
                    cells.append(cell)
                if (config["size_adjustment"].get("enabled", False)
                        and study.get("type", "estimation") != "estimation"
                        and calibration_key not in seen_calibration):
                    cell = {**base, "study": "size_calibration", "study_type": "calibration",
                            "role": "calibration", "gap_type": "fixed", "gap_param": 0.0,
                            "gap": 0.0, "delta_target": delta0,
                            "repetitions": int(config["size_adjustment"]["calibration_reps"])}
                    cell["cell_id"] = short_hash(cell)
                    cells.append(cell)
                    seen_calibration.add(calibration_key)
    for cell in cells:
        oracle = truth(cell["error"], config["alpha"], cell["delta_target"],
                       config["oracle_nodes"], config["mixture_p"], config["mixture_a"],
                       config["error_base"], config.get("fixed_beta"),
                       config.get("emg_lambda", 1.5))
        cell.update(oracle)
        if config.get("fixed_beta") is not None:
            cell["delta_target"] = oracle["delta_true"]
            cell["delta0"] = oracle["delta_true"]
    return cells


def run_replication(cell: dict[str, Any], rep: int, config: dict[str, Any]) -> list[dict[str, Any]]:
    law = ErrorLaw(cell["error"], config["mixture_p"], config["mixture_a"],
                   config["error_base"], config.get("emg_lambda", 1.5))
    if config["error_base"] == "normal":
        data_key = (cell["error"], cell["n"], f"{cell['delta_target']:.12g}",
                    cell["role"], rep)
    else:
        data_key = (config["error_base"], config.get("emg_lambda", 1.5),
                    cell["error"], cell["n"],
                    f"{cell['delta_target']:.12g}", cell["role"], rep)
    rng = stable_rng(config["seed"], "data", *data_key)
    data = generate(cell["n"], cell["beta"], law, rng, config["mean_fn"])
    meta = {key: value for key, value in cell.items() if key != "repetitions"}
    meta.update({"rep": rep, "alpha": config["alpha"], "test_level": config["test_level"],
                 "ci_level": config["ci_level"], "variance_choice": config["variance"],
                 "g_estimator": config["g_estimator"], "mean_fn": config["mean_fn"]})
    every = int(config["diagnostics"].get("every", 1))
    do_diagnostics = bool(config["diagnostics"].get("enabled", True) and rep % every == 0)
    output = []
    for layer in config["layers"]:
        mode = layer.get("shift", "positive")
        if mode == "rademacher":
            sign_rng = stable_rng(config["seed"], "controlled_signs", *data_key)
            signs = tuple(float(s) for s in sign_rng.choice([-1, 1], size=2))
        else:
            signs = (-1.0, -1.0) if mode == "negative" else (1.0, 1.0)
        layer_meta = {"layer": layer["id"], "kind": layer["kind"],
                      "gamma_m": layer.get("gamma_m", math.nan),
                      "gamma_f": layer.get("gamma_f", math.nan),
                      "c_m": layer.get("c_m", math.nan),
                      "c_f": layer.get("c_f", math.nan), "shift_mode": mode,
                      "diagnostics_computed": do_diagnostics}
        try:
            results = estimate_layer(data, law, layer, config, cell, signs,
                                    do_diagnostics, (cell["cell_id"], rep))
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            tail = " | ".join(traceback.format_exc().splitlines()[-4:])
            results = [{"method": method, "status": "exception", "error_message": message,
                        "error_trace_tail": tail, "estimator_valid": False,
                        "inference_valid": False} for method in config["methods"]]
        output.extend({**meta, **layer_meta, **row} for row in results)
    return output


@dataclass(frozen=True)
class Batch:
    cell: dict[str, Any]
    first: int
    stop: int
    filename: str


def make_batches(cells: list[dict[str, Any]], batch_size: int) -> list[Batch]:
    return [Batch(cell, start, min(start+batch_size, cell["repetitions"]),
                  f"{cell['cell_id']}_{start:06d}_{min(start+batch_size, cell['repetitions']):06d}.csv")
            for cell in cells for start in range(0, cell["repetitions"], batch_size)]


def run_batch(batch: Batch, config: dict[str, Any], raw_dir: str) -> dict[str, Any]:
    rows = []
    for rep in range(batch.first, batch.stop):
        rows.extend(run_replication(batch.cell, rep, config))
    frame = pd.DataFrame(rows)
    path = Path(raw_dir)/batch.filename
    temporary = path.with_suffix(f".tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)
    return {"file": path.name, "rows": len(frame),
            "failed_rows": int((frame["status"] != "ok").sum())}


def execute(config: dict[str, Any], output: str | Path, jobs: int = 1,
            resume: bool = False, dry_run: bool = False) -> dict[str, Any]:
    config = validate_config(config)
    cells = make_cells(config)
    batches = make_batches(cells, int(config["batch_size"]))
    expected_datasets = sum(cell["repetitions"] for cell in cells)
    expected_rows = expected_datasets*len(config["layers"])*len(config["methods"])
    description = {"cells": len(cells), "datasets": expected_datasets,
                   "layers": len(config["layers"]), "raw_rows": expected_rows,
                   "batches": len(batches)}
    print(json.dumps(description, ensure_ascii=False, indent=2), flush=True)
    if dry_run:
        return description
    output = Path(output)
    raw = output/"raw"
    snapshot = output/"config_resolved.json"
    fingerprint = short_hash(config)
    if snapshot.exists():
        if not resume:
            raise FileExistsError(f"{output} already contains a run; use --resume or a new output path")
        old = json.loads(snapshot.read_text(encoding="utf-8"))
        if short_hash(old) != fingerprint:
            raise ValueError("resume requires an identical resolved configuration, including repetition counts")
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to reuse a nonempty, unrecognized output directory: {output}")
    raw.mkdir(parents=True, exist_ok=True)
    snapshot.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(cells).to_csv(output/"truth_cells.csv", index=False)
    manifest = {**description, "config_hash": fingerprint,
                "last_started_utc": datetime.now(timezone.utc).isoformat(),
                "python": platform.python_version(), "platform": platform.platform(),
                "packages": {name: importlib.metadata.version(name)
                             for name in ("numpy", "scipy", "pandas", "matplotlib", "joblib")},
                "jobs": jobs, "resume": resume}
    (output/"run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    todo = []
    for batch in batches:
        path = raw/batch.filename
        if path.exists():
            # Checkpoints are written atomically, but verify their row count as well.
            nrows = len(pd.read_csv(path, usecols=["method"]))
            expected = ((batch.stop-batch.first)*len(config["layers"])
                        *len(config["methods"]))
            if nrows != expected:
                raise RuntimeError(f"invalid checkpoint row count: {path}")
        else:
            todo.append(batch)
    done = len(batches)-len(todo)
    print(f"Completed checkpoints: {done}/{len(batches)}", flush=True)
    with parallel_config(backend="loky", inner_max_num_threads=1):
        stream = Parallel(n_jobs=jobs, return_as="generator_unordered")(
            delayed(run_batch)(batch, config, str(raw)) for batch in todo)
        for result in stream:
            done += 1
            if done % max(1, len(batches)//50) == 0 or result["failed_rows"]:
                print(f"[{done}/{len(batches)}] {result['file']}: "
                      f"{result['rows']} rows, {result['failed_rows']} failed", flush=True)
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["completed_batches"] = len(batches)
    (output/"run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
