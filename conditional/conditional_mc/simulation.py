"""Conditional Monte Carlo driver with atomic batch checkpoints."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import traceback
from typing import Any

import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import norm

from marginal_mc.dgp import MEAN_FNS, ErrorLaw, generate, stable_rng
from marginal_mc.simulation import short_hash, Batch, make_batches

from .estimators import estimate_at_s0, estimate_curve
from .oracle import conditional_truth


DEFAULTS = {
    "seed": 20260906, "alpha": 0.10, "test_level": 0.05, "ci_level": 0.95,
    "mixture_p": 0.2, "mixture_a": 5.0, "error_base": "normal", "emg_lambda": 1.5,
    "mean_fn": "paper", "oracle_nodes": 160, "batch_size": 10,
    "c_semi": 1.5, "c_conf": 1.5, "c_n_factor": 1.0,
    "mean_bw_by_s0": True, "mean_cv_radius": 0.25,
    "methods": ["semi", "conf", "conf_hsemi"],
    "locations": [-0.5, 0.0, 0.5],
    "errors": ["left", "right"],
    "sample_sizes": [1000, 2000, 4000, 8000, 16000],
    "repetitions": 1000,
    "fixed_beta": None,
}


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    c = deepcopy(config)
    for key, value in DEFAULTS.items():
        c.setdefault(key, value)
    if not 0 < float(c["alpha"]) < 1:
        raise ValueError("alpha must lie in (0,1)")
    if float(c["c_semi"]) <= 0 or float(c["c_conf"]) <= 0 or float(c["c_n_factor"]) <= 0:
        raise ValueError("bandwidth constants and c_n_factor must be positive")
    if float(c["mean_cv_radius"]) <= 0:
        raise ValueError("mean_cv_radius must be positive")
    if c.get("mean_fn", "paper") not in MEAN_FNS:
        raise ValueError("mean_fn must be paper, quadratic, quad_half, or sine_quad")
    allowed = {"semi", "conf", "conf_hsemi"}
    if not c["methods"] or any(m not in allowed for m in c["methods"]):
        raise ValueError(f"methods must be a nonempty subset of {sorted(allowed)}")
    if not c.get("locations"):
        raise ValueError("locations required")
    if any(abs(float(s)) >= 1 for s in c["locations"]):
        raise ValueError("locations must be interior points of (-1,1)")
    if not c.get("errors") or not set(c["errors"]) <= {"symmetric", "right", "left"}:
        raise ValueError("invalid errors")
    if not c.get("sample_sizes") or any(int(n) < 40 or int(n) % 2 for n in c["sample_sizes"]):
        raise ValueError("sample_sizes must be even and >=40")
    if int(c["repetitions"]) < 1 or int(c["batch_size"]) < 1:
        raise ValueError("repetitions and batch_size must be positive")
    if not c.get("layers"):
        raise ValueError("at least one nuisance layer is required")
    if not c.get("studies"):
        raise ValueError("at least one study is required")
    for study in c["studies"]:
        if study.get("type", "estimation") not in {"estimation", "inference", "acquisition"}:
            raise ValueError("study type must be estimation, inference, or acquisition")
        if study.get("type", "estimation") == "inference":
            if float(study["delta0"]) <= 0:
                raise ValueError("positive delta0 required")
            if not study.get("gaps"):
                raise ValueError("inference studies need gaps")
        if study.get("type") == "acquisition":
            if c.get("fixed_beta") is None:
                raise ValueError("acquisition requires fixed_beta")
            if float(study["delta0"]) <= 0:
                raise ValueError("positive delta0 required")
        if c.get("fixed_beta") is not None and study.get("type") == "inference":
            raise ValueError("fixed_beta cannot be combined with inference")
    if c.get("fixed_beta") is not None:
        beta = float(c["fixed_beta"])
        if not math.isfinite(beta) or beta <= 0:
            raise ValueError("fixed_beta must be finite and positive")
        c["fixed_beta"] = beta
    return c


def make_cells(config: dict[str, Any]) -> list[dict[str, Any]]:
    if config.get("frozen_cells"):
        return [dict(cell) for cell in config["frozen_cells"]]
    cells = []
    for error in config["errors"]:
        for n in config["sample_sizes"]:
            for study in config["studies"]:
                stype = study.get("type", "estimation")
                if stype == "estimation":
                    locations = list(config["locations"])
                    beta = float(config["fixed_beta"])
                    truth0 = conditional_truth(
                        error, float(locations[0]), config["alpha"], beta, None,
                        config["oracle_nodes"], config["mixture_p"], config["mixture_a"],
                        config["error_base"], config.get("emg_lambda", 1.5),
                    )
                    cell = {
                        "error": error, "n": int(n), "study": study["name"],
                        "study_type": "estimation", "beta": beta,
                        "gap": 0.0, "delta0": None, "s0": None,
                        "locations": [float(s) for s in locations],
                        "repetitions": int(config["repetitions"]),
                        "error_base": config["error_base"],
                    }
                    cell["cell_id"] = short_hash({k: cell[k] for k in cell if k != "repetitions"})
                    cell["truth_by_s0"] = {
                        f"{s:.6g}": conditional_truth(
                            error, float(s), config["alpha"], beta, None,
                            config["oracle_nodes"], config["mixture_p"], config["mixture_a"],
                            config["error_base"], config.get("emg_lambda", 1.5),
                        ) for s in locations
                    }
                    cell["delta_true"] = truth0["delta_true"]
                    cells.append(cell)
                elif stype == "acquisition":
                    locations = [float(s) for s in config["locations"]]
                    beta = float(config["fixed_beta"])
                    delta0 = float(study["delta0"])
                    truth_by_s0 = {
                        f"{s:.6g}": conditional_truth(
                            error, float(s), config["alpha"], beta, None,
                            config["oracle_nodes"], config["mixture_p"], config["mixture_a"],
                            config["error_base"], config.get("emg_lambda", 1.5),
                        ) for s in locations
                    }
                    cell = {
                        "error": error, "n": int(n), "study": study["name"],
                        "study_type": "acquisition", "beta": beta,
                        "gap": 0.0, "delta0": delta0, "s0": None,
                        "locations": locations,
                        "repetitions": int(config["repetitions"]),
                        "error_base": config["error_base"],
                        "truth_by_s0": truth_by_s0,
                    }
                    cell["cell_id"] = short_hash({
                        k: cell[k] for k in cell
                        if k not in {"repetitions", "truth_by_s0"}
                    })
                    cell["delta_true"] = truth_by_s0[f"{locations[0]:.6g}"]["delta_true"]
                    cells.append(cell)
                else:
                    delta0 = float(study["delta0"])
                    for s0 in config["locations"]:
                        for gap in study["gaps"]:
                            target = delta0 + float(gap)
                            if target <= 0:
                                raise ValueError("all simulated Delta(s0) must be positive")
                            truth = conditional_truth(
                                error, float(s0), config["alpha"], None, target,
                                config["oracle_nodes"], config["mixture_p"],
                                config["mixture_a"], config["error_base"],
                                config.get("emg_lambda", 1.5),
                            )
                            cell = {
                                "error": error, "n": int(n), "study": study["name"],
                                "study_type": "inference", "beta": truth["beta"],
                                "s0": float(s0), "locations": [float(s0)],
                                "gap": float(gap), "delta0": delta0,
                                "delta_target": target, "delta_true": truth["delta_true"],
                                "zeta_x": truth["zeta_x"], "zeta_w": truth["zeta_w"],
                                "repetitions": int(config["repetitions"]),
                                "error_base": config["error_base"],
                                "truth_by_s0": {f"{float(s0):.6g}": truth},
                            }
                            cell["cell_id"] = short_hash({
                                k: cell[k] for k in cell
                                if k not in {"repetitions", "truth_by_s0"}
                            })
                            cells.append(cell)
    return cells


def run_replication(cell: dict[str, Any], rep: int, config: dict[str, Any]) -> list[dict[str, Any]]:
    law = ErrorLaw(cell["error"], config["mixture_p"], config["mixture_a"],
                   config["error_base"], config.get("emg_lambda", 1.5))
    s0_key = "all" if cell.get("s0") is None else f"{float(cell['s0']):.6g}"
    data_key = (config["error_base"], cell["error"], cell["n"],
                f"{cell['beta']:.12g}", cell["study_type"],
                s0_key, f"{cell.get('gap', 0.0):.6g}", rep)
    rng = stable_rng(config["seed"], "cond_data", *data_key)
    data = generate(cell["n"], cell["beta"], law, rng, config["mean_fn"])
    z_test = float(norm.ppf(1.0 - config["test_level"]))
    z_ci = float(norm.ppf(0.5 + 0.5 * config["ci_level"]))
    meta = {
        "error": cell["error"], "n": cell["n"], "rep": rep, "beta": cell["beta"],
        "study": cell["study"], "study_type": cell["study_type"],
        "gap": cell.get("gap", 0.0), "delta0": cell.get("delta0"),
        "alpha": config["alpha"], "mean_fn": config["mean_fn"],
        "c_semi": config["c_semi"], "c_conf": config["c_conf"],
        "error_base": config["error_base"],
    }
    output = []
    tested = cell["study_type"] in {"inference", "acquisition"}

    def _consume(layer: dict[str, Any], rows: list[dict[str, Any]]) -> None:
        for row in rows:
            truth = cell["truth_by_s0"][f"{float(row['s0']):.6g}"]
            packed = {**meta, **row, "layer": layer["id"], "kind": layer["kind"]}
            packed.update({
                "zeta_x_true": truth["zeta_x"],
                "zeta_w_true": truth["zeta_w"],
                "delta_true": truth["delta_true"],
            })
            if cell["study_type"] == "acquisition":
                packed["gap"] = float(truth["delta_true"]) - float(cell["delta0"])
            if packed.get("estimator_valid"):
                packed["bias_x"] = packed["zeta_x_est"] - truth["zeta_x"]
                packed["bias_w"] = packed["zeta_w_est"] - truth["zeta_w"]
                packed["bias_delta"] = packed["delta_est"] - truth["delta_true"]
            if packed.get("inference_valid"):
                se_d = packed["se_delta"]
                null = (float(cell["delta0"]) if tested else truth["delta_true"])
                packed["T"] = (packed["delta_est"] - null) / se_d
                if tested:
                    packed["reject_nominal"] = float(packed["T"] > z_test)
                packed["ci_x_cover"] = float(
                    abs(packed["zeta_x_est"] - truth["zeta_x"]) <= z_ci * packed["se_x"])
                packed["ci_w_cover"] = float(
                    abs(packed["zeta_w_est"] - truth["zeta_w"]) <= z_ci * packed["se_w"])
                packed["ci_delta_cover"] = float(
                    abs(packed["delta_est"] - truth["delta_true"]) <= z_ci * se_d)
                packed["ci_delta_length"] = 2.0 * z_ci * se_d
            output.append(packed)

    for layer in config["layers"]:
        if cell["study_type"] == "acquisition":
            try:
                _consume(layer, estimate_curve(
                    data, law, layer, config, cell["locations"], cell["beta"],
                    cell["truth_by_s0"],
                ))
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                tail = " | ".join(traceback.format_exc().splitlines()[-4:])
                failed = [{"method": m, "s0": float(s0), "status": "exception",
                           "error_message": message, "error_trace_tail": tail,
                           "estimator_valid": False, "inference_valid": False}
                          for s0 in cell["locations"] for m in config["methods"]]
                _consume(layer, failed)
            continue
        for s0 in cell["locations"]:
            truth = cell["truth_by_s0"][f"{float(s0):.6g}"]
            try:
                rows = estimate_at_s0(data, law, layer, config, float(s0),
                                      cell["beta"], truth)
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                tail = " | ".join(traceback.format_exc().splitlines()[-4:])
                rows = [{"method": m, "s0": float(s0), "status": "exception",
                         "error_message": message, "error_trace_tail": tail,
                         "estimator_valid": False, "inference_valid": False}
                        for m in config["methods"]]
            _consume(layer, rows)
    return output


def run_batch(batch: Batch, config: dict[str, Any], raw_dir: str) -> dict[str, Any]:
    rows = []
    for rep in range(batch.first, batch.stop):
        rows.extend(run_replication(batch.cell, rep, config))
    frame = pd.DataFrame(rows)
    path = Path(raw_dir) / batch.filename
    temporary = path.with_suffix(f".tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)
    return {"file": path.name, "rows": len(frame),
            "failed_rows": int((frame["status"] != "ok").sum())}


def _shard_bucket(filename: str, spec: str) -> tuple[int, int, int]:
    part, total = spec.split("/")
    keep, modulus = int(part), int(total)
    if modulus < 1 or not 0 <= keep < modulus:
        raise ValueError("shard must look like K/N with 0 <= K < N")
    bucket = int(hashlib.md5(filename.encode()).hexdigest(), 16) % modulus
    return bucket, keep, modulus


def execute(config: dict[str, Any], output: str | Path, jobs: int = 1,
            resume: bool = False, dry_run: bool = False,
            shard: str | None = None, shard_except: str | None = None,
            only_files: str | Path | None = None) -> dict[str, Any]:
    config = validate_config(config)
    cells = make_cells(config)
    batches = make_batches(cells, int(config["batch_size"]))
    expected = sum(cell["repetitions"] for cell in cells)
    n_loc = [len(cell["locations"]) for cell in cells]
    description = {
        "cells": len(cells), "datasets": expected,
        "location_evals": int(sum(n * cell["repetitions"] for n, cell in zip(n_loc, cells))),
        "layers": len(config["layers"]), "methods": len(config["methods"]),
        "batches": len(batches), "c_semi": config["c_semi"], "c_conf": config["c_conf"],
    }
    print(json.dumps(description, indent=2), flush=True)
    if dry_run:
        return description
    out = Path(output)
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    (out / "config_resolved.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    manifest = {
        **description,
        "config_hash": short_hash(config),
        "last_started_utc": datetime.now(timezone.utc).isoformat(),
        "jobs": jobs, "resume": resume,
    }
    todo = batches
    if resume:
        done = {p.name for p in raw.glob("*.csv")}
        todo = [b for b in batches if b.filename not in done]
    print(f"Completed checkpoints: {len(batches)-len(todo)}/{len(batches)}", flush=True)
    if shard and shard_except:
        raise ValueError("pass only one of shard or shard_except")
    spec = shard or shard_except
    if spec:
        exclude = shard_except is not None
        selected = []
        for batch in todo:
            bucket, keep, _modulus = _shard_bucket(batch.filename, spec)
            if (bucket != keep) if exclude else (bucket == keep):
                selected.append(batch)
        print(
            f"Shard {'except ' if exclude else ''}{spec}: "
            f"{len(selected)}/{len(todo)} remaining batches",
            flush=True,
        )
        todo = selected
    if only_files:
        allowed = {line.strip() for line in Path(only_files).read_text(encoding="utf-8").splitlines() if line.strip()}
        todo = [batch for batch in todo if batch.filename in allowed]
        print(f"Only-files: {len(todo)} remaining batches", flush=True)
        manifest["only_files"] = str(only_files)
    if todo:
        stream = Parallel(n_jobs=jobs, return_as="generator_unordered")(
            delayed(run_batch)(batch, config, str(raw)) for batch in todo
        )
        assigned = len(todo)
        for i, result in enumerate(stream, 1):
            if i % max(1, assigned // 20) == 0 or result["failed_rows"] or i == assigned:
                print(f"[{i}/{assigned}] {result['file']}: "
                      f"{result['rows']} rows, {result['failed_rows']} failed", flush=True)
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["completed_batches"] = len(list(raw.glob("*.csv")))
    if spec:
        manifest["shard"] = ("except " if shard_except else "") + spec
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
