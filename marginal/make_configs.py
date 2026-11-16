#!/usr/bin/env python3
"""Write editable presets; executing this does not run Monte Carlo experiments."""
from __future__ import annotations
from copy import deepcopy
import json
from pathlib import Path

ORACLE = {"id": "oracle", "kind": "oracle"}
LEARNED = {"id": "learned", "kind": "learned", "m_bw_mult": 0.8,
           "f_bw_mult": 0.65, "y_bw_mult": 0.8, "density_positive_part": True,
           "min_neighbors": 12, "bandwidth_pilot_size": 256}

LEARNED_CV = deepcopy(LEARNED)
LEARNED_CV.update({
    "id": "learned_cv",
    "bandwidth_selection": {
        "method": "inner_cv",
        "folds": 3,
        "max_points": 512,
        "m_multipliers": [0.5, 0.75, 1.0, 1.25, 1.5],
        "f_multipliers": [0.75, 1.0, 1.25],
        "y_multipliers": [0.75, 1.0, 1.25],
    },
})

LEARNED_RESIDUAL_CV = deepcopy(LEARNED_CV)
LEARNED_RESIDUAL_CV.update({
    "id": "learned_residual_cv",
    "density_estimator": "residual_loo",
    "density_feature_selection": "inner_cv",
    "density_feature_mode": "all",
    "density_feature_candidates": ["all", "x_only"],
    "direct_contrast": True,
})

LEARNED_RESIDUAL_JOINT_R8 = deepcopy(LEARNED_RESIDUAL_CV)
LEARNED_RESIDUAL_JOINT_R8.update({
    "id": "learned_residual_cv_joint_r8",
    "mean_bandwidth": "anisotropic",
})
LEARNED_RESIDUAL_JOINT_R8["bandwidth_selection"] = {
    "method": "inner_cv",
    "folds": 3,
    "max_points": 512,
    "m_multipliers": [0.5, 0.75, 1.0],
    "z_scales": [4.0, 5.0, 6.0, 8.0],
    "f_multipliers": [0.75, 1.0, 1.25],
    "y_multipliers": [0.75, 1.0, 1.25],
}

# Drop unused large c (1.25-2 never selected) and extend r to 12.
LEARNED_RESIDUAL_JOINT_R12 = deepcopy(LEARNED_RESIDUAL_JOINT_R8)
LEARNED_RESIDUAL_JOINT_R12.update({
    "id": "learned_residual_cv_joint_r12",
})
LEARNED_RESIDUAL_JOINT_R12["bandwidth_selection"] = {
    "method": "inner_cv",
    "folds": 3,
    "max_points": 512,
    "m_multipliers": [0.5, 0.75, 1.0],
    "z_scales": [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0],
    "f_multipliers": [0.75, 1.0, 1.25],
    "y_multipliers": [0.75, 1.0, 1.25],
}

# Density CV was pinned at y=0.75 and f_w=0.75; widen downward and keep f_x room.
LEARNED_RESIDUAL_JOINT_R8_DENSITY_WIDE = deepcopy(LEARNED_RESIDUAL_JOINT_R8)
LEARNED_RESIDUAL_JOINT_R8_DENSITY_WIDE.update({
    "id": "learned_residual_cv_joint_r8_density_wide",
})
LEARNED_RESIDUAL_JOINT_R8_DENSITY_WIDE["bandwidth_selection"] = {
    **deepcopy(LEARNED_RESIDUAL_JOINT_R8["bandwidth_selection"]),
    "f_multipliers": [0.5, 0.75, 1.0, 1.25],
    "y_multipliers": [0.5, 0.625, 0.75],
}

# Next density step: expand only past the pinned sides, drop unused y.
LEARNED_RESIDUAL_JOINT_R8_DENSITY_WIDE2 = deepcopy(LEARNED_RESIDUAL_JOINT_R8)
LEARNED_RESIDUAL_JOINT_R8_DENSITY_WIDE2.update({
    "id": "learned_residual_cv_joint_r8_density_wide2",
})
LEARNED_RESIDUAL_JOINT_R8_DENSITY_WIDE2["bandwidth_selection"] = {
    **deepcopy(LEARNED_RESIDUAL_JOINT_R8["bandwidth_selection"]),
    "f_multipliers": [0.35, 0.5, 0.75, 1.0, 1.25, 1.5],
    "y_multipliers": [0.35, 0.5],
}

# y=0.35 was still the selected edge, especially for the w residual density.
LEARNED_RESIDUAL_JOINT_R8_DENSITY_WIDE3 = deepcopy(LEARNED_RESIDUAL_JOINT_R8_DENSITY_WIDE2)
LEARNED_RESIDUAL_JOINT_R8_DENSITY_WIDE3.update({
    "id": "learned_residual_cv_joint_r8_density_wide3",
})
LEARNED_RESIDUAL_JOINT_R8_DENSITY_WIDE3["bandwidth_selection"] = {
    **deepcopy(LEARNED_RESIDUAL_JOINT_R8_DENSITY_WIDE2["bandwidth_selection"]),
    "y_multipliers": [0.2, 0.25, 0.35, 0.5],
}


def controlled(gamma: float, cm: float = 1.0, shift: str = "positive") -> dict:
    label = f"controlled_g{round(gamma*100):03d}"
    if cm != 1:
        label += f"_cm{cm:g}"
    if shift != "positive":
        label += "_"+shift
    return {"id": label, "kind": "controlled", "gamma_m": gamma, "gamma_f": gamma,
            "c_m": cm, "c_f": 1.0, "shift": shift, "dagger_sd": 2.0}


def base(name: str) -> dict:
    return {"name": name, "seed": 20260906, "alpha": 0.10, "test_level": 0.05,
            "ci_level": 0.95, "mixture_p": 0.20, "mixture_a": 5.0,
            "oracle_nodes": 160, "errors": ["symmetric", "right", "left"],
            "sample_sizes": [500, 1000, 2000, 4000], "repetitions": 1000,
            "batch_size": 10, "g_estimator": "spacing", "variance": "empirical",
            "layers": [deepcopy(ORACLE), controlled(0.35), deepcopy(LEARNED)],
            "studies": [{"name": "estimation_d020", "type": "estimation",
                         "delta0": 0.20, "gaps": [0.0]}],
            "diagnostics": {"enabled": True, "every": 10, "grid_side": 15,
                            "density_points": 21, "density_radius": 0.35,
                            "b_nodes": 24, "b_check_nodes": 0},
            "size_adjustment": {"enabled": False, "calibration_reps": 0}}


def configs() -> dict[str, dict]:
    presets = {}
    smoke = base("smoke")
    smoke.update({"sample_sizes": [200, 400], "repetitions": 12, "batch_size": 4})
    smoke["studies"].append({"name": "inference_d020", "type": "inference",
                             "delta0": 0.20, "gaps": [0.0, 0.10]})
    smoke["size_adjustment"] = {"enabled": True, "calibration_reps": 20}
    smoke["diagnostics"].update({"every": 4, "grid_side": 9,
                                   "density_points": 11, "b_nodes": 16})
    smoke["note"] = "Only software/data-flow checks; R=12/20 cannot assess nominal size or power."
    presets["smoke"] = smoke

    oracle = base("main_oracle")
    oracle.update({"layers": [deepcopy(ORACLE)], "repetitions": 2000,
                   "sample_sizes": [500, 1000, 2000, 4000, 8000, 16000]})
    presets["main_oracle"] = oracle

    control = base("main_controlled")
    control.update({"layers": [controlled(g) for g in (0.35, 0.30, 0.25, 0.20)],
                    "repetitions": 2000,
                    "sample_sizes": [500, 1000, 2000, 4000, 8000, 16000]})
    presets["main_controlled"] = control

    learned = base("main_learned")
    learned["layers"] = [deepcopy(LEARNED)]
    presets["main_learned"] = learned

    learned_cv = base("learned_cv_asymmetric")
    learned_cv.update({
        "errors": ["right", "left"],
        "sample_sizes": [500, 1000, 2000, 4000, 8000, 16000],
        "layers": [deepcopy(LEARNED_CV)],
        "repetitions": 1000,
    })
    presets["learned_cv_asymmetric"] = learned_cv

    learned_residual_cv = deepcopy(learned_cv)
    learned_residual_cv["name"] = "learned_residual_cv"
    learned_residual_cv["layers"] = [deepcopy(LEARNED_RESIDUAL_CV)]
    presets["learned_residual_cv"] = learned_residual_cv

    learned_residual_cv_t4_right = deepcopy(learned_cv)
    learned_residual_cv_t4_right.update({
        "name": "learned_residual_cv_t4_right_inference",
        "error_base": "t4",
        "errors": ["right"],
        "sample_sizes": [1000, 2000, 4000, 8000, 16000],
        "layers": [deepcopy(LEARNED_RESIDUAL_CV)],
        "repetitions": 1000,
    })
    learned_residual_cv_t4_right["studies"] = [{
        "name": "inference_d020",
        "type": "inference",
        "delta0": 0.20,
        "gaps": [-0.05, 0.0, 0.05, 0.10, 0.20],
    }]
    learned_residual_cv_t4_right["diagnostics"] = {
        "enabled": False, "every": 20, "grid_side": 15,
        "density_points": 21, "density_radius": 0.35,
        "b_nodes": 24, "b_check_nodes": 0,
    }
    learned_residual_cv_t4_right["size_adjustment"] = {
        "enabled": False, "calibration_reps": 0,
    }
    presets["learned_residual_cv_t4_right_inference"] = learned_residual_cv_t4_right

    quadratic_m0 = deepcopy(learned_cv)
    quadratic_m0.update({
        "name": "learned_residual_cv_quadratic_m0_estimation",
        "mean_fn": "quadratic",
        "errors": ["right", "left"],
        "sample_sizes": [1000, 2000, 4000, 8000, 16000],
        "layers": [deepcopy(LEARNED_RESIDUAL_CV)],
        "repetitions": 1000,
        "methods": ["semi", "conf"],
    })
    quadratic_m0["studies"] = [{
        "name": "estimation_d020",
        "type": "estimation",
        "delta0": 0.20,
        "gaps": [0.0],
    }]
    quadratic_m0["diagnostics"] = {
        "enabled": True, "every": 10, "grid_side": 15,
        "density_points": 0, "density_radius": 0.35,
        "b_nodes": 32, "b_check_nodes": 0, "b_mc_size": 20000,
    }
    quadratic_m0["size_adjustment"] = {"enabled": False, "calibration_reps": 0}
    presets["learned_residual_cv_quadratic_m0_estimation"] = quadratic_m0

    quadratic_m0_a10 = deepcopy(quadratic_m0)
    quadratic_m0_a10["name"] = "learned_residual_cv_quadratic_m0_a10_estimation"
    quadratic_m0_a10["mixture_a"] = 10.0
    presets["learned_residual_cv_quadratic_m0_a10_estimation"] = quadratic_m0_a10

    quadratic_m0_a10_b07 = deepcopy(quadratic_m0_a10)
    quadratic_m0_a10_b07.update({
        "name": "learned_residual_cv_quadratic_m0_a10_b07_right_estimation",
        "errors": ["right"],
        "fixed_beta": 0.7,
    })
    quadratic_m0_a10_b07["studies"] = [{
        "name": "estimation_b07",
        "type": "estimation",
        "delta0": 0.20,
        "gaps": [0.0],
    }]
    presets["learned_residual_cv_quadratic_m0_a10_b07_right_estimation"] = quadratic_m0_a10_b07

    quadratic_m0_a10_b05 = deepcopy(quadratic_m0_a10_b07)
    quadratic_m0_a10_b05.update({
        "name": "learned_residual_cv_quadratic_m0_a10_b05_right_estimation",
        "fixed_beta": 0.5,
    })
    quadratic_m0_a10_b05["studies"] = [{
        "name": "estimation_b05",
        "type": "estimation",
        "delta0": 0.20,
        "gaps": [0.0],
    }]
    presets["learned_residual_cv_quadratic_m0_a10_b05_right_estimation"] = quadratic_m0_a10_b05

    paper_m0_a10_b05 = deepcopy(learned_residual_cv)
    paper_m0_a10_b05.update({
        "name": "learned_residual_cv_paper_m0_a10_b05_estimation",
        "mean_fn": "paper",
        "mixture_a": 10.0,
        "fixed_beta": 0.5,
        "errors": ["left", "right"],
        "sample_sizes": [1000, 2000, 4000, 8000, 16000],
        "methods": ["semi", "conf"],
    })
    paper_m0_a10_b05["studies"] = [{
        "name": "estimation_b05",
        "type": "estimation",
        "delta0": 0.20,
        "gaps": [0.0],
    }]
    presets["learned_residual_cv_paper_m0_a10_b05_estimation"] = paper_m0_a10_b05

    paper_a10_joint_est = deepcopy(paper_m0_a10_b05)
    paper_a10_joint_est.update({
        "name": "learned_residual_cv_paper_a10_joint_r8_estimation",
        "layers": [deepcopy(LEARNED_RESIDUAL_JOINT_R8)],
        "methods": ["semi", "conf"],
    })
    paper_a10_joint_est["diagnostics"] = {
        "enabled": True, "every": 20, "grid_side": 15,
        "density_points": 0, "density_radius": 0.35,
        "b_nodes": 24, "b_check_nodes": 0,
    }
    presets["learned_residual_cv_paper_a10_joint_r8_estimation"] = paper_a10_joint_est

    paper_a10_joint_inf = deepcopy(paper_a10_joint_est)
    paper_a10_joint_inf.update({
        "name": "learned_residual_cv_paper_a10_joint_r8_inference",
        "errors": ["right", "left"],
    })
    paper_a10_joint_inf.pop("fixed_beta", None)
    paper_a10_joint_inf["studies"] = [{
        "name": "inference_d020",
        "type": "inference",
        "delta0": 0.20,
        "gaps": [-0.05, 0.0, 0.05, 0.10, 0.20],
    }]
    paper_a10_joint_inf["diagnostics"] = {
        "enabled": False, "every": 20, "grid_side": 15,
        "density_points": 0, "density_radius": 0.35,
        "b_nodes": 24, "b_check_nodes": 0,
    }
    paper_a10_joint_inf["size_adjustment"] = {"enabled": False, "calibration_reps": 0}
    presets["learned_residual_cv_paper_a10_joint_r8_inference"] = paper_a10_joint_inf

    paper_a10_joint_inf_size = deepcopy(paper_a10_joint_inf)
    paper_a10_joint_inf_size["name"] = "learned_residual_cv_paper_a10_joint_r8_inference_size"
    paper_a10_joint_inf_size["studies"] = [{
        "name": "inference_d020",
        "type": "inference",
        "delta0": 0.20,
        "gaps": [0.0],
    }]
    presets["learned_residual_cv_paper_a10_joint_r8_inference_size"] = paper_a10_joint_inf_size

    paper_a10_joint_m_oracle_size = deepcopy(paper_a10_joint_inf_size)
    paper_a10_joint_m_oracle_size.update({
        "name": "learned_m_oracle_density_paper_a10_joint_r8_inference_size",
        "layers": [{
            **deepcopy(LEARNED_RESIDUAL_JOINT_R8),
            "id": "learned_joint_r8_oracle_density",
            "kind": "learned_mean_oracle_density",
        }],
    })
    presets["learned_m_oracle_density_paper_a10_joint_r8_inference_size"] = paper_a10_joint_m_oracle_size

    # Quick size check of trimmed-c / r<=12 vs the r<=8 oracle-A Table 2.
    paper_a10_joint_r12_m_oracle_size = deepcopy(paper_a10_joint_m_oracle_size)
    paper_a10_joint_r12_m_oracle_size.update({
        "name": "learned_m_oracle_density_paper_a10_joint_r12_inference_size",
        "sample_sizes": [1000, 4000, 16000],
        "layers": [{
            **deepcopy(LEARNED_RESIDUAL_JOINT_R12),
            "id": "learned_joint_r12_oracle_density",
            "kind": "learned_mean_oracle_density",
        }],
    })
    presets["learned_m_oracle_density_paper_a10_joint_r12_inference_size"] = paper_a10_joint_r12_m_oracle_size

    paper_a10_density_wide_size = deepcopy(paper_a10_joint_inf_size)
    paper_a10_density_wide_size.update({
        "name": "learned_residual_cv_paper_a10_density_wide_inference_size",
        "sample_sizes": [1000, 4000, 16000],
        "layers": [deepcopy(LEARNED_RESIDUAL_JOINT_R8_DENSITY_WIDE)],
    })
    presets["learned_residual_cv_paper_a10_density_wide_inference_size"] = paper_a10_density_wide_size

    paper_a10_density_wide2_size = deepcopy(paper_a10_density_wide_size)
    paper_a10_density_wide2_size.update({
        "name": "learned_residual_cv_paper_a10_density_wide2_inference_size",
        "layers": [deepcopy(LEARNED_RESIDUAL_JOINT_R8_DENSITY_WIDE2)],
    })
    presets["learned_residual_cv_paper_a10_density_wide2_inference_size"] = paper_a10_density_wide2_size

    paper_a10_both_oracle_size = deepcopy(paper_a10_density_wide_size)
    paper_a10_both_oracle_size.update({
        "name": "oracle_paper_a10_inference_size",
        "layers": [deepcopy(ORACLE)],
    })
    presets["oracle_paper_a10_inference_size"] = paper_a10_both_oracle_size

    paper_a10_fake_both_size = deepcopy(paper_a10_both_oracle_size)
    paper_a10_fake_both_size.update({
        "name": "fake_mean_fake_A_paper_a10_inference_size",
        "layers": [{
            "id": "fake_mean_fake_A",
            "kind": "fake_mean_fake_A",
            "gamma_m_x": 0.4,
            "gamma_m_w": 1.0 / 3.0,
            "gamma_a": 1.0 / 3.0,
            "c_m": 1.0,
            "c_a": 1.0,
            "shift": "positive",
        }],
    })
    presets["fake_mean_fake_A_paper_a10_inference_size"] = paper_a10_fake_both_size

    paper_a10_fake_mean_oracle_A_size = deepcopy(paper_a10_both_oracle_size)
    paper_a10_fake_mean_oracle_A_size.update({
        "name": "fake_mean_oracle_density_paper_a10_inference_size",
        "layers": [{
            "id": "fake_mean_oracle_density",
            "kind": "fake_mean_oracle_density",
            "gamma_m_x": 0.4,
            "gamma_m_w": 1.0 / 3.0,
            "c_m": 1.0,
            "shift": "positive",
        }],
    })
    presets["fake_mean_oracle_density_paper_a10_inference_size"] = paper_a10_fake_mean_oracle_A_size

    paper_emg15_est = deepcopy(paper_a10_joint_est)
    paper_emg15_est.update({
        "name": "learned_residual_cv_emg15_density_wide2_estimation",
        "error_base": "exponnorm",
        "emg_lambda": 1.5,
        "layers": [deepcopy(LEARNED_RESIDUAL_JOINT_R8_DENSITY_WIDE2)],
        "fixed_beta": 0.5,
        "errors": ["left", "right"],
        "sample_sizes": [1000, 2000, 4000, 8000, 16000],
    })
    presets["learned_residual_cv_emg15_density_wide2_estimation"] = paper_emg15_est

    paper_emg15_inf = deepcopy(paper_a10_joint_inf_size)
    paper_emg15_inf.update({
        "name": "learned_residual_cv_emg15_density_wide2_inference_size",
        "error_base": "exponnorm",
        "emg_lambda": 1.5,
        "layers": [deepcopy(LEARNED_RESIDUAL_JOINT_R8_DENSITY_WIDE2)],
        "errors": ["right", "left"],
        "sample_sizes": [1000, 2000, 4000, 8000, 16000],
    })
    paper_emg15_inf.pop("fixed_beta", None)
    presets["learned_residual_cv_emg15_density_wide2_inference_size"] = paper_emg15_inf

    paper_emg20_est = deepcopy(paper_emg15_est)
    paper_emg20_est.update({
        "name": "learned_residual_cv_emg20_density_wide2_estimation",
        "emg_lambda": 2.0,
        "sample_sizes": [1000, 4000, 16000],
    })
    presets["learned_residual_cv_emg20_density_wide2_estimation"] = paper_emg20_est

    paper_emg20_inf = deepcopy(paper_emg15_inf)
    paper_emg20_inf.update({
        "name": "learned_residual_cv_emg20_density_wide2_inference_size",
        "emg_lambda": 2.0,
        "sample_sizes": [1000, 4000, 16000],
    })
    presets["learned_residual_cv_emg20_density_wide2_inference_size"] = paper_emg20_inf

    paper_a5_est = deepcopy(paper_a10_joint_est)
    paper_a5_est.update({
        "name": "learned_residual_cv_paper_a5_density_wide2_estimation",
        "mixture_a": 5.0,
        "layers": [deepcopy(LEARNED_RESIDUAL_JOINT_R8_DENSITY_WIDE2)],
        "sample_sizes": [1000, 4000, 8000],
        "fixed_beta": 0.5,
        "errors": ["left", "right"],
        "methods": ["semi", "conf"],
    })
    presets["learned_residual_cv_paper_a5_density_wide2_estimation"] = paper_a5_est

    paper_a5_quadhalf_est = deepcopy(paper_a5_est)
    paper_a5_quadhalf_est.update({
        "name": "learned_residual_cv_quadhalf_a5_density_wide2_estimation",
        "mean_fn": "quad_half",
        "repetitions": 500,
    })
    presets["learned_residual_cv_quadhalf_a5_density_wide2_estimation"] = paper_a5_quadhalf_est

    quadratic_a6_est = deepcopy(paper_a5_est)
    quadratic_a6_est.update({
        "name": "learned_residual_cv_quadratic_a6_density_wide2_estimation",
        "mean_fn": "quadratic",
        "mixture_a": 6.0,
        "repetitions": 100,
    })
    presets["learned_residual_cv_quadratic_a6_density_wide2_estimation"] = quadratic_a6_est

    quadratic_a6_est_extra = deepcopy(quadratic_a6_est)
    quadratic_a6_est_extra.update({
        "name": "learned_residual_cv_quadratic_a6_density_wide2_estimation_nextra",
        "sample_sizes": [2000, 16000],
    })
    presets["learned_residual_cv_quadratic_a6_density_wide2_estimation_nextra"] = quadratic_a6_est_extra

    quadratic_a6_b02_est = deepcopy(quadratic_a6_est)
    quadratic_a6_b02_est.update({
        "name": "learned_residual_cv_quadratic_a6_b02_density_wide2_estimation",
        "fixed_beta": 0.2,
        "sample_sizes": [1000, 2000, 4000, 8000],
        "repetitions": 100,
    })
    quadratic_a6_b02_est["studies"] = [{
        "name": "estimation_b02",
        "type": "estimation",
        "delta0": 0.20,
        "gaps": [0.0],
    }]
    presets["learned_residual_cv_quadratic_a6_b02_density_wide2_estimation"] = quadratic_a6_b02_est

    quadratic_a5_inf_size = deepcopy(paper_a5_est)
    quadratic_a5_inf_size.update({
        "name": "learned_residual_cv_quadratic_a5_density_wide2_inference_size",
        "mean_fn": "quadratic",
        "mixture_a": 5.0,
        "sample_sizes": [1000, 2000, 4000, 8000],
        "repetitions": 100,
        "errors": ["right", "left"],
        "methods": ["semi", "conf"],
    })
    quadratic_a5_inf_size.pop("fixed_beta", None)
    quadratic_a5_inf_size["studies"] = [{
        "name": "inference_d020",
        "type": "inference",
        "delta0": 0.20,
        "gaps": [0.0],
    }]
    quadratic_a5_inf_size["diagnostics"] = {
        "enabled": False, "every": 20, "grid_side": 15,
        "density_points": 0, "density_radius": 0.35,
        "b_nodes": 24, "b_check_nodes": 0,
    }
    presets["learned_residual_cv_quadratic_a5_density_wide2_inference_size"] = quadratic_a5_inf_size

    paper_ns = [1000, 2000, 4000, 8000, 16000]
    quadratic_a5_full_est = deepcopy(paper_a5_est)
    quadratic_a5_full_est.update({
        "name": "learned_residual_cv_quadratic_a5_density_wide2_estimation_b02",
        "mean_fn": "quadratic",
        "mixture_a": 5.0,
        "fixed_beta": 0.2,
        "sample_sizes": list(paper_ns),
        "repetitions": 1000,
        "errors": ["left", "right"],
        "methods": ["semi", "conf"],
        "layers": [deepcopy(LEARNED_RESIDUAL_JOINT_R8_DENSITY_WIDE2)],
    })
    quadratic_a5_full_est["studies"] = [{
        "name": "estimation_b02",
        "type": "estimation",
        "delta0": 0.20,
        "gaps": [0.0],
    }]
    presets["learned_residual_cv_quadratic_a5_density_wide2_estimation_b02"] = quadratic_a5_full_est

    quadratic_a5_full_inf = deepcopy(quadratic_a5_full_est)
    quadratic_a5_full_inf.update({
        "name": "learned_residual_cv_quadratic_a5_density_wide2_inference_d020",
        "errors": ["right", "left"],
    })
    quadratic_a5_full_inf.pop("fixed_beta", None)
    quadratic_a5_full_inf["studies"] = [{
        "name": "inference_d020",
        "type": "inference",
        "delta0": 0.20,
        "gaps": [0.0],
    }]
    quadratic_a5_full_inf["diagnostics"] = {
        "enabled": False, "every": 20, "grid_side": 15,
        "density_points": 0, "density_radius": 0.35,
        "b_nodes": 24, "b_check_nodes": 0,
    }
    presets["learned_residual_cv_quadratic_a5_density_wide2_inference_d020"] = quadratic_a5_full_inf

    paper_a5_full_est = deepcopy(quadratic_a5_full_est)
    paper_a5_full_est.update({
        "name": "learned_residual_cv_paper_a5_density_wide2_estimation_b02",
        "mean_fn": "paper",
    })
    presets["learned_residual_cv_paper_a5_density_wide2_estimation_b02"] = paper_a5_full_est

    paper_a5_full_inf = deepcopy(quadratic_a5_full_inf)
    paper_a5_full_inf.update({
        "name": "learned_residual_cv_paper_a5_density_wide2_inference_d020",
        "mean_fn": "paper",
    })
    presets["learned_residual_cv_paper_a5_density_wide2_inference_d020"] = paper_a5_full_inf

    paper_a5_power = deepcopy(paper_a5_full_inf)
    paper_a5_power["name"] = "learned_residual_cv_paper_a5_density_wide2_inference_d020_power"
    paper_a5_power["studies"] = [{
        "name": "inference_d020",
        "type": "inference",
        "delta0": 0.20,
        "gaps": [-0.05, 0.05, 0.10, 0.20],
    }]
    presets["learned_residual_cv_paper_a5_density_wide2_inference_d020_power"] = paper_a5_power

    paper_a5_t4_est = deepcopy(paper_a5_full_est)
    paper_a5_t4_est.update({
        "name": "learned_residual_cv_paper_a5_t4_density_wide2_estimation_b02",
        "error_base": "t4",
    })
    presets["learned_residual_cv_paper_a5_t4_density_wide2_estimation_b02"] = paper_a5_t4_est

    paper_a5_t4_inf = deepcopy(paper_a5_full_inf)
    paper_a5_t4_inf.update({
        "name": "learned_residual_cv_paper_a5_t4_density_wide2_inference_d020",
        "error_base": "t4",
    })
    paper_a5_t4_inf["studies"] = [{
        "name": "inference_d020",
        "type": "inference",
        "delta0": 0.20,
        "gaps": [-0.05, 0.0, 0.05, 0.10, 0.20],
    }]
    presets["learned_residual_cv_paper_a5_t4_density_wide2_inference_d020"] = paper_a5_t4_inf

    paper_a5_sym_est = deepcopy(paper_a5_full_est)
    paper_a5_sym_est.update({
        "name": "learned_residual_cv_paper_a5_density_wide2_estimation_b02_sym",
        "errors": ["symmetric"],
    })
    presets["learned_residual_cv_paper_a5_density_wide2_estimation_b02_sym"] = paper_a5_sym_est

    paper_a5_sym_inf = deepcopy(paper_a5_full_inf)
    paper_a5_sym_inf.update({
        "name": "learned_residual_cv_paper_a5_density_wide2_inference_d020_sym",
        "errors": ["symmetric"],
    })
    presets["learned_residual_cv_paper_a5_density_wide2_inference_d020_sym"] = paper_a5_sym_inf

    paper_a5_t4_sym_est = deepcopy(paper_a5_t4_est)
    paper_a5_t4_sym_est.update({
        "name": "learned_residual_cv_paper_a5_t4_density_wide2_estimation_b02_sym",
        "errors": ["symmetric"],
    })
    presets["learned_residual_cv_paper_a5_t4_density_wide2_estimation_b02_sym"] = paper_a5_t4_sym_est

    paper_a5_t4_sym_inf = deepcopy(paper_a5_full_inf)
    paper_a5_t4_sym_inf.update({
        "name": "learned_residual_cv_paper_a5_t4_density_wide2_inference_d020_sym",
        "errors": ["symmetric"],
        "error_base": "t4",
    })
    presets["learned_residual_cv_paper_a5_t4_density_wide2_inference_d020_sym"] = paper_a5_t4_sym_inf

    paper_m0_bcheck = deepcopy(quadratic_m0)
    paper_m0_bcheck["name"] = "learned_residual_cv_paper_m0_bcheck_estimation"
    paper_m0_bcheck["mean_fn"] = "paper"
    presets["learned_residual_cv_paper_m0_bcheck_estimation"] = paper_m0_bcheck

    learned_m_oracle = deepcopy(learned_cv)
    learned_m_oracle.update({
        "name": "learned_m_oracle_density_semi_right_size",
        "errors": ["right"],
        "sample_sizes": [1000, 2000, 4000, 8000, 16000],
        "layers": [{
            "id": "learned_m_oracle_density",
            "kind": "learned_mean_oracle_density",
            "m_bw_mult": 0.8,
            "f_bw_mult": 0.65,
            "y_bw_mult": 0.8,
            "density_positive_part": True,
            "min_neighbors": 12,
            "bandwidth_pilot_size": 256,
            "bandwidth_selection": deepcopy(LEARNED_CV["bandwidth_selection"]),
        }],
        "repetitions": 1000,
        "methods": ["semi"],
    })
    learned_m_oracle["studies"] = [{
        "name": "inference_d020",
        "type": "inference",
        "delta0": 0.20,
        "gaps": [0.0],
    }]
    learned_m_oracle["diagnostics"] = {
        "enabled": True, "every": 1, "grid_side": 15,
        "density_points": 0, "density_radius": 0.35,
        "b_nodes": 32, "b_check_nodes": 0,
    }
    learned_m_oracle["size_adjustment"] = {"enabled": False, "calibration_reps": 0}
    presets["learned_m_oracle_density_semi_right_size"] = learned_m_oracle

    learned_local_cubic = deepcopy(learned_m_oracle)
    learned_local_cubic.update({
        "name": "learned_local_cubic_oracle_density_semi_right_size",
        "layers": [{
            **deepcopy(learned_m_oracle["layers"][0]),
            "id": "learned_local_cubic_oracle_density",
            "mean_degree": 3,
            "min_neighbors": 24,
            "bandwidth_selection": {
                "method": "inner_cv",
                "folds": 3,
                "max_points": 512,
                "m_multipliers": [0.75, 1.0, 1.25, 1.5, 2.0, 2.5],
                "f_multipliers": [0.75, 1.0, 1.25],
                "y_multipliers": [0.75, 1.0, 1.25],
            },
        }],
    })
    presets["learned_local_cubic_oracle_density_semi_right_size"] = learned_local_cubic

    learned_ll_largeh = deepcopy(learned_m_oracle)
    learned_ll_largeh.update({
        "name": "learned_ll_largeh_oracle_density_semi_right_size",
        "layers": [{
            **deepcopy(learned_m_oracle["layers"][0]),
            "id": "learned_ll_largeh_oracle_density",
            "bandwidth_selection": {
                "method": "inner_cv",
                "folds": 3,
                "max_points": 512,
                "m_multipliers": [1.0, 1.5, 2.0, 2.5, 3.0],
                "f_multipliers": [0.75, 1.0, 1.25],
                "y_multipliers": [0.75, 1.0, 1.25],
            },
        }],
    })
    presets["learned_ll_largeh_oracle_density_semi_right_size"] = learned_ll_largeh

    learned_ll_aniso = deepcopy(learned_m_oracle)
    learned_ll_aniso.update({
        "name": "learned_ll_aniso_oracle_density_semi_right_size",
        "layers": [{
            **deepcopy(learned_m_oracle["layers"][0]),
            "id": "learned_ll_aniso_oracle_density",
            "mean_bandwidth": "anisotropic",
            "z_scale": 3.0,
        }],
    })
    presets["learned_ll_aniso_oracle_density_semi_right_size"] = learned_ll_aniso

    learned_ll_joint_bw = deepcopy(learned_m_oracle)
    learned_ll_joint_bw.update({
        "name": "learned_ll_joint_bw_oracle_density_semi_right_size",
        "layers": [{
            **deepcopy(learned_m_oracle["layers"][0]),
            "id": "learned_ll_joint_bw_oracle_density",
            "mean_bandwidth": "anisotropic",
            "bandwidth_selection": {
                "method": "inner_cv",
                "folds": 3,
                "max_points": 512,
                "m_multipliers": [0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
                "z_scales": [1.0, 1.5, 2.0, 3.0, 4.0],
                "f_multipliers": [0.75, 1.0, 1.25],
                "y_multipliers": [0.75, 1.0, 1.25],
            },
        }],
    })
    presets["learned_ll_joint_bw_oracle_density_semi_right_size"] = learned_ll_joint_bw

    learned_ll_joint_bw_r8 = deepcopy(learned_ll_joint_bw)
    learned_ll_joint_bw_r8.update({
        "name": "learned_ll_joint_bw_r8_oracle_density_semi_right_size",
        "layers": [{
            **deepcopy(learned_ll_joint_bw["layers"][0]),
            "id": "learned_ll_joint_bw_r8_oracle_density",
            "bandwidth_selection": {
                **deepcopy(learned_ll_joint_bw["layers"][0]["bandwidth_selection"]),
                "z_scales": [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0],
            },
        }],
    })
    presets["learned_ll_joint_bw_r8_oracle_density_semi_right_size"] = learned_ll_joint_bw_r8

    learned_residual_joint_r8 = deepcopy(learned_ll_joint_bw_r8)
    learned_residual_joint_r8.update({
        "name": "learned_residual_cv_joint_r8_semi_right_size",
        "layers": [deepcopy(LEARNED_RESIDUAL_JOINT_R8)],
    })
    presets["learned_residual_cv_joint_r8_semi_right_size"] = learned_residual_joint_r8

    fake_mean_oracle = deepcopy(learned_m_oracle)
    fake_mean_oracle.update({
        "name": "fake_mean_oracle_density_semi_right_size",
        "layers": [{
            "id": "fake_mean_oracle_density",
            "kind": "fake_mean_oracle_density",
            "gamma_m_x": 0.4,
            "gamma_m_w": 1.0 / 3.0,
            "c_m": 1.0,
            "shift": "positive",
        }],
    })
    presets["fake_mean_oracle_density_semi_right_size"] = fake_mean_oracle

    fake_mean_both_n25 = deepcopy(fake_mean_oracle)
    fake_mean_both_n25.update({
        "name": "fake_mean_both_n25_oracle_density_semi_right_size",
        "layers": [{
            "id": "fake_mean_both_n25",
            "kind": "fake_mean_oracle_density",
            "gamma_m_x": 0.4,
            "gamma_m_w": 0.4,
            "c_m": 1.0,
            "shift": "positive",
        }],
    })
    presets["fake_mean_both_n25_oracle_density_semi_right_size"] = fake_mean_both_n25

    fake_mean_c25 = deepcopy(fake_mean_oracle)
    fake_mean_c25.update({
        "name": "fake_mean_c25_oracle_density_semi_right_size",
        "layers": [{
            "id": "fake_mean_c25",
            "kind": "fake_mean_oracle_density",
            "gamma_m_x": 0.4,
            "gamma_m_w": 1.0 / 3.0,
            "c_m": 2.5,
            "shift": "positive",
        }],
    })
    presets["fake_mean_c25_oracle_density_semi_right_size"] = fake_mean_c25

    learned_cv_symmetric = deepcopy(learned_cv)
    learned_cv_symmetric["name"] = "learned_cv_symmetric"
    learned_cv_symmetric["errors"] = ["symmetric"]
    presets["learned_cv_symmetric"] = learned_cv_symmetric

    learned_cv_asymmetric_a10 = deepcopy(learned_cv)
    learned_cv_asymmetric_a10["name"] = "learned_cv_asymmetric_a10"
    learned_cv_asymmetric_a10["mixture_a"] = 10.0
    presets["learned_cv_asymmetric_a10"] = learned_cv_asymmetric_a10

    robustness = base("controlled_robustness")
    robustness["layers"] = [controlled(0.35, 2), controlled(0.35, shift="negative"),
                             controlled(0.35, shift="rademacher")]
    robustness["repetitions"] = 2000
    presets["controlled_robustness"] = robustness

    inference = base("main_inference")
    inference.update({"sample_sizes": [1000, 2000, 4000], "repetitions": 2000})
    inference["studies"] = [{"name": "inference_d020", "type": "inference",
                             "delta0": 0.20,
                             "gaps": [-0.05, 0.0, 0.05, 0.10, 0.20]}]
    inference["size_adjustment"] = {"enabled": True, "calibration_reps": 2000}
    inference["diagnostics"]["every"] = 20
    presets["main_inference"] = inference

    learned_cv_inference = base("learned_cv_asymmetric_inference")
    learned_cv_inference.update({
        "errors": ["right", "left"],
        "sample_sizes": [1000, 2000, 4000, 8000, 16000],
        "layers": [deepcopy(LEARNED_CV)],
        "repetitions": 1000,
    })
    learned_cv_inference["studies"] = [
        {"name": "inference_d020", "type": "inference",
         "delta0": 0.20,
         "gaps": [-0.05, 0.0, 0.05, 0.10, 0.20]}
    ]
    learned_cv_inference["size_adjustment"] = {
        "enabled": True,
        "calibration_reps": 1000,
    }
    learned_cv_inference["diagnostics"]["every"] = 20
    presets["learned_cv_asymmetric_inference"] = learned_cv_inference

    local = base("local_power")
    local.update({"sample_sizes": [500, 1000, 2000, 4000], "repetitions": 2000})
    local["studies"] = [{"name": "local_power_d020", "type": "local_power",
                          "delta0": 0.20, "local_cs": [0.0, 0.5, 1.0, 2.0, 4.0]}]
    local["size_adjustment"] = {"enabled": True, "calibration_reps": 2000}
    local["diagnostics"]["every"] = 20
    presets["local_power"] = local

    size = deepcopy(inference)
    size["name"] = "boundary_size_5000"
    size["repetitions"] = 5000
    size["size_adjustment"] = {"enabled": False, "calibration_reps": 0}
    for study in size["studies"]:
        study["gaps"] = [0.0]
    presets["boundary_size_5000"] = size

    audit = base("nuisance_audit")
    audit["layers"] = [deepcopy(LEARNED)]
    audit["sample_sizes"] = [500, 1000, 2000, 4000, 8000]
    audit["repetitions"] = 100
    audit["diagnostics"].update({"every": 1, "grid_side": 25, "density_points": 41,
                                  "b_nodes": 32, "b_check_nodes": 48})
    presets["nuisance_audit"] = audit

    # A moderately replicated analytic-layer check, not a replacement for the main runs.
    validation = base("analytic_validation")
    validation["layers"] = [deepcopy(ORACLE), controlled(0.35)]
    validation["sample_sizes"] = [500, 2000, 8000]
    validation["repetitions"] = 150
    validation["diagnostics"].update({"every": 25, "grid_side": 7, "density_points": 7})
    presets["analytic_validation"] = validation

    learned_check = base("learned_validation")
    learned_check.update({"sample_sizes": [500, 2000, 4000], "repetitions": 10,
                          "layers": [deepcopy(LEARNED)], "batch_size": 5})
    learned_check["diagnostics"].update({"every": 5, "grid_side": 11, "density_points": 11,
                                        "b_nodes": 24, "b_check_nodes": 32})
    learned_check["note"] = "Ten repetitions per cell: numerical and diagnostic checks, not formal size/power evidence."
    presets["learned_validation"] = learned_check
    return presets


def main() -> None:
    destination = Path(__file__).parent/"configs"
    destination.mkdir(exist_ok=True)
    for name, config in configs().items():
        (destination/f"{name}.json").write_text(json.dumps(config, indent=2, ensure_ascii=False),
                                               encoding="utf-8")
    print(f"Wrote {len(configs())} editable configurations to {destination}")


if __name__ == "__main__":
    main()
