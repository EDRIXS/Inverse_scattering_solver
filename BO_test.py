"""
Minimal one-off evaluator (NEW method)
-------------------------------------
Usage:
    python eval_single_params.py --config-py NiCl2_config.py

This loads your material-specific config, uses its RIXS_runner instance (rixs_funV)
and experimental reference map (rixs_ref), and evaluates the positive L1 distance
between L1-normalized simulation and reference at a single parameter set.
"""

import argparse
import importlib.util
import numpy as np
import sys
from types import ModuleType

def load_config_module(path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location("cfg_module", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load config from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def l1_distance(sim: np.ndarray, ref: np.ndarray) -> float:
    sim = sim.astype(float)
    ref = ref.astype(float)
    # global L1 normalization (matches original objective)
    sim_norm = sim / np.sum(np.abs(sim))
    ref_norm = ref / np.sum(np.abs(ref))
    return float(np.sum(np.abs(sim_norm - ref_norm)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-py", required=True, help="Path to material config (e.g., NiCl2_config.py)")
    args = ap.parse_args()

    cfg = load_config_module(args.config_py)

    # --- Required objects from config ---
    # rixs_funV: an instance of RIXS_runner configured for this material
    # rixs_ref: experimental reference map (2D numpy array)
    try:
        rixs_runner = cfg.rixs_funV
        rixs_ref = cfg.rixs_ref
    except AttributeError as e:
        raise RuntimeError("Config must define 'rixs_funV' (RIXS_runner instance) and 'rixs_ref' (2D array).") from e

    # --- Choose a single parameter set ---
    # Fill these with the values you want to evaluate.

    params = {
        "F2_dd": 7.61573947,
        "F2_dp": 2.58015028,
        "F4_dd": 5.78267154,
        "G1_dp": 4.36028284 ,
        "G3_dp":  2.58147238 ,
        "tenDq": 0.91287146,
        "xoffset": -5.24678316,
        "soc_v_i":  0.083,
        "soc_v_n": 0.102,
        "soc_c": 11.507 ,
        # The runner expects these fixed knobs too:
        "Gam_c": 0.4125,
        "sigma": 0.066,
    }

    # If your config exposes default fixed params, use them to fill in any missing keys
    for name in ("fixed_params_greedy", "fixed_params"):
        if hasattr(cfg, name):
            fixed = getattr(cfg, name)
            if isinstance(fixed, dict):
                for k, v in fixed.items():
                    params.setdefault(k, v)

    # Run the new path (RIXS_runner)
    sim_map = rixs_runner(params)  # shape matches cfg target (usually [N_eloss, N_ominc])

    # Compute positive L1 distance versus the reference
    dist = l1_distance(sim_map, rixs_ref)

    # Print results
    print("Evaluated parameters:")
    for k in ["F2_dd","F2_dp","F4_dd","G1_dp","G3_dp","tenDq","xoffset","soc_v_i","soc_v_n","soc_c","Gam_c","sigma"]:
        if k in params:
            print(f"  {k:>8s} = {params[k]}")
    print(f"\nPositive L1 distance (L1-normalized maps): {dist:.12g}")

if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)
    sys.exit(main())