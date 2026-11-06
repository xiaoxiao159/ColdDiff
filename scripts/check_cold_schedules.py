"""
Sanity check for the Cold Diffusion alpha-bar schedules.

Verifies the contract that all downstream consumers (two-stage loss, improved
sampler, visualization) rely on:
    - shape (T+1,)
    - alpha_bar[0] == 1.0, alpha_bar[T] == 0.0
    - strictly monotonically decreasing  (so delta_alpha = a[t-1]-a[t] > 0)

and prints the four curves + per-step delta_alpha side by side so the shape
differences (linear uniform / cosine symmetric / sigmoid steep-mid / quad
slow-start-fast-end) are visible at a glance.

Run from the repo root:
    python scripts/check_cold_schedules.py
"""
import os
import sys

# Make the repo root importable when run as a plain script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from runners.schedules import get_cold_alpha_schedule, get_cmpd_alpha_schedule

T = 10
SCHEMES = ["linear", "cosine", "cmpd", "sigmoid", "quad"]

print(f"Cold Diffusion alpha_bar schedules (T={T})")
print("=" * 78)

curves = {}
for sch in SCHEMES:
    ab = get_cold_alpha_schedule(sch, T)
    arr = ab.numpy()

    # ---- contract checks ----
    assert ab.shape == (T + 1,), f"{sch}: shape {ab.shape} != ({T+1},)"
    assert abs(arr[0] - 1.0) < 1e-6, f"{sch}: alpha_bar[0]={arr[0]} != 1.0"
    assert abs(arr[-1] - 0.0) < 1e-6, f"{sch}: alpha_bar[T]={arr[-1]} != 0.0"
    diffs = np.diff(arr)
    assert np.all(diffs < 0), f"{sch}: not strictly decreasing, diffs={diffs}"
    curves[sch] = arr

# alias check: cmpd must equal cosine exactly
assert np.allclose(curves["cmpd"], curves["cosine"]), "cmpd should be an alias of cosine"
print("Contract OK for all schedules: shape (T+1,), endpoints 1->0, strictly decreasing.")
print("Alias OK: 'cmpd' == 'cosine'.\n")

# ---- alpha_bar table ----
header = " t |" + "".join(f" {sch:>9s} |" for sch in SCHEMES)
print(header)
print("-" * len(header))
for t in range(T + 1):
    row = f"{t:2d} |" + "".join(f" {curves[sch][t]:9.4f} |" for sch in SCHEMES)
    print(row)

# ---- per-step delta_alpha (error-propagation coefficient in the improved sampler) ----
print("\nPer-step delta_alpha = alpha_bar[t-1] - alpha_bar[t]  (improved-sampler error coef)")
header = "step |" + "".join(f" {sch:>9s} |" for sch in SCHEMES)
print(header)
print("-" * len(header))
for t in range(T, 0, -1):
    row = f"{t:4d}|" + "".join(f" {curves[sch][t-1]-curves[sch][t]:9.4f} |" for sch in SCHEMES)
    print(row)

print("\nBackward-compat: get_cmpd_alpha_schedule matches get_cold_alpha_schedule('cmpd'):",
      np.allclose(get_cmpd_alpha_schedule(T).numpy(), curves["cmpd"]))
