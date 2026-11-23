"""Reference drop targets for the unprotected (bare) robot shell on a wood floor, to replace the steel-anvil calibration.

Why: the measured bare drops are a steel impactor on a steel anvil (runs/bare_measured_peaks.csv); the robot falls on a
floor, and with an aluminum shell on wood the softest element is the wood (crushing strength across the grain
~5-20 MPa << aluminum yield ~275 MPa), so the floor sets the bare contact stiffness.

Model (same impactor as the protector drops: 5.487 kg, hemispherical face R = 53 mm): wood crushes under the ball
(cellular solid, no lateral flow, constraint factor ~1): contact radius a^2 ~ 2 R d, F = sigma_c * pi a^2 = 2 pi R sigma_c d,
i.e. a linear spring k = 2 pi R sigma_c during loading. Peak F = v sqrt(k m), max indentation d = v sqrt(m / k),
loading time pi/2 sqrt(m / k). The elastic Hertz stage (E_perp ~1 GPa) ends at a few tens of micrometres and is ignored.
Assumes the flooring (>= 10 mm) is not crushed through; d at 1 m is 5.7 mm for sigma_c = 10 MPa.

Abaqus ground truth can replace this file: same columns (mass optional, default 5.487 kg; h, Fpk_N; pen_mm optional).
Usage: bare_reference.py [--sigma-c 10e6] [--radius 0.053] [--out runs/bare_reference_wood_sc10.csv]
"""
import argparse

import numpy as np
import pandas as pd

HEIGHTS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
MASS = 5.487


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--sigma-c", type=float, default=10e6, help="wood crushing strength across the grain (Pa)")
  ap.add_argument("--radius", type=float, default=0.053)
  ap.add_argument("--masses", type=float, nargs="*", default=[MASS],
                  help="impactor masses (kg); several masses pin F ~ sqrt(m) (fit_material.py fits each mass)")
  ap.add_argument("--out", default=None)
  a = ap.parse_args()
  k = 2 * np.pi * a.radius * a.sigma_c
  rows = []
  for M in a.masses:
   for h in HEIGHTS:
    v = np.sqrt(2 * 9.81 * h)
    rows.append(dict(mass=M, h=h, v=v, Fpk_N=v * np.sqrt(k * M), pen_mm=v * np.sqrt(M / k) * 1e3,
                     T_load_ms=np.pi / 2 * np.sqrt(M / k) * 1e3, k_N_per_m=k,
                     source=f"wood crush sigma_c {a.sigma_c / 1e6:g} MPa, R {a.radius * 1e3:g} mm"))
  df = pd.DataFrame(rows)
  tag = "" if a.masses == [MASS] else "_multimass"
  out = a.out or f"runs/bare_reference_wood_sc{a.sigma_c / 1e6:g}{tag}.csv"
  df.to_csv(out, index=False)
  print(out); print(df.round(3).to_string(index=False))


if __name__ == "__main__":
  main()
