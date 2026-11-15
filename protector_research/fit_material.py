"""Fit one mass-normalized contact material (same parameters for every link) to drop-test targets.

Materials:
  srm       data1 measurements (rho 0.4) + SRM CHR targets above 1 m (Fig. 2c relation)
  silicone  fitted_parameters_silicone.json (F_after = C t^a F_before, gamma = 1) at 0.1-3 m;
            12 mm @ 1 m (~16 kN) is consistent with the measured Fig. 2a comparison
  bare      unprotected shell: F = m v / 0.26554 ms (CHR's own unprotected definition), 0.1-1 m
All at the drop-test mass 5.487 kg. Heavy-mass rows (10/20/40 kg) are constraint-only:
penetration <= 0.9 t (bare: 2 mm) and restitution <= 0.6, so the pad stiffens before bottoming out.

Contact model (see fit_m2_direct.material): solref = (-K(t)/m, -B(t)/m), solimp shape shared.

Usage: .venv/bin/python fit_material.py --material silicone --seed 0 --out runs/mat_silicone_s0.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import fit_m2
from drop import BARE_PULSE_S, IMPACTOR_MASS, load_targets
from fit_m2 import FastDrop, add_chr_targets, optimize, score
from fit_m2_direct import BOUNDS, NAMES, material

WS = Path("/home/wang22/engineai/engineai_ros2_workspace/scripts/ThicknessCalculate")
BARE_T = 1.0  # nominal "thickness" (mm) that scales width for the bare shell material
BARE_PEN = 0.002


def targets(mat):
  hs = [0.1, 0.4, 0.8, 1.0]
  if mat == "srm":
    tg = load_targets()
    tg = tg[tg.density == 0.4]
    return add_chr_targets(tg, [1.5, 2.0, 2.5, 3.0], 0.4).reset_index(drop=True)
  rows = []
  if mat == "silicone":
    p = json.loads((WS / "fitted_parameters_silicone.json").read_text())
    for t in (6.0, 12.0, 24.0):
      for h in hs + [1.5, 2.0, 2.5, 3.0]:
        v = float(np.sqrt(2 * 9.81 * h)); fb = IMPACTOR_MASS * v / BARE_PULSE_S / 1e3
        rows.append(dict(t_mm=t, h=h, v=v, Fpk_N=p["C"] * t ** p["alpha"] * fb ** p["gamma"] * 1e3, source="silicone_fit"))
  if mat == "bare":
    for h in hs:
      v = float(np.sqrt(2 * 9.81 * h))
      rows.append(dict(t_mm=BARE_T, h=h, v=v, Fpk_N=IMPACTOR_MASS * v / BARE_PULSE_S, source="m*v/dt"))
  return pd.DataFrame(rows)


def pen_lim(mat, t):
  return BARE_PEN if mat == "bare" else 0.9 * t * 1e-3


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--material", choices=["srm", "silicone", "bare"], required=True)
  ap.add_argument("--dt", type=float, default=0.0005)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--maxiter", type=int, default=150)
  ap.add_argument("--out", type=Path, required=True)
  a = ap.parse_args()
  tg = targets(a.material)
  sim = FastDrop(a.dt, IMPACTOR_MASS)
  speeds, tgt = tg.v.to_numpy(), tg.Fpk_N.to_numpy()
  limits = np.array([pen_lim(a.material, t) for t in tg.t_mm])
  contacts = lambda x: [material(x, t, IMPACTOR_MASS) for t in tg.t_mm]
  thick = sorted(set(tg.t_mm))
  heavy = [(t, m, v) for t in thick for m in (10.0, 20.0, 40.0) for v in (2.0, 4.3)]
  heavy_sims = {m: FastDrop(a.dt, m) for m in (10.0, 20.0, 40.0)}

  def heavy_penalty(x):
    viol = 0.0
    for t, m, v in heavy:
      s = heavy_sims[m]; s.set(material(x, t, m)); f, pen, e = s.run_avg(v, 2)
      if not np.isfinite(f):
        return 1e3
      # bare: a rigid shell cannot stop a fast heavy body within 2 mm in one 0.5 ms step (v*dt alone is
      # 2.15 mm at 4.3 m/s), so the heavy-row limit is relaxed to 1.5 steps of travel
      lim = max(BARE_PEN, 1.5 * v * a.dt) if a.material == "bare" else pen_lim(a.material, t)
      viol += max(pen / lim - 1.0, 0.0) ** 2 + max(e - fit_m2.MAX_RESTITUTION, 0.0) ** 2
    return fit_m2.PENALTY * viol

  def loss(x):
    l = score(sim, contacts(x), speeds, tgt, limits, 4)[0]
    return l + (heavy_penalty(x) if l < 1e2 else 0.0)

  t0 = time.time()
  res = optimize(loss, BOUNDS, a.maxiter, a.seed)
  l, diag = score(sim, contacts(res.x), speeds, tgt, limits, 4)
  err = np.log(diag[:, 0] / tgt)
  out = {"material": a.material, "x": res.x.tolist(), "x_names": NAMES, "dt": a.dt, "seed": a.seed,
         "rms": float(np.sqrt(np.mean(err ** 2))), "max": float(np.abs(err).max()),
         "heavy_penalty": heavy_penalty(res.x), "bare_t_mm": BARE_T if a.material == "bare" else None,
         "constraints_ok": bool(np.all(diag[:, 1] <= limits * 1.001) and np.all(diag[:, 2] <= fit_m2.MAX_RESTITUTION + 1e-3)),
         "rows": [dict(t_mm=r.t_mm, h=r.h, src=r.source, target_kN=r.Fpk_N / 1e3, sim_kN=f / 1e3, pen_mm=p * 1e3, e=e)
                  for (_, r), (f, p, e) in zip(tg.iterrows(), diag)],
         "fit_seconds": time.time() - t0}
  a.out.write_text(json.dumps(out, indent=1))
  print(json.dumps({k: out[k] for k in ["material", "seed", "rms", "max", "heavy_penalty", "constraints_ok"]}))


if __name__ == "__main__":
  main()
