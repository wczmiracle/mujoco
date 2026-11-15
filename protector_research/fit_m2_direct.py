"""Mass-normalized M2: one material for every link.

Contact uses MuJoCo's direct (negative) solref, scaled per contact by its effective mass m:
  solref = (-K(t) / m, -B(t) / m)   so the reference acceleration times m gives a force that depends
                                     only on penetration and its rate, like a material, not on m.
  K(t) = K12 * (t / 12 mm)^kexp,  B(t) = B12 * (t / 12 mm)^bexp
  solimp = (d0, d1, width = w * t, midpoint, power), shared across thickness.
Fitted at the drop-test mass (5.487 kg) to data1 (rho 0.4) plus CHR targets above 1 m (user decision).

Usage: .venv/bin/python fit_m2_direct.py --seed 0 --out runs/m2d_dt0.0005_s0.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

import fit_m2
from drop import IMPACTOR_MASS, Contact, load_targets
from fit_m2 import FastDrop, add_chr_targets, optimize, pen_limit, score

NAMES = ["log10_K12", "kexp", "log10_B12", "bexp", "d0", "s1", "width_over_t", "midpoint", "power"]
BOUNDS = [(4.0, 8.5), (-3.0, 2.0), (0.5, 5.5), (-3.0, 2.0), (0.0001, 0.95), (0.0, 1.0), (0.1, 1.5),
          (0.02, 0.98), (1.0, 12.0)]


def material(x, t_mm, mass):
  K = 10 ** x[0] * (t_mm / 12.0) ** x[1]
  B = 10 ** x[2] * (t_mm / 12.0) ** x[3]
  d0 = x[4]
  d1 = d0 + (0.9999 - d0) * x[5]
  return Contact((-K / mass, -B / mass), (d0, d1, max(x[6] * t_mm * 1e-3, 1e-4), x[7], x[8]))


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--dt", type=float, default=0.0005)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--maxiter", type=int, default=150)
  ap.add_argument("--out", type=Path, required=True)
  ap.add_argument("--heavy", action="store_true",
                  help="add constraint-only rows (no force target): 10/20/40 kg up to 4.3 m/s must keep "
                       "penetration <= 0.9 t and restitution <= max-e (pad must stiffen before bottoming out)")
  a = ap.parse_args()
  tg = load_targets()
  tg = tg[tg.density == 0.4]
  tg = add_chr_targets(tg, [1.5, 2.0, 2.5, 3.0], 0.4).reset_index(drop=True)
  sim = FastDrop(a.dt, IMPACTOR_MASS)
  speeds, targets = tg.v.to_numpy(), tg.Fpk_N.to_numpy()
  limits = np.array([pen_limit(t) for t in tg.t_mm])
  contacts = lambda x: [material(x, t, IMPACTOR_MASS) for t in tg.t_mm]
  t0 = time.time()
  heavy = [(t, m, v) for t in (6.0, 12.0, 18.0, 24.0) for m in (10.0, 20.0, 40.0) for v in (2.0, 4.3)] if a.heavy else []
  heavy_sims = {m: FastDrop(a.dt, m) for m in {h[1] for h in heavy}}

  def heavy_penalty(x):
    viol = 0.0
    for t, m, v in heavy:
      sm = heavy_sims[m]
      sm.set(material(x, t, m))
      f, pen, e = sm.run_avg(v, 2)
      if not np.isfinite(f):
        return 1e3
      viol += max(pen / pen_limit(t) - 1.0, 0.0) ** 2 + max(e - fit_m2.MAX_RESTITUTION, 0.0) ** 2
    return fit_m2.PENALTY * viol

  def loss(x):
    l = score(sim, contacts(x), speeds, targets, limits, 4)[0]
    return l + (heavy_penalty(x) if heavy and l < 1e2 else 0.0)

  res = optimize(loss, BOUNDS, a.maxiter, a.seed)
  loss, diag = score(sim, contacts(res.x), speeds, targets, limits, 4)
  err = np.log(diag[:, 0] / targets)
  d1 = (tg.source == "data1").to_numpy()
  out = {"x": res.x.tolist(), "x_names": NAMES, "dt": a.dt, "seed": a.seed, "loss": loss, "heavy": a.heavy,
         "heavy_penalty": heavy_penalty(res.x) if heavy else 0.0,
         "data1_rms": float(np.sqrt(np.mean(err[d1] ** 2))), "data1_max": float(np.abs(err[d1]).max()),
         "chr_rms": float(np.sqrt(np.mean(err[~d1] ** 2))), "chr_max": float(np.abs(err[~d1]).max()),
         "constraints_ok": bool(np.all(diag[:, 1] <= limits * 1.001) and np.all(diag[:, 2] <= fit_m2.MAX_RESTITUTION + 1e-3)),
         "rows": [dict(t_mm=r.t_mm, h=r.h, src=r.source, target_kN=r.Fpk_N / 1e3, sim_kN=f / 1e3, pen_mm=p * 1e3, e=e)
                  for (_, r), (f, p, e) in zip(tg.iterrows(), diag)],
         "fit_seconds": time.time() - t0}
  a.out.write_text(json.dumps(out, indent=1))
  print(json.dumps({k: out[k] for k in ["seed", "loss", "data1_rms", "data1_max", "chr_rms", "chr_max", "constraints_ok"]}))


if __name__ == "__main__":
  main()
