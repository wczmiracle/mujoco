"""Fit M2 contact parameters (solref + nonlinear solimp) to the measured drop tests.

Peak force must match the measurements, subject to physical constraints that keep the
contact usable on the robot:
  - max penetration <= PEN_FRAC * pad thickness (bare ground: <= BARE_PEN_M), i.e. no 穿模
  - coefficient of restitution <= MAX_RESTITUTION (momentum cannot be invented to fake a peak)

Modes:
  group  one parameter set per (thickness, density)                 7 params / 4 points
  joint  one material per density, thickness enters via width and timeconst  8 params / 12-16 points

Usage:
  .venv/bin/python fit_m2.py --dt 0.0001 --out runs/m2_group_dt0.0001.json
  .venv/bin/python fit_m2.py --joint --dt 0.0001 --out runs/m2_joint_dt0.0001.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import differential_evolution

from drop import IMPACTOR_MASS, IMPACTOR_RADIUS, Contact, build_xml, load_targets

PEN_FRAC = 0.9
BARE_PEN_M = 0.002
MAX_RESTITUTION = 0.6
PENALTY = 10.0


class FastDrop:
  """Reuses one compiled model and rewrites the pair's contact parameters in place."""

  def __init__(self, dt: float, mass: float = IMPACTOR_MASS):
    self.model = mujoco.MjModel.from_xml_string(
        build_xml(Contact((0.02, 1), (0.9, 0.95, 0.001)), dt, mass))
    self.data = mujoco.MjData(self.model)
    self.dt = dt
    self.f6 = np.zeros(6)

  def set(self, c: Contact) -> None:
    self.model.pair_solref[0, :] = c.solref
    si = np.array([0.9, 0.95, 0.001, 0.5, 2.0])
    si[:len(c.solimp)] = c.solimp
    self.model.pair_solimp[0, :] = si
    self.model.pair_margin[0] = c.margin

  def run(self, speed: float, phase: float, t_max: float = 0.05):
    """Returns (peak normal force N, max penetration m, restitution)."""
    m, d = self.model, self.data
    mujoco.mj_resetData(m, d)
    d.qpos[2] = IMPACTOR_RADIUS + m.pair_margin[0] + (2.0 - phase) * self.dt * speed
    d.qvel[2] = -speed
    peak, pen, touched, rest = 0.0, 0.0, False, 0.0
    for _ in range(int(np.ceil(t_max / self.dt))):
      mujoco.mj_step(m, d)
      fn = 0.0
      for i in range(d.ncon):
        mujoco.mj_contactForce(m, d, i, self.f6)
        fn += self.f6[0]
        pen = max(pen, -d.contact[i].dist)
      if fn > 1e-6:
        touched = True
        peak = max(peak, fn)
      elif touched:
        rest = max(d.qvel[2], 0.0) / speed
        break
    else:
      rest = max(d.qvel[2], 0.0) / speed  # still in contact at t_max: e.g. stuck in the pad
    if not np.isfinite(d.qvel).all():
      return np.inf, np.inf, np.inf
    return peak, pen, rest

  def run_avg(self, speed: float, phases: int):
    r = np.array([self.run(speed, k / phases) for k in range(phases)])
    return float(r[:, 0].mean()), float(r[:, 1].max()), float(r[:, 2].max())


def score(sim: FastDrop, contacts, speeds, targets, pen_limits, phases):
  """Mean squared log error plus constraint penalties; also returns per-row diagnostics."""
  rows = []
  for c, v in zip(contacts, speeds):
    sim.set(c)
    rows.append(sim.run_avg(v, phases))
  rows = np.array(rows)
  f, pen, rest = rows[:, 0], rows[:, 1], rows[:, 2]
  if not np.all(np.isfinite(rows)) or np.any(f <= 0):
    return 1e3, rows
  err = np.mean(np.log(f / targets) ** 2)
  viol = (np.maximum(pen / pen_limits - 1.0, 0.0) ** 2).sum() + \
         (np.maximum(rest - MAX_RESTITUTION, 0.0) ** 2).sum()
  return float(err + PENALTY * viol), rows


def pen_limit(t_mm: float) -> float:
  return PEN_FRAC * t_mm * 1e-3 if t_mm > 0 else BARE_PEN_M


# group: x = [log10 timeconst, dampratio, d0, s1, width/thickness, midpoint, power]
BOUNDS = [(-4.5, -1.0), (0.05, 3.0), (0.0001, 0.95), (0.0, 1.0), (0.1, 1.5), (0.02, 0.98), (1.0, 12.0)]


def decode(x: np.ndarray, thickness_m: float) -> Contact:
  d0 = x[2]
  d1 = d0 + (0.9999 - d0) * x[3]
  width = max(x[4] * thickness_m, 1e-4)
  return Contact((10 ** x[0], x[1]), (d0, d1, width, x[5], x[6]))


# joint: x = [log10 tc12, tc_exp, dampratio, d0, s1, width/t, midpoint, power]
# timeconst(t) = tc12 * (t / 12 mm) ** tc_exp
JOINT_BOUNDS = [(-4.5, -1.0), (-2.0, 3.0), (0.05, 3.0), (0.0001, 0.95), (0.0, 1.0), (0.1, 1.5),
                (0.02, 0.98), (1.0, 12.0)]
JOINT_NAMES = ["log10_tc12", "tc_exp", "dampratio", "d0", "s1", "width_over_t", "midpoint", "power"]


def decode_joint(x: np.ndarray, t_mm: float) -> Contact:
  tc = 10 ** x[0] * (t_mm / 12.0) ** x[1]
  return decode(np.array([np.log10(tc), x[2], x[3], x[4], x[5], x[6], x[7]]), t_mm * 1e-3)


def optimize(loss, bounds, maxiter, seed):
  return differential_evolution(loss, bounds, maxiter=maxiter, popsize=12, seed=seed, tol=1e-5,
                                polish=True, updating="deferred", workers=1)


def report_rows(rows_df, diag):
  out = []
  for (_, r), (f, pen, rest) in zip(rows_df.iterrows(), diag):
    out.append({"t_mm": r.t_mm, "h": r.h, "v": r.v, "meas_kN": r.Fpk_N / 1e3, "sim_kN": f / 1e3,
                "pen_mm": pen * 1e3, "pen_limit_mm": pen_limit(r.t_mm) * 1e3, "restitution": rest})
  return out


def add_chr_targets(tg, heights, density):
  """Synthetic high-force targets from the CHR fit used by the simulator (user decision 2026-11-11:
  no new drop tests; approximate the high-force range with the CHR function)."""
  import pandas as pd
  p = json.loads(Path("/home/wang22/engineai/engineai_ros2_workspace/scripts/ThicknessCalculate/"
                      "fitted_parameters.json").read_text())
  from drop import BARE_PULSE_S
  rows = []
  for t in [6.0, 12.0, 18.0, 24.0]:
    for h in heights:
      v = float(np.sqrt(2 * 9.81 * h))
      fb_kN = IMPACTOR_MASS * v / BARE_PULSE_S / 1e3
      f_kN = p["C"] * t ** p["alpha"] * density ** p["beta"] * fb_kN ** p["gamma"]
      rows.append(dict(t_mm=t, density=density, h=h, v=v, Fpk_N=f_kN * 1e3, source="chr"))
  tg = tg.copy()
  tg["source"] = "data1"
  return pd.concat([tg, pd.DataFrame(rows)], ignore_index=True)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--dt", type=float, required=True)
  ap.add_argument("--out", type=Path, required=True)
  ap.add_argument("--maxiter", type=int, default=100)
  ap.add_argument("--joint", action="store_true")
  ap.add_argument("--only", type=str, default="", help="group mode: e.g. 12:0.4")
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--chr-heights", type=float, nargs="*", default=[],
                  help="add CHR-formula targets at these drop heights (m) for t=6/12/18/24 mm (high-force range)")
  ap.add_argument("--max-e", type=float, default=None, help="override MAX_RESTITUTION (default 0.6)")
  ap.add_argument("--density", type=float, default=None, help="joint mode: fit only this density")
  ap.add_argument("--holdout-h", type=float, default=None,
                  help="fit without this drop height, then report error on it (overfitting check)")
  args = ap.parse_args()

  global MAX_RESTITUTION
  if args.max_e is not None:
    MAX_RESTITUTION = args.max_e
  phases = 1 if args.dt <= 2e-5 else 4
  sim = FastDrop(args.dt)
  tg = load_targets()
  if args.density is not None:
    tg = tg[(tg.density == args.density) | (tg.t_mm == 0)]
  if args.chr_heights:
    tg = add_chr_targets(tg, args.chr_heights, args.density if args.density is not None else 0.4)
  results = []

  if args.joint:
    groups = tg[tg.t_mm > 0].groupby("density")
  else:
    groups = tg.groupby(["t_mm", "density"])

  for key, rows in groups:
    if not args.joint and args.only and args.only != f"{key[0]:g}:{key[1]:g}":
      continue
    all_rows = rows.reset_index(drop=True)
    if args.holdout_h is not None:
      rows = all_rows[~np.isclose(all_rows.h, args.holdout_h)].reset_index(drop=True)
    else:
      rows = all_rows
    speeds = rows.v.to_numpy()
    targets = rows.Fpk_N.to_numpy()
    limits = np.array([pen_limit(t) for t in rows.t_mm])
    t0 = time.time()
    if args.joint:
      contacts_of = lambda x: [decode_joint(x, t) for t in rows.t_mm]  # rows is read at call time
      bounds = JOINT_BOUNDS
    else:
      th = key[0] * 1e-3 if key[0] > 0 else 1e-3
      contacts_of = lambda x, th=th: [decode(x, th)] * len(rows)
      bounds = BOUNDS
    res = optimize(lambda x: score(sim, contacts_of(x), speeds, targets, limits, phases)[0],
                   bounds, args.maxiter, seed=args.seed)
    fit_rows = rows
    if args.holdout_h is not None:  # evaluate on all heights, held-out ones included
      rows = all_rows
      speeds = rows.v.to_numpy()
      targets = rows.Fpk_N.to_numpy()
      limits = np.array([pen_limit(t) for t in rows.t_mm])
    loss, diag = score(sim, contacts_of(res.x), speeds, targets, limits, phases)
    log_err = np.log(diag[:, 0] / targets)
    item = {
        "mode": "joint" if args.joint else "group",
        "density": float(key) if args.joint else float(key[1]),
        "t_mm": None if args.joint else float(key[0]),
        "dt": args.dt, "x": res.x.tolist(),
        "rms_log_err": float(np.sqrt(np.mean(log_err ** 2))),
        "max_abs_log_err": float(np.abs(log_err).max()),
        "loss_with_penalty": loss,
        "constraints_ok": bool(np.all(diag[:, 1] <= limits * 1.001) and
                               np.all(diag[:, 2] <= MAX_RESTITUTION + 1e-3)),
        "params": {f"{t:g}": {"solref": list(c.solref), "solimp": list(c.solimp)}
                   for t, c in zip(rows.t_mm, contacts_of(res.x))},
        "rows": report_rows(rows, diag),
        "fit_seconds": time.time() - t0,
        "holdout_h": args.holdout_h,
        "max_restitution": MAX_RESTITUTION,
    }
    if args.holdout_h is not None:
      held = np.isclose(rows.h.to_numpy(), args.holdout_h)
      item["holdout_rms_log_err"] = float(np.sqrt(np.mean(log_err[held] ** 2)))
      item["holdout_max_abs_log_err"] = float(np.abs(log_err[held]).max())
      item["train_rms_log_err"] = float(np.sqrt(np.mean(log_err[~held] ** 2)))
    if args.joint:
      item["x_names"] = JOINT_NAMES
    results.append(item)
    print(json.dumps(item), flush=True)
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
  main()
