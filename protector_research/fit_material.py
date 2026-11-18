"""Fit one mass-normalized contact material (same parameters for every link) to drop-test targets.

Materials:
  srm       data1 measurements (rho 0.4) + SRM CHR targets above 1 m (Fig. 2c relation)
  silicone  fitted_parameters_silicone.json (F_after = C t^a F_before, gamma = 1) at 0.1-3 m;
            12 mm @ 1 m (~16 kN) is consistent with the measured Fig. 2a comparison
  bare      unprotected shell: F = m v / 0.26554 ms (CHR's own unprotected definition), 0.1-1 m;
            --bare-meas: the measured unprotected drops (runs/bare_measured_peaks.csv)
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


def targets(mat, chr_heights=(1.5, 2.0, 2.5, 3.0), meas_weight=1, bare_meas=False):
  hs = [0.1, 0.4, 0.8, 1.0]
  if mat == "srm":
    tg = load_targets()
    tg = tg[tg.density == 0.4]
    tg = pd.concat([tg] * meas_weight).assign(source="data1")  # repeat measured rows = weight in the mean squared log error
    return add_chr_targets(tg, list(chr_heights), 0.4).reset_index(drop=True) if chr_heights else tg.reset_index(drop=True)
  rows = []
  if mat == "silicone":
    p = json.loads((WS / "fitted_parameters_silicone.json").read_text())
    for t in (6.0, 12.0, 24.0):
      for h in hs + [1.5, 2.0, 2.5, 3.0]:
        v = float(np.sqrt(2 * 9.81 * h)); fb = IMPACTOR_MASS * v / BARE_PULSE_S / 1e3
        rows.append(dict(t_mm=t, h=h, v=v, Fpk_N=p["C"] * t ** p["alpha"] * fb ** p["gamma"] * 1e3, source="silicone_fit"))
  if mat == "bare" and bare_meas:
    m = pd.read_csv(Path(__file__).parent / "runs/bare_measured_peaks.csv")
    for h, f in zip(m.h, m.Fpk_N):
      # meas_weight > 1: repeat the robot-range rows (h <= 0.5 m, v <= 3.1 m/s) that many times
      for _ in range(meas_weight if h <= 0.5 else 1):
        rows.append(dict(t_mm=BARE_T, h=h, v=float(np.sqrt(2 * 9.81 * h)), Fpk_N=float(f), source="measured_bare"))
  elif mat == "bare":
    for h in hs:
      v = float(np.sqrt(2 * 9.81 * h))
      rows.append(dict(t_mm=BARE_T, h=h, v=v, Fpk_N=IMPACTOR_MASS * v / BARE_PULSE_S, source="m*v/dt"))
  return pd.DataFrame(rows)


class BottomDrop(FastDrop):
  """FastDrop with a second contact pair: the bare shell, active once the pad is compressed beyond t
  (includemargin = -t; MuJoCo's reference then acts on dist + t). Mirrors the simulator's bottom_out."""

  def __init__(self, dt, mass):
    import mujoco
    from drop import IMPACTOR_RADIUS
    self.model = mujoco.MjModel.from_xml_string(f"""<mujoco><option timestep="{dt:.9g}" tolerance="1e-10" gravity="0 0 -9.81" cone="elliptic" impratio="10"/>
<worldbody><geom name="floor" type="plane" size="0 0 1" contype="0" conaffinity="0"/>
<body name="ball" pos="0 0 1"><freejoint/><geom name="ball" type="sphere" size="{IMPACTOR_RADIUS:.9g}" mass="{mass:.9g}" contype="0" conaffinity="0"/></body></worldbody>
<contact><pair geom1="floor" geom2="ball" condim="3" friction="1 1 0.005 0.0001 0.0001"/>
<pair geom1="floor" geom2="ball" condim="3" friction="1 1 0.005 0.0001 0.0001"/></contact></mujoco>""")
    self.data = mujoco.MjData(self.model)
    self.dt = dt
    self.f6 = np.zeros(6)

  def set(self, c):
    pad, shell, t_mm = c
    FastDrop.set(self, pad)
    self.model.pair_solref[1, :] = shell.solref
    self.model.pair_solimp[1, :] = shell.solimp
    self.model.pair_margin[1] = -t_mm * 1e-3


REST_CASES = ((1.0, 50.0, 0.5), (1.8, 200.0, 1.0), (5.5, 300.0, 1.0), (20.0, 500.0, 1.0))  # (kg, push N, v0 m/s)


def rest_check(x, t_mm, dt, cases=REST_CASES, dur=0.2, settle=0.05):
  """A body pressed onto the material (constant push) must settle: after `settle` s the contact force must
  equal m g + push without chatter. Returns [(m, mean/expected, std/expected)]. A peak-only fit can give a
  material that bounces a light pressed body forever with 100 kN spikes (robot hands, 2026-11-19)."""
  import mujoco
  from drop import IMPACTOR_RADIUS
  out = []
  for m, push, v0 in cases:
    model = mujoco.MjModel.from_xml_string(f"""<mujoco><option timestep="{dt:.9g}" tolerance="1e-10" gravity="0 0 -9.81" cone="elliptic" impratio="10"/>
<worldbody><geom name="floor" type="plane" size="0 0 1" contype="0" conaffinity="0"/>
<body name="ball" pos="0 0 1"><freejoint/><geom name="ball" type="sphere" size="{IMPACTOR_RADIUS:.9g}" mass="{m:.9g}" contype="0" conaffinity="0"/></body></worldbody>
<contact><pair geom1="floor" geom2="ball" condim="3" friction="1 1 0.005 0.0001 0.0001"/></contact></mujoco>""")
    data = mujoco.MjData(model)
    c = material(x, t_mm, m)
    model.pair_solref[0] = c.solref
    model.pair_solimp[0] = c.solimp
    data.qpos[2] = IMPACTOR_RADIUS + 1e-4
    data.qvel[2] = -v0
    data.xfrc_applied[1, 2] = -push
    f6, fs = np.zeros(6), []
    for k in range(int(dur / dt)):
      mujoco.mj_step(model, data)
      if not np.all(np.isfinite(data.qpos)):
        out.append((m, np.inf, np.inf)); break
      if k * dt >= settle:
        fn = 0.0
        for i in range(data.ncon):
          mujoco.mj_contactForce(model, data, i, f6); fn += f6[0]
        fs.append(fn)
    else:
      exp = m * 9.81 + push
      out.append((m, float(np.mean(fs) / exp), float(np.std(fs) / exp)))
  return out


def rest_penalty(x, t_mm, dt):
  v = sum(max(abs(mu - 1) - 0.05, 0.0) ** 2 + max(sd - 0.05, 0.0) ** 2 if np.isfinite(mu) else 1e2
          for _, mu, sd in rest_check(x, t_mm, dt))
  return fit_m2.PENALTY * v


def bare_x():
  import yaml
  mats = yaml.safe_load(open("/home/wang22/engineai/engineai_ros2_workspace/src/simulation/mujoco/assets/config/protector_materials.yaml"))["materials"]
  return np.array([mats["bare"][k] for k in NAMES]), mats["bare"].get("t_mm_nominal", 1.0)


def pen_lim(mat, t):
  return BARE_PEN if mat == "bare" else 0.9 * t * 1e-3


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--material", choices=["srm", "silicone", "bare"], required=True)
  ap.add_argument("--dt", type=float, default=0.0005)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--maxiter", type=int, default=150)
  ap.add_argument("--out", type=Path, required=True)
  ap.add_argument("--chr-heights", type=float, nargs="*", default=[1.5, 2.0, 2.5, 3.0],
                  help="srm: heights (m) of CHR-extrapolated targets above the measured 1 m; none = measured only")
  ap.add_argument("--meas-weight", type=int, default=1, help="srm: weight of measured rows relative to CHR rows; bare (--bare-meas): weight of the h <= 0.5 m rows")
  ap.add_argument("--no-heavy", action="store_true", help="drop the 10/20/40 kg constraint rows")
  ap.add_argument("--bottom-out", action="store_true",
                  help="pad + bare-shell contact once compressed beyond t (simulator bottom_out); no pen limit")
  ap.add_argument("--max-e", type=float, default=None, help="override the restitution cap (fit_m2.MAX_RESTITUTION, 0.6)")
  ap.add_argument("--rest", action="store_true",
                  help="add the resting-contact constraint (pressed body must settle, no chatter); see rest_check")
  ap.add_argument("--bare-meas", action="store_true",
                  help="bare: fit the measured unprotected drops (runs/bare_measured_peaks.csv, 0.1-1 m) instead of m v / dt")
  a = ap.parse_args()
  if a.max_e is not None:
    fit_m2.MAX_RESTITUTION = a.max_e
  tg = targets(a.material, a.chr_heights, a.meas_weight, a.bare_meas)
  speeds, tgt = tg.v.to_numpy(), tg.Fpk_N.to_numpy()
  if a.bottom_out:
    sim = BottomDrop(a.dt, IMPACTOR_MASS)
    xb, tb = bare_x()
    limits = np.full(len(tg), np.inf)  # bottoming out is allowed: the shell contact takes over
    contacts = lambda x: [(material(x, t, IMPACTOR_MASS), material(xb, tb, IMPACTOR_MASS), t) for t in tg.t_mm]
  else:
    sim = FastDrop(a.dt, IMPACTOR_MASS)
    limits = np.array([pen_lim(a.material, t) for t in tg.t_mm])
    contacts = lambda x: [material(x, t, IMPACTOR_MASS) for t in tg.t_mm]
  thick = sorted(set(tg.t_mm))
  heavy = [] if a.no_heavy else [(t, m, v) for t in thick for m in (10.0, 20.0, 40.0) for v in (2.0, 4.3)]
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

  rest_ts = sorted(set(tg.t_mm)) if a.rest else []

  def loss(x):
    l = score(sim, contacts(x), speeds, tgt, limits, 4)[0]
    if l < 1e2:
      l += heavy_penalty(x) if heavy else 0.0
      l += sum(rest_penalty(x, t, a.dt) for t in rest_ts)
    return l

  t0 = time.time()
  res = optimize(loss, BOUNDS, a.maxiter, a.seed)
  l, diag = score(sim, contacts(res.x), speeds, tgt, limits, 4)
  err = np.log(diag[:, 0] / tgt)
  out = {"material": a.material, "x": res.x.tolist(), "x_names": NAMES, "dt": a.dt, "seed": a.seed,
         "rms": float(np.sqrt(np.mean(err ** 2))), "max": float(np.abs(err).max()),
         "heavy_penalty": heavy_penalty(res.x), "rest": a.rest, "max_e": fit_m2.MAX_RESTITUTION,
         "rest_check": {str(t): rest_check(res.x, t, a.dt) for t in sorted(set(tg.t_mm))}, "chr_heights": a.chr_heights, "meas_weight": a.meas_weight, "no_heavy": a.no_heavy, "bottom_out": a.bottom_out, "bare_meas": a.bare_meas, "bare_t_mm": BARE_T if a.material == "bare" else None,
         "constraints_ok": bool(np.all(diag[:, 1] <= limits * 1.001) and np.all(diag[:, 2] <= fit_m2.MAX_RESTITUTION + 1e-3)),
         "rows": [dict(t_mm=r.t_mm, h=r.h, src=r.source, target_kN=r.Fpk_N / 1e3, sim_kN=f / 1e3, pen_mm=p * 1e3, e=e)
                  for (_, r), (f, p, e) in zip(tg.iterrows(), diag)],
         "fit_seconds": time.time() - t0}
  a.out.write_text(json.dumps(out, indent=1))
  print(json.dumps({k: out[k] for k in ["material", "seed", "rms", "max", "heavy_penalty", "constraints_ok"]}))


if __name__ == "__main__":
  main()
