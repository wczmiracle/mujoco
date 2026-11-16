"""Replay a logged closed-loop run from a state shortly before a launch, with the Python material replica.

pm_data.npy holds [t, qpos] every physics step (0.5 ms); with the Euler integrator qvel_n is exactly
differentiatePos(qpos_{n-1}, qpos_n) / h, so the full state is recovered. During damping mode the simulator
applies ctrl = -kd * qvel each step (kp = 0), reproduced here.

Usage: .venv/bin/python replay_launch.py <run_dir> --t0 3.76 --dur 0.04 --clamp new [--mn_floor 0] [--fix ...]
"""
import argparse

import mujoco
import numpy as np
import pandas as pd

from mat_fall import KD_DAMP, MatSim, apply_model_fix


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("run_dir")
  ap.add_argument("--t0", type=float, required=True)
  ap.add_argument("--dur", type=float, default=0.04)
  ap.add_argument("--clamp", default="new")
  ap.add_argument("--fix", default="none")
  ap.add_argument("--quiet", action="store_true")
  ap.add_argument("--json", action="store_true", help="print per-body peak ground force and max vz as JSON")
  a = ap.parse_args()
  arr = np.load(a.run_dir + "/pm_data.npy"); t = arr[:, 0]; Q = arr[:, 1:]
  sim = MatSim("srm", 6.0, a.clamp); sim.fix = a.fix
  m, d = sim.m, sim.d; h = m.opt.timestep
  apply_model_fix(m, a.fix)
  n0 = int(np.argmin(abs(t - a.t0)))
  d.qpos[:] = Q[n0]; v = np.zeros(m.nv); mujoco.mj_differentiatePos(m, v, h, Q[n0 - 1], Q[n0]); d.qvel[:] = v
  d.time = t[n0]; kd = np.array(KD_DAMP, float); f6 = np.zeros(6)
  lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
  vz_log = lambda n: (Q[n, 2] - Q[n - 1, 2]) / h
  pert = pd.read_csv(a.run_dir + "/perturbation_data.csv")
  # push log can skip rows; apply the last logged force over the whole push window, like the simulator
  pt = pert.timestamp.to_numpy(); pf = pert[["world_force_x", "world_force_y", "world_force_z"]].to_numpy()
  p_end = pert.start_time.iloc[0] + pert.duration.iloc[0]
  def push_at(tt):
    i = np.searchsorted(pt, tt + 1e-9) - 1
    return pf[max(i, 0)] if pt[0] - 1e-9 <= tt <= p_end + 1e-9 else (0.0, 0.0, 0.0)
  torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, pert.body_name.iloc[0].strip('"'))
  worst = -1e9; peaks = {}
  for k in range(int(a.dur / h)):
    d.ctrl[:] = np.clip(-kd * d.qvel[6:], lo, hi)
    d.xfrc_applied[torso, :3] = push_at(t[n0 + k + 1])
    sim.step()
    n = n0 + k + 1
    fw = 0.0
    for i in range(d.ncon):
      mujoco.mj_contactForce(m, d, i, f6); fw = max(fw, f6[0])
      c = d.contact[i]; b1, b2 = m.geom_bodyid[c.geom[0]], m.geom_bodyid[c.geom[1]]
      if (b1 == 0) != (b2 == 0):
        name = m.body(b1 or b2).name
        peaks[name] = max(peaks.get(name, 0.0), f6[0])
    worst = max(worst, d.qvel[2])
    if not a.quiet and k % 2 == 1:
      print(f"t={d.time:.4f} vz replay {d.qvel[2]:7.2f}  logged {vz_log(n):7.2f}   z {d.qpos[2]:.3f}/{Q[n, 2]:.3f}"
            f"  Fmax {fw / 1e3:8.1f} kN  ncon {d.ncon}")
  if a.json:
    import json
    print("JSON " + json.dumps({"fix": a.fix, "vz_max": worst, "peaks": peaks}))
  err = np.abs(d.qpos - Q[n0 + int(a.dur / h)]).max()
  print(f"clamp={a.clamp} fix={a.fix}: final |qpos err| {err:.1e}; max vz replay {worst:.2f} m/s, logged {max(vz_log(n0 + k + 1) for k in range(int(a.dur / h))):.2f}")


if __name__ == "__main__":
  main()
