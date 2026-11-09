"""M2 behaviour beyond the drop-test range: sweep drop height (energy) at the robot timestep and
check that peak force, penetration and restitution vary smoothly (no single-step spikes).
Measured range: 0.1-1.0 m with the 5.487 kg impactor (5.4-54 J)."""
import json
import mujoco
import numpy as np
import pandas as pd
from fit_m2 import decode_joint
from drop import IMPACTOR_RADIUS, Contact, build_xml, load_targets

dt = 0.0005
x = np.array([r for r in json.load(open("runs/m2_joint_dt0.0005.json")) if r["density"] == 0.4][0]["x"])
tg = load_targets(); tg = tg[tg.density == 0.4]


def drop(c, h, mass=5.487):
  m = mujoco.MjModel.from_xml_string(build_xml(c, dt, mass)); d = mujoco.MjData(m)
  v = np.sqrt(2 * 9.81 * h)
  d.qpos[2] = IMPACTOR_RADIUS + 1.5 * dt * v; d.qvel[2] = -v
  f6 = np.zeros(6); F = []; pen = 0
  for _ in range(int(0.05 / dt)):
    mujoco.mj_step(m, d)
    fn = 0
    for i in range(d.ncon):
      mujoco.mj_contactForce(m, d, i, f6); fn += f6[0]; pen = max(pen, -d.contact[i].dist)
    F.append(fn)
    if len(F) > 2 and fn == 0 and max(F) > 0: break
  F = np.array(F); pk = F.max()
  width = int((F > 0.5 * pk).sum())  # steps above half peak
  return pk / 1e3, pen * 1e3, max(d.qvel[2], 0) / v, width, 0.5 * mass * v * v


rows = []
for t in [6, 12, 18, 24]:
  c = decode_joint(x, t)
  for h in [0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0]:
    pk, pen, e, w, E = drop(c, h)
    meas = tg[(tg.t_mm == t) & np.isclose(tg.h, h)].Fpk_N
    rows.append(dict(t=t, h=h, E_J=round(E, 1), sim_kN=round(pk, 2), meas_kN=round(meas.iloc[0] / 1e3, 2) if len(meas) else None,
                     pen_mm=round(pen, 1), pen_lim=0.9 * t, e=round(e, 2), steps_above_half=w))
df = pd.DataFrame(rows)
print(df.to_string(index=False))
df.to_csv("runs/extrapolate_m2_dt0.0005.csv", index=False)
