"""M3 drop test with the experimental Stable Neo-Hookean material (mjsFlex.elastic3d = 1, MuJoCo >= 3.15)."""
import numpy as np
import mujoco
import drop_m3
from drop_m3 import Pad, build_xml, IMPACTOR_RADIUS


def drop_snh(pad, h, dt=1e-4, t_max=0.05, snh=True):
  spec = mujoco.MjSpec.from_string(build_xml(pad, dt))
  if snh:
    spec.flexes[0].elastic3d = 1
  m = spec.compile(); d = mujoco.MjData(m)
  ball = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ball")
  qa = m.jnt_qposadr[m.body_jntadr[ball]]; da = m.jnt_dofadr[m.body_jntadr[ball]]
  g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "ball")
  v = np.sqrt(2 * 9.81 * h); z0 = pad.t_mm * 1e-3 + IMPACTOR_RADIUS
  d.qpos[qa + 2] = z0 + 2 * dt * v; d.qvel[da + 2] = -v
  f6 = np.zeros(6); pk = 0; touched = False; comp = 0
  for _ in range(int(t_max / dt)):
    mujoco.mj_step(m, d); fn = 0
    for i in range(d.ncon):
      if g in (d.contact[i].geom[0], d.contact[i].geom[1]):
        mujoco.mj_contactForce(m, d, i, f6); fn += f6[0]
    comp = max(comp, z0 - d.qpos[qa + 2])
    if fn > 1e-6: touched = True; pk = max(pk, fn)
    elif touched: break
    if not np.isfinite(d.qvel).all(): return np.inf, np.inf, np.inf
  return pk / 1e3, comp * 1e3, max(d.qvel[da + 2], 0) / v


if __name__ == "__main__":
  meas = {0.1: 1.13, 0.4: 4.33, 0.8: 10.07, 1.0: 12.77}
  res = []
  for E in [3e5, 1e6, 3e6, 1e7]:
    for nu in [0.0, 0.3, 0.45]:
      for damp in [0.002, 0.01]:
        r = {h: drop_snh(Pad(12, E, nu, damp), h) for h in meas}
        if not all(np.isfinite(r[h][0]) and r[h][0] > 0 for h in meas): continue
        err = np.sqrt(np.mean([np.log(r[h][0] / meas[h]) ** 2 for h in meas]))
        res.append((err, E, nu, damp, [round(r[h][0], 2) for h in meas], [round(r[h][2], 2) for h in meas], [round(r[h][1], 1) for h in meas]))
  for x in sorted(res)[:6]:
    print(f"rms log err {x[0]:.2f} E={x[1]:.0e} nu={x[2]} damp={x[3]} kN={x[4]} e={x[5]} comp={x[6]}")
  print("measured", list(meas.values()))
