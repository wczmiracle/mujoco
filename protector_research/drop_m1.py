"""Emulate the current M1 pipeline (2 ms, default contact, CHR scaling of efc_force with the
simulator's min-scale clamp and two-rate low-pass) on the drop test, to compare its accuracy
with M2 on the same measured data.

Mirrors SimManager::ApplyProtectionToContactForces(): after mj_fwdConstraint the contact's
efc_force rows are multiplied by the filtered scale, qfrc_constraint and qacc are recomputed,
then Euler integration.
"""
import json
import mujoco
import numpy as np
from drop import IMPACTOR_MASS, IMPACTOR_RADIUS, Contact, build_xml, load_targets

P = json.load(open("/home/wang22/engineai/engineai_ros2_workspace/scripts/ThicknessCalculate/fitted_parameters.json"))
MIN_HIGH, MIN_MID = 0.01, 0.03  # pm_v2.yaml chr_min_scale_high / mid


def min_scale(fn):
  n1, n2, n3 = 500.0, 2000.0, 10000.0
  if fn <= n1:
    return 1.0
  if fn <= n2:
    return 1.0 + (fn - n1) / (n2 - n1) * (MIN_MID - 1.0)
  if fn <= n3:
    return MIN_MID + (fn - n2) / (n3 - n2) * (MIN_HIGH - MIN_MID)
  return MIN_HIGH


def drop_m1(t_mm, v, dt=0.002, density=0.4, phase=0.0, scale_on=True):
  c = Contact((0.02, 1.0), (0.9, 0.95, 0.001, 0.5, 2.0))  # MuJoCo defaults (robot links and floor)
  m = mujoco.MjModel.from_xml_string(build_xml(c, dt)); d = mujoco.MjData(m)
  d.qpos[2] = IMPACTOR_RADIUS + (2.0 - phase) * dt * v; d.qvel[2] = -v
  f6 = np.zeros(6); prev = 1.0; peak_applied = peak_unprot = 0.0; touched = False
  minv = np.zeros(m.nv)
  for _ in range(int(0.08 / dt)):
    mujoco.mj_step1(m, d)
    mujoco.mj_fwdActuation(m, d); mujoco.mj_fwdAcceleration(m, d); mujoco.mj_fwdConstraint(m, d)
    fn = 0.0
    if d.ncon:
      mujoco.mj_contactForce(m, d, 0, f6); fn = f6[0]
    if fn > 1e-6 and scale_on:
      fb = fn / 1e3
      fa = P["C"] * t_mm ** P["alpha"] * density ** P["beta"] * fb ** P["gamma"] if t_mm >= 6 else fb
      s = min(1.0, max(0.0, fa / fb)); s = max(s, min_scale(fn))
      a = 0.1 if (s < prev or fn > 2000.0) else 0.8
      s = min(1.0, max(0.0, a * prev + (1 - a) * s)); prev = s
      con = d.contact[0]; adr, n = con.efc_address, (2 * (con.dim - 1) if m.opt.cone == 0 else con.dim)
      d.efc_force[adr:adr + n] *= s
      mujoco.mj_mulJacTVec(m, d, d.qfrc_constraint, d.efc_force)
      mujoco.mj_solveM(m, d, minv.reshape(1, -1), d.qfrc_constraint.reshape(1, -1))
      d.qacc[:] = d.qacc_smooth + minv
    elif fn <= 1e-6:
      prev = 1.0
    if fn > 1e-6:
      touched = True; peak_unprot = max(peak_unprot, fn); peak_applied = max(peak_applied, fn * (prev if scale_on else 1))
    elif touched:
      break
    mujoco.mj_Euler(m, d)
  return peak_applied / 1e3, peak_unprot / 1e3


if __name__ == "__main__":
  tg = load_targets(); tg = tg[tg.density == 0.4]
  errs = []
  print(f"{'t':>3} {'h':>4} {'meas kN':>8} {'M1 applied':>10} {'ratio':>6} | {'F_unprot (sim)':>14} {'F_before (CHR def)':>18}")
  for _, r in tg.iterrows():
    res = [drop_m1(r.t_mm, r.v, phase=k / 4) for k in range(4)]
    fa = np.mean([x[0] for x in res]); fu = np.mean([x[1] for x in res])
    fb = IMPACTOR_MASS * r.v / 0.00026554 / 1e3
    errs.append(np.log(fa / (r.Fpk_N / 1e3)))
    print(f"{r.t_mm:3.0f} {r.h:4.1f} {r.Fpk_N/1e3:8.2f} {fa:10.2f} {fa/(r.Fpk_N/1e3):6.2f} | {fu:14.2f} {fb:18.1f}")
  e = np.abs(errs)
  print(f"M1 on drop test: RMS {np.exp(np.sqrt(np.mean(np.square(errs))))-1:.0%}, max {np.exp(e.max())-1:.0%}")
