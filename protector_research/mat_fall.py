"""Passive falls with the per-contact mass-normalized contact material (Python replica of the simulator's
SimManager::Step1 + ApplyContactMaterials, MuJoCo 3.2.3), to reproduce and debug launches seen in P2d.

Robot: all non-foot links padded (P2d: protection.enabled, use_protected_links false, 6 mm srm);
PD hold at the homing keyframe, shove at the torso, after tilt > 0.5 rad switch to damping mode
(kp = 0, kd = passive_damping of rl_basic_param_XZL.yaml), like falling_switch: damping.

Usage: .venv/bin/python mat_fall.py --clamp new --push 300 --t_end 3
"""
import argparse

import mujoco
import numpy as np
import yaml

RES = "/home/wang22/engineai/engineai_ros2_workspace/src/simulation/mujoco/assets/resource/pm_v2_mesh.xml"
# collision_model_type: mesh -> pm_v2_mesh.xml + ApplySolref6mmMeshCollisionPreset (pm_v2.yaml solref_6mm)
SOLREF_6MM = (0.0005, 1.0)
SOLREF_6MM_BODIES = ["LINK_ELBOW_PITCH_L", "LINK_ELBOW_PITCH_R", "LINK_KNEE_PITCH_L", "LINK_KNEE_PITCH_R",
                     "LINK_HIP_YAW_L", "LINK_HIP_YAW_R", "LINK_SHOULDER_YAW_L", "LINK_SHOULDER_YAW_R"]
MATS = "/home/wang22/engineai/engineai_ros2_workspace/src/simulation/mujoco/assets/config/protector_materials.yaml"
KD_DAMP = [4, 4, 4, 4, 3, 3] * 2 + [2] + [3] * 10 + [1]


def parse_fix(s):
  """'selftc:0.004,mnfloor:5' -> {'selftc': 0.004, 'mnfloor': 5.0}; bare words map to True."""
  out = {}
  for tok in filter(None, (s or "none").split(",")):
    k, _, v = tok.partition(":")
    out[k] = float(v) if v else True
  out.pop("none", None)
  return out


def apply_model_fix(m, fix):
  """Model-level variants: selftc:<s> keeps robot geoms' solref time constant >= s (self/foot contacts)."""
  fx = parse_fix(fix)
  if "selftc" in fx:
    for g in range(m.ngeom):
      if m.geom_bodyid[g] != 0 and m.geom_solref[g, 0] > 0:
        m.geom_solref[g, 0] = max(m.geom_solref[g, 0], fx["selftc"])
  if "no6mm" in fx:
    for g in range(m.ngeom): m.geom_solref[g] = (0.02, 1.0)


def impedance(si, pos, margin=0.0):
  if si[0] == si[1] or si[2] <= mujoco.mjMINVAL:
    return 0.5 * (si[0] + si[1])
  x = abs((pos - margin) / si[2])
  if x >= 1: return si[1]
  if x <= 0: return si[0]
  mid, p = si[3], si[4]
  if p == 1: y = x
  elif x <= mid: y = x ** p / mid ** (p - 1)
  else: y = 1 - (1 - x) ** p / (1 - mid) ** (p - 1)
  return si[0] + y * (si[1] - si[0])


class MatSim:
  def __init__(self, pad="srm", t_pad=6.0, clamp="new", dt=0.0005):
    self.m = m = mujoco.MjModel.from_xml_path(RES)
    m.opt.timestep = dt; m.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC; m.opt.impratio = 10
    for g in range(m.ngeom):
      if m.geom_contype[g] and m.body(m.geom_bodyid[g]).name in SOLREF_6MM_BODIES:
        m.geom_solref[g] = SOLREF_6MM
    self.d = mujoco.MjData(m)
    mats = yaml.safe_load(open(MATS))["materials"]
    self.clamp = clamp
    self.geom = {}
    for g in range(m.ngeom):
      b = m.geom_bodyid[g]
      if b == 0 or not (m.geom_contype[g] or m.geom_conaffinity[g]): continue
      name = m.body(b).name
      if "FOOT" in name or "ANKLE" in name: continue
      p, t = (mats[pad], t_pad) if pad else (mats["bare"], mats["bare"]["t_mm_nominal"])
      K = 10 ** p["log10_K12"] * (t / 12) ** p["kexp"]; B = 10 ** p["log10_B12"] * (t / 12) ** p["bexp"]
      si = np.array([p["d0"], p["d0"] + (0.9999 - p["d0"]) * p["s1"], max(p["width_over_t"] * t * 1e-3, 1e-4),
                     p["midpoint"], p["power"]])
      self.geom[g] = (K, B, si)
    self.jacp = np.zeros((3, m.nv)); self.n_clamped = 0

  def apply_materials(self):
    m, d = self.m, self.d
    fx = parse_fix(getattr(self, "fix", "none"))
    if "nomat" in fx: return
    mn_floor = fx.get("mnfloor", 0.0); bmax = fx.get("clampb", 1.5)
    h = m.opt.timestep
    for i in range(d.ncon):
      c = d.contact[i]
      b1, b2 = m.geom_bodyid[c.geom[0]], m.geom_bodyid[c.geom[1]]
      if b1 == 0 and b2 != 0: g, b = c.geom[1], b2
      elif b2 == 0 and b1 != 0: g, b = c.geom[0], b1
      else: continue
      if g not in self.geom: continue
      K, B, si = self.geom[g]
      mujoco.mj_jac(m, d, self.jacp, None, c.pos, b)
      jn = c.frame[:3] @ self.jacp
      x = np.zeros(m.nv); mujoco.mj_solveM(m, d, x.reshape(1, -1), jn.reshape(1, -1))
      A = jn @ x; Ahat = m.body_invweight0[b, 0]
      dimp = impedance(si, c.dist, c.includemargin)
      inv_mn = dimp * A + (1 - dimp) * Ahat
      if not inv_mn > mujoco.mjMINVAL: continue
      mn = max(1 / inv_mn, mn_floor)
      k, bb = K / mn, B / mn
      if self.clamp == "old":
        km, bm = (1.5 / h) ** 2, 4.0 / h
        if k > km or bb > bm: self.n_clamped += 1
        k, bb = min(k, km), min(bb, bm)
      elif self.clamp == "new":
        rd = dimp / si[1]; be = rd * bb * h; cl = False
        if be > bmax: bb *= bmax / be; be = bmax; cl = True
        ke = rd * rd * k * h * h; kmax = 0.8 * (4 - 2 * be)
        if ke > kmax: k *= kmax / ke; cl = True
        self.n_clamped += cl
      c.solref[:] = [-k, -bb]; c.solimp[:] = si

  def step(self):
    m, d = self.m, self.d
    for f in (mujoco.mj_checkPos, mujoco.mj_checkVel, mujoco.mj_kinematics, mujoco.mj_comPos, mujoco.mj_camlight,
              mujoco.mj_flex, mujoco.mj_tendon, mujoco.mj_crb, mujoco.mj_factorM, mujoco.mj_collision):
      f(m, d)
    self.apply_materials()
    for f in (mujoco.mj_makeConstraint, mujoco.mj_transmission, mujoco.mj_projectConstraint, mujoco.mj_sensorPos,
              mujoco.mj_energyPos, mujoco.mj_fwdVelocity, mujoco.mj_sensorVel, mujoco.mj_energyVel):
      f(m, d)
    mujoco.mj_step2(m, d)


def fall(sim, ang_deg, push=300.0, push_t=0.4, t_end=3.0, kp=150.0, kd=5.0, trace=False):
  m, d = sim.m, sim.d
  mujoco.mj_resetDataKeyframe(m, d, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "floating_base_homing"))
  q0 = d.qpos[7:].copy(); kdd = np.array(KD_DAMP, float)
  torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "LINK_TORSO_YAW")
  a = np.deg2rad(ang_deg); damping = False
  vz_max, fmax, log = -1e9, 0.0, []
  f6 = np.zeros(6)
  for _ in range(int(t_end / m.opt.timestep)):
    qw = d.qpos[3:7]; tilt = np.arccos(np.clip(1 - 2 * (qw[1] ** 2 + qw[2] ** 2), -1, 1))
    damping = damping or tilt > 0.5
    d.qfrc_applied[6:] = -kdd * d.qvel[6:] if damping else kp * (q0 - d.qpos[7:]) - kd * d.qvel[6:]
    d.xfrc_applied[torso, :3] = [push * np.cos(a), push * np.sin(a), 0] if d.time < push_t else 0
    sim.step()
    if not np.all(np.isfinite(d.qvel)): return dict(ang=ang_deg, diverged=True, vz_max=np.inf, fmax=np.inf, log=log)
    vz_max = max(vz_max, d.qvel[2])
    fw = 0.0
    for i in range(d.ncon):
      mujoco.mj_contactForce(m, d, i, f6); fw = max(fw, f6[0])
    fmax = max(fmax, fw)
    if trace: log.append((d.time, d.qpos[2], d.qvel[2], fw, d.ncon))
  return dict(ang=ang_deg, diverged=False, vz_max=vz_max, fmax=fmax, log=log)


if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("--clamp", choices=["none", "old", "new"], default="new")
  ap.add_argument("--pad", default="srm"); ap.add_argument("--t_pad", type=float, default=6.0)
  ap.add_argument("--push", type=float, default=300.0); ap.add_argument("--t_end", type=float, default=3.0)
  ap.add_argument("--angles", default="0,45,90,135,180,225,270,315")
  a = ap.parse_args()
  sim = MatSim(a.pad or None, a.t_pad, a.clamp)
  for ang in map(float, a.angles.split(",")):
    r = fall(sim, ang, a.push, t_end=a.t_end)
    print(f"clamp={a.clamp} ang={ang:5.0f}  diverged={r['diverged']}  vz_max={r['vz_max']:6.2f} m/s  "
          f"Fmax={r['fmax'] / 1e3:8.1f} kN  clamped_steps={sim.n_clamped}", flush=True)
