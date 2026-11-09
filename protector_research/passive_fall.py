"""Passive falls without the policy: joints held at the homing keyframe by PD, robot shoved in 8 directions.
Same per-link contact params as the simulator (PROTECTOR_M2_YAML semantics), dt 0.5 ms, elliptic cone.
Separates contact dynamics from policy reaction when comparing slip between M0s and M2s.
"""
import sys
import mujoco
import numpy as np
import yaml

RES = "/home/wang22/engineai/engineai_ros2_workspace/src/simulation/mujoco/assets/resource/pm_v2.xml"


def load(links_yaml, impratio=10.0, dt=0.0005):
  m = mujoco.MjModel.from_xml_path(RES)
  m.opt.timestep = dt; m.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC; m.opt.impratio = impratio
  floor_mu = max(m.geom_friction[g, 0] for g in range(m.ngeom) if m.geom_bodyid[g] == 0)
  if links_yaml:
    for link, p in yaml.safe_load(open(links_yaml))["links"].items():
      b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, link)
      if b < 0: continue
      for g in range(m.ngeom):
        if m.geom_bodyid[g] == b and (m.geom_contype[g] or m.geom_conaffinity[g]):
          m.geom_solref[g] = p["solref"]; m.geom_solimp[g] = p["solimp"]; m.geom_priority[g] = 1
          m.geom_friction[g, 0] = max(m.geom_friction[g, 0], floor_mu)
  return m


def run(m, ang_deg, push=300.0, push_t=0.4, t_end=3.0, kp=150.0, kd=5.0):
  d = mujoco.MjData(m)
  k = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "floating_base_homing")
  mujoco.mj_resetDataKeyframe(m, d, k if k >= 0 else 0)
  q0 = d.qpos[7:].copy()
  torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "LINK_TORSO_YAW")
  a = np.deg2rad(ang_deg); f6 = np.zeros(6)
  slip, slip_window, t_touch, peak = 0.0, 0.0, None, 0.0
  jac = np.zeros((3, m.nv))
  for _ in range(int(t_end / m.opt.timestep)):
    d.qfrc_applied[6:] = kp * (q0 - d.qpos[7:]) - kd * d.qvel[6:]
    d.xfrc_applied[torso, :3] = [push * np.cos(a), push * np.sin(a), 0] if d.time < push_t else 0
    mujoco.mj_step(m, d)
    worst = 0.0
    for i in range(d.ncon):
      c = d.contact[i]
      g = c.geom[1] if m.geom_bodyid[c.geom[0]] == 0 else c.geom[0] if m.geom_bodyid[c.geom[1]] == 0 else -1
      if g < 0: continue
      b = m.geom_bodyid[g]; name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
      if "FOOT" in name or "ANKLE" in name: continue
      mujoco.mj_contactForce(m, d, i, f6)
      if f6[0] < 50: continue
      t_touch = t_touch if t_touch is not None else d.time
      peak = max(peak, f6[0])
      mujoco.mj_jac(m, d, jac, None, np.array([c.pos[0], c.pos[1], 0.0]), b)
      v = jac @ d.qvel
      vt = np.linalg.norm([v @ c.frame[3:6], v @ c.frame[6:9]])
      worst = max(worst, vt)
    slip += worst * m.opt.timestep
    if t_touch is not None and d.time < t_touch + 0.25:
      slip_window += worst * m.opt.timestep
  return slip * 1e3, slip_window * 1e3, peak / 1e3


if __name__ == "__main__":
  for name, y in [("M0s", "runs/m0s_links_dt0.0005.yaml"), ("M2s", "runs/m2s_links_dt0.0005.yaml")]:
    m = load(y)
    res = np.array([run(m, ang) for ang in range(0, 360, 45)])
    print(f"{name}: slip total med {np.median(res[:,0]):.1f} mm, first 250ms med {np.median(res[:,1]):.1f} mm, peak force med {np.median(res[:,2]):.1f} kN"
          f"   per-dir total: {' '.join(f'{x:.0f}' for x in res[:,0])}")
