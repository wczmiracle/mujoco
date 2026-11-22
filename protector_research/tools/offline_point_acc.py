"""Accelerometer readings at any body point from logged data, offline.
Logged link velocities (link_kinetic_energy_data.csv) are MuJoCo cvel: angular w and linear velocity of the body-fixed
point at c = subtree_com[root] (whole-robot COM). With qpos (pm_data.npy) kinematics: v_p = v_c + w x (p - c);
a_p = d v_p / dt (forward difference, one-step shift like the integrator), + [0,0,9.81] (specific force, as
mj_objectAcceleration/cacc). Points: pelvis 'imu' site, torso = LINK_TORSO_YAW COM, 'head_imu' site."""
import mujoco, numpy as np, pandas as pd
XML = '/home/wang22/engineai/engineai_ros2_workspace/src/simulation/mujoco/assets/resource/pm_v2_mesh.xml'
_m = mujoco.MjModel.from_xml_path(XML); _d = mujoco.MjData(_m)
_IMU = mujoco.mj_name2id(_m, mujoco.mjtObj.mjOBJ_SITE, 'imu')
_HEAD = mujoco.mj_name2id(_m, mujoco.mjtObj.mjOBJ_SITE, 'head_imu')
_TORSO = mujoco.mj_name2id(_m, mujoco.mjtObj.mjOBJ_BODY, 'LINK_TORSO_YAW')
_BASE = mujoco.mj_name2id(_m, mujoco.mjtObj.mjOBJ_BODY, 'LINK_BASE')
G = np.array([0, 0, 9.81])
def point_acc(pm, ke):
  """pm: (n, 32) time+qpos; ke: link_kinetic_energy DataFrame. Returns t, {name: (n,3) acc}, {name: (n,3) acc from pos}."""
  t, q = pm[:, 0], pm[:, 1:]
  kt = ke.timestamp.to_numpy(float)
  idx = np.searchsorted(t, kt - 1e-7); ok = (idx < len(t)) & (np.abs(t[np.clip(idx, 0, len(t) - 1)] - kt) < 1e-6)
  kt, idx = kt[ok], idx[ok]; ke = ke.loc[ok]
  bodies = {'pelvis': ('LINK_BASE', lambda d: d.site_xpos[_IMU]), 'torso': ('LINK_TORSO_YAW', lambda d: d.xipos[_TORSO]),
            'head': ('LINK_HEAD_YAW', lambda d: d.site_xpos[_HEAD])}
  V = {k: np.empty((len(kt), 3)) for k in bodies}; P = {k: np.empty((len(kt), 3)) for k in bodies}
  W = {k: ke[[f'{b}_angvel_{a}' for a in 'xyz']].to_numpy(float) for k, (b, _) in bodies.items()}
  L = {k: ke[[f'{b}_vel_{a}' for a in 'xyz']].to_numpy(float) for k, (b, _) in bodies.items()}
  for j, i in enumerate(idx):
    _d.qpos[:] = q[i]; mujoco.mj_kinematics(_m, _d); mujoco.mj_comPos(_m, _d)
    c = _d.subtree_com[_BASE]
    for k, (b, f) in bodies.items():
      p = f(_d); P[k][j] = p; V[k][j] = L[k][j] + np.cross(W[k][j], p - c)
  A, AP = {}, {}
  for k in bodies:
    a = np.zeros_like(V[k]); a[:-1] = np.diff(V[k], axis=0) / np.diff(kt)[:, None]; a[-1] = a[-2]; A[k] = a + G
    ap = np.zeros_like(P[k]); dt = np.median(np.diff(kt)); ap[1:-1] = (P[k][2:] - 2 * P[k][1:-1] + P[k][:-2]) / dt**2; AP[k] = ap + G
  return kt, A, AP
