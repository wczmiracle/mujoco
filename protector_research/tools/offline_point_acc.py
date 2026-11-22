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



def qpos_from_logs(ke, joint_state, contact):
  """Rebuild pm_data-like (time + qpos) on the link_kinetic_energy timestamps for logs without pm_data.npy.
  Joints: joint_state_data.csv (joint_i_position = qpos[7+i], every step), linear interpolation.
  Base orientation: integrate the logged LINK_BASE world angular velocity, re-anchored at each base_link_q* sample
  in contact_data.csv (sparse). Base position: interpolated (does not affect accelerations, only r = p - c matters).
  Validated against pm_data on old P3_S1 runs (pelvis corr 1.0000, torso >= 0.997, peaks within 0.5%)."""
  t = ke.timestamp.to_numpy(float)
  bc = ['base_link_x', 'base_link_y', 'base_link_z', 'base_link_qw', 'base_link_qx', 'base_link_qy', 'base_link_qz']
  b = contact[['timestamp'] + bc].drop_duplicates('timestamp').sort_values('timestamp')
  bt = b.timestamp.to_numpy(float); bq = b[bc[3:]].to_numpy(float)
  pos = np.column_stack([np.interp(t, bt, b[k]) for k in bc[:3]])
  w = ke[[f'LINK_BASE_angvel_{a}' for a in 'xyz']].to_numpy(float)
  quat = np.empty((len(t), 4)); k = 0
  q = bq[0].copy() if len(bq) else np.array([1.0, 0, 0, 0])
  for i in range(len(t)):
    while k < len(bt) and bt[k] <= t[i] + 1e-9:
      q = bq[k].copy(); k += 1
    quat[i] = q
    if i + 1 < len(t):
      mujoco.mju_quatIntegrate(q, _rot_local(q, w[i]), t[i + 1] - t[i])
      q /= np.linalg.norm(q)
  js = joint_state.drop_duplicates('timestamp').sort_values('timestamp')
  nj = _m.nq - 7
  jj = np.column_stack([np.interp(t, js.timestamp, js[f'joint_{i}_position']) for i in range(nj)])
  return np.column_stack([t, pos, quat, jj])


def _rot_local(q, w_world):
  """World angular velocity -> body frame (mju_quatIntegrate takes local angular velocity)."""
  qi = np.zeros(4); mujoco.mju_negQuat(qi, q); out = np.zeros(3); mujoco.mju_rotVecQuat(out, w_world, qi); return out
