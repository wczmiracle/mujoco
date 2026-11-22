"""Old-data vibration at the pelvis 'imu' site, keeping the protector (CHR) scaling of the old simulator.

Old logs (before fe4978bc) store base_link_lin_acc / base_link_ang_acc = d->cacc of LINK_BASE, i.e. the spatial
acceleration at c = subtree_com[root], computed after ApplyProtectionToContactForces (mj_rnePostConstraint with the
scaled efc_force). Moving it to the IMU point p like mj_objectAcceleration:
  a_p = lin_c + alpha x (p - c) + w x v_p,   v_p = v_c + w x (p - c)
keeps the protector scaling and fixes the measurement point. Validated on unprotected old P3_S1 runs against the
velocity-derived pelvis acceleration (offline_point_acc.point_acc): corr 1.0000, peaks within ~1 %.
Per run: ratio/risk for 'scaled' (this transform) and 'vel' (velocity-derived, unscaled motion), same collision
windows / 20 Hz filter as vib_by_imu_body.py.
Usage: vib_pelvis_from_log.py <tar> <job> <out.csv>"""
import io, sys, tarfile
from pathlib import Path
import mujoco
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, '/home/wang22/engineai/engineai_ros2_workspace/scripts')
import offline_point_acc as o
from vib_by_imu_body import collisions, ratio, COLS
from calculate_fall_risk import filter_torso_acceleration, calculate_acceleration_risk

ANG = [f'base_link_ang_acc_{a}' for a in 'xyz']


def scaled_pelvis(pm, ke, s, t):
  """Logged (protector-scaled) cacc of LINK_BASE moved to the imu site, on times t."""
  s = s.drop_duplicates('timestamp').set_index('timestamp').reindex(t, method='nearest', tolerance=1e-6)
  lin, alp = s[COLS].to_numpy(float), s[ANG].to_numpy(float)
  kt = ke.timestamp.to_numpy(float); ii = np.searchsorted(kt, t - 1e-7)
  w = ke[[f'LINK_BASE_angvel_{a}' for a in 'xyz']].to_numpy(float)[ii]
  vc = ke[[f'LINK_BASE_vel_{a}' for a in 'xyz']].to_numpy(float)[ii]
  jj = np.searchsorted(pm[:, 0], t - 1e-7); r = np.empty((len(t), 3))
  for n, i in enumerate(jj):
    o._d.qpos[:] = pm[i, 1:]; mujoco.mj_kinematics(o._m, o._d); mujoco.mj_comPos(o._m, o._d)
    r[n] = o._d.site_xpos[o._IMU] - o._d.subtree_com[o._BASE]
  a = lin + np.cross(alp, r) + np.cross(w, vc + np.cross(w, r))
  ok = np.isfinite(a).all(1)
  return t[ok], a[ok]


def process(rec, p):
  c = pd.read_csv(io.BytesIO(p['contact_data.csv']))
  col = collisions(c)
  if col.size == 0:
    return rec
  s = pd.read_csv(io.BytesIO(p['sensor_vibration_data.csv']), usecols=['timestamp'] + COLS + ANG)
  ke = pd.read_csv(io.BytesIO(p['link_kinetic_energy_data.csv']))
  if 'pm_data.npy' in p:
    pm = np.load(io.BytesIO(p['pm_data.npy'])); rec['qpos_src'] = 'pm_data'
  else:
    pm = o.qpos_from_logs(ke, pd.read_csv(io.BytesIO(p['joint_state_data.csv'])), c); rec['qpos_src'] = 'rebuilt'
  t, A, _ = o.point_acc(pm, ke)
  ts, a_s = scaled_pelvis(pm, ke, s, t)
  sigs = {'scaled': pd.DataFrame({'timestamp': ts, **{COLS[i]: a_s[:, i] for i in range(3)}}),
          'vel': pd.DataFrame({'timestamp': t, **{COLS[i]: A['pelvis'][:, i] for i in range(3)}})}
  for k, df in sigs.items():
    f = filter_torso_acceleration(df)
    rec[f'ratio_{k}'] = ratio(f, col); rec[f'risk_{k}'] = float(calculate_acceleration_risk(f, col))
  return rec


def main(tarp, job, out):
  want = {'pm_data.npy', 'sensor_vibration_data.csv', 'link_kinetic_energy_data.csv', 'contact_data.csv', 'joint_state_data.csv'}
  base = {'sensor_vibration_data.csv', 'link_kinetic_energy_data.csv', 'contact_data.csv'}
  rows = []; cur, p = None, {}

  def flush():
    if cur is None or not base.issubset(p) or ('pm_data.npy' not in p and 'joint_state_data.csv' not in p):
      return
    rec = dict(job=job, run=Path(cur).parts[-2])
    try:
      rows.append(process(rec, p))
    except Exception as e:
      rec['error'] = repr(e)[:200]; rows.append(rec)

  with tarfile.open(tarp, 'r|gz') as tf:
    for mem in tf:
      fn = Path(mem.name).name
      if fn not in want: continue
      par = str(Path(mem.name).parent)
      if par != cur:
        flush(); cur, p = par, {}
      p[fn] = tf.extractfile(mem).read()
    flush()
  pd.DataFrame(rows).to_csv(out, index=False); print(out, len(rows), flush=True)


if __name__ == '__main__':
  main(*sys.argv[1:])
