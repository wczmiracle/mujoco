"""Old-data vibration with the IMU on different bodies. Per run: collision windows as calculate_fall_risk
(non-foot contacts > 400 N), 20 Hz filter + 200 Hz hold (filter_torso_acceleration), then
  ratio_* = peak |a| in the 20 ms post-collision windows / 100 (as fig2 vibration load ratio)
  risk_*  = calculate_acceleration_risk (the vibration risk score)
for: log (logged base_link acc, as the published figure), pelvis (imu site, recomputed), torso (LINK_TORSO_YAW COM), head (head_imu).
Usage: torso_batch.py <tar> <job> <out.csv>"""
import io, sys, tarfile
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, '/home/wang22/engineai/engineai_ros2_workspace/scripts')
from offline_point_acc import point_acc
from calculate_fall_risk import filter_torso_acceleration, calculate_acceleration_risk
COLS = [f'base_link_lin_acc_{a}' for a in 'xyz']
def collisions(c):
  b = (c.body1_name.astype(str) + '|' + c.body2_name.astype(str)).str.lower()
  nf = ~b.str.contains('foot|ankle|toe|heel')
  return np.unique(c.timestamp.to_numpy(float)[nf.to_numpy() & (c.force_magnitude.to_numpy(float) > 400)])
def ratio(f, col):
  t = f.timestamp.to_numpy(float); mag = np.linalg.norm(f[COLS].to_numpy(float), axis=1); w = 0.02
  g = [[col[0]]]
  for x in col[1:]:
    (g[-1].append(x) if x - g[-1][-1] < w else g.append([x]))
  pk = 0.0
  for grp in g:
    a, b = min(grp), (max(grp) + w if len(grp) > 1 else min(grp) + w)
    m = (t >= a) & (t <= b)
    if m.any(): pk = max(pk, float(mag[m].max()))
  return pk / 100.0
def main(tarp, job, out):
  need = {'pm_data.npy', 'sensor_vibration_data.csv', 'link_kinetic_energy_data.csv', 'contact_data.csv'}; pend = {}; rows = []
  with tarfile.open(tarp, 'r|gz') as tf:
    for mem in tf:
      fn = Path(mem.name).name
      if fn not in need: continue
      par = str(Path(mem.name).parent); p = pend.setdefault(par, {}); p[fn] = tf.extractfile(mem).read()
      if not need.issubset(p): continue
      del pend[par]
      rec = dict(job=job, run=Path(par).parts[-2])
      try:
        c = pd.read_csv(io.BytesIO(p['contact_data.csv']), usecols=['timestamp', 'body1_name', 'body2_name', 'force_magnitude'])
        s = pd.read_csv(io.BytesIO(p['sensor_vibration_data.csv']), usecols=['timestamp'] + COLS)
        col = collisions(c)
        if col.size == 0:
          rows.append(rec); continue
        t, A, _ = point_acc(np.load(io.BytesIO(p['pm_data.npy'])), pd.read_csv(io.BytesIO(p['link_kinetic_energy_data.csv'])))
        sigs = {'log': s}
        for k in ('pelvis', 'torso', 'head'):
          sigs[k] = pd.DataFrame({'timestamp': t, **{COLS[i]: A[k][:, i] for i in range(3)}})
        for k, df in sigs.items():
          f = filter_torso_acceleration(df)
          rec[f'ratio_{k}'] = ratio(f, col); rec[f'risk_{k}'] = float(calculate_acceleration_risk(f, col))
      except Exception as e:  # keep going, record the failure
        rec['error'] = repr(e)[:200]
      rows.append(rec)
  pd.DataFrame(rows).to_csv(out, index=False); print(out, len(rows), flush=True)
if __name__ == '__main__':
  main(*sys.argv[1:])
