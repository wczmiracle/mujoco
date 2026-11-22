"""Per-run vibration load ratio (same rule as _refresh_fig2_filtered_vibration._vibration_peak_ratio) plus
which link hits the ground at the peak. Usage: vibcmp.py <tar> <tag> <job> <out.csv>"""
import io, re, sys, tarfile
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, '/home/wang22/engineai/engineai_ros2_workspace/scripts')
from calculate_fall_risk import filter_torso_acceleration
FOOT = 'foot|ankle|toe|heel'
def metric(c, s):
  b1, b2 = c.body1_name.astype(str), c.body2_name.astype(str)
  nonfoot = ~(b1.str.lower().str.contains(FOOT) | b2.str.lower().str.contains(FOOT))
  F = c.force_magnitude.to_numpy(float); ts = c.timestamp.to_numpy(float)
  col = np.unique(ts[nonfoot.to_numpy() & (F > 400)])
  if col.size == 0 or s.empty:
    return dict(load_ratio=0.0)
  f = filter_torso_acceleration(s)
  t = f.timestamp.to_numpy(float)
  mag = np.sqrt(sum(f[f'base_link_lin_acc_{a}'].to_numpy(float) ** 2 for a in 'xyz'))
  w = 0.02; groups = [[col[0]]]
  for x in col[1:]:
    (groups[-1].append(x) if x - groups[-1][-1] < w else groups.append([x]))
  peak, tpk = 0.0, np.nan
  for g in groups:
    a, b = min(g), (max(g) + w if len(g) > 1 else min(g) + w)
    m = (t >= a) & (t <= b)
    if m.any():
      i = np.nanargmax(np.where(m, mag, -1))
      if mag[i] > peak: peak, tpk = float(mag[i]), t[i]
  world = (b1.str.lower().isin(['world', 'ground']) | b2.str.lower().isin(['world', 'ground'])).to_numpy()
  link = np.where(b1.str.startswith('LINK_'), b1, b2)
  wm = world & nonfoot.to_numpy() & (ts >= tpk - 0.05) & (ts <= tpk)
  dom = ''
  if wm.any():
    imp = pd.Series(F[wm]).groupby(link[wm]).sum()
    dom = re.sub(r'^LINK_', '', imp.idxmax())
  hand = np.char.find(link.astype(str), 'ELBOW_END') >= 0
  big = world & nonfoot.to_numpy() & (F > 400)
  th = ts[big & hand].min() if (big & hand).any() else np.inf
  to = ts[big & ~hand].min() if (big & ~hand).any() else np.inf
  return dict(load_ratio=peak / 100.0, t_peak=tpk - col[0], dom_link=dom, hand_first=bool(th < to),
              hand_touch=bool(np.isfinite(th)))
def main(tarp, tag, job, out):
  need = {'contact_data.csv', 'sensor_vibration_data.csv'}; pend = {}; rows = []
  with tarfile.open(tarp, 'r|gz') as tf:
    for mem in tf:
      fn = Path(mem.name).name
      if fn not in need: continue
      par = str(Path(mem.name).parent); p = pend.setdefault(par, {}); p[fn] = tf.extractfile(mem).read()
      if not need.issubset(p): continue
      c = pd.read_csv(io.BytesIO(p['contact_data.csv']), usecols=['timestamp', 'body1_name', 'body2_name', 'force_magnitude'])
      s = pd.read_csv(io.BytesIO(p['sensor_vibration_data.csv']), usecols=['timestamp'] + [f'base_link_lin_acc_{a}' for a in 'xyz'])
      rows.append(dict(tag=tag, job=job, run=Path(par).parts[-2], **metric(c, s))); del pend[par]
  pd.DataFrame(rows).to_csv(out, index=False); print(out, len(rows), flush=True)
if __name__ == '__main__':
  main(*sys.argv[1:])
