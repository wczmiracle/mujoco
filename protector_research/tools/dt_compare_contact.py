"""Old vs new contact data: is the contact-risk / load-ratio change from the finer dt?
Streams the first N contact_data.csv of a tar, computes (same rules as calculate_contact_force_risk, vectorized):
  risk (row-level, collision mask +-1 ms), peak non-foot force, impulse above thr, total non-foot world impulse,
and for fine-dt data the same after decimating rows to a 2 ms grid (what a 2 ms log would have seen of the same physics)."""
import io, sys, tarfile, json
import numpy as np, pandas as pd
FOOT = ('foot', 'ankle', 'toe', 'heel')
def thr_of(link):
  u = link.upper()
  return 400.0 if ('HEAD' in u or 'ELBOW_END' in u) else (1000.0 if 'TORSO' in u else 2000.0)
def metrics(df, dt):
  df = df[['timestamp', 'body1_name', 'body2_name', 'force_magnitude']].copy()
  b1, b2 = df.body1_name.astype(str), df.body2_name.astype(str)
  nonfoot = ~(b1.str.lower().str.contains('|'.join(FOOT)) | b2.str.lower().str.contains('|'.join(FOOT)))
  ts = df.timestamp.to_numpy(float); F = df.force_magnitude.to_numpy(float)
  tc = np.unique(ts[nonfoot.to_numpy() & (F > 400)])
  if len(tc) == 0:
    return dict(risk=0, peak=0, imp_over=0, imp_all=0, ratio_max=0)
  i = np.searchsorted(tc, ts); lo = np.abs(ts - tc[np.clip(i - 1, 0, len(tc) - 1)]); hi = np.abs(ts - tc[np.clip(i, 0, len(tc) - 1)])
  m = (np.minimum(lo, hi) <= 0.001) & nonfoot.to_numpy()
  link = np.where(b1.str.startswith('LINK_'), b1, np.where(b2.str.startswith('LINK_'), b2, b2))
  world = (b1.str.lower().isin(['world', 'ground']) | b2.str.lower().isin(['world', 'ground'])).to_numpy()
  thr = np.array([thr_of(l) for l in link])
  r = np.clip((F - thr) / thr, 0, None)
  risk = round(float((r[m]).sum() * dt), 4)
  ratio = F / thr
  return dict(risk=risk, peak=float(F[m].max()) if m.any() else 0.0, imp_over=float((np.clip(F - thr, 0, None)[m]).sum() * dt),
              imp_all=float(F[nonfoot.to_numpy() & world].sum() * dt), ratio_max=float(ratio[m].max()) if m.any() else 0.0)
def main(tarp, n, label):
  out = []
  with tarfile.open(tarp, 'r|gz') as tf:
    for mem in tf:
      if not mem.name.endswith('contact_data.csv'):
        continue
      df = pd.read_csv(io.BytesIO(tf.extractfile(mem).read()), usecols=['timestamp', 'body1_name', 'body2_name', 'force_magnitude'])
      t = np.unique(df.timestamp.to_numpy(float)); dt = round(float(np.median(np.diff(t))), 5)
      rec = dict(label=label, run=mem.name.split('/')[1], dt=dt, **metrics(df, dt))
      if dt < 0.0015:
        k = int(round(0.002 / dt)); keep = set(t[::k])
        d2 = df[df.timestamp.isin(keep)]
        rec.update({f'dec2ms_{a}': b for a, b in metrics(d2, 0.002).items()})
      out.append(rec); print(json.dumps(rec), flush=True)
      if len(out) >= n:
        break
if __name__ == '__main__':
  main(sys.argv[1], int(sys.argv[2]), sys.argv[3])
