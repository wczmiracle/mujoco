"""Split late (>0.5 s after first >2 kN impact) hand-ground contact into episodes (gap > 2 ms = separated)
and attribute hand contact risk to short impacts (<30 ms) vs sustained support (>=30 ms)."""
import glob, os, shutil, subprocess, sys, tempfile
import numpy as np, pandas as pd
job = sys.argv[1]
tar = glob.glob(os.path.join(job, "*.tar.gz"))[0]
tmp = tempfile.mkdtemp()
subprocess.run(["tar", "-xzf", tar, "-C", tmp, "--wildcards", "*contact_data.csv"], check=True)
eps_per_run, short_ex, long_ex, early_ex = [], 0.0, 0.0, 0.0
long_meanF, short_peak, durs = [], [], []
for f in glob.glob(tmp + "/*/*/*/contact_data.csv"):
  c = pd.read_csv(f, usecols=["timestamp", "body1_name", "body2_name", "force_magnitude"])
  c = c[c.body1_name.str.contains("world")]
  if c.empty: continue
  dt = float(np.median(np.diff(np.unique(c.timestamp))))
  t1 = c[c.force_magnitude > 2000].timestamp.min()
  h = c[c.body2_name.str.contains("ELBOW_END")]
  n = 0
  for b, g in h.groupby("body2_name"):
    s = g.groupby("timestamp").force_magnitude.sum().sort_index()
    t = s.index.values; F = s.values
    ex = np.clip(F - 400, 0, None) / 400 * dt
    early_ex += ex[t <= t1 + 0.5].sum()
    m = t > t1 + 0.5
    t, F, ex = t[m], F[m], ex[m]
    if len(t) == 0: continue
    cut = np.where(np.diff(t) > 0.002)[0] + 1
    for ti, Fi, ei in zip(np.split(t, cut), np.split(F, cut), np.split(ex, cut)):
      d = ti[-1] - ti[0] + dt; n += 1
      if ei.sum() == 0: continue
      durs.append(d)
      if d < 0.03: short_ex += ei.sum(); short_peak.append(Fi.max())
      else: long_ex += ei.sum(); long_meanF.append(Fi.mean())
  eps_per_run.append(n)
shutil.rmtree(tmp)
tot = short_ex + long_ex
print(f"{os.path.basename(os.path.dirname(job))}: runs {len(eps_per_run)} | late hand episodes/run median {np.median(eps_per_run):.0f}"
      f" | late hand risk: short(<30ms) {100*short_ex/tot:.0f}%  sustained {100*long_ex/tot:.0f}% | late/total hand {100*tot/(tot+early_ex):.0f}%")
if short_peak: print(f"   short episodes w/ risk: n {len(short_peak)}  peak median {np.median(short_peak):.0f} N  p90 {np.percentile(short_peak,90):.0f}")
if long_meanF: print(f"   sustained episodes w/ risk: n {len(long_meanF)}  mean force median {np.median(long_meanF):.0f} N  dur median {np.median([d for d in durs if d>=0.03]):.2f} s")
