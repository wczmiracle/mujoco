"""Count runs whose first non-foot ground contact happens before the push starts (spontaneous falls).
Usage: python check_prepush_falls.py EXTRACTED_BATCH_DIR"""
import glob, os, sys
import numpy as np, pandas as pd
rows = []
for c in glob.glob(sys.argv[1] + "/*/*/*/contact_data.csv"):
  d = os.path.dirname(c)
  try:
    t_push = pd.read_csv(d + "/perturbation_data.csv", usecols=["start_time"]).start_time.min()
  except Exception:
    t_push = np.nan
  cd = pd.read_csv(c, usecols=["timestamp", "body1_name", "body2_name"])
  w1 = cd.body1_name.str.contains("world", na=False); w2 = cd.body2_name.str.contains("world", na=False)
  robot = cd.body2_name.where(w1, cd.body1_name)
  nf = cd[(w1 | w2) & ~robot.str.contains("ANKLE|FOOT", na=False)]
  rows.append((d.split("/")[-2], t_push, nf.timestamp.min() if len(nf) else np.nan))
df = pd.DataFrame(rows, columns=["run", "t_push", "t_first_nonfoot"])
early = df.t_first_nonfoot < df.t_push
print(f"{len(df)} runs; pre-push non-foot ground contact: {int(early.sum())} ({early.mean():.1%}); no push record: {int(df.t_push.isna().sum())}")
if early.any():
  print(df[early].sort_values("t_first_nonfoot").head(10).to_string(index=False))
