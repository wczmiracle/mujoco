"""Where the contact risk of a sample_plan job comes from: share from the hands (ELBOW_END) and share
occurring more than 0.5 s after the first > 2 kN ground impact (post-fall chatter), per the risk definition
(excess over 2 kN, 400 N for HEAD/ELBOW_END, times the logged dt).

Usage: .venv/bin/python tools/contact_risk_share.py <job dir with the batch .tar.gz>
"""
import glob
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd


def main():
  tar = glob.glob(os.path.join(sys.argv[1], "*.tar.gz"))[0]
  tmp = tempfile.mkdtemp(prefix="share_")
  try:
    subprocess.run(["tar", "-xzf", tar, "-C", tmp, "--wildcards", "*contact_data.csv"], check=True)
    tot, hand, late = [], 0.0, 0.0
    for f in glob.glob(tmp + "/*/*/*/contact_data.csv"):
      c = pd.read_csv(f, usecols=["timestamp", "body1_name", "body2_name", "force_magnitude"])
      c = c[c.body1_name.str.contains("world")]
      if c.empty:
        continue
      dt = float(np.median(np.diff(np.unique(c.timestamp))))
      thr = np.where(c.body2_name.str.contains("HEAD|ELBOW_END"), 400.0, 2000.0)
      ex = np.clip(c.force_magnitude - thr, 0, None) / thr * dt
      t1 = c[c.force_magnitude > 2000].timestamp.min()
      tot.append(ex.sum())
      hand += ex[c.body2_name.str.contains("ELBOW_END")].sum()
      late += ex[c.timestamp > t1 + 0.5].sum()
    s = np.sum(tot)
    print(f"runs {len(tot)} | contact risk median {np.median(tot):.2f} | hand share {100 * hand / s:.0f}% | "
          f"> 0.5 s after first impact {100 * late / s:.0f}%")
  finally:
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
  main()
