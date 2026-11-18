"""Hand-contact chatter check of one sample_plan job: per run and hand, the number of logged samples with
world-hand force > 2 kN (a hand pressed on the ground should carry ~100 N), plus the risk medians.

Usage: .venv/bin/python tools/check_hand_chatter.py <job dir with the batch .tar.gz>
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
  tmp = tempfile.mkdtemp(prefix="chat_")
  try:
    subprocess.run(["tar", "-xzf", tar, "-C", tmp, "--wildcards", "*contact_data.csv", "*per_run_risk.csv"], check=True)
    n = []
    for f in glob.glob(tmp + "/*/*/*/contact_data.csv"):
      c = pd.read_csv(f, usecols=["timestamp", "body1_name", "body2_name", "force_magnitude"])
      for L in ("LINK_ELBOW_END_R", "LINK_ELBOW_END_L"):
        w = c[(c.body2_name == L) & c.body1_name.str.contains("world")].groupby("timestamp").force_magnitude.sum()
        n.append(int((w > 2000).sum()))
    print(f"hand samples > 2 kN per run-hand: median {np.median(n):.0f}  p90 {np.percentile(n, 90):.0f}  max {max(n)}")
    r = pd.read_csv(glob.glob(tmp + "/*/per_run_risk.csv")[0])
    print(r[["contact_risk", "motor_risk", "jointforce_risk", "all_risk", "max_contact_force_n"]].median().round(3).to_string())
  finally:
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
  main()
