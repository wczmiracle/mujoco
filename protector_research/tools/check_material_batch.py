"""Data-quality check of one sample_plan job (material re-run): launches, peak forces, risk, per push direction.

Usage: .venv/bin/python tools/check_material_batch.py <job dir with the batch .tar.gz, or an extracted batch dir>
A .tar.gz is unpacked (contact_data.csv + per_run_risk.csv only) into a temp dir that is removed afterwards.
"""
import glob
import os
import shutil
import subprocess
import sys
import tempfile

import pandas as pd


def batch_dir(path):
  tars = glob.glob(os.path.join(path, "*.tar.gz"))
  if not tars:
    return path, None
  tmp = tempfile.mkdtemp(prefix="chk_")
  subprocess.run(["tar", "-xzf", tars[0], "-C", tmp, "--wildcards", "*contact_data.csv", "*per_run_risk.csv"], check=True)
  return os.path.dirname(glob.glob(tmp + "/**/per_run_risk.csv", recursive=True)[0]), tmp


def main():
  d, tmp = batch_dir(sys.argv[1])
  try:
    rows = []
    for f in sorted(glob.glob(d + "/*/2026*/contact_data.csv")):
      c = pd.read_csv(f, usecols=["body1_name", "body2_name", "force_magnitude", "base_link_vel_z"])
      w = c.body1_name.str.contains("world") | c.body2_name.str.contains("world")
      run = f.split("/")[-3]
      rows.append(dict(run=run, dir=run.split("-")[0], vz_max=c.base_link_vel_z.max(), Fw=c[w].force_magnitude.max(),
                       Fs=c[~w].force_magnitude.max() if (~w).any() else 0.0))
    r = pd.DataFrame(rows)
    risk = pd.read_csv(d + "/per_run_risk.csv")
    risk["run"] = risk.run_folder.str.split("/").str[0]
    r = r.merge(risk[["run", "contact_risk", "all_risk"]], on="run", how="left")
    print(f"runs {len(r)} | vz_max>3 m/s: {(r.vz_max > 3).sum()} | Fw>2e5: {(r.Fw > 2e5).sum()} | Fself>1e5: {(r.Fs > 1e5).sum()}")
    print(f"Fw median {r.Fw.median():.0f}  p95 {r.Fw.quantile(.95):.0f}  max {r.Fw.max():.0f}")
    print(f"all_risk median {r.all_risk.median():.2f}  mean {r.all_risk.mean():.2f}  max {r.all_risk.max():.1f}  >20: {(r.all_risk > 20).sum()}")
    print(r.groupby("dir").agg(n=("vz_max", "size"), vz_med=("vz_max", "median"), vz_max=("vz_max", "max"),
                               risk_mean=("all_risk", "mean"), risk_max=("all_risk", "max")).round(2).to_string())
    print(r.sort_values("all_risk", ascending=False).head(5).round(2).to_string(index=False))
  finally:
    if tmp:
      shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
  main()
