"""Per-impact check of protected-link forces in closed-loop runs.

For every protected body's ground impact (approach speed >= VMIN) take the approach speed v and
the normal effective mass m_eff at first touch, and the peak normal force within WINDOW after touch.
Compare the simulated peak with:
  CHR   : F_after = C t^a rho^b F_before^g with F_before = m_eff * v / 0.26554 ms (CHR's own definition)
  drop  : the drop-test model with the same contact parameters, impactor mass m_eff, speed v

Usage: .venv/bin/python analyze_impacts.py RUNS_GLOB LINKS_YAML OUT_CSV
"""
import glob
import json
import sys

import numpy as np
import pandas as pd
import yaml

from drop import BARE_PULSE_S, Contact
from fit_m2 import FastDrop

VMIN = 0.5  # m/s
WINDOW = 0.05  # s
DT = 0.0005
P = json.load(open("/home/wang22/engineai/engineai_ros2_workspace/scripts/ThicknessCalculate/fitted_parameters.json"))


def chr_kN(m, v, t):
  fb = m * v / BARE_PULSE_S / 1e3
  return P["C"] * t ** P["alpha"] * 0.4 ** P["beta"] * fb ** P["gamma"], fb


def main():
  runs, links_yaml, out_csv = sys.argv[1], sys.argv[2], sys.argv[3]
  links = yaml.safe_load(open(links_yaml))["links"]
  rows = []
  for q in sorted(glob.glob(runs + "/contact_quality.csv")):
    c = pd.read_csv(q)
    back = np.where(np.diff(c.time.to_numpy()) < -1e-9)[0]
    if len(back):
      c = c.iloc[back[-1] + 1:]
    for body, g in c.groupby("body"):
      t_mm = links.get(body, {}).get("thickness_mm", 0)
      if t_mm <= 0:
        continue
      s = g.groupby("time").agg(fn=("fn", "sum"), vn=("vn", "min"), m=("m_eff", "median"))
      # split into contact episodes separated by > 20 ms without contact
      times = s.index.to_numpy()
      starts = np.r_[0, np.where(np.diff(times) > 0.02)[0] + 1]
      for k, i0 in enumerate(starts):
        t0 = times[i0]
        w = s[(s.index >= t0) & (s.index <= t0 + WINDOW)]
        v = float(-w.vn.iloc[:4].min())  # approach speed over the first steps
        if v < VMIN:
          continue
        m_eff = float(w.m.iloc[0])
        rows.append(dict(run=q.split("/")[-2], body=body, t_mm=t_mm, t0=t0, v=v, m_eff=m_eff,
                         F_sim=float(w.fn.max() / 1e3), E_J=0.5 * m_eff * v * v))
  df = pd.DataFrame(rows)
  sims = {}
  F_drop = []
  for _, r in df.iterrows():
    key = (r.t_mm, round(r.m_eff, 1))
    if key not in sims:
      sims[key] = FastDrop(DT, max(r.m_eff, 0.2))
      lk = next(v for v in links.values() if v["thickness_mm"] == r.t_mm)
      sims[key].set(Contact(tuple(lk["solref"]), tuple(lk["solimp"])))
    F_drop.append(sims[key].run_avg(r.v, 4)[0] / 1e3)
  df["F_drop"] = F_drop
  df["F_chr"], df["F_before"] = zip(*[chr_kN(r.m_eff, r.v, r.t_mm) for _, r in df.iterrows()])
  df.to_csv(out_csv, index=False)
  df["sim/chr"] = df.F_sim / df.F_chr
  df["sim/drop"] = df.F_sim / df.F_drop
  print(f"{len(df)} impacts with v >= {VMIN} m/s")
  agg = df.groupby("t_mm").agg(n=("F_sim", "size"), v_med=("v", "median"), m_med=("m_eff", "median"),
                               F_sim_med=("F_sim", "median"), F_chr_med=("F_chr", "median"),
                               ratio_chr_med=("sim/chr", "median"),
                               ratio_chr_p25=("sim/chr", lambda x: x.quantile(0.25)),
                               ratio_chr_p75=("sim/chr", lambda x: x.quantile(0.75)),
                               ratio_drop_med=("sim/drop", "median"))
  print(agg.round(2).to_string())
  hi = df[df.F_before > 30]
  print(f"\nhigh-energy impacts (CHR F_before > 30 kN, i.e. like >= 0.1 m drop of 5.5 kg): n={len(hi)}, "
        f"sim/CHR median {hi['sim/chr'].median():.2f} (IQR {hi['sim/chr'].quantile(.25):.2f}-{hi['sim/chr'].quantile(.75):.2f})")


if __name__ == "__main__":
  main()
