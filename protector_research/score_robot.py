"""Score one closed-loop fall run from contact_quality.csv (PLAN.md section 2).

Per robot body touching the ground (feet excluded):
  pen_max_mm     max penetration -dist
  fn_peak_kN     max normal force on that body (sum over its contacts in a step)
  slip_mm        tangential sliding distance while loaded (fn > FN_LOADED), sum of slip_speed * dt
  vn_impact      normal approach speed at first touch
  m_eff_impact   normal effective mass at first touch
Plus run totals and a stability flag (non-finite or exploding values).

Usage: .venv/bin/python score_robot.py RUN_DIR [RUN_DIR ...]  (prints a table, writes RUN_DIR/score.json)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FN_LOADED = 50.0  # N: contact counts as loaded above this
FEET = ("ANKLE", "FOOT")


def score(run: Path) -> dict:
  df = pd.read_csv(run / "contact_quality.csv")
  df = df[~df.body.str.upper().str.contains("|".join(FEET))]
  out = {"run": run.name, "bodies": {}}
  if df.empty:
    out["fell"] = False
    return out
  out["fell"] = True
  out["stable"] = bool(np.isfinite(df[["dist", "fn", "slip_speed"]].to_numpy()).all()
                       and df.fn.max() < 1e6)
  dt = float(df.dt.iloc[0])
  for body, g in df.groupby("body"):
    per_step = g.groupby("time").agg(fn=("fn", "sum"), pen=("dist", lambda d: max(0.0, -d.min())),
                                     slip=("slip_speed", "max"), vn=("vn", "min"),
                                     m_eff=("m_eff", "median"))
    loaded = per_step[per_step.fn > FN_LOADED]
    first = per_step.iloc[0]
    out["bodies"][body] = {
        "pen_max_mm": float(per_step.pen.max() * 1e3),
        "fn_peak_kN": float(per_step.fn.max() / 1e3),
        "slip_mm": float(loaded.slip.sum() * dt * 1e3),
        "loaded_ms": float(len(loaded) * dt * 1e3),
        "vn_impact": float(-first.vn),
        "m_eff_impact": float(first.m_eff),
    }
  b = out["bodies"].values()
  out["total"] = {
      "pen_max_mm": max(x["pen_max_mm"] for x in b),
      "fn_peak_kN": max(x["fn_peak_kN"] for x in b),
      "slip_mm_sum": sum(x["slip_mm"] for x in b),
      "dt": dt,
  }
  return out


def main():
  for arg in sys.argv[1:]:
    run = Path(arg).expanduser()
    s = score(run)
    (run / "score.json").write_text(json.dumps(s, indent=1))
    print(f"\n== {run}  fell={s['fell']}  stable={s.get('stable')}  total={s.get('total')}")
    print(f"{'body':24} {'pen mm':>7} {'Fn kN':>7} {'slip mm':>8} {'load ms':>8} {'vn m/s':>7} {'m_eff':>6}")
    for k, v in sorted(s["bodies"].items(), key=lambda kv: -kv[1]["fn_peak_kN"]):
      print(f"{k:24} {v['pen_max_mm']:7.2f} {v['fn_peak_kN']:7.2f} {v['slip_mm']:8.1f} "
            f"{v['loaded_ms']:8.0f} {v['vn_impact']:7.2f} {v['m_eff_impact']:6.1f}")


if __name__ == "__main__":
  main()
