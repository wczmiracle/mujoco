"""Closed-loop comparison plot across configs of one batch tag: penetration, peak force, slip.

Usage: .venv/bin/python plot_robot.py ~/data/mujoco_logs/protector_research/baseline
Writes runs/figs/robot_<tag>.png
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path(sys.argv[1]).expanduser()
order = ["M0_dt0.002", "M1_dt0.002", "M1L_dt0.002", "M0_dt0.0005", "M1_dt0.0005", "M1L_dt0.0005", "M2_dt0.0005"]
cfgs = [c for c in order if (root / c).is_dir()] + sorted(
    p.name for p in root.iterdir() if p.is_dir() and p.name not in order)

metrics = [("pen_max_mm", "max penetration (mm)", False),
           ("fn_peak_kN", "peak normal force (kN)", True),
           ("slip_mm_sum", "slip while loaded, sum (mm)", True)]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
for ax, (key, label, logy) in zip(axes, metrics):
  data, names = [], []
  for c in cfgs:
    vals = [json.loads(f.read_text())["total"][key] for f in sorted((root / c).glob("*/score.json"))
            if json.loads(f.read_text()).get("fell")]
    if vals:
      data.append(vals)
      names.append(f"{c}\n(n={len(vals)})")
  ax.boxplot(data, showfliers=True)
  ax.set_xticks(range(1, len(names) + 1), names, rotation=30, fontsize=7)
  ax.set_title(label)
  if logy:
    ax.set_yscale("log")
  ax.grid(axis="y", alpha=0.3)
fig.suptitle(f"PM01 AMP falls, 8 directions x 200/300 N ({root.name})")
fig.tight_layout()
out = Path(__file__).resolve().parent / "runs/figs" / f"robot_{root.name}.png"
fig.savefig(out, dpi=130)
print(out)
