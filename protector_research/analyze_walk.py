"""Summarize locomotion runs from tools/run_walk.py: policy rate in sim time, speed tracking, stability.

Usage: .venv/bin/python analyze_walk.py ~/data/mujoco_logs/protector_research/walk
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

T0 = 5.0  # discard start-up and transition


def yaw_of(q):
  w, x, y, z = q.T
  return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def roll_pitch(q):
  w, x, y, z = q.T
  roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
  pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
  return roll, pitch


rows = []
for run in sorted(Path(sys.argv[1]).expanduser().glob("*_vx*_r*")):
  f = run / "base_state.csv"
  if not f.exists():
    continue
  df = pd.read_csv(f)
  back = np.where(np.diff(df.time.to_numpy()) < -1e-9)[0]  # keep after last reset
  if len(back):
    df = df.iloc[back[-1] + 1:]
  df = df.drop_duplicates("time", keep="last")
  fell = bool((df.z < 0.5).any())
  w = df[df.time >= T0]
  if len(w) < 100:
    continue
  dur = w.time.iloc[-1] - w.time.iloc[0]
  updates = int((np.abs(np.diff(w.cmd_sum.to_numpy())) > 1e-9).sum())
  q = w[["qw", "qx", "qy", "qz"]].to_numpy()
  yaw = yaw_of(q)
  vfwd = w.vx * np.cos(yaw) + w.vy * np.sin(yaw)  # world velocity projected on heading
  vlat = -w.vx * np.sin(yaw) + w.vy * np.cos(yaw)
  roll, pitch = roll_pitch(q)
  m = re.match(r"(bug|fixed)_vx([\d.]+)_r(\d+)", run.name)
  rows.append(dict(mode=m.group(1), vx_cmd=float(m.group(2)), rep=int(m.group(3)), sim_s=round(dur, 1),
                   policy_hz=updates / dur, v_fwd=vfwd.mean(), v_lat=vlat.mean(),
                   yaw_rate_deg_s=np.degrees(np.unwrap(yaw)[-1] - np.unwrap(yaw)[0]) / dur,
                   z_mean=w.z.mean(), z_std_mm=w.z.std() * 1e3,
                   roll_std_deg=np.degrees(roll.std()), pitch_std_deg=np.degrees(pitch.std()), fell=fell))
df = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(df.round(3).to_string(index=False))
print("\nmean over repeats:")
print(df.groupby(["vx_cmd", "mode"]).mean(numeric_only=True).drop(columns="rep").round(3).to_string())
