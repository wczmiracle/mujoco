"""Render a logged sample_plan run (pm_data.npy = time + full qpos every step) to mp4, offline (no simulation).

The hand collision balls (ELBOW_END) are drawn and turn red with the logged hand-ground force (contact_data.csv);
the overlay shows the logged left/right hand force (max over each frame interval) against the 400 N threshold.
Real time from `--t0` to the end, then a slow-motion replay of the `--slow-len` s window with the most hand
force above 400 N.

Usage: MUJOCO_GL=egl .venv/bin/python tools/render_replay.py <run dir> <out.mp4> [--title TEXT]
"""
import argparse
import glob
import os
import subprocess

import mujoco
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

XML = "/home/wang22/engineai/engineai_ros2_workspace/src/simulation/mujoco/assets/resource/pm_v2_mesh.xml"
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
THR = 400.0
W, H, FPS = 960, 720, 30
PANEL = 130
FMAX = 20000.0


def hand_force(run):
  c = pd.read_csv(glob.glob(os.path.join(run, "contact_data.csv"))[0],
                  usecols=["timestamp", "body1_name", "body2_name", "force_magnitude"])
  c = c[c.body1_name.str.contains("world")]
  out = {}
  for side in "LR":
    h = c[c.body2_name.str.endswith(f"ELBOW_END_{side}")]
    out[side] = h.groupby("timestamp").force_magnitude.sum()
  t_fall = c[c.force_magnitude > 2000].timestamp.min()
  return out, t_fall


def window_max(s, a, b):
  v = s[(s.index >= a) & (s.index < b)]
  return float(v.max()) if len(v) else 0.0


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("run")
  ap.add_argument("out")
  ap.add_argument("--title", default="")
  ap.add_argument("--t0", type=float, default=None, help="start time (default: 1.5 s before the first > 2 kN impact)")
  ap.add_argument("--slow", type=float, default=0.1, help="slow-motion factor")
  ap.add_argument("--slow-len", type=float, default=1.0)
  ap.add_argument("--azim", type=float, default=135.0)
  ap.add_argument("--dist", type=float, default=1.9)
  ap.add_argument("--elev", type=float, default=-35.0)
  ap.add_argument("--focus", choices=["L", "R"], default=None, help="close-up following this hand ball")
  ap.add_argument("--alpha", type=float, default=1.0, help="robot body opacity (hand balls stay opaque)")
  ap.add_argument("--slow-start", type=float, default=None, help="slow-motion window start (default: most hand force)")
  ap.add_argument("--no-realtime", action="store_true")
  a = ap.parse_args()

  pm = np.load(glob.glob(os.path.join(a.run, "pm_data.npy"))[0])
  t, qpos = pm[:, 0], pm[:, 1:]
  hf, t_fall = hand_force(a.run)
  harr = {k: (v.index.values, v.values) for k, v in hf.items()}
  t0 = a.t0 if a.t0 is not None else max(0.0, t_fall - 1.5)

  # slow-motion window: most hand force excess above 400 N
  step = 0.05
  best, ts = -1.0, t_fall
  for s0 in np.arange(t_fall, t[-1] - a.slow_len, step):
    e = sum(np.clip(v[(v.index >= s0) & (v.index < s0 + a.slow_len)] - THR, 0, None).sum() for v in hf.values())
    if e > best:
      best, ts = e, s0

  m = mujoco.MjModel.from_xml_path(XML)
  m.vis.global_.offwidth, m.vis.global_.offheight = W, H
  d = mujoco.MjData(m)
  hand_geoms = {}
  for side in "LR":
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"LINK_ELBOW_END_{side}")
    g = [i for i in range(m.ngeom) if m.geom_bodyid[i] == b][0]
    m.geom_group[g] = 0
    hand_geoms[side] = g
  if a.alpha < 1.0:
    for i in range(m.ngeom):
      if m.geom_bodyid[i] != 0 and i not in hand_geoms.values():
        rgba = m.mat_rgba[m.geom_matid[i]].copy() if m.geom_matid[i] >= 0 else m.geom_rgba[i].copy()
        m.geom_matid[i] = -1
        m.geom_rgba[i] = [rgba[0], rgba[1], rgba[2], rgba[3] * a.alpha]
  r = mujoco.Renderer(m, H, W)
  cam = mujoco.MjvCamera()
  cam.type = mujoco.mjtCamera.mjCAMERA_FREE
  cam.distance, cam.azimuth, cam.elevation = a.dist, a.azim, a.elev
  font = ImageFont.truetype(FONT, 22)
  small = ImageFont.truetype(FONT, 18)
  ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                         "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", a.out],
                        stdin=subprocess.PIPE)
  look = None

  def frame(tc, dtf, label):
    nonlocal look
    i = min(np.searchsorted(t, tc), len(t) - 1)
    d.qpos[:] = qpos[i]
    mujoco.mj_kinematics(m, d)
    base = d.geom_xpos[hand_geoms[a.focus]].copy() if a.focus else d.xpos[1].copy()
    look = base if look is None else 0.85 * look + 0.15 * base
    cam.lookat[:] = [look[0], look[1], 0.06 if a.focus else min(look[2], 0.6)]
    f = {s: window_max(hf[s], tc - dtf, tc) for s in "LR"}
    for s in "LR":
      x = min(f[s] / 5000.0, 1.0)
      m.geom_rgba[hand_geoms[s]] = [0.3 + 0.7 * x, 0.6 * (1 - x), 0.6 * (1 - x), 0.9] if f[s] > THR else [0.2, 0.6, 1.0, 0.6]
    r.update_scene(d, cam)
    img = Image.fromarray(r.render())
    dr = ImageDraw.Draw(img)
    dr.rectangle([0, 0, W, 34], fill=(0, 0, 0))
    dr.text((10, 4), f"{a.title}   t = {tc:5.2f} s   {label}", font=font, fill=(255, 255, 255))
    # bottom panel: numbers + force trace of the last `span` s (both hands, 0-20 kN, yellow = 400 N)
    span = 60 * dtf
    y0 = H - PANEL
    dr.rectangle([0, y0, W, H], fill=(0, 0, 0))
    for k, s in enumerate("LR"):
      col = (255, 80, 80) if f[s] > THR else (150, 200, 255)
      dr.text((10, y0 + 10 + 30 * k), f"{'左手' if s == 'L' else '右手'} {f[s] / 1000:6.2f} kN", font=small,
              fill=col)
    x0, x1, yb, yt = 200, W - 20, H - 14, y0 + 12
    dr.rectangle([x0, yt, x1, yb], outline=(90, 90, 90))
    ythr = yb - THR / FMAX * (yb - yt)
    dr.line([x0, ythr, x1, ythr], fill=(255, 255, 0), width=1)
    dr.text((x0 + 4, yt), f"0–{FMAX / 1000:g} kN，最近 {span:.2f} s；黄线 400 N", font=small, fill=(200, 200, 200))
    for s, col in (("L", (255, 120, 120)), ("R", (120, 200, 255))):
      tt, ff_ = harr[s]
      j0, j1 = np.searchsorted(tt, tc - span), np.searchsorted(tt, tc)
      if j1 > j0:
        xs = x0 + (tt[j0:j1] - (tc - span)) / span * (x1 - x0)
        ys = yb - np.minimum(ff_[j0:j1] / FMAX, 1.0) * (yb - yt)
        for xx, yy in zip(xs, ys):
          dr.line([xx, yb, xx, yy], fill=col, width=2)
    ff.stdin.write(np.asarray(img).tobytes())

  if a.slow_start is not None:
    ts = a.slow_start
  for tc in ([] if a.no_realtime else np.arange(t0 + 1 / FPS, t[-1], 1 / FPS)):
    frame(tc, 1 / FPS, "实时")
  look = None
  dtf = a.slow / FPS
  for _ in range(FPS):  # 1 s title card frame hold
    frame(ts, dtf, f"下面 {a.slow:g}× 慢放")
  look = None
  for tc in np.arange(ts + dtf, ts + a.slow_len, dtf):
    frame(tc, dtf, f"{a.slow:g}× 慢放")
  ff.stdin.close()
  ff.wait()
  print(a.out, f"fall {t_fall:.2f}s, slow window {ts:.2f}-{ts + a.slow_len:.2f}s")


if __name__ == "__main__":
  main()
