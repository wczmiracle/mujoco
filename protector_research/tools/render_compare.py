"""Side-by-side offline replay of two logged runs (pm_data.npy = time + qpos), e.g. P5 vs P6 of the same S4 scene.

--sync switch (default): each run aligned at its own walking -> fall-policy switch (policy_switch.csv; runs with the
same scene name do NOT share the start state or the trip time, so absolute time is meaningless);
--sync fall: each run at its own first non-foot contact > 2 kN; --sync abs: same simulation time. Links whose ground contact force exceeds the
fall-risk threshold (head/hand 400 N, torso 1000 N, others 2000 N) are tinted red (darker = higher F/threshold,
max over the frame interval). Top text: torso / shoulder / hand force; bottom: torso force trace of both runs.

Usage: MUJOCO_GL=egl .venv/bin/python tools/render_compare.py <runA> <runB> <out.mp4> --labels P5 P6 [--title T]
       [--azim 90] [--pre 0.6] [--post 1.6] [--slow 0.25]
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
PW, PH, FPS, PANEL = 640, 600, 30, 150
TORSO_THR = 1000.0
FOOT = ("foot", "ankle", "toe", "heel")


def thr(link):
  u = link.upper()
  return 400.0 if ("HEAD" in u or "ELBOW_END" in u) else (1000.0 if "TORSO" in u else 2000.0)


def load(run):
  pm = np.load(glob.glob(os.path.join(run, "pm_data.npy"))[0])
  c = pd.read_csv(os.path.join(run, "contact_data.csv"), usecols=["timestamp", "body1_name", "body2_name", "force_magnitude"])
  c = c[c.body1_name.astype(str).str.contains("world")]
  c = c[~c.body2_name.astype(str).str.lower().str.contains("|".join(FOOT))]
  f = c.groupby(["body2_name", "timestamp"]).force_magnitude.sum()
  t_fall = c[c.force_magnitude > 2000].timestamp.min()
  per = {b: (s.index.get_level_values(1).to_numpy(), s.to_numpy()) for b, s in f.groupby(level=0)}
  return pm, per, float(t_fall)


def wmax(per, b, a, z):
  if b not in per:
    return 0.0
  t, v = per[b]
  i, j = np.searchsorted(t, a), np.searchsorted(t, z)
  return float(v[i:j].max()) if j > i else 0.0


def group(per, key, a, z):
  return max([wmax(per, b, a, z) for b in per if key in b.upper()] or [0.0])


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("runA"); ap.add_argument("runB"); ap.add_argument("out")
  ap.add_argument("--labels", nargs=2, default=["A", "B"]); ap.add_argument("--title", default="")
  ap.add_argument("--azim", type=float, default=90.0); ap.add_argument("--elev", type=float, default=-20.0)
  ap.add_argument("--dist", type=float, default=2.4)
  ap.add_argument("--pre", type=float, default=0.6); ap.add_argument("--post", type=float, default=1.6)
  ap.add_argument("--slow", type=float, default=0.25)
  ap.add_argument("--sync", choices=("switch", "fall", "abs"), default="switch")
  a = ap.parse_args()

  m = mujoco.MjModel.from_xml_path(XML)
  m.vis.global_.offwidth, m.vis.global_.offheight = PW, PH
  base_rgba = {}
  for g in range(m.ngeom):
    if m.geom_bodyid[g] != 0:
      base_rgba[g] = (m.mat_rgba[m.geom_matid[g]].copy() if m.geom_matid[g] >= 0 else m.geom_rgba[g].copy())
      m.geom_matid[g] = -1
      m.geom_rgba[g] = base_rgba[g]
  body_geoms = {}
  for g in base_rgba:
    body_geoms.setdefault(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]), []).append(g)
  runs = [load(r) for r in (a.runA, a.runB)]
  if a.sync == "switch":
    def t_switch(run):
      ps = pd.read_csv(os.path.join(run, "policy_switch.csv"))
      ps = ps[(ps.from_mode == "walking") & (ps.to_mode != "walking")]
      return float(ps.timestamp.iloc[-1])
    runs = [(pm, per, t_switch(r)) for (pm, per, _), r in zip(runs, (a.runA, a.runB))]
  if a.sync == "abs":
    t0 = min(x[2] for x in runs)
    runs = [(pm, per, t0) for pm, per, _ in runs]
  datas = [mujoco.MjData(m) for _ in runs]
  r = mujoco.Renderer(m, PH, PW)
  font = ImageFont.truetype(FONT, 20); small = ImageFont.truetype(FONT, 16)
  W, H = 2 * PW, PH + PANEL + 40
  ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                         "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", a.out],
                        stdin=subprocess.PIPE)
  dtf = a.slow / FPS
  looks = [None, None]
  for tr in np.arange(-a.pre, a.post, dtf):
    canvas = Image.new("RGB", (W, H), (0, 0, 0))
    for k, ((pm, per, tf), d) in enumerate(zip(runs, datas)):
      tc = tf + tr
      i = min(np.searchsorted(pm[:, 0], tc), len(pm) - 1)
      d.qpos[:] = pm[i, 1:]
      mujoco.mj_kinematics(m, d)
      base = d.xpos[1].copy()
      looks[k] = base if looks[k] is None else 0.9 * looks[k] + 0.1 * base
      cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
      cam.distance, cam.azimuth, cam.elevation = a.dist, a.azim, a.elev
      cam.lookat[:] = [looks[k][0], looks[k][1], min(looks[k][2], 0.55)]
      for b, gs in body_geoms.items():
        x = wmax(per, b, tc - dtf, tc) / thr(b) if b else 0.0
        for g in gs:
          if x > 1.0:
            s = min((x - 1.0) / 9.0, 1.0)
            m.geom_rgba[g] = [0.95, 0.55 * (1 - s), 0.55 * (1 - s), 1.0]
          else:
            m.geom_rgba[g] = base_rgba[g]
      r.update_scene(d, cam)
      img = Image.fromarray(r.render())
      dr = ImageDraw.Draw(img)
      win = (tc - 4 * dtf, tc)
      ft, fs, fh = group(per, "TORSO", *win), group(per, "SHOULDER", *win), group(per, "ELBOW_END", *win)
      dr.rectangle([0, 0, PW, 30], fill=(0, 0, 0))
      dr.text((8, 3), f"{a.labels[k]}   胸 {ft/1e3:5.1f} kN   肩 {fs/1e3:5.1f} kN   手 {fh/1e3:4.2f} kN", font=small,
              fill=(255, 120, 120) if ft > TORSO_THR else (230, 230, 230))
      canvas.paste(img, (k * PW, 40))
    dr = ImageDraw.Draw(canvas)
    note = {"switch": "（0 = 各自切换到摔倒策略）", "abs": "（两边同一仿真时刻）", "fall": "（0 = 各自首次 > 2 kN 着地）"}[a.sync]
    dr.text((10, 8), f"{a.title}   t = {tr:+.3f} s{note}  {a.slow:g}× 慢放   红 = 超阈（深 = F/阈值大）",
            font=font, fill=(255, 255, 255))
    # torso force trace of both runs
    x0, x1, yb, yt = 60, W - 20, H - 16, PH + 54
    fmax = 10000.0
    dr.rectangle([x0, yt, x1, yb], outline=(90, 90, 90))
    ythr = yb - TORSO_THR / fmax * (yb - yt)
    dr.line([x0, ythr, x1, ythr], fill=(255, 255, 0))
    dr.text((x0 + 4, yt + 2), "胸部（TORSO）接触力 0–10 kN，黄线 1000 N", font=small, fill=(200, 200, 200))
    span = a.pre + a.post
    xc = x0 + (tr + a.pre) / span * (x1 - x0)
    dr.line([xc, yt, xc, yb], fill=(150, 150, 150))
    for k, col in enumerate(((150, 200, 255), (255, 140, 140))):
      pm, per, tf = runs[k]
      ts = [b for b in per if "TORSO" in b.upper()]
      for b in ts:
        t, v = per[b]
        sel = (t >= tf - a.pre) & (t <= tf + tr)
        xs = x0 + (t[sel] - tf + a.pre) / span * (x1 - x0)
        ys = yb - np.minimum(v[sel] / fmax, 1.0) * (yb - yt)
        for xx, yy in zip(xs, ys):
          dr.line([xx, yb, xx, yy], fill=col, width=2)
      dr.text((x1 - 150, yt + 2 + 18 * k), f"— {a.labels[k]}", font=small, fill=col)
    ff.stdin.write(np.asarray(canvas).tobytes())
  ff.stdin.close(); ff.wait()
  print(a.out)


if __name__ == "__main__":
  main()
