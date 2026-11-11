"""Side-by-side fall video: current M1 (left) vs candidate M2 (right), same push directions.

For each direction: trim pm_data.npy to the segment after the last reset (continuous mode prepends
the previous experiment's tail), cut [push-1 s, push+6 s], render with the workspace renderer
(scripts/render_pm_data_video.py) using one fixed camera per direction (fitted to the M1
trajectory, reused for M2), label, hstack, then concatenate all directions.

Usage: .venv/bin/python tools/make_fall_video.py --root ~/data/mujoco_logs/protector_research/video10
"""
import argparse
import glob
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

WS = Path("/home/wang22/engineai/engineai_ros2_workspace")
RENDER = WS / "scripts/render_pm_data_video.py"
PY = "/home/wang22/anaconda3/envs/engineai/bin/python"
FONT = "/usr/share/fonts/truetype/arphic/uming.ttc"
LEFT = ("M1_dt0.002", "现行 M1：CHR 缩放，dt 2 ms")
RIGHT = ("M2s+e10_dt0.0005", "候选方案：M2 接触参数，dt 0.5 ms")


def trimmed_run(run: Path, work: Path, pre=1.0, post=6.0) -> tuple[Path, float]:
  pm = next(run.glob("*/pm_data.npy"))
  arr = np.load(pm)
  back = np.where(np.diff(arr[:, 0]) < -1e-9)[0]
  if len(back):
    arr = arr[back[-1] + 1:]
  pf = list(run.glob("*/perturbation_data.csv"))
  t_push = pd.read_csv(pf[0], usecols=["start_time"]).start_time.min() if pf else 4.0
  arr = arr[(arr[:, 0] >= t_push - pre) & (arr[:, 0] <= t_push + post)]
  out = work / run.name
  out.mkdir(parents=True, exist_ok=True)
  np.save(out / "pm_data.npy", arr)
  return out, t_push


def render(run_dir: Path, out: Path, azimuth: float, cam_json: Path, reuse: bool, w: int, h: int):
  cmd = [PY, str(RENDER), str(run_dir), "-o", str(out), "--width", str(w), "--height", str(h), "--fps", "30",
         "--elevation", "0", "--far-clip", "100.0"]
  if reuse:
    cmd += ["--camera-json", str(cam_json)]
  else:
    cmd += ["--auto-frame", "--azimuth", str(azimuth), "--target-height", "0.65", "--frame-margin", "0.88",
            "--min-distance", "2.6", "--write-camera-json", str(cam_json)]
  subprocess.run(cmd, check=True, cwd=WS, capture_output=True)


def label(src: Path, dst: Path, text: str, sub: str):
  vf = (f"drawtext=fontfile={FONT}:text='{text}':x=24:y=20:fontsize=34:fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=10,"
        f"drawtext=fontfile={FONT}:text='{sub}':x=24:y=76:fontsize=26:fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=8")
  subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-vf", vf, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                  "-crf", "20", str(dst)], check=True)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--root", type=Path, required=True)
  ap.add_argument("--width", type=int, default=1280)
  ap.add_argument("--height", type=int, default=720)
  a = ap.parse_args()
  root = a.root.expanduser()
  work = root / "_video_work"
  shutil.rmtree(work, ignore_errors=True)
  work.mkdir()
  pairs = []
  for run_l in sorted((root / LEFT[0]).glob("dir*")):
    run_r = root / RIGHT[0] / run_l.name
    if not run_r.exists():
      continue
    d = float(run_l.name[3:].split("_")[0])
    az = (90.0 - d) % 360.0
    cam = work / f"{run_l.name}_camera.json"
    clips = []
    for (run, (tag, title)), reuse in zip(((run_l, LEFT), (run_r, RIGHT)), (False, True)):
      tr, _ = trimmed_run(run, work / tag)
      raw = work / f"{tag}_{run_l.name}_raw.mp4"
      render(tr, raw, az, cam, reuse, a.width, a.height)
      lab = work / f"{tag}_{run_l.name}.mp4"
      label(raw, lab, title, f"推力方向 {d:.0f}°，250 N，推前 1 s 至推后 6 s")
      clips.append(lab)
    pair = work / f"pair_{run_l.name}.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(clips[0]), "-i", str(clips[1]), "-filter_complex",
                    "[0:v][1:v]hstack=inputs=2[v]", "-map", "[v]", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                    str(pair)], check=True)
    pairs.append(pair)
    print("done", run_l.name, flush=True)
  lst = work / "list.txt"
  lst.write_text("".join(f"file '{p}'\n" for p in pairs))
  final = root / "fall10_M1_vs_candidate.mp4"
  subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(final)],
                 check=True)
  print(final)


if __name__ == "__main__":
  main()
