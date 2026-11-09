"""Run the 16-scenario fall set (8 directions x 2 push forces) for several (method, dt) configs.

Usage: .venv/bin/python tools/batch_robot.py --tag baseline --configs M0:0.002 M1:0.002 M0:0.0005 M1:0.0005
Runs are sequential (the simulator config file is global). Each run is scored right after it ends.
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
DIRS = [0, 45, 90, 135, 180, 225, 270, 315]

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--configs", nargs="+", required=True, help="METHOD:DT, e.g. M0:0.002")
ap.add_argument("--forces", nargs="+", type=float, default=[200.0, 300.0])
ap.add_argument("--m2-yaml", default=None, help="for M2; or METHOD=path,... e.g. M0s=a.yaml,M2s=b.yaml")
ap.add_argument("--lean", action="store_true")
args = ap.parse_args()

def yaml_for(method):
  if not args.m2_yaml:
    return None
  if "=" not in args.m2_yaml:
    return args.m2_yaml if method == "M2" else None
  return dict(kv.split("=", 1) for kv in args.m2_yaml.split(",")).get(method)


root = Path.home() / "data/mujoco_logs/protector_research" / args.tag
for cfg in args.configs:
  method_tag, dt = cfg.split(":")
  method, _, ir = method_tag.partition("+e")  # e.g. M2s+e10 -> elliptic cone, impratio 10
  for force in args.forces:
    for d in DIRS:
      out = root / f"{method_tag}_dt{dt}" / f"dir{d:03d}_F{force:g}"
      if (out / "score.json").exists():
        continue
      print(f"[batch] {method_tag} dt={dt} dir={d} F={force}", flush=True)
      r = subprocess.run([sys.executable, str(HERE / "tools/run_robot.py"), "--method", method,
                          "--dt", dt, "--direction", str(d), "--force", str(force), "--out", str(out)]
                         + (["--m2-yaml", yaml_for(method)] if yaml_for(method) else [])
                         + (["--lean"] if args.lean else [])
                         + (["--impratio", ir] if ir else []))
      if r.returncode == 0 and (out / "contact_quality.csv").exists():
        subprocess.run([sys.executable, str(HERE / "score_robot.py"), str(out)], stdout=subprocess.DEVNULL)
print("[batch] done", root)
