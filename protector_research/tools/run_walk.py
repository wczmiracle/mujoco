"""Locomotion check: does the simulator pacing bug change normal walking?

Runs the AMP controller (walking policy from rl_basic_param_XZL.yaml) without any push, at a commanded
forward speed, with or without PROTECTOR_EMULATE_PACING_BUG, and logs base state every physics step
(PROTECTOR_BASE_CSV). Only pm_v2.yaml (installed symlink) and rl_basic_param_XZL.yaml
initial_velocity are edited; both are backed up and restored.

Usage: .venv/bin/python tools/run_walk.py --tag walk --speeds 0 0.5 1.0 --repeats 2 --wall 25
"""

import argparse
import itertools
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "tools"))
from run_robot import ENV_SETUP, KILL_PATTERNS, PM_INST, WS, kill_all, set_in_block  # noqa: E402

XZL = WS / "src/interface_example/config/pm01/rl_basic/basic/rl_basic_param_XZL.yaml"


def set_speed(text: str, vx: float) -> str:
  out, n = re.subn(r"^(initial_velocity:\s*\n\s+linear:\s*)\[[^\]]*\]", rf"\g<1>[{vx:.2f}, 0.00, 0.0]", text,
                   count=1, flags=re.M)
  if n != 1:
    raise RuntimeError("initial_velocity.linear not found in XZL yaml")
  return out


def one_run(out: Path, vx: float, bug: bool, wall: float):
  if any(subprocess.run(["pgrep", "-f", p], capture_output=True).stdout for p in KILL_PATTERNS):
    raise SystemExit("another simulator/controller is running")
  out.mkdir(parents=True, exist_ok=True)
  pm_bak, xzl_bak = out / "pm_v2.yaml.orig", out / "xzl.yaml.orig"
  shutil.copy2(PM_INST, pm_bak)
  shutil.copy2(XZL, xzl_bak)
  env = dict(os.environ, PROTECTOR_BASE_CSV=str(out / "base_state.csv"))
  if bug:
    env["PROTECTOR_EMULATE_PACING_BUG"] = "1"
  procs = []
  try:
    pm = PM_INST.read_text()
    pm = set_in_block(pm, "perturbation", "auto_sampling", "false")
    pm = set_in_block(pm, "protection", "enabled", "false")
    PM_INST.write_text(pm)
    XZL.write_text(set_speed(XZL.read_text(), vx))
    procs.append(subprocess.Popen(
        ["bash", "-c", f"{ENV_SETUP} && exec ros2 launch interface_example rl_basic_example_AMP.launch.py"],
        cwd=WS, env=env, stdout=open(out / "controller.log", "w"), stderr=subprocess.STDOUT, start_new_session=True))
    time.sleep(3)
    procs.append(subprocess.Popen(
        ["bash", "-c", f"{ENV_SETUP} && exec ros2 launch mujoco_simulator mujoco_simulator.launch.py headless:=true "
                       f"save_policy_switch_csv:=true csv_format:=csv csv_file_path:={out}"],
        cwd=WS, env=env, stdout=open(out / "sim.log", "w"), stderr=subprocess.STDOUT, start_new_session=True))
    time.sleep(wall)
  finally:
    for p in procs:
      try:
        os.killpg(p.pid, signal.SIGINT)
      except ProcessLookupError:
        pass
    time.sleep(3)
    kill_all()
    shutil.copy2(pm_bak, PM_INST)
    shutil.copy2(xzl_bak, XZL)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--tag", default="walk")
  ap.add_argument("--speeds", nargs="+", type=float, default=[0.0, 0.5, 1.0])
  ap.add_argument("--repeats", type=int, default=2)
  ap.add_argument("--wall", type=float, default=25.0, help="wall-clock seconds per run after launch")
  a = ap.parse_args()
  root = Path.home() / "data/mujoco_logs/protector_research" / a.tag
  for vx, bug, rep in itertools.product(a.speeds, [False, True], range(a.repeats)):
    out = root / f"{'bug' if bug else 'fixed'}_vx{vx:.1f}_r{rep}"
    if (out / "base_state.csv").exists():
      continue
    print(f"[walk] {out.name}", flush=True)
    one_run(out, vx, bug, a.wall)
  print("[walk] done", root, flush=True)


if __name__ == "__main__":
  main()
