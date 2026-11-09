"""One closed-loop fall run of PM01 with the AMP policy in the ROS MuJoCo simulator.

Edits only the *installed* pm_v2.yaml (backed up and always restored), launches the AMP
controller and a headless simulator, lets auto_sampling push once, then shuts everything down.

Outputs in --out: the simulator's usual CSVs plus contact_quality.csv
(written by SimManager::RecordContactQuality on branch protector-sim).

Usage:
  .venv/bin/python tools/run_robot.py --method M0 --dt 0.002 --direction 0 --force 200 \
      --out ~/data/mujoco_logs/protector_research/smoke
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import signal
import subprocess
import time
from pathlib import Path

WS = Path("/home/wang22/engineai/engineai_ros2_workspace")
PM_INST = WS / "install/mujoco_simulator/share/mujoco_simulator/assets/config/pm_v2.yaml"
ENV_SETUP = (
    'eval "$(conda shell.bash hook)" && conda activate engineai_ros2 && '
    f"source {WS}/install/setup.bash && "
    'export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"'
)
# Anchored to the executable (first argv word, or the script after a python interpreter) so they
# never match unrelated shells whose command lines merely mention these names.
KILL_PATTERNS = [
    r"^[^ ]*/(mujoco_simulator|rl_basic_example_AMP|robot_state_publisher|static_transform_publisher)( |$)",
    r"^[^ ]*python[0-9.]* [^ ]*/joint_state_converter\.py( |$)",
]


def set_in_block(text: str, block: str, key: str, value: str) -> str:
  """Set `key: value` inside the top-level YAML `block:` (keeps indentation and comments)."""
  out, inside, done = [], False, False
  for line in text.splitlines(True):
    if re.match(rf"^{block}:\s*(#.*)?$", line):
      inside = True
    elif inside and line.strip() and not line[0].isspace():
      inside = False
    m = re.match(rf"^(\s+){key}:\s*([^#\n]*?)(\s*#.*)?$", line.rstrip("\n")) if inside else None
    if m and not done:
      line = f"{m.group(1)}{key}: {value}{m.group(3) or ''}\n"
      done = True
    out.append(line)
  if not done:
    raise KeyError(f"{block}.{key} not found in {PM_INST}")
  return "".join(out)


def configure(method: str, direction: float, force: float, delay: float) -> str:
  text = PM_INST.read_text()
  text = set_in_block(text, "perturbation", "auto_sampling", "true")
  text = set_in_block(text, "perturbation", "auto_direction_angle", f"{direction:g}")
  text = set_in_block(text, "perturbation", "default_force_magnitude", f"{force:.1f}")
  text = set_in_block(text, "perturbation", "auto_delay", f"{delay:.1f}")
  text = set_in_block(text, "perturbation", "random_direction_jitter", "false")
  if method == "M0":
    text = set_in_block(text, "protection", "enabled", "false")
  elif method == "M1":  # current pipeline: CHR scaling with the protector map
    text = set_in_block(text, "protection", "enabled", "true")
    text = set_in_block(text, "protection", "use_protector_map", "true")
    text = set_in_block(text, "protection", "force_method", "chr")
  elif method == "M1L":  # CHR scaling, per-link thickness (protected_links + link_thickness_mm)
    text = set_in_block(text, "protection", "enabled", "true")
    text = set_in_block(text, "protection", "use_protector_map", "false")
    text = set_in_block(text, "protection", "use_protected_links", "true")
    text = set_in_block(text, "protection", "force_method", "chr")
  elif method in ("M2", "M2s", "M0s"):  # per-link contact params from PROTECTOR_M2_YAML; no force scaling
    text = set_in_block(text, "protection", "enabled", "false")
  else:
    raise ValueError(method)
  return text


def kill_all():
  for p in KILL_PATTERNS:
    subprocess.run(["pkill", "-TERM", "-f", p], capture_output=True)
  time.sleep(2)
  for p in KILL_PATTERNS:
    subprocess.run(["pkill", "-KILL", "-f", p], capture_output=True)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--method", choices=["M0", "M1", "M1L", "M2", "M2s", "M0s"], required=True)
  ap.add_argument("--m2-yaml", type=Path, default=None, help="per-link contact params for M2 (make_m2_yaml.py)")
  ap.add_argument("--dt", type=float, default=0.0, help="override timestep (s); 0 = keep XML")
  ap.add_argument("--direction", type=float, default=0.0)
  ap.add_argument("--force", type=float, default=200.0)
  ap.add_argument("--delay", type=float, default=4.0, help="seconds before the push")
  ap.add_argument("--after", type=float, default=6.0, help="seconds recorded after the push")
  ap.add_argument("--out", type=Path, required=True)
  ap.add_argument("--no-quality", action="store_true", help="do not write contact_quality.csv")
  ap.add_argument("--csv-min-force", type=float, default=0.0,
                  help="simulator contact CSV threshold in N (user's collection scripts use 400)")
  ap.add_argument("--extra-sim-args", default="", help="appended to the simulator launch (later args win)")
  ap.add_argument("--impratio", type=float, default=0.0, help=">0: elliptic cone with this impratio")
  ap.add_argument("--lean", action="store_true",
                  help="disable the simulator's own CSVs and contact export (keeps joint_state + contact_quality)")
  args = ap.parse_args()

  if any(subprocess.run(["pgrep", "-f", p], capture_output=True).stdout for p in KILL_PATTERNS):
    raise SystemExit("another simulator/controller is running; refusing to touch pm_v2.yaml")

  out = args.out.expanduser().resolve()
  out.mkdir(parents=True, exist_ok=True)
  backup = out / "pm_v2.yaml.orig"
  shutil.copy2(PM_INST, backup)
  env = dict(os.environ)
  if not args.no_quality:
    env["PROTECTOR_QUALITY_CSV"] = str(out / "contact_quality.csv")
  if args.method in ("M2", "M2s", "M0s"):
    if not args.m2_yaml:
      raise SystemExit("--m2-yaml is required for M2/M2s/M0s")
    env["PROTECTOR_M2_YAML"] = str(args.m2_yaml.resolve())
    shutil.copy2(args.m2_yaml, out / "m2_links.yaml")
  if args.impratio > 0:
    env["PROTECTOR_CONE"] = "elliptic"
    env["PROTECTOR_IMPRATIO"] = f"{args.impratio:g}"
  if args.dt > 0:
    env["PROTECTOR_SIM_TIMESTEP"] = f"{args.dt:.9g}"
  procs = []
  try:
    PM_INST.write_text(configure(args.method, args.direction, args.force, args.delay))
    shutil.copy2(PM_INST, out / "pm_v2.yaml.used")
    ctrl_log = open(out / "controller.log", "w")
    procs.append(subprocess.Popen(
        ["bash", "-c", f"{ENV_SETUP} && exec ros2 launch interface_example rl_basic_example_AMP.launch.py"],
        cwd=WS, env=env, stdout=ctrl_log, stderr=subprocess.STDOUT, start_new_session=True))
    time.sleep(3)
    sim_log = open(out / "sim.log", "w")
    if args.lean:
      csv_args = ("export_contact:=false save_contact_csv:=false save_perturbation_csv:=true "
                  "save_joint_state_csv:=true save_policy_switch_csv:=false save_link_kinetic_energy_csv:=false "
                  "save_joint_forces_csv:=false save_sensor_vibration_csv:=false")
    else:
      csv_args = ("export_contact:=true save_contact_csv:=true save_perturbation_csv:=true "
                  "save_joint_state_csv:=true save_policy_switch_csv:=true save_link_kinetic_energy_csv:=true "
                  f"csv_min_force_n:={args.csv_min_force:g}")
    sim_cmd = (f"{ENV_SETUP} && exec ros2 launch mujoco_simulator mujoco_simulator.launch.py "
               f"headless:=true {csv_args} csv_format:=csv csv_file_path:={out} {args.extra_sim_args}")
    procs.append(subprocess.Popen(["bash", "-c", sim_cmd], cwd=WS, env=env, stdout=sim_log,
                                  stderr=subprocess.STDOUT, start_new_session=True))
    # Wall clock ≈ sim time because the simulator is paced to real time.
    t_end = time.time() + 5.0 + args.delay + 0.4 + args.after
    while time.time() < t_end:
      if any(p.poll() is not None for p in procs):
        print("a process exited early; see logs in", out)
        break
      time.sleep(0.5)
  finally:
    for p in procs:
      try:
        os.killpg(p.pid, signal.SIGINT)
      except ProcessLookupError:
        pass
    time.sleep(3)
    kill_all()
    shutil.copy2(backup, PM_INST)
  print("done:", out)


if __name__ == "__main__":
  main()
