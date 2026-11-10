"""Continuous-mode fall batch, matching scripts/automated_collection_continuous.sh.

The controller and simulator are launched once per (config, force). A warm-up segment (discarded)
absorbs start-up transients; every scored experiment is started by publishing
/mujoco/next_direction_angle then /mujoco/new_experiment, which resets the robot and switches all
CSVs (including contact_quality.csv) to the experiment's directory.

Usage:
  .venv/bin/python tools/batch_continuous.py --tag cont --configs M1:0.002 M2s+e10:0.0005 \
      --forces 200 300 --m2-yaml M2s=runs/m2s_links_dt0.0005.yaml
"""

import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "tools"))
from run_robot import ENV_SETUP, KILL_PATTERNS, PM_INST, WS, configure, kill_all  # noqa: E402

DIRS = [0, 45, 90, 135, 180, 225, 270, 315]
FORCE_DURATION = 0.4


def ros_pub(topic, msg_type, data):
  subprocess.run(["bash", "-c", f"{ENV_SETUP} && ros2 topic pub --once {topic} {msg_type} \"{data}\""],
                 cwd=WS, capture_output=True, timeout=30)


def run_session(method, ir, dt, force, dirs, root, m2_yaml, delay, after, lean):
  sess = root / f"{method}{'+e' + ir if ir else ''}_dt{dt}"
  todo = [(d, sess / (f"dir{d:05.1f}_F{force:g}")) for d in dirs]
  todo = [(d, o) for d, o in todo if not (o / "score.json").exists()]
  if not todo:
    return
  if any(subprocess.run(["pgrep", "-f", p], capture_output=True).stdout for p in KILL_PATTERNS):
    raise SystemExit("another simulator/controller is running")
  sess.mkdir(parents=True, exist_ok=True)
  warm = sess / f"_warmup_F{force:g}"
  warm.mkdir(exist_ok=True)
  backup = sess / f"pm_v2.yaml.orig_F{force:g}"
  shutil.copy2(PM_INST, backup)
  env = dict(os.environ, PROTECTOR_QUALITY_CSV=str(warm / "contact_quality.csv"))
  if method in ("M2", "M2s", "M0s"):
    env["PROTECTOR_M2_YAML"] = str(Path(m2_yaml).resolve())
  if ir:
    env["PROTECTOR_CONE"] = "elliptic"
    env["PROTECTOR_IMPRATIO"] = ir
  if float(dt) > 0:
    env["PROTECTOR_SIM_TIMESTEP"] = dt
  csv_args = ("export_contact:=true save_contact_csv:=true save_perturbation_csv:=true save_joint_state_csv:=true "
              "save_policy_switch_csv:=true save_link_kinetic_energy_csv:=" + ("false" if lean else "true") +
              " csv_min_force_n:=400")
  procs = []
  try:
    PM_INST.write_text(configure(method, todo[0][0], force, delay))
    procs.append(subprocess.Popen(
        ["bash", "-c", f"{ENV_SETUP} && exec ros2 launch interface_example rl_basic_example_AMP.launch.py"],
        cwd=WS, env=env, stdout=open(sess / f"controller_F{force:g}.log", "w"), stderr=subprocess.STDOUT,
        start_new_session=True))
    time.sleep(3)
    procs.append(subprocess.Popen(
        ["bash", "-c", f"{ENV_SETUP} && exec ros2 launch mujoco_simulator mujoco_simulator.launch.py "
                       f"headless:=true {csv_args} csv_format:=csv csv_file_path:={warm}"],
        cwd=WS, env=env, stdout=open(sess / f"sim_F{force:g}.log", "w"), stderr=subprocess.STDOUT,
        start_new_session=True))
    time.sleep(5.0 + delay + FORCE_DURATION + after)  # warm-up segment, discarded
    for d, out in todo:
      if any(p.poll() is not None for p in procs):
        print("[cont] a process exited early; stopping session", flush=True)
        break
      out.mkdir(parents=True, exist_ok=True)
      shutil.copy2(PM_INST, out / "pm_v2.yaml.used")
      ros_pub("/mujoco/next_direction_angle", "std_msgs/msg/Float64", f"{{data: {d}}}")
      time.sleep(0.2)
      ros_pub("/mujoco/new_experiment", "std_msgs/msg/String", f"{{data: '{out}'}}")
      print(f"[cont] {sess.name} F={force} dir={d}", flush=True)
      time.sleep(1.0 + delay + FORCE_DURATION + after)
  finally:
    for p in procs:
      try:
        os.killpg(p.pid, signal.SIGINT)
      except ProcessLookupError:
        pass
    time.sleep(3)
    kill_all()
    shutil.copy2(backup, PM_INST)
  for d, out in todo:
    if (out / "contact_quality.csv").exists():
      subprocess.run([sys.executable, str(HERE / "score_robot.py"), str(out)], stdout=subprocess.DEVNULL)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--tag", required=True)
  ap.add_argument("--configs", nargs="+", required=True, help="METHOD[+eIMPRATIO]:DT")
  ap.add_argument("--forces", nargs="+", type=float, default=[200.0, 300.0])
  ap.add_argument("--dirs", nargs="+", type=float, default=DIRS)
  ap.add_argument("--m2-yaml", default="", help="METHOD=path,... for M2/M2s/M0s")
  ap.add_argument("--delay", type=float, default=4.0)
  ap.add_argument("--after", type=float, default=6.0)
  ap.add_argument("--lean", action="store_true", help="disable link kinetic energy CSV (needed for dt 0.2 ms)")
  a = ap.parse_args()
  yamls = dict(kv.split("=", 1) for kv in a.m2_yaml.split(",") if kv)
  root = Path.home() / "data/mujoco_logs/protector_research" / a.tag
  for cfg in a.configs:
    tag, dt = cfg.split(":")
    method, _, ir = tag.partition("+e")
    for force in a.forces:
      run_session(method, ir, dt, force, a.dirs, root, yamls.get(method), a.delay, a.after, a.lean)
  print("[cont] done", root, flush=True)


if __name__ == "__main__":
  main()
