"""Write protector_materials.yaml for the simulator's contact_material mode.

Picks, per material, the seed with the lowest (rms + heavy-penalty) among runs/mat_<material>_s*.json
(SRM: runs/mat_srm_selected.json = mass-normalized fit with heavy-mass constraints).

Usage: .venv/bin/python make_materials_yaml.py --out <workspace>/src/simulation/mujoco/assets/config/protector_materials.yaml
"""
import argparse
import datetime as dt
import glob
import json
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent


def best(pattern):
  runs = [(json.load(open(f)), f) for f in glob.glob(str(HERE / pattern))]
  r, f = min(runs, key=lambda rf: rf[0].get("rms", rf[0].get("data1_rms", 9)) + rf[0].get("heavy_penalty", 0))
  return r, f


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", type=Path, required=True)
  ap.add_argument("--srm", default="runs/mat_srm_selected.json", help="SRM fit json (relative to this directory)")
  ap.add_argument("--bare", default="runs/mat_bare_s*.json", help="bare fit json or glob (relative to this directory)")
  ap.add_argument("--dt", type=float, default=0.0005, help="timestep the fits were calibrated at (set contact_material.timestep to it)")
  ap.add_argument("--native-bodies", nargs="*", default=[],
                  help="body-name substrings whose unprotected geoms keep the model's own solref/solimp (e.g. ELBOW_END)")
  ap.add_argument("--bottom-out", action="store_true",
                  help="write bottom_out: true (pad + bare-shell contact beyond the pad thickness)")
  a = ap.parse_args()
  mats, src = {}, {}
  for name, pattern in [("srm", a.srm), ("silicone", "runs/mat_silicone_s*.json"),
                        ("bare", a.bare)]:
    r, f = best(pattern)
    p = dict(zip(r["x_names"], [float(v) for v in r["x"]]))
    if name == "bare":
      p["t_mm_nominal"] = float(r.get("bare_t_mm") or 1.0)
    mats[name] = p
    src[name] = Path(f).name
  doc = {
      "generated": dt.datetime.now().isoformat(timespec="seconds"),
      "generator": "mujoco/protector_research/make_materials_yaml.py (fork wczmiracle/mujoco, branch protector-research)",
      "sources": src,
      "dt_calibrated": a.dt,
      "model": "per contact: solref = (-K(t)/m_n, -B(t)/m_n), m_n = 1/(d*A + (1-d)*Ahat); "
               "K(t) = 10^log10_K12 * (t/12mm)^kexp, B(t) = 10^log10_B12 * (t/12mm)^bexp; "
               "solimp = (d0, d0 + (0.9999-d0)*s1, width_over_t * t, midpoint, power)",
      "materials": mats,
  }
  if a.bottom_out:
    doc["bottom_out"] = True
    doc["bottom_out_note"] = ("pad compressed beyond its thickness t: a second contact with includemargin = -t and the "
                              "bare material (shell reaches the ground); the SRM fit includes it")
  if a.native_bodies:
    doc["native_bodies"] = list(a.native_bodies)
    doc["native_bodies_note"] = ("unprotected geoms of these bodies keep the robot model's own contact parameters "
                                 "(rubber hand ball; the bare material was measured on a hard shell)")
  a.out.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True))
  print(a.out)
  print(yaml.safe_dump(src))


if __name__ == "__main__":
  main()
