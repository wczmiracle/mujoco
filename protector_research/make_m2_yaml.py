"""Write the per-link M2 contact parameters the simulator loads via PROTECTOR_M2_YAML.

Thickness per link follows the simulator's per-link protection layout in pm_v2.yaml:
protected_links, each with link_thickness_mm[link] or the global protection.thickness.
Parameters come from the joint material model fitted at the given timestep and density.

Usage: .venv/bin/python make_m2_yaml.py --fit runs/m2_joint_dt0.0005.json --density 0.4 --out runs/m2_links_dt0.0005.yaml
"""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from fit_m2 import decode_joint

PM = Path("/home/wang22/engineai/engineai_ros2_workspace/src/simulation/mujoco/assets/config/pm_v2.yaml")

ap = argparse.ArgumentParser()
ap.add_argument("--fit", type=Path, required=True)
ap.add_argument("--density", type=float, default=0.4)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--bare-all", action="store_true",
                help="M0s: every non-foot robot body gets the bare-ground group fit (t=0) instead")
args = ap.parse_args()

if args.bare_all:
  import mujoco
  fit = [r for r in json.loads(args.fit.read_text()) if r["mode"] == "group" and r["t_mm"] == 0]
  assert len(fit) == 1, "need the group fit file with the bare (t=0) entry"
  prm = fit[0]["params"]["0"]
  m = mujoco.MjModel.from_xml_path(str(PM.parent.parent / "resource/pm_v2.xml"))
  bodies = sorted({mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) for g in range(m.ngeom)
                   if m.geom_bodyid[g] > 0 and (m.geom_contype[g] or m.geom_conaffinity[g])})
  bodies = [b for b in bodies if not any(k in b.upper() for k in ("FOOT", "ANKLE"))]
  links = {b: {"thickness_mm": 0.0, "solref": prm["solref"], "solimp": prm["solimp"]} for b in bodies}
  doc = {"source_fit": str(args.fit), "mode": "bare_all", "dt": fit[0]["dt"], "links": links}
  args.out.write_text(yaml.safe_dump(doc, sort_keys=False))
  print(args.out, len(links), "bodies", prm)
  raise SystemExit

prot = yaml.safe_load(PM.read_text())["protection"]
fit = [r for r in json.loads(args.fit.read_text()) if r["mode"] == "joint" and r["density"] == args.density]
assert len(fit) == 1, "need exactly one joint fit for this density"
x = np.array(fit[0]["x"])
links = {}
for link in prot["protected_links"]:
  t = float(prot.get("link_thickness_mm", {}).get(link, prot["thickness"]))
  c = decode_joint(x, t)
  links[link] = {"thickness_mm": t, "solref": [float(v) for v in c.solref],
                 "solimp": [float(v) for v in c.solimp]}
doc = {"source_fit": str(args.fit), "density": args.density, "dt": fit[0]["dt"], "links": links}
args.out.write_text(yaml.safe_dump(doc, sort_keys=False))
print(args.out); print(yaml.safe_dump(links, sort_keys=False))
