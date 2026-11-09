"""Creep test (sphere on x/z slide joints, no rotation): a resting body pushed tangentially below the Coulomb limit should not move.

Measures horizontal creep over 1 s for given contact params, cone and impratio.
"""
import mujoco
import numpy as np
import yaml


def creep(solref, solimp, dt=0.0005, cone="pyramidal", impratio=1.0, mass=10.0, frac=0.5, mu=1.0, t=1.0):
  sr = " ".join(map(str, solref)); si = " ".join(map(str, solimp))
  xml = f"""<mujoco><option timestep="{dt}" cone="{cone}" impratio="{impratio}"/>
  <worldbody><geom name="floor" type="plane" size="0 0 1" contype="0" conaffinity="0"/>
  <body pos="0 0 0.05"><joint name="x" type="slide" axis="1 0 0"/><joint name="z" type="slide" axis="0 0 1"/><geom name="b" type="sphere" size="0.05" mass="{mass}" contype="0" conaffinity="0"/></body></worldbody>
  <contact><pair geom1="floor" geom2="b" condim="3" friction="{mu} {mu} 0.005 0.0001 0.0001" solref="{sr}" solimp="{si}"/></contact></mujoco>"""
  m = mujoco.MjModel.from_xml_string(xml); d = mujoco.MjData(m)
  for _ in range(int(0.3 / dt)):  # settle
    mujoco.mj_step(m, d)
  x0 = d.qpos[0]; pen = -d.contact[0].dist if d.ncon else 0
  d.xfrc_applied[1, 0] = frac * mu * mass * 9.81
  for _ in range(int(t / dt)):
    mujoco.mj_step(m, d)
  return (d.qpos[0] - x0) * 1e3, pen * 1e3


m2 = yaml.safe_load(open("runs/m2_links_dt0.0005.yaml"))["links"]
bare = yaml.safe_load(open("runs/m0s_links_dt0.0005.yaml"))["links"]["LINK_BASE"]
cases = {
    "M0 default (0.02 1 / 0.9 0.95 0.001)": ((0.02, 1), (0.9, 0.95, 0.001, 0.5, 2)),
    "M0s bare stiff": (bare["solref"], bare["solimp"]),
    "M2 6mm": (m2["LINK_HIP_PITCH_L"]["solref"], m2["LINK_HIP_PITCH_L"]["solimp"]),
    "M2 12mm": (m2["LINK_ELBOW_PITCH_L"]["solref"], m2["LINK_ELBOW_PITCH_L"]["solimp"]),
    "M2 18mm": (m2["LINK_KNEE_PITCH_L"]["solref"], m2["LINK_KNEE_PITCH_L"]["solimp"]),
}
print(f"{'case':40} {'cone':10} {'impratio':>8} {'static pen mm':>13} {'creep mm/s':>10}")
for name, (sr, si) in cases.items():
  for cone, ir in [("pyramidal", 1), ("elliptic", 1), ("elliptic", 10), ("elliptic", 100)]:
    c, p = creep(sr, si, cone=cone, impratio=ir)
    print(f"{name:40} {cone:10} {ir:8g} {p:13.2f} {c:10.2f}")
