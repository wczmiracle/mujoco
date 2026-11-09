"""Oblique impact: sphere on x/z slide joints hits the floor with (vx, vz). Measures slip during the
loaded phase and the tangential velocity left at the end, for bare vs protector contact params.
Coulomb expectation: if mu * Jn >= m * vx the sphere sticks; slip ~ vx * (time to stop) / 2.
"""
import mujoco
import numpy as np
import yaml


def oblique(solref, solimp, vx=1.0, vz=2.0, dt=0.0005, mass=10.0, mu=1.0, cone="elliptic", impratio=10):
  sr = " ".join(map(str, solref)); si = " ".join(map(str, solimp))
  xml = f"""<mujoco><option timestep="{dt}" cone="{cone}" impratio="{impratio}"/>
  <worldbody><geom name="floor" type="plane" size="0 0 1" contype="0" conaffinity="0"/>
  <body pos="0 0 0.0505"><joint type="slide" axis="1 0 0"/><joint type="slide" axis="0 0 1"/>
  <geom name="b" type="sphere" size="0.05" mass="{mass}" contype="0" conaffinity="0"/></body></worldbody>
  <contact><pair geom1="floor" geom2="b" condim="3" friction="{mu} {mu} 0.005 0.0001 0.0001" solref="{sr}" solimp="{si}"/></contact></mujoco>"""
  m = mujoco.MjModel.from_xml_string(xml); d = mujoco.MjData(m)
  d.qvel[:] = [vx, -vz]
  f6 = np.zeros(6); slip = 0.0; t_stop = None; t0 = None; peak = 0.0; pen = 0.0
  for i in range(int(0.1 / dt)):
    mujoco.mj_step(m, d)
    fn = 0.0
    for k in range(d.ncon):
      mujoco.mj_contactForce(m, d, k, f6); fn += f6[0]; pen = max(pen, -d.contact[k].dist)
    if fn > 1.0:
      t0 = t0 if t0 is not None else d.time
      slip += abs(d.qvel[0]) * dt
      peak = max(peak, fn)
      if t_stop is None and abs(d.qvel[0]) < 0.01:
        t_stop = d.time - t0
  return slip * 1e3, (t_stop or float("nan")) * 1e3, peak / 1e3, pen * 1e3, d.qvel[0]


m2 = yaml.safe_load(open("runs/m2s_links_dt0.0005.yaml"))["links"]
cases = {"bare stiff": m2["LINK_BASE"], "M2 6mm": m2["LINK_HIP_PITCH_L"],
         "M2 12mm": m2["LINK_ELBOW_PITCH_L"], "M2 18mm": m2["LINK_KNEE_PITCH_L"]}
print(f"{'case':12} {'slip mm':>8} {'t_stop ms':>9} {'peak kN':>8} {'pen mm':>7} {'vx_end':>7} {'Coulomb est slip mm':>20}")
for name, c in cases.items():
  s, ts, pk, pen, vxe = oblique(c["solref"], c["solimp"])
  print(f"{name:12} {s:8.2f} {ts:9.2f} {pk:8.2f} {pen:7.2f} {vxe:7.3f} {1.0 * ts / 2:20.2f}")
