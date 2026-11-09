"""M3 drop test (MuJoCo 4.0): spherical impactor onto a 3D flex foam pad whose bottom layer is pinned.

The pad is a flexcomp box with `elasticity` (young, poisson, damping, optional yield). The transmitted
force is taken as the impactor-pad contact force (pad mass ~0.07 kg, negligible).

Run with protector_research/.venv4 (MuJoCo 4.0 Python bindings built by tools/build_mj4_python.sh).
Usage: .venv4/bin/python drop_m3.py --t 12 --young 2e5 --poisson 0.2 --damping 0.002 --h 0.4
"""

from __future__ import annotations

import argparse
import dataclasses

import mujoco
import numpy as np

IMPACTOR_MASS = 5.487
IMPACTOR_RADIUS = 0.053


@dataclasses.dataclass
class Pad:
  t_mm: float
  young: float  # Pa
  poisson: float = 0.2
  damping: float = 0.0  # Rayleigh, s
  yield_strain: float = 0.0  # 0 = elastic
  side: float = 0.12  # m, square footprint
  nxy: int = 9  # vertices per side
  layer_mm: float = 3.0  # target layer thickness
  density: float = 400.0  # kg/m^3 (foam ~0.4 g/cm^3)


def build_xml(pad: Pad, dt: float, solver: str = "Newton") -> str:
  t = pad.t_mm * 1e-3
  nz = max(2, int(round(pad.t_mm / pad.layer_mm)) + 1)
  sp = pad.side / (pad.nxy - 1)
  spz = t / (nz - 1)
  mass = pad.density * pad.side ** 2 * t
  y = f' yield="{pad.yield_strain:g}"' if pad.yield_strain > 0 else ""
  # Bottom layer pinned: grid ids are z-fastest? pin by gridrange instead of ids.
  return f"""
<mujoco>
  <option timestep="{dt:g}" integrator="discrete" solver="{solver}" tolerance="1e-10" gravity="0 0 -9.81"/>
  <size memory="200M"/>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 1" contype="0" conaffinity="0"/>
    <flexcomp name="pad" type="grid" dim="3" count="{pad.nxy} {pad.nxy} {nz}"
              spacing="{sp:g} {sp:g} {spz:g}" pos="0 0 {t / 2:g}" radius="0.0005" mass="{mass:g}" dof="full">
      <elasticity young="{pad.young:g}" poisson="{pad.poisson:g}" damping="{pad.damping:g}"{y}/>
      <contact condim="3" selfcollide="none" contype="1" conaffinity="2" solref="0.0005 1" solimp="0.95 0.99 0.0001"/>
      <pin gridrange="0 0 0 {pad.nxy - 1} {pad.nxy - 1} 0"/>
    </flexcomp>
    <body name="ball" pos="0 0 1">
      <freejoint/>
      <geom name="ball" type="sphere" size="{IMPACTOR_RADIUS:g}" mass="{IMPACTOR_MASS:g}" contype="2" conaffinity="1"
            solref="0.0005 1" solimp="0.95 0.99 0.0001"/>
    </body>
  </worldbody>
</mujoco>"""


def drop(pad: Pad, h: float, dt: float = 1e-4, t_max: float = 0.05):
  m = mujoco.MjModel.from_xml_string(build_xml(pad, dt))
  d = mujoco.MjData(m)
  ball = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ball")
  qadr = m.jnt_qposadr[m.body_jntadr[ball]]
  dadr = m.jnt_dofadr[m.body_jntadr[ball]]
  v = np.sqrt(2 * 9.81 * h)
  d.qpos[qadr + 2] = pad.t_mm * 1e-3 + IMPACTOR_RADIUS + 2 * dt * v
  d.qvel[dadr + 2] = -v
  ball_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "ball")
  f6 = np.zeros(6)
  peak, touched, z0 = 0.0, False, pad.t_mm * 1e-3 + IMPACTOR_RADIUS
  max_comp = 0.0
  for _ in range(int(t_max / dt)):
    mujoco.mj_step(m, d)
    fn = 0.0
    for i in range(d.ncon):
      if ball_geom in (d.contact[i].geom[0], d.contact[i].geom[1]):
        mujoco.mj_contactForce(m, d, i, f6)
        fn += f6[0]
    max_comp = max(max_comp, z0 - d.qpos[qadr + 2])
    if fn > 1e-6:
      touched = True
      peak = max(peak, fn)
    elif touched:
      break
    if not np.isfinite(d.qvel).all():
      return dict(peak_kN=np.inf, comp_mm=np.inf, e=np.inf)
  return dict(peak_kN=peak / 1e3, comp_mm=max_comp * 1e3, e=max(d.qvel[dadr + 2], 0) / v)


if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("--t", type=float, default=12)
  ap.add_argument("--young", type=float, default=2e5)
  ap.add_argument("--poisson", type=float, default=0.2)
  ap.add_argument("--damping", type=float, default=0.002)
  ap.add_argument("--yield-strain", type=float, default=0.0)
  ap.add_argument("--h", type=float, nargs="+", default=[0.1, 0.4, 0.8, 1.0])
  ap.add_argument("--dt", type=float, default=1e-4)
  a = ap.parse_args()
  pad = Pad(a.t, a.young, a.poisson, a.damping, a.yield_strain)
  for h in a.h:
    print(h, drop(pad, h, a.dt))
