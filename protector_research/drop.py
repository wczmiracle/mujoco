"""Drop-test harness: rigid spherical impactor hitting a plane through a contact pair.

The protector is modeled by the pair's contact parameters (method M2), so this measures
what peak normal force MuJoCo produces for given solref/solimp, timestep, mass and speed.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import mujoco
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WS = Path("/home/wang22/engineai/engineai_ros2_workspace")
DATA1 = Path("/home/wang22/engineai/PressureCalculation/force/data1.csv")

IMPACTOR_MASS = 5.487  # kg, data1.csv column m
IMPACTOR_RADIUS = 0.053  # m, same as scripts/FreeBallTest 5 kg ball
BARE_PULSE_S = 0.00026554  # s, impact duration used by force_fit.py for F_before = m v / dt


@dataclasses.dataclass
class Contact:
  solref: tuple[float, float]
  solimp: tuple[float, ...]
  margin: float = 0.0

  def attrs(self) -> str:
    sr = " ".join(f"{x:.9g}" for x in self.solref)
    si = " ".join(f"{x:.9g}" for x in self.solimp)
    return f'solref="{sr}" solimp="{si}" margin="{self.margin:.9g}"'


def build_xml(contact: Contact, dt: float, mass: float = IMPACTOR_MASS,
              radius: float = IMPACTOR_RADIUS) -> str:
  # Solver/integrator defaults match the robot model (serial_pm_v2_new.xml: Euler, pyramidal, Newton).
  return f"""
<mujoco>
  <option timestep="{dt:.9g}" tolerance="1e-10" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 1" contype="0" conaffinity="0"/>
    <body name="ball" pos="0 0 1">
      <freejoint/>
      <geom name="ball" type="sphere" size="{radius:.9g}" mass="{mass:.9g}" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
  <contact>
    <pair geom1="floor" geom2="ball" condim="3" friction="1 1 0.005 0.0001 0.0001" {contact.attrs()}/>
  </contact>
</mujoco>"""


@dataclasses.dataclass
class DropResult:
  peak_force: float  # N, max normal force over steps
  max_penetration: float  # m, max of -dist
  rebound_speed: float  # m/s, upward speed after separation
  contact_time: float  # s, steps with nonzero normal force times dt


def drop(contact: Contact, dt: float, speed: float, mass: float = IMPACTOR_MASS,
         phase: float = 0.0, t_max: float = 0.05) -> DropResult:
  """Impactor starts above the floor so contact begins `phase` (0..1) of a step after a step boundary."""
  model = mujoco.MjModel.from_xml_string(build_xml(contact, dt, mass))
  data = mujoco.MjData(model)
  # Ignore gravity during the short approach: place so the gap closes in (1 - phase) steps + 1 step.
  approach = (1.0 + (1.0 - phase)) * dt * speed
  data.qpos[2] = IMPACTOR_RADIUS + contact.margin + approach
  data.qvel[2] = -speed
  force6 = np.zeros(6)
  peak = 0.0
  max_pen = 0.0
  steps_in_contact = 0
  separated_after_contact = False
  n = int(np.ceil(t_max / dt))
  for _ in range(n):
    mujoco.mj_step(model, data)
    fn = 0.0
    for i in range(data.ncon):
      mujoco.mj_contactForce(model, data, i, force6)
      fn += force6[0]
      max_pen = max(max_pen, -data.contact[i].dist)
    if fn > 1e-6:
      steps_in_contact += 1
      peak = max(peak, fn)
    elif steps_in_contact > 0:
      separated_after_contact = True
      break
  if not np.isfinite(data.qvel).all():
    peak = np.inf
  return DropResult(peak, max_pen, float(data.qvel[2]) if separated_after_contact else 0.0,
                    steps_in_contact * dt)


def peak_force(contact: Contact, dt: float, speed: float, mass: float = IMPACTOR_MASS,
               phases: int = 4) -> float:
  """Peak force averaged over contact phases within a step (removes step-alignment luck)."""
  if dt <= 1e-5:
    phases = 1
  return float(np.mean([drop(contact, dt, speed, mass, phase=k / phases).peak_force
                        for k in range(phases)]))


def load_targets() -> pd.DataFrame:
  """Measured drop tests: thickness t (mm), density p, height h (m), speed v (m/s), peak Fpk (N)."""
  df = pd.read_csv(DATA1, encoding="utf-8-sig")
  df.columns = [c.strip() for c in df.columns]
  out = pd.DataFrame({
      "t_mm": df["t"].astype(float),
      "density": df["p"].astype(float),
      "h": df["h"].astype(float),
      "v": df["v"].astype(float),
      "Fpk_N": df["Fpk"].astype(float) * 1e3,
  })
  bare = pd.DataFrame({"h": sorted(out["h"].unique())})
  bare["v"] = np.sqrt(2 * 9.81 * bare["h"])
  bare["t_mm"] = 0.0
  bare["density"] = 0.0
  bare["Fpk_N"] = IMPACTOR_MASS * bare["v"] / BARE_PULSE_S  # same F_before as the CHR fit
  return pd.concat([bare, out], ignore_index=True)
