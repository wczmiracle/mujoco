"""Drop tests with the OLD (bench) simulator's unprotected contact, to identify the ball/ground stiffness it implies.

Contact exactly as in the bench robot model (pm_v2_mesh.xml, mujoco 3.2.3): the ball geom carries the robot collision
class (solref 0.0005 1, solimp 0.9 0.95 [0.001 0.5 2], condim 6), the floor is a plane with MuJoCo defaults
(solref 0.02 1); equal solmix/priority -> mixed contact solref 0.01025. Options as serial_pm_v2_mesh.xml:
timestep 2 ms, tolerance 1e-10, Euler, pyramidal, Newton (defaults). No CHR scaling (protection off).
Aluminum balls (rho 2700) of several masses from several heights; contact starts at 4 phases of a step.

Writes runs/old_contact_drop/: one CSV per case (t_s, F_N = normal contact force, pen_m = penetration,
z_m, vz_mps), summary.csv (peak force, contact time, max penetration, restitution, impulse), README.txt,
and old_contact_drop.zip of all of it.
"""
import io
import os
import zipfile

import mujoco
import numpy as np
import pandas as pd

OUT = os.path.join(os.path.dirname(__file__), "..", "runs", "old_contact_drop")
MASSES = (1.0, 2.0, 5.0, 10.0, 20.0, 40.0)
HEIGHTS = (0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0)
PHASES = (0.0, 0.25, 0.5, 0.75)
RHO_AL = 2700.0
DT = 0.002


def xml(mass, radius):
  return f"""
<mujoco>
  <option timestep="{DT}" tolerance="1e-10"/>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 1" condim="3" friction="1"/>
    <body name="ball" pos="0 0 1">
      <freejoint/>
      <geom name="ball" type="sphere" size="{radius:.9g}" mass="{mass:.9g}" condim="6"
            friction="1 0.005 0.0001" solimp="0.9 0.95" solref="0.0005 1"/>
    </body>
  </worldbody>
</mujoco>"""


def run(mass, h, phase):
  r = (3 * mass / (4 * np.pi * RHO_AL)) ** (1 / 3)
  m = mujoco.MjModel.from_xml_string(xml(mass, r))
  d = mujoco.MjData(m)
  v0 = np.sqrt(2 * 9.81 * h)
  d.qpos[2] = r + (2.0 - phase) * DT * v0  # first contact `phase` of a step after the 2nd step
  d.qvel[2] = -v0
  f6 = np.zeros(6)
  rows, touched, sep = [], False, 0
  for _ in range(int(0.5 / DT)):
    mujoco.mj_step(m, d)
    F = 0.0
    if d.ncon:
      mujoco.mj_contactForce(m, d, 0, f6)
      F = f6[0]
      touched = True
    pen = max(0.0, r - d.qpos[2])
    rows.append((d.time, F, pen, d.qpos[2], d.qvel[2]))
    if touched and F == 0.0 and d.qvel[2] > 0:
      sep += 1
      if sep > 5:
        break
  df = pd.DataFrame(rows, columns=["t_s", "F_N", "pen_m", "z_m", "vz_mps"])
  # impact phase: first loaded step until separation, or until the force falls back to 1.1 m g after the peak (ball at rest)
  ld = np.flatnonzero(df.F_N.to_numpy() > 0)
  ipk = int(df.F_N.idxmax())
  after = np.flatnonzero((np.arange(len(df)) > ipk) & (df.F_N.to_numpy() <= 1.1 * mass * 9.81))
  end = after[0] if len(after) else len(df)
  c = df.iloc[ld[0]:end]
  rest = df.F_N.iloc[-1] > 0
  summ = dict(mass_kg=mass, radius_m=r, h_m=h, v_in_mps=v0, phase=phase, Fpk_N=df.F_N.max(),
              contact_s=len(c) * DT, pen_max_m=df.pen_m.max(), impulse_Ns=float(c.F_N.sum() * DT),
              v_out_mps=0.0 if rest else float(df.vz_mps.iloc[-1]),
              restitution=0.0 if rest else float(df.vz_mps.iloc[-1] / v0), rests=bool(rest),
              n_steps_loaded=len(c))
  return df, summ


def main():
  os.makedirs(OUT, exist_ok=True)
  summ = []
  zbuf = zipfile.ZipFile(os.path.join(OUT, "old_contact_drop.zip"), "w", zipfile.ZIP_DEFLATED)
  for mass in MASSES:
    for h in HEIGHTS:
      for ph in PHASES:
        df, s = run(mass, h, ph)
        name = f"m{mass:g}kg_h{h:g}m_phase{ph:g}.csv"
        zbuf.writestr(f"curves/{name}", df.to_csv(index=False))
        summ.append(s)
  S = pd.DataFrame(summ)
  S.to_csv(os.path.join(OUT, "summary.csv"), index=False)
  zbuf.writestr("summary.csv", S.to_csv(index=False))
  readme = (__doc__ + "\nColumns of summary.csv: mass_kg, radius_m (aluminum sphere), h_m, v_in_mps, phase (contact start "
            "within a step), Fpk_N, contact_s (impact phase: first contact to separation, or to F <= 1.1 m g after the peak when the ball comes to rest), pen_max_m, impulse_Ns (sum F dt over the impact phase), v_out_mps, restitution "
            "(v_out / v_in; 0 if the ball rests), rests, n_steps_loaded.\nCurves: one row per 2 ms step: t_s, F_N, pen_m (radius - centre height), z_m, vz_mps.\n")
  zbuf.writestr("README.txt", readme)
  zbuf.close()
  open(os.path.join(OUT, "README.txt"), "w").write(readme)
  g = S.groupby(["mass_kg", "h_m"]).agg(Fpk_kN=("Fpk_N", lambda x: x.mean() / 1e3), contact_ms=("contact_s", lambda x: x.mean() * 1e3),
                                         pen_mm=("pen_max_m", lambda x: x.mean() * 1e3), e=("restitution", "mean"),
                                         I_over_mv=("impulse_Ns", "mean"))
  g["I_over_mv"] = g.I_over_mv / (g.index.get_level_values(0) * np.sqrt(2 * 9.81 * g.index.get_level_values(1)))
  pd.set_option("display.width", 200)
  print(g.round(3).to_string())


if __name__ == "__main__":
  main()
