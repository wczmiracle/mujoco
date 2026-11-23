"""Elastic Hertz sphere-on-half-space impact (no plasticity, rigid support): peak force, contact time, max indentation.
Used to compare the measured steel-on-steel drops with aluminum-on-concrete before the Abaqus runs.
F_max = 4/3 E* sqrt(R) d^1.5,  d = (15 m v^2 / (16 E* sqrt(R)))^0.4,  T = 2.94 d / v."""
import numpy as np
import pandas as pd

CASES = {  # m kg, rho, E1, nu1, E2, nu2
  "steel 5.487 kg / steel": (5.487, 7860, 200e9, .30, 200e9, .30),
  "steel 5.487 kg / concrete 30 GPa": (5.487, 7860, 200e9, .30, 30e9, .20),
  "Al 5 kg / steel": (5, 2700, 70e9, .33, 200e9, .30),
  "Al 5 kg / concrete 30 GPa": (5, 2700, 70e9, .33, 30e9, .20),
}


def hertz(m, rho, E1, n1, E2, n2, h):
  R = (3 * m / (4 * np.pi * rho)) ** (1 / 3)
  Es = 1 / ((1 - n1**2) / E1 + (1 - n2**2) / E2)
  v = np.sqrt(2 * 9.81 * h)
  d = (15 * m * v**2 / (16 * Es * np.sqrt(R))) ** 0.4
  return R, 4 / 3 * Es * np.sqrt(R) * d**1.5, 2.94 * d / v, d


meas = pd.read_csv("runs/bare_measured_peaks.csv").set_index("h").Fpk_N
rows = []
for name, a in CASES.items():
  for h in (0.1, 0.4, 1.0):
    R, F, T, d = hertz(*a, h)
    rows.append(dict(case=name, h=h, R_mm=R * 1e3, F_kN=F / 1e3, T_ms=T * 1e3, d_um=d * 1e6, meas_steel_kN=meas[h] / 1e3))
df = pd.DataFrame(rows)
print(df.round(3).to_string(index=False))
k = np.polyfit(np.log(meas.index), np.log(meas.values), 1)[0]
print(f"measured steel/steel: F ~ h^{k:.2f} (Hertz: h^0.60)")
