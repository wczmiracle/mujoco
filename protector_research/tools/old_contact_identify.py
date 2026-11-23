"""Identify the stiffness the old (bench) unprotected contact implies, from runs/old_contact_drop/old_contact_drop.zip,
and compare it with simple aluminum-on-wood references.

Fit per mass, over every loaded 2 ms step of all heights/phases: F = k*pen + c*pen_dot (linear Kelvin-Voigt) and
F = K*pen^1.5 (Hertz form). References for the same ball (aluminum sphere, rho 2700) and drop:
  hertz_wood : elastic Hertz, Al (70 GPa, 0.33) on wood across the grain E = 1 GPa, nu 0.35 (local contact only)
  crush_wood : wood crushing under the ball, F = 2 pi R p_y pen, p_y = 10 MPa (perpendicular-to-grain crushing);
               a linear spring k = 2 pi R p_y, peak F = v sqrt(k m)
  floor k    : static point stiffness of timber joist floors, 1 kN -> 0.5..4 mm (EN 1995-1-1 vibration check range),
               i.e. 0.25..2 MN/m, for a suspended timber floor; wood laid on a concrete slab is much stiffer."""
import io, os, zipfile
import numpy as np, pandas as pd

D = os.path.join(os.path.dirname(__file__), "..", "runs", "old_contact_drop")
z = zipfile.ZipFile(os.path.join(D, "old_contact_drop.zip"))
S = pd.read_csv(io.BytesIO(z.read("summary.csv")))
rows = []
for m, g in S.groupby("mass_kg"):
  X = []
  for _, r in g.iterrows():
    c = pd.read_csv(io.BytesIO(z.read(f"curves/m{r.mass_kg:g}kg_h{r.h_m:g}m_phase{r.phase:g}.csv")))
    pd_ = np.gradient(c.pen_m.to_numpy(), c.t_s.to_numpy())
    l = c.F_N.to_numpy() > 1.5 * m * 9.81
    X.append(np.c_[c.pen_m.to_numpy()[l], pd_[l], c.F_N.to_numpy()[l]])
  X = np.vstack(X)
  (k, cdamp), *_ = np.linalg.lstsq(X[:, :2], X[:, 2], rcond=None)
  K = np.linalg.lstsq(X[:, :1] ** 1.5, X[:, 2], rcond=None)[0][0]
  rows.append(dict(mass_kg=m, k_N_per_m=k, c_Ns_per_m=cdamp, k_over_m=k / m, K_hertz=K,
                   r2=1 - np.sum((X[:, :2] @ [k, cdamp] - X[:, 2]) ** 2) / np.sum((X[:, 2] - X[:, 2].mean()) ** 2)))
fit = pd.DataFrame(rows)

def hertz(m, R, h, E2=1e9, n2=0.35):
  Es = 1 / ((1 - .33**2) / 70e9 + (1 - n2**2) / E2); v = np.sqrt(2 * 9.81 * h)
  d = (15 * m * v**2 / (16 * Es * np.sqrt(R))) ** 0.4
  return 4 / 3 * Es * np.sqrt(R) * d**1.5, 2.94 * d / v, d

cmp = []
for (m, h), g in S.groupby(["mass_kg", "h_m"]):
  R = g.radius_m.iloc[0]; v = np.sqrt(2 * 9.81 * h)
  Fh, Th, dh = hertz(m, R, h)
  kc = 2 * np.pi * R * 10e6
  cmp.append(dict(mass_kg=m, h_m=h, old_F_kN=g.Fpk_N.mean() / 1e3, old_T_ms=g.contact_s.mean() * 1e3, old_pen_mm=g.pen_max_m.mean() * 1e3,
                  old_k_secant=g.Fpk_N.mean() / g.pen_max_m.mean(),
                  hertz_wood_F_kN=Fh / 1e3, hertz_wood_T_ms=Th * 1e3, hertz_wood_pen_mm=dh * 1e3,
                  crush_wood_F_kN=v * np.sqrt(kc * m) / 1e3, crush_wood_T_ms=np.pi / 2 * np.sqrt(m / kc) * 1e3, crush_k=kc,
                  floor_F_kN_lo=v * np.sqrt(0.25e6 * m) / 1e3, floor_F_kN_hi=v * np.sqrt(2e6 * m) / 1e3))
cmp = pd.DataFrame(cmp)
fit.to_csv(os.path.join(D, "identified_fit.csv"), index=False)
cmp.to_csv(os.path.join(D, "compare_references.csv"), index=False)
pd.set_option("display.width", 250)
print(fit.round(3).to_string(index=False))
print(cmp[cmp.h_m.isin([0.2, 1.0])].round(2).to_string(index=False))
