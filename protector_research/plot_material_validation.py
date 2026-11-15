"""Fig. 2c-style validation of the contact materials used by the simulator (dt 0.5 ms, 5.487 kg).

Panel SRM: measured peak force (data1, rho 0.4, circles), the fitted relation F_pk = 0.0452 t^-1.16 F^1.88
(lines, as in Fig. 2c) and MuJoCo with the calibrated SRM material (squares), versus the input force proxy
F = m v / 0.26554 ms. Panels silicone and bare: the same for their targets.
Usage: .venv/bin/python plot_material_validation.py <protector_materials.yaml>
"""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib import font_manager

from drop import BARE_PULSE_S, IMPACTOR_MASS, load_targets
from fit_m2 import FastDrop
from fit_m2_direct import NAMES, material

font_manager.fontManager.addfont("/usr/share/fonts/truetype/arphic/uming.ttc")
plt.rcParams["font.family"] = font_manager.FontProperties(fname="/usr/share/fonts/truetype/arphic/uming.ttc").get_name()
plt.rcParams.update({"axes.unicode_minus": False, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
COL = {6: "#2a78d6", 12: "#eb6834", 18: "#1baf7a", 24: "#6250d6", 1: "#52514e"}
WS = "/home/wang22/engineai/engineai_ros2_workspace/scripts/ThicknessCalculate/"

mats = yaml.safe_load(open(sys.argv[1]))["materials"]
x_of = lambda name: np.array([mats[name][k] for k in NAMES])
sim = FastDrop(0.0005, IMPACTOR_MASS)
heights = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0])
F_in = IMPACTOR_MASS * np.sqrt(2 * 9.81 * heights) / BARE_PULSE_S / 1e3
P = json.load(open(WS + "fitted_parameters.json"))
PS = json.load(open(WS + "fitted_parameters_silicone.json"))
tg = load_targets()
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
report = []


def sim_curve(name, t):
  out = []
  for h in heights:
    sim.set(material(x_of(name), t, IMPACTOR_MASS))
    out.append(sim.run_avg(np.sqrt(2 * 9.81 * h), 4)[0] / 1e3)
  return np.array(out)


ax = axes[0]
for t in (6, 12, 18, 24):
  fit = P["C"] * t ** P["alpha"] * 0.4 ** P["beta"] * F_in ** P["gamma"]
  ax.plot(F_in, fit, "-", color=COL[t], lw=1.5, alpha=0.7, label=f"{t} mm 拟合关系" if t != 18 else "18 mm 拟合关系（无实测）")
  s = sim_curve("srm", t)
  ax.plot(F_in, s, "s", color=COL[t], ms=5, mfc="none", mew=1.5, label=f"{t} mm MuJoCo")
  m = tg[(tg.density == 0.4) & (tg.t_mm == t)]
  if len(m):
    fm = IMPACTOR_MASS * m.v / BARE_PULSE_S / 1e3
    ax.plot(fm, m.Fpk_N / 1e3, "o", color=COL[t], ms=8, label=f"{t} mm 实测")
    for (_, r), f in zip(m.iterrows(), fm):
      i = np.argmin(np.abs(heights - r.h)); report.append(("srm", t, r.h, r.Fpk_N / 1e3, s[i]))
ax.set_title("SRM（ρ0.4）：实测（圆）、Fig.2c 拟合关系（线）、MuJoCo（方框）")
ax = axes[1]
for t in (6, 12, 24):
  fit = PS["C"] * t ** PS["alpha"] * F_in ** PS["gamma"]
  ax.plot(F_in, fit, "-", color=COL[t], lw=1.5, alpha=0.7, label=f"{t} mm 目标（硅胶关系）")
  ax.plot(F_in, sim_curve("silicone", t), "s", color=COL[t], ms=5, mfc="none", mew=1.5, label=f"{t} mm MuJoCo")
ax.plot([IMPACTOR_MASS * np.sqrt(2 * 9.81) / BARE_PULSE_S / 1e3], [16.1], "o", color=COL[12], ms=8, label="12 mm @1 m（Fig.2a 对应值）")
ax.set_title("硅胶：目标关系（线）与 MuJoCo（方框）")
ax = axes[2]
ax.plot(F_in, F_in, "-", color=COL[1], lw=1.5, alpha=0.7, label="无护具：m·v/Δt")
sb = sim_curve("bare", mats["bare"].get("t_mm_nominal", 1.0))
ax.plot(F_in, sb, "s", color=COL[1], ms=5, mfc="none", mew=1.5, label="MuJoCo（0.5 ms）")
ax.plot([IMPACTOR_MASS * np.sqrt(2 * 9.81) / BARE_PULSE_S / 1e3], [84.8], "o", color=COL[1], ms=8, label="实测 non.csv（约 1 m）")
ax.set_title("无护具外壳：m·v/Δt（线）与 MuJoCo（方框）")
for ax in axes:
  ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(color="#e6e5e1", lw=0.6)
  ax.set_xlabel("输入冲击力 F = m·v/Δt (kN)"); ax.set_ylabel("传递峰值力 F_pk (kN)")
  ax.axvspan(IMPACTOR_MASS * np.sqrt(2 * 9.81 * 0.1) / BARE_PULSE_S / 1e3, IMPACTOR_MASS * np.sqrt(2 * 9.81) / BARE_PULSE_S / 1e3,
             color="#e6e5e1", alpha=0.5, lw=0)
  ax.legend(fontsize=6.5, frameon=False, ncol=1, loc="upper left")
  from matplotlib.ticker import FuncFormatter
  ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}")); ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
fig.suptitle("材料接触模型验证（dt 0.5 ms，5.487 kg 冲头；灰底为实测覆盖的 0.1–1 m）", fontsize=11)
fig.tight_layout()
out = "runs/figs/material_validation_fig2c_style.png"
fig.savefig(out, dpi=150, facecolor="#fcfcfb")
print(out)
print("SRM measured vs MuJoCo:")
for _, t, h, meas, s in report:
  print(f"  {t:2d} mm h={h:3.1f}: meas {meas:6.2f} kN  MuJoCo {s:6.2f} kN  ({s/meas-1:+.0%})")
