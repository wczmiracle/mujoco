"""Drop-weight check of the old and the new simulator contact against the measured impacts (5.487 kg ball on the anvil).

measured : no protector  runs/bare_measured_peaks.csv (冲击力截取.zip, 0.1-1 m)
           SRM rho 0.4   data1 (drop.load_targets, 6/12/24 mm x 0.1/0.4/0.8/1.0 m; per-trial curves confirmed 2026-11-19)
old sim  : the pipeline of the old robot data (nmi_dataset): dt 2 ms, MuJoCo default contact solref [0.02, 1] /
           solimp [0.9, 0.95, 0.001, 0.5, 2], protector = CHR scaling of efc_force with the simulator's min-scale limits
           (drop_m1.py, RESULTS 2026-11-11 21:00, 4 phases). No protector = the unscaled force of the same model.
new sim  : the current simulator (contact_material, protector_materials.yaml as installed): dt 0.25 ms, bare = the
           measured-fit bare material, SRM = H_s0 pad + bare shell once compressed beyond t (bottom_out), 4 phases.
Writes runs/figs/drop_old_vs_new_vs_measured.png and runs/drop_old_vs_new.csv.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib import font_manager

from drop import IMPACTOR_MASS, load_targets
from fit_m2 import FastDrop
from fit_m2_direct import NAMES, material
from fit_material import BottomDrop, BARE_T

font_manager.fontManager.addfont("/usr/share/fonts/truetype/arphic/uming.ttc")
plt.rcParams["font.family"] = font_manager.FontProperties(fname="/usr/share/fonts/truetype/arphic/uming.ttc").get_name()
plt.rcParams.update({"axes.unicode_minus": False, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

MATS = "/home/wang22/engineai/engineai_ros2_workspace/src/simulation/mujoco/assets/config/protector_materials.yaml"
DT_NEW = 0.00025
H4 = (0.1, 0.4, 0.8, 1.0)
# old pipeline (drop_m1.py, RESULTS 2026-11-11 21:00): kN at 0.1/0.4/0.8/1.0 m
OLD_BARE = dict(zip(H4, (0.8, 1.6, 5.6, 6.2)))
OLD_SRM = {6: (0.67, 0.57, 0.28, 0.32), 12: (0.66, 0.58, 0.28, 0.31), 24: (0.64, 0.59, 0.28, 0.31)}
COL = {6: "#2a78d6", 12: "#eb6834", 24: "#6250d6"}

mats = yaml.safe_load(open(MATS))["materials"]
x = {k: np.array([mats[k][n] for n in NAMES]) for k in ("srm", "bare")}
tb = mats["bare"].get("t_mm_nominal", BARE_T)
v = lambda h: float(np.sqrt(2 * 9.81 * h))

bare = pd.read_csv("runs/bare_measured_peaks.csv")
fd = FastDrop(DT_NEW, IMPACTOR_MASS)
fd.set(material(x["bare"], tb, IMPACTOR_MASS))
rows = []
for _, r in bare.iterrows():
  rows.append(dict(case="无护具", t_mm=0, h=r.h, measured_kN=r.Fpk_N / 1e3, old_kN=OLD_BARE.get(round(r.h, 2), np.nan),
                   new_kN=fd.run_avg(v(r.h), 4)[0] / 1e3))
bd = BottomDrop(DT_NEW, IMPACTOR_MASS)
tg = load_targets()
for t in (6, 12, 24):
  bd.set((material(x["srm"], t, IMPACTOR_MASS), material(x["bare"], tb, IMPACTOR_MASS), t))
  m = tg[(tg.density == 0.4) & (tg.t_mm == t)].sort_values("h")
  for _, r in m.iterrows():
    rows.append(dict(case=f"SRM {t} mm", t_mm=t, h=r.h, measured_kN=r.Fpk_N / 1e3,
                     old_kN=OLD_SRM[t][H4.index(round(r.h, 2))], new_kN=bd.run_avg(v(r.h), 4)[0] / 1e3))
df = pd.DataFrame(rows)
df["old_err"] = df.old_kN / df.measured_kN - 1
df["new_err"] = df.new_kN / df.measured_kN - 1
df.to_csv("runs/drop_old_vs_new.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
ax = axes[0]
b = df[df.t_mm == 0]
ax.plot(b.h, b.measured_kN, "o-", color="#222222", label="实测")
ax.plot(b.h, b.new_kN, "s--", color="#1baf7a", mfc="none", mew=1.5, label="新仿真（0.25 ms，材料模型）")
ob = b.dropna(subset=["old_kN"])
ax.plot(ob.h, ob.old_kN, "^:", color="#c0392b", mfc="none", mew=1.5, label="旧仿真（2 ms，MuJoCo 默认接触）")
ax.set_title("无护具：5.5 kg 落锤砸砧板")
ax = axes[1]
for t in (6, 12, 24):
  s = df[df.t_mm == t]
  ax.plot(s.h, s.measured_kN, "o-", color=COL[t], label=f"{t} mm 实测")
  ax.plot(s.h, s.new_kN, "s--", color=COL[t], mfc="none", mew=1.5, label=f"{t} mm 新仿真")
  ax.plot(s.h, s.old_kN, "^:", color=COL[t], mfc="none", mew=1.2, alpha=0.8, label=f"{t} mm 旧仿真（CHR 缩放）")
ax.set_title("SRM（ρ0.4）护具：5.5 kg 落锤")
for ax in axes:
  ax.set_yscale("log"); ax.set_xlabel("落高 (m)"); ax.set_ylabel("峰值力 (kN)"); ax.grid(color="#e6e5e1", lw=0.6, which="both")
axes[0].legend(fontsize=7, frameon=False, loc="lower right")
axes[1].legend(fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
fig.tight_layout()
fig.savefig("runs/figs/drop_old_vs_new_vs_measured.png", dpi=200)
pd.set_option("display.width", 200)
print(df.round(3).to_string(index=False))
for name, g in (("无护具", df[df.t_mm == 0]), ("SRM", df[df.t_mm > 0])):
  o = g.dropna(subset=["old_kN"])
  print(f"{name}: 新仿真误差 {g.new_err.min():+.0%} ~ {g.new_err.max():+.0%}（rms {np.sqrt((np.log1p(g.new_err)**2).mean()):.0%}），"
        f"旧仿真 {o.old_err.min():+.0%} ~ {o.old_err.max():+.0%}")
