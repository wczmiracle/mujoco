"""Per-body impact attenuation on the same 48 scenarios: stiff bare (M0s+e10) vs candidate (M2s+e10),
with the CHR-expected protected force computed from the bare peak (CHR's F_before)."""
import collections, glob, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import yaml
font_manager.fontManager.addfont("/usr/share/fonts/truetype/arphic/uming.ttc")
plt.rcParams["font.family"] = font_manager.FontProperties(fname="/usr/share/fonts/truetype/arphic/uming.ttc").get_name()
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "axes.unicode_minus": False})
R = "/home/wang22/data/mujoco_logs/protector_research/validate_chr/"
P = json.load(open("/home/wang22/engineai/engineai_ros2_workspace/scripts/ThicknessCalculate/fitted_parameters.json"))
links = yaml.safe_load(open("runs/m2schr_links_dt0.0005.yaml"))["links"]
def per_body(cfg):
  a = collections.defaultdict(list)
  for f in glob.glob(R + cfg + "/*/score.json"):
    for b, v in json.load(open(f))["bodies"].items():
      a[b].append(v["fn_peak_kN"])
  return a
B, Pr = per_body("M0s+e10_dt0.0005"), per_body("M2s+e10_dt0.0005")
bodies = [b for b in ["LINK_SHOULDER_YAW_R", "LINK_SHOULDER_YAW_L", "LINK_HIP_PITCH_L", "LINK_HIP_PITCH_R", "LINK_KNEE_PITCH_L",
                      "LINK_KNEE_PITCH_R", "LINK_ELBOW_PITCH_L", "LINK_ELBOW_PITCH_R", "LINK_HIP_YAW_L", "LINK_HIP_YAW_R"] if b in B and b in Pr]
names = {"SHOULDER_YAW": "肩", "HIP_PITCH": "髋俯仰", "KNEE_PITCH": "膝", "ELBOW_PITCH": "肘", "HIP_YAW": "髋偏航"}
fig, ax = plt.subplots(figsize=(11, 4.6))
y = np.arange(len(bodies))
for i, b in enumerate(bodies):
  t = links[b]["thickness_mm"]; fb = np.median(B[b]); fp = np.median(Pr[b])
  fc = P["C"] * t ** P["alpha"] * 0.4 ** P["beta"] * fb ** P["gamma"]
  ax.plot([fp, fb], [i, i], color="#c3c2b7", lw=2, zorder=1)
  ax.scatter(fb, i, s=70, color="#52514e", zorder=3, label="无护具（刚性裸地）" if i == 0 else None)
  ax.scatter(fp, i, s=70, color="#2a78d6", zorder=3, label="候选方案（护具）" if i == 0 else None)
  ax.scatter(fc, i, s=80, facecolor="none", edgecolor="#eb6834", lw=1.8, zorder=4, label="CHR 预期（护具）" if i == 0 else None)
  ax.text(max(fb, fp) * 1.25, i, f"×{fp/fb:.2f}（CHR ×{fc/fb:.2f}）", va="center", fontsize=8, color="#52514e")
part = lambda b: next(v for k, v in names.items() if k in b) + ("右" if b.endswith("_R") else "左")
ax.set_yticks(y, [f"{part(b)}  {links[b]['thickness_mm']:.0f} mm" for b in bodies])
ax.invert_yaxis(); ax.set_xscale("log"); ax.set_xlim(0.003, 600)
from matplotlib.ticker import FuncFormatter, LogLocator
ax.xaxis.set_major_locator(LogLocator(base=10)); ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
ax.set_xlabel("峰值法向力中位数 (kN)，48 个推倒场景")
ax.grid(axis="x", color="#e6e5e1"); ax.legend(frameon=False, loc="lower left", fontsize=8)
ax.set_title("护具衰减效果：同一组场景下无护具 vs 候选方案；橙圈为 CHR 按无护具峰值算出的预期", fontsize=10)
fig.tight_layout(); fig.savefig("runs/figs/attenuation_per_body.png", dpi=140, facecolor="#fcfcfb"); print("runs/figs/attenuation_per_body.png")
