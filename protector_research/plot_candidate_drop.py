"""Candidate contact params (dt 0.5 ms) on the drop test vs measurements, for 0/6/12/24 mm.

Top: peak force vs drop height (5.487 kg). Bare: candidate stiff-bare params vs non.csv (5 kg steel
ball on steel, ~1 m) and the CHR input definition m*v/0.26554 ms. Pads: candidate M2 (CHR high-force
fit) vs data1 (rho 0.4) and CHR targets above 1 m.
Bottom: bare 1 m force-time history, simulation at 0.5 ms vs non.csv (sampled every 0.04 ms).
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
font_manager.fontManager.addfont("/usr/share/fonts/truetype/arphic/uming.ttc")
plt.rcParams["font.family"] = font_manager.FontProperties(fname="/usr/share/fonts/truetype/arphic/uming.ttc").get_name()
plt.rcParams["axes.unicode_minus"] = False
import mujoco
import numpy as np
import pandas as pd
import yaml
from drop import IMPACTOR_MASS, IMPACTOR_RADIUS, BARE_PULSE_S, Contact, build_xml, load_targets
from fit_m2 import FastDrop, decode_joint

DT = 0.0005
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})

bare = yaml.safe_load(open("runs/m0s_links_dt0.0005.yaml"))["links"]["LINK_BASE"]
x = np.array(json.load(open("runs/m2_joint_dt0.0005_rho0.4_chr_s2.json"))[0]["x"])
P = json.load(open("/home/wang22/engineai/engineai_ros2_workspace/scripts/ThicknessCalculate/fitted_parameters.json"))
params = {0: Contact(tuple(bare["solref"]), tuple(bare["solimp"]))}
params.update({t: decode_joint(x, t) for t in (6, 12, 24)})
tg = load_targets(); tg = tg[tg.density == 0.4]
heights = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0])

sim = FastDrop(DT)
fig = plt.figure(figsize=(13, 7.2))
gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 1], hspace=0.42, wspace=0.28)
table = []
for i, t in enumerate((0, 6, 12, 24)):
  ax = fig.add_subplot(gs[0, i])
  sim.set(params[t])
  f = np.array([sim.run_avg(np.sqrt(2 * 9.81 * h), 4)[0] for h in heights]) / 1e3
  ax.plot(heights, f, color=BLUE, lw=2, marker="o", ms=4, label="仿真（候选参数，0.5 ms）")
  if t == 0:
    fb = IMPACTOR_MASS * np.sqrt(2 * 9.81 * heights) / BARE_PULSE_S / 1e3
    ax.plot(heights, fb, color=ORANGE, lw=1.2, ls="--", label="m·v/Δt（CHR 的缓冲前力）")
    ax.plot([1.0], [84.8], "o", color=ORANGE, ms=8, label="实测 non.csv（5 kg 钢球, 约 1 m）")
    ax.set_title("0 mm（钢球直接砸钢地）", color=INK)
    for h, a, b in zip(heights, f, fb):
      table.append((t, h, a, b, "m·v/Δt"))
  else:
    m = tg[tg.t_mm == t]
    ax.plot(m.h, m.Fpk_N / 1e3, "o", color=ORANGE, ms=8, label="实测 data1（ρ0.4）")
    hh = heights[heights > 1.0]
    fb = IMPACTOR_MASS * np.sqrt(2 * 9.81 * hh) / BARE_PULSE_S / 1e3
    chr_ = P["C"] * t ** P["alpha"] * 0.4 ** P["beta"] * fb ** P["gamma"]
    ax.plot(hh, chr_, "o", mfc="none", mec=ORANGE, mew=1.5, ms=8, label="CHR 目标（> 1 m）")
    ax.set_title(f"{t} mm 护具", color=INK)
    for h, a in zip(heights, f):
      mm = m[np.isclose(m.h, h)]
      ref = mm.Fpk_N.iloc[0] / 1e3 if len(mm) else (P["C"] * t ** P["alpha"] * 0.4 ** P["beta"] * (IMPACTOR_MASS * np.sqrt(2 * 9.81 * h) / BARE_PULSE_S / 1e3) ** P["gamma"] if h > 1 else np.nan)
      table.append((t, h, a, ref, "data1" if len(mm) else ("CHR" if h > 1 else "")))
  ax.axvspan(0.1, 1.0, color=GRID, alpha=0.6, lw=0)
  ax.set_yscale("log"); ax.set_xlabel("落高 (m)"); ax.grid(axis="y", color=GRID, lw=0.8)
  if i == 0:
    ax.set_ylabel("峰值力 (kN)")
  ax.legend(fontsize=7, frameon=False, loc="upper left")

# bare 1 m time history: sim vs non.csv
ax = fig.add_subplot(gs[1, :])
m_ = mujoco.MjModel.from_xml_string(build_xml(params[0], DT)); d = mujoco.MjData(m_)
v = np.sqrt(2 * 9.81 * 1.0); d.qpos[2] = IMPACTOR_RADIUS + 1.5 * DT * v; d.qvel[2] = -v
f6 = np.zeros(6); ts, fs = [], []
for _ in range(40):
  mujoco.mj_step(m_, d); fn = 0
  for k in range(d.ncon):
    mujoco.mj_contactForce(m_, d, k, f6); fn += f6[0]
  ts.append(d.time); fs.append(fn / 1e3)
ts, fs = np.array(ts), np.array(fs)
non = pd.read_csv("/home/wang22/engineai/engineai_ros2_workspace/scripts/FreeBallTest/non.csv")
tn = non["time(s)"].to_numpy(); fn_ = non["Force(kN)"].to_numpy()
ipk = fn_.argmax(); tn_ms = (tn - tn[ipk]) * 1e3
ts_ms = (ts - ts[fs.argmax()]) * 1e3
ax.plot(tn_ms, fn_, color=ORANGE, lw=2, marker="o", ms=3, label="实测 non.csv（采样 0.04 ms）")
ax.step(ts_ms, fs, where="mid", color=BLUE, lw=2, label="仿真 0 mm（0.5 ms 一步）")
ax.set_xlim(-2, 3); ax.set_xlabel("相对峰值时刻 (ms)"); ax.set_ylabel("力 (kN)")
ax.set_title("钢球 1 m 砸钢地：力-时间曲线", color=INK); ax.grid(axis="y", color=GRID, lw=0.8)
ax.legend(fontsize=8, frameon=False)
fig.suptitle("候选接触参数（dt 0.5 ms）落锤仿真 vs 实测；灰底为实测数据覆盖的 0.1–1 m", color=INK, fontsize=11)
out = "runs/figs/candidate_drop_vs_measured.png"
fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="#fcfcfb")
print(out)
df = pd.DataFrame(table, columns=["t_mm", "h", "sim_kN", "ref_kN", "ref"])
df["ratio"] = df.sim_kN / df.ref_kN
print(df[df.ref != ""].round(2).to_string(index=False))
print("bare sim 1 m peak", fs.max().round(1), "kN; pulse > half peak:", int((fs > fs.max() / 2).sum()), "steps")
