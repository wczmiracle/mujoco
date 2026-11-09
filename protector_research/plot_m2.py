"""Plot M2 fit accuracy vs timestep and the joint-model fit at a chosen timestep."""
import glob, json, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

out = Path("runs/figs"); out.mkdir(parents=True, exist_ok=True)
rows = [r for f in glob.glob("runs/m2_*_dt*.json") for r in json.load(open(f))]

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for mode, mk in [("group", "o"), ("joint", "s")]:
  for label, sel in [("bare", lambda r: r["t_mm"] == 0), ("pads", lambda r: r["t_mm"] != 0)]:
    rs = [r for r in rows if r["mode"] == mode and sel(r)]
    if not rs: continue
    dts = sorted({r["dt"] for r in rs})
    worst = [max(r["max_abs_log_err"] for r in rs if r["dt"] == dt) for dt in dts]
    ax[0].plot(np.array(dts) * 1e3, np.exp(worst) - 1, marker=mk, label=f"{mode} {label}")
ax[0].axhline(np.exp(0.25) - 1, color="gray", ls="--", lw=1, label="criterion ±28%")
ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].invert_xaxis()
ax[0].set_xlabel("timestep (ms)"); ax[0].set_ylabel("worst relative peak-force error")
ax[0].set_title("M2 accuracy vs timestep (constrained)"); ax[0].legend(fontsize=8)

dt = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0005
for r in [r for r in rows if r["mode"] == "joint" and r["dt"] == dt]:
  for t in sorted({x["t_mm"] for x in r["rows"]}):
    xs = [x for x in r["rows"] if x["t_mm"] == t]
    l, = ax[1].plot([x["v"] for x in xs], [x["meas_kN"] for x in xs], "o", label=f"{t:g}mm ρ{r['density']:g} meas")
    ax[1].plot([x["v"] for x in xs], [x["sim_kN"] for x in xs], "-", color=l.get_color())
ax[1].set_yscale("log"); ax[1].set_xlabel("impact speed (m/s)"); ax[1].set_ylabel("peak force (kN)")
ax[1].set_title(f"joint model, dt={dt*1e3:g} ms (dots: measured, lines: sim)"); ax[1].legend(fontsize=6, ncol=2)
fig.tight_layout(); p = out / f"m2_fit_summary.png"; fig.savefig(p, dpi=130); print(p)
