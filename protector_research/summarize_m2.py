"""Summarize constrained M2 fits: accuracy and constraint status per timestep."""
import glob, json, re
rows = []
for f in sorted(glob.glob("runs/m2_*_dt*.json")):
  for r in json.load(open(f)):
    rows.append(r)
def key(r): return (r["mode"], -r["dt"], r["density"], r["t_mm"] or 0)
print(f"{'mode':5} {'dt':>7} {'rho':>4} {'t':>4} {'rms':>6} {'max':>6} {'ok':>3}  worst rows (h: sim/meas, pen/limit mm, e)")
for r in sorted(rows, key=key):
  worst = sorted(r["rows"], key=lambda x: -abs(__import__('math').log(x["sim_kN"]/x["meas_kN"])))[:2]
  w = "; ".join(f"t{x['t_mm']:g} h{x['h']:g}: {x['sim_kN']/x['meas_kN']:.2f}, {x['pen_mm']:.1f}/{x['pen_limit_mm']:.1f}, e{x['restitution']:.2f}" for x in worst)
  t = "-" if r["t_mm"] is None else f"{r['t_mm']:g}"
  print(f"{r['mode']:5} {r['dt']:7g} {r['density']:4g} {t:>4} {r['rms_log_err']:6.3f} {r['max_abs_log_err']:6.3f} {'Y' if r['constraints_ok'] else 'N':>3}  {w}")
