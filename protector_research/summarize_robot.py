"""Aggregate score.json files of a batch: per config, distribution of penetration / force / slip."""
import json, sys
from pathlib import Path
import numpy as np
root = Path(sys.argv[1]).expanduser()
print(f"{'config':16} {'n':>3} {'fell':>4} {'stable':>6} | {'pen max mm (med/p90/max)':>26} | {'Fn peak kN (med/max)':>20} | {'slip mm sum (med/max)':>21} | worst-pen body")
for cfg in sorted(p for p in root.iterdir() if p.is_dir()):
  S = [json.loads(f.read_text()) for f in sorted(cfg.glob("*/score.json"))]
  F = [s for s in S if s.get("fell")]
  if not F: print(cfg.name, "no falls"); continue
  pen = np.array([s["total"]["pen_max_mm"] for s in F]); fn = np.array([s["total"]["fn_peak_kN"] for s in F])
  sl = np.array([s["total"]["slip_mm_sum"] for s in F])
  worst = max(F, key=lambda s: s["total"]["pen_max_mm"]); wb = max(worst["bodies"].items(), key=lambda kv: kv[1]["pen_max_mm"])[0]
  print(f"{cfg.name:16} {len(S):3d} {len(F):4d} {sum(s['stable'] for s in F):6d} | {np.median(pen):8.1f} {np.percentile(pen,90):8.1f} {pen.max():8.1f} | {np.median(fn):9.2f} {fn.max():9.2f} | {np.median(sl):9.0f} {sl.max():10.0f} | {wb} ({worst['run']})")
