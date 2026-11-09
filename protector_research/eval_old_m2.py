"""Score the earlier hand-tuned M2 contact parameters against the measured drop tests."""
import numpy as np
from drop import Contact, load_targets, peak_force

OLD = {  # from scripts/FreeBallTest/iron_ball_drop.xml comments
    0: Contact((0.0005, 1), (0.9, 0.95, 0.001)),
    6: Contact((0.0015, 1), (0.7, 0.95, 0.001)),
    12: Contact((0.0034, 1), (0.7, 0.95, 0.001)),
    18: Contact((0.0041, 1), (0.7, 0.95, 0.001)),
    24: Contact((0.0085, 1), (0.7, 0.95, 0.001)),
}
tg = load_targets()
tg = tg[(tg.density == 0) | (tg.density == 0.4) | ((tg.density == 0.3) & (tg.t_mm == 18))]
for dt in [0.002, 0.0005, 0.0001, 0.00001]:
  errs = []
  print(f"\n=== dt={dt}")
  print(f"{'t':>3} {'rho':>4} {'h':>4} {'meas kN':>8} {'sim kN':>8} {'ratio':>6}")
  for _, r in tg.iterrows():
    f = peak_force(OLD[int(r.t_mm)], dt, r.v)
    errs.append(np.log(f / r.Fpk_N))
    print(f"{r.t_mm:3.0f} {r.density:4.1f} {r.h:4.1f} {r.Fpk_N/1e3:8.2f} {f/1e3:8.2f} {f/r.Fpk_N:6.2f}")
  e = np.abs(errs)
  print(f"mean |ln err| = {e.mean():.3f}  max = {e.max():.3f}")
