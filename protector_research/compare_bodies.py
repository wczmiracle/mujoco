"""Per-body comparison of a protected config against its bare reference, with the CHR expectation.

CHR (foam, fitted_parameters.json): F_after = C * t^alpha * rho^beta * F_before^gamma (kN, mm).
Usage: .venv/bin/python compare_bodies.py ROOT BARE_CFG PROT_CFG LINKS_YAML
"""
import collections, glob, json, sys
import numpy as np, yaml

root, bare_cfg, prot_cfg, links_yaml = sys.argv[1:5]
chr_p = json.load(open("/home/wang22/engineai/engineai_ros2_workspace/scripts/ThicknessCalculate/fitted_parameters.json"))
thick = {k: v["thickness_mm"] for k, v in yaml.safe_load(open(links_yaml))["links"].items()}


def per_body(cfg):
  agg = collections.defaultdict(lambda: collections.defaultdict(list))
  for f in glob.glob(f"{root}/{cfg}/*/score.json"):
    for b, v in json.load(open(f))["bodies"].items():
      for k, x in v.items():
        agg[b][k].append(x)
  return agg


B, P = per_body(bare_cfg), per_body(prot_cfg)
print(f"{'body':22} {'t':>3} {'n b/p':>6} {'F bare':>7} {'F prot':>7} {'ratio':>6} {'CHR':>6} | {'pen b':>5} {'pen p':>5} {'lim':>5} | {'slip b':>6} {'slip p':>6} | {'vn b':>5} {'vn p':>5}")
for b in sorted(set(B) & set(P), key=lambda b: -np.median(B[b]["fn_peak_kN"])):
  t = thick.get(b, 0.0)
  fb, fp = np.median(B[b]["fn_peak_kN"]), np.median(P[b]["fn_peak_kN"])
  chr_ratio = (chr_p["C"] * t ** chr_p["alpha"] * 0.4 ** chr_p["beta"] * fb ** chr_p["gamma"] / fb) if t >= 6 else float("nan")
  lim = 0.9 * t if t > 0 else 2.0
  print(f"{b:22} {t:3.0f} {len(B[b]['fn_peak_kN']):2d}/{len(P[b]['fn_peak_kN']):<3d} {fb:7.1f} {fp:7.1f} {fp/fb:6.2f} {min(chr_ratio,9.99):6.2f} | "
        f"{np.median(B[b]['pen_max_mm']):5.1f} {max(P[b]['pen_max_mm']):5.1f} {lim:5.1f} | "
        f"{np.median(B[b]['slip_mm']):6.0f} {np.median(P[b]['slip_mm']):6.0f} | {np.median(B[b]['vn_impact']):5.2f} {np.median(P[b]['vn_impact']):5.2f}")
