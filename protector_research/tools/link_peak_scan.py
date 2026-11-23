"""Per-run, per-link peak ground-contact force (non-foot links, world contacts) from a run tar.
Usage: link_peak_scan.py <tar> <out.csv>"""
import io, sys, tarfile
from pathlib import Path
import pandas as pd

FOOT = ("FOOT", "ANKLE", "TOE", "HEEL")


def main(tarp, out):
  rows = []
  with tarfile.open(tarp, "r|gz") as tf:
    for mem in tf:
      if not mem.name.endswith("/contact_data.csv"):
        continue
      run = str(Path(mem.name).parent)
      c = pd.read_csv(io.BytesIO(tf.extractfile(mem).read()), usecols=["timestamp", "body1_name", "body2_name", "force_magnitude"])
      w = c.body1_name.astype(str).str.lower().eq("world")
      if not w.any():
        continue
      c = c[w].assign(link=c.loc[w, "body2_name"].astype(str).str.upper())
      c = c[~c.link.map(lambda s: any(k in s for k in FOOT))]
      for link, g in c.groupby("link"):
        f = g.groupby("timestamp").force_magnitude.sum()  # several contacts of one link at one instant
        rows.append(dict(run=run, link=link, Fmax=float(g.force_magnitude.max()), Fsum_max=float(f.max())))
  pd.DataFrame(rows).to_csv(out, index=False)
  print(out, len(rows), flush=True)


if __name__ == "__main__":
  main(*sys.argv[1:])
