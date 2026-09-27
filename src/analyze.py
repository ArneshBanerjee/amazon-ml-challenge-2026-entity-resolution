"""Dump false positives and false negatives on C1 for a prediction table (error analysis)."""
import sys

import polars as pl

from common import ART, ROOT
from select_sets import one_to_one
from stack import labels

pred = sys.argv[1]
thr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
p = labels(pl.read_parquet(ART / "pred" / f"{pred}_train.parquet"))
k = one_to_one(p).filter(pl.col("grp") == "C1")
raw = pl.read_parquet(ART / "records_train.parquet", columns=["entity_id", "name", "addr"]).with_row_index("i")
fp = k.filter((pl.col("p") >= thr) & (pl.col("y") == 0)).sort("p", descending=True)
# false negatives: true pairs of C1 S1s that were not selected
gt_all = p.filter((pl.col("grp") == "C1") & (pl.col("y") == 1))
sel = k.filter(pl.col("p") >= thr).select("i1", "i2")
fn = gt_all.join(sel, on=["i1", "i2"], how="anti").sort("p", descending=True)


def show(df, n, f):
    for r in df.head(n).iter_rows(named=True):
        a, b = raw.row(r["i1"]), raw.row(r["i2"])
        f.write(f"p={r['p']:.3f} | {a[2]} | {a[3]}\n          {b[1][:2]} | {b[2]} | {b[3]}\n")
        # for FPs show the record's true S1 if any
        f.write("\n")


out = ROOT / "analysis" / f"errors_{pred}.txt"
with open(out, "w") as f:
    f.write(f"C1 FP={fp.height} FN(not selected true pairs)={fn.height}\n\n== top false positives\n")
    show(fp, 60, f)
    f.write("\n== false negatives with highest p (lost to one-to-one or threshold)\n")
    show(fn, 40, f)
    f.write("\n== false negatives sample\n")
    show(fn.sample(min(40, fn.height), seed=0), 40, f)
print(out)
