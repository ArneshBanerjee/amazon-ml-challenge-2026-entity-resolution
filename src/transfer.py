"""Cross-country transfer check: train the stacker on one country's B entities, score C1 of each
country with the same selection rule. A small gap suggests the design transfers to unseen countries."""
import os
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

from common import ART
from evaluate import s1_group, score
from select_sets import one_to_one, to_ids
from stack import DROP, labels, load

feat = sys.argv[1]
extra = [e for e in (sys.argv[2] if len(sys.argv) > 2 else "").split(",") if e]
drop = {d for d in (sys.argv[3] if len(sys.argv) > 3 else "").split(",") if d}
pairs_to_run = (sys.argv[4] if len(sys.argv) > 4 else "US,India,both").split(",")
tr = labels(load(feat, extra, "train"))
ctry = pl.read_parquet(ART / "norm_train.parquet", columns=["country"]).with_row_index("i1")
tr = tr.join(ctry, on="i1")
feats = [c for c in tr.columns if c not in DROP | {"country"} | drop]
c1 = s1_group(["C1"])
for src in pairs_to_run:
    fit = tr.filter((pl.col("grp") == "B") & ((pl.col("country") == src) | (src == "both")))
    m = lgb.train(dict(objective="binary", learning_rate=0.1, num_leaves=127, min_data_in_leaf=50, num_threads=os.cpu_count(), verbose=-1),
                  lgb.Dataset(fit.select(feats).to_numpy().astype(np.float32), fit["y"].to_numpy()), 400)
    ev = tr.filter(pl.col("grp") == "C1")
    ev = ev.with_columns(pl.Series("p", m.predict(ev.select(feats).to_numpy().astype(np.float32))))
    sel = one_to_one(ev).filter(pl.col("p") >= 0.6)
    for dst in [d for d in ["US", "India"] if d != src or len(pairs_to_run) > 2]:
        score(to_ids(sel, "train"), c1.filter(pl.col("country") == dst), f"train {src} -> test {dst}")
