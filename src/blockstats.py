"""Blocking recall vs candidate count on C for a kNN table (analysis helper)."""
import sys
import polars as pl
from common import ART
from evaluate import s1_group

def truth_idx(split_groups):
    ids = pl.read_parquet(ART / "norm_train.parquet", columns=["entity_id"]).with_row_index("i")
    s = s1_group(split_groups)
    gt = pl.read_parquet(ART / "gt_pairs.parquet").join(s.select("s1", "country"), on="s1")
    gt = gt.join(ids.rename({"i": "i1", "entity_id": "s1"}), on="s1").join(ids.rename({"i": "i2", "entity_id": "rec"}), on="rec")
    s = s.join(ids.rename({"i": "i1", "entity_id": "s1"}), on="s1")
    return gt.select("i1", "i2", "country"), s

if __name__ == "__main__":
    tag = sys.argv[1]
    gt, s = truth_idx(["C1", "C2"])
    k = pl.read_parquet(ART / "knn" / f"{tag}_train.parquet").join(s.select("i1"), on="i1")
    j = gt.join(k, on=["i1", "i2"], how="left").with_columns(pl.col("rf").fill_null(255), pl.col("rr").fill_null(255))
    print("pairs in C:", gt.height, " any hit:", (j["cos"].is_not_null()).mean())
    for kf in [1, 2, 3, 5, 10, 20, 30]:
        for kr in [0, 1, 2, 3, 5, 10]:
            m = (pl.col("rf") < kf) | (pl.col("rr") < kr)
            rec = j.select(m.mean()).item()
            n = k.filter(m).height / s.height
            print(f"kf={kf:2d} kr={kr:2d} recall={rec:.5f} avg_cand={n:.2f}")
