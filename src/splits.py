"""Split train S1 entities into A (80%), B (10%), C (10%), stratified by country.

C is further split into C1 (tuning, 80% of C) and C2 (untouched final check).
Every S2/S3 record inherits the group of its true S1. Distractors get group "X".
Output: artifacts/split_s1.parquet (s1, country, grp), artifacts/split_rec.parquet (rec, grp)
"""
import numpy as np
import polars as pl

from common import ART, log

SEED = 20260927


def main():
    rec = pl.read_parquet(ART / "records_train.parquet", columns=["entity_id", "src", "country"])
    s1 = rec.filter(pl.col("src") == 1).select(pl.col("entity_id").alias("s1"), "country").sort("s1")
    rng = np.random.default_rng(SEED)
    grp = np.empty(s1.height, dtype=object)
    for c in s1["country"].unique().sort().to_list():
        idx = np.where((s1["country"] == c).to_numpy())[0]
        idx = rng.permutation(idx)
        n = len(idx)
        a, b = int(0.8 * n), int(0.9 * n)
        c1 = b + int(0.8 * (n - b))
        grp[idx[:a]] = "A"
        grp[idx[a:b]] = "B"
        grp[idx[b:c1]] = "C1"
        grp[idx[c1:]] = "C2"
    s1 = s1.with_columns(pl.Series("grp", grp.astype(str)))
    s1.write_parquet(ART / "split_s1.parquet")
    log(s1.group_by("country", "grp").len().sort("country", "grp"))

    gt = pl.read_parquet(ART / "gt_pairs.parquet")
    others = rec.filter(pl.col("src") != 1).select(pl.col("entity_id").alias("rec"))
    assert gt["rec"].is_in(others["rec"].implode()).all(), "gt id missing from S2/S3"
    r = others.join(gt.join(s1.select("s1", "grp"), on="s1"), on="rec", how="left").with_columns(
        pl.col("grp").fill_null("X")
    )
    r.select("rec", "s1", "grp").write_parquet(ART / "split_rec.parquet")
    log(r.group_by("grp").len().sort("grp"))


if __name__ == "__main__":
    main()
