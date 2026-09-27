"""Candidate generation from kNN tables (forward and reverse ranks), optional exact-key blocks.

Output: artifacts/cand/<out>_<split>.parquet with i1, i2 and retrieval features.
"""
import argparse

import polars as pl

from common import ART, done, log


def from_knn(tags, split, kf, kr):
    """Union of kNN tables; each tag contributes cos_<tag>, rf_<tag>, rr_<tag>."""
    out = None
    for t in tags:
        k = pl.read_parquet(ART / "knn" / f"{t}_{split}.parquet")
        k = k.filter((pl.col("rf") < kf) | (pl.col("rr") < kr)).rename(
            {"cos": f"cos_{t}", "rf": f"rf_{t}", "rr": f"rr_{t}"}
        )
        out = k if out is None else out.join(k, on=["i1", "i2"], how="full", coalesce=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--knn", required=True, help="comma separated knn tags")
    ap.add_argument("--kf", type=int, default=10)
    ap.add_argument("--kr", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default="train,test")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    (ART / "cand").mkdir(exist_ok=True)
    tags = a.knn.split(",")
    for split in a.splits.split(","):
        path = ART / "cand" / f"{a.out}_{split}.parquet"
        if done(path, a.force):
            continue
        c = from_knn(tags, split, a.kf, a.kr)
        # a single cos column for context features: the first tag
        c = c.with_columns(pl.col(f"cos_{tags[0]}").fill_null(0).alias("cos"))
        c.write_parquet(path)
        log("wrote", path, c.height)


if __name__ == "__main__":
    main()
