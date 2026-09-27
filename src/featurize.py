"""Compute pair features for a candidate table, in chunks, plus context features.

Input:  artifacts/cand/<cand>_<split>.parquet  (i1, i2, retrieval columns)
Output: artifacts/feat/<cand>_<split>.parquet
"""
import argparse

import polars as pl

from common import ART, done, log
from features import context_features, pair_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True)
    ap.add_argument("--splits", default="train,test")
    ap.add_argument("--chunk", type=int, default=3_000_000)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    (ART / "feat").mkdir(exist_ok=True)
    for split in a.splits.split(","):
        out = ART / "feat" / f"{a.cand}_{split}.parquet"
        if done(out, a.force):
            log("skip", out)
            continue
        cand = pl.read_parquet(ART / "cand" / f"{a.cand}_{split}.parquet")
        parts_dir = ART / "feat" / f"{a.cand}_{split}_parts"
        parts_dir.mkdir(exist_ok=True)
        parts = []
        for k, s in enumerate(range(0, cand.height, a.chunk)):
            p = parts_dir / f"{k:04d}.parquet"
            if not p.exists():
                pair_features(split, cand.slice(s, a.chunk)).write_parquet(p)
                log(split, "chunk", k, "done")
            parts.append(p)
        df = pl.concat([pl.read_parquet(p) for p in parts])
        df = context_features(df, "cos" if "cos" in df.columns else "cos_c", "kc")
        df = context_features(df, "n_ratio", "kn")
        df = context_features(df, "a_tset", "ka")
        df.write_parquet(out)
        for p in parts:
            p.unlink()
        parts_dir.rmdir()
        log("wrote", out, df.shape)


if __name__ == "__main__":
    main()
