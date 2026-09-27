"""Read the raw TSVs once and store them as parquet.

Outputs (in artifacts/):
  records_train.parquet, records_test.parquet  columns: entity_id, src, name, addr, country
  gt_pairs.parquet                              columns: s1, rec   (one row per true pair)
  gt_s1.parquet                                 columns: s1, n_match
"""
import argparse
import subprocess

import polars as pl

from common import ART, DATA, done, log, read_tsv


def load_split(split):
    parts = []
    for s in (1, 2, 3):
        path = DATA / split / f"{split}_source{s}.tsv"
        df = read_tsv(path)
        n_lines = int(subprocess.check_output(["wc", "-l", str(path)]).split()[0])
        assert df.height == n_lines - 1, (path, df.height, n_lines)
        assert df.columns == ["entity_id", "business_name", "business_address", "country"]
        assert df["entity_id"].str.starts_with(f"S{s}-").all()
        assert df["entity_id"].n_unique() == df.height
        log(split, s, df.height, "rows ok")
        parts.append(
            df.select(
                pl.col("entity_id"),
                pl.lit(s, dtype=pl.Int8).alias("src"),
                pl.col("business_name").alias("name"),
                pl.col("business_address").alias("addr"),
                pl.col("country"),
            )
        )
    return pl.concat(parts)


def main(force=False):
    for split in ("train", "test"):
        out = ART / f"records_{split}.parquet"
        if done(out, force):
            continue
        load_split(split).write_parquet(out)
        log("wrote", out)

    if not done(ART / "gt_pairs.parquet", force):
        path = DATA / "train" / "train_ground_truth.tsv"
        gt = read_tsv(path)
        n_lines = int(subprocess.check_output(["wc", "-l", str(path)]).split()[0])
        assert gt.height == n_lines - 1
        assert gt["source1_entity_id"].n_unique() == gt.height
        gt = gt.rename({"source1_entity_id": "s1", "matched_entity_ids": "m"})
        pairs = (
            gt.filter(pl.col("m") != "")
            .with_columns(pl.col("m").str.split(","))
            .explode("m")
            .rename({"m": "rec"})
        )
        s1n = gt.select("s1", pl.when(pl.col("m") == "").then(0).otherwise(pl.col("m").str.count_matches(",") + 1).alias("n_match"))
        pairs.write_parquet(ART / "gt_pairs.parquet")
        s1n.write_parquet(ART / "gt_s1.parquet")
        log("gt pairs", pairs.height, "unique recs", pairs["rec"].n_unique(), "s1", s1n.height)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    main(ap.parse_args().force)
