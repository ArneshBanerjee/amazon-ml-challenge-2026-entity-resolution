"""Combine two finished outputs by country.

Countries that appear in train take their rows from the ensemble output; countries without training
labels take their rows from a single run. Reason: on the public leaderboard, accepting extra
low-margin pairs for the unlabelled country lowered the score, and almost all extra pairs that the
ensemble accepts for that country are of exactly that kind (see the documentation). The candidate
file is the union of both candidate files, so every matched id is a candidate.

Usage: python splice.py --labelled <dir> --unlabelled <dir> --out-dir <dir>
"""
import argparse
from pathlib import Path

import polars as pl

from common import ART, log, read_tsv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labelled", required=True, help="output folder used for countries present in train")
    ap.add_argument("--unlabelled", required=True, help="output folder used for countries absent from train")
    ap.add_argument("--unlabelled2", default="", help="second single-run output folder (for --mode intersect / union)")
    ap.add_argument("--mode", default="single", choices=["single", "intersect", "union"],
                    help="unlabelled countries: rows of one run, pairs both runs accept, or pairs either run accepts")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    train_c = set(pl.read_parquet(ART / "records_train.parquet", columns=["country"])["country"].unique().to_list())
    s1 = pl.read_parquet(ART / "records_test.parquet", columns=["entity_id", "src", "country"]).filter(pl.col("src") == 1)
    s1 = s1.select(pl.col("entity_id").alias("source1_entity_id"), pl.col("country").is_in(list(train_c)).alias("lab"))
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    m_lab = read_tsv(Path(a.labelled) / "matching_results.tsv")
    m_unl = read_tsv(Path(a.unlabelled) / "matching_results.tsv")
    if a.mode != "single":
        def pairs(df):
            return (df.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
                    .filter(pl.col("matched_entity_ids") != ""))
        p1 = pairs(m_unl)
        p2 = pairs(read_tsv(Path(a.unlabelled2) / "matching_results.tsv"))
        pp = p1.join(p2, on=["source1_entity_id", "matched_entity_ids"]) if a.mode == "intersect" else pl.concat([p1, p2]).unique()
        m_unl = pp.group_by("source1_entity_id").agg(pl.col("matched_entity_ids").sort().str.join(","))
        log("unlabelled countries:", a.mode, "pairs", pp.height)
    m = (s1.join(m_lab, on="source1_entity_id", how="left")
         .join(m_unl, on="source1_entity_id", how="left", suffix="_u")
         .select("source1_entity_id",
                 pl.when(pl.col("lab")).then(pl.col("matched_entity_ids")).otherwise(pl.col("matched_entity_ids_u"))
                 .fill_null("").alias("matched_entity_ids")))
    c_lab = read_tsv(Path(a.labelled) / "candidate_pairs.tsv")
    c_unl = read_tsv(Path(a.unlabelled) / "candidate_pairs.tsv")
    if a.mode != "single":
        c_unl = pl.concat([c_unl, read_tsv(Path(a.unlabelled2) / "candidate_pairs.tsv")])

    def explode(df):
        return (df.with_columns(pl.col("candidate_entity_ids").str.split(","))
                .explode("candidate_entity_ids").filter(pl.col("candidate_entity_ids") != ""))

    c = pl.concat([explode(c_lab), explode(c_unl)]).unique()
    c = c.group_by("source1_entity_id").agg(pl.col("candidate_entity_ids").sort().str.join(","))
    c = s1.select("source1_entity_id").join(c, on="source1_entity_id", how="left").with_columns(
        pl.col("candidate_entity_ids").fill_null(""))
    for df, name, col in ((m, "matching_results.tsv", "matched_entity_ids"), (c, "candidate_pairs.tsv", "candidate_entity_ids")):
        with open(out / name, "w") as f:
            f.write(f"source1_entity_id\t{col}\n")
            for x, y in df.select("source1_entity_id", col).iter_rows():
                f.write(f"{x}\t{y}\n")
    log("wrote", out, "labelled rows", int(s1["lab"].sum()), "unlabelled rows", int((~s1["lab"]).sum()))


if __name__ == "__main__":
    main()
