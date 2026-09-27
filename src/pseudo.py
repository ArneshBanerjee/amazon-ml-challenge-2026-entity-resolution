"""Pseudo-labels for countries that have no training labels (France in this test set).

Takes very confident test pairs (one-to-one winner with high probability and a clear margin) for
every test country that never appears in train, and
  1. mines alias maps for that country with the same code used on the training pairs
     (for example department vs region, street abbreviations), merged into artifacts/maps.json
  2. writes the pairs to artifacts/pseudo_<pred>.parquet for optional fine-tuning

This only uses the provided test files and our own predictions.
"""
import argparse
import json

import polars as pl

from common import ART, log
from collections import Counter

from rapidfuzz import fuzz

import normalize
from normalize import ORD, addr_components, clean_punct, mine_pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--p", type=float, default=0.98)
    ap.add_argument("--margin", type=float, default=0.9)
    ap.add_argument("--neg-p", type=float, default=0.02)
    ap.add_argument("--n", type=int, default=150_000, help="pseudo positives and negatives to keep")
    ap.add_argument("--no-maps", action="store_true")
    ap.add_argument("--min-comp", type=int, default=500)
    a = ap.parse_args()
    train_c = set(pl.read_parquet(ART / "records_train.parquet", columns=["country"])["country"].unique().to_list())
    rec = pl.read_parquet(ART / "records_test.parquet").with_row_index("i")
    new_c = sorted(set(rec["country"].unique().to_list()) - train_c)
    log("countries without labels:", new_c)
    p = pl.read_parquet(ART / "pred" / f"{a.pred}_test.parquet")
    p = p.with_columns(pl.col("p").top_k(2).over("i2", mapping_strategy="join").alias("t2")).with_columns(
        (pl.col("p") - pl.col("t2").list.get(1, null_on_oob=True).fill_null(0)).alias("margin")
    )
    conf = p.filter((pl.col("p") >= a.p) & (pl.col("margin") >= a.margin)).select("i1", "i2")
    r1 = rec.select(pl.col("i").alias("i1"), pl.col("name").alias("n1"), pl.col("addr").alias("a1"), "country")
    r2 = rec.select(pl.col("i").alias("i2"), pl.col("name").alias("n2"), pl.col("addr").alias("a2"))
    pairs = conf.join(r1, on="i1").join(r2, on="i2").filter(pl.col("country").is_in(new_c))
    log("confident pairs in new countries", pairs.height)
    # pseudo negatives: candidate pairs of the same countries the model clearly rejects
    ctry = rec.select(pl.col("i").alias("i1"), "country")
    neg = p.join(ctry, on="i1").filter(pl.col("country").is_in(new_c) & (pl.col("p") <= a.neg_p)).select("i1", "i2")
    pos = pairs.select("i1", "i2")
    out = pl.concat([
        pos.sample(min(a.n, pos.height), seed=0).with_columns(pl.lit(1, pl.Int8).alias("y")),
        neg.sample(min(a.n, neg.height), seed=0).with_columns(pl.lit(0, pl.Int8).alias("y")),
    ])
    out.write_parquet(ART / f"pseudo_{a.pred}.parquet")
    log("pseudo pairs written", out.height, "neg available", neg.height)
    if a.no_maps:
        return
    maps = json.loads((ART / "maps.json").read_text())
    maps.setdefault("drop_comp", {})
    # optional address components: frequent, and present on only one side of most confident pairs
    # (for France: region names that records often drop, departments that only records use)
    for c in new_c:
        both, one = Counter(), Counter()
        sub = pairs.filter(pl.col("country") == c)
        for a1, a2 in zip(sub["a1"].to_list(), sub["a2"].to_list()):
            c1 = {clean_punct(ORD.sub(r"\1", x)) for x in addr_components(a1)[0]}
            c2 = {clean_punct(ORD.sub(r"\1", x)) for x in addr_components(a2)[0]}
            if not c1 or not c2:
                continue
            both.update(c1 & c2)
            one.update(c1 ^ c2)
        shared = [k for k, v in both.most_common(200) if v >= a.min_comp and v / (v + one[k]) >= 0.5]
        drop = sorted(k for k in set(both) | set(one)
                      if both[k] + one[k] >= a.min_comp and one[k] / (both[k] + one[k]) >= 0.5
                      and not any(ch.isdigit() for ch in k)
                      # a spelling variant of a shared component (st nazaire vs saint nazaire) is an alias, not optional
                      and max((fuzz.ratio(k, b) for b in shared), default=0) < 70)
        maps["drop_comp"][c] = drop
        normalize.MAPS["drop_comp"][c] = drop
        log("optional components for", c, drop)
    mined = mine_pairs(pairs)
    for kind, d in mined.items():
        for c, m in d.items():
            maps[kind][c] = m
            log("mined", kind, c, len(m), list(m.items())[:30])
    (ART / "maps.json").write_text(json.dumps(maps, ensure_ascii=False, indent=0, sort_keys=True))


if __name__ == "__main__":
    main()
