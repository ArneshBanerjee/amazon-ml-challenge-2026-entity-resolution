"""Group (context) features from a first-stage prediction table.

A true record usually agrees with the other confident records of the same S1 entity, while a
distractor is a perturbed copy of the entity with a deviation nobody else shares. For every
candidate pair (s1, r) we compare r with the other "strong" records of s1 (stage-1 p >= 0.5 and
r' assigned to s1 under one-to-one), and add record-side competition features.

Output: artifacts/group/<out>_<split>.parquet (i1, i2, g_* features)
"""
import argparse

import numpy as np
import polars as pl
import torch
from rapidfuzz import fuzz

from common import ART, done, log
from features import cp, norm_table


@torch.no_grad()
def pair_cos(emb_tag, split, a, b, chunk=1_000_000):
    """Cosine of row pairs. The embedding stays in CPU memory; chunks go to the GPU."""
    e = np.load(ART / "emb" / f"{emb_tag}_{split}.npy")
    out = np.empty(len(a), dtype=np.float32)
    for s in range(0, len(a), chunk):
        x = torch.from_numpy(e[a[s : s + chunk].astype(np.int64)]).cuda().float()
        y = torch.from_numpy(e[b[s : s + chunk].astype(np.int64)]).cuda().float()
        out[s : s + chunk] = (x * y).sum(1).cpu().numpy()
    del e
    torch.cuda.empty_cache()
    return out


def build(pred, split, emb_tag):
    p = pl.read_parquet(ART / "pred" / f"{pred}_{split}.parquet")
    p = p.with_columns(
        (pl.col("p") == pl.col("p").max().over("i2")).alias("is_arg"),
        pl.col("p").top_k(2).over("i2", mapping_strategy="join").alias("top2"),
    ).with_columns(
        pl.when(pl.col("is_arg")).then(pl.col("top2").list.get(1, null_on_oob=True)).otherwise(pl.col("top2").list.get(0))
        .fill_null(0).alias("g_p_other_best"),
    ).drop("top2")
    p = p.with_columns(
        (pl.col("p") - pl.col("g_p_other_best")).alias("g_p_margin"),
        pl.col("p").sum().over("i1").alias("g_psum1"),
        (pl.col("p") >= 0.5).sum().over("i1").cast(pl.Float32).alias("g_nstrong1"),
        pl.col("p").rank("ordinal", descending=True).over("i1").cast(pl.Float32).alias("g_prank1"),
        pl.len().over("i2").cast(pl.Float32).alias("g_n2"),
    )
    strong = p.filter((pl.col("p") >= 0.5) & pl.col("is_arg")).select("i1", pl.col("i2").alias("j"))
    pairs = p.select("i1", "i2").join(strong, on="i1").filter(pl.col("i2") != pl.col("j"))
    log(split, "group pairs", pairs.height)
    n = norm_table(split)
    a, b = pairs["i2"].to_numpy(), pairs["j"].to_numpy()
    A, B = n[a], n[b]
    f1 = A["nums"].str.split(" ").list.first()
    f2 = B["nums"].str.split(" ").list.first()
    pairs = pairs.with_columns(
        pl.Series("rc", pair_cos(emb_tag, split, a, b)),
        pl.Series("rn", cp(A["name_n"].to_list(), B["name_n"].to_list(), fuzz.token_sort_ratio)),
        pl.Series("rcore", cp(A["core"].to_list(), B["core"].to_list(), fuzz.token_set_ratio)),
        pl.Series("ra", np.where((A["addr_n"] == "").to_numpy() | (B["addr_n"] == "").to_numpy(), np.nan,
                                 cp(A["addr_n"].to_list(), B["addr_n"].to_list(), fuzz.token_set_ratio)).astype(np.float32)),
        pl.Series("rnum", ((f1 == f2) & (f1 != "")).cast(pl.Float32).to_numpy()),
        pl.Series("rnum_ok", ((f1 != "") & (f2 != "")).to_numpy()),
    )
    g = pairs.group_by("i1", "i2").agg(
        pl.col("rc").max().alias("g_rc_max"), pl.col("rc").mean().alias("g_rc_mean"),
        pl.col("rn").max().alias("g_rn_max"), pl.col("rn").mean().alias("g_rn_mean"),
        pl.col("rcore").max().alias("g_rcore_max"), pl.col("rcore").mean().alias("g_rcore_mean"),
        pl.col("ra").max().alias("g_ra_max"), pl.col("ra").mean().alias("g_ra_mean"),
        pl.col("rnum").filter(pl.col("rnum_ok")).mean().alias("g_num_support"),
        pl.len().cast(pl.Float32).alias("g_nothers"),
    )
    # does the S1's own first number agree with its strong records (tells whether S1 is the noisy one)
    s1n = n.select(pl.col("nums").str.split(" ").list.first().alias("fn")).with_row_index("i")
    ss = strong.join(s1n.rename({"i": "i1", "fn": "f1"}), on="i1").join(s1n.rename({"i": "j", "fn": "f2"}), on="j")
    ss = ss.filter((pl.col("f1") != "") & (pl.col("f2") != "")).group_by("i1").agg(
        (pl.col("f1") == pl.col("f2")).mean().alias("g_s1num_support"))
    out = (
        p.drop("p", "is_arg").join(g, on=["i1", "i2"], how="left").join(ss, on="i1", how="left")
        .with_columns(pl.col("g_nothers").fill_null(0))
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--emb", default="r2c")
    ap.add_argument("--splits", default="train,test")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    (ART / "group").mkdir(exist_ok=True)
    for split in a.splits.split(","):
        path = ART / "group" / f"{a.out}_{split}.parquet"
        if done(path, a.force):
            continue
        build(a.pred, split, a.emb).write_parquet(path)
        log("wrote", path)


if __name__ == "__main__":
    main()
