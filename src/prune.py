"""Candidate generation v2: union of multi-view kNN lists plus exact-key blocks, then a cheap
LightGBM pruning model. Its output is the final candidate set (the input of the matcher).

Steps per split:
  1. union of kNN tables for each view (forward rank < kf or reverse rank < kr)
  2. exact-key blocks (acronym of S1 core = record core, domain stem = S1 core without spaces),
     capped per key so frequent keys do not explode. Same-core-name blocks were tried and dropped:
     the kNN lists already contain almost all of those pairs.
  3. cheap features: cosine in every view (recomputed from embeddings for all pairs), ranks in both
     directions, a few rapidfuzz scores, rank/gap context
  4. LightGBM trained on split-B entities, keep pairs with score >= threshold (chosen on C1 for the
     target pair recall), with a per-S1 cap

Outputs:
  artifacts/cand/<out>_pre_<split>.parquet  (all union pairs with cheap features and prune score)
  artifacts/cand/<out>_<split>.parquet      (pruned candidate set)
"""
import os
import argparse
import json

import lightgbm as lgb
import numpy as np
import polars as pl
import torch
from rapidfuzz import fuzz

from common import ART, done, log
from evaluate import blocking, s1_group
from features import cp, norm_table

VIEWS = {"c": "combined", "n": "name", "a": "addr"}


def acronym(s):
    return "".join(t[0] for t in s.split() if t)


def key_blocks(split, cap=20):
    n = norm_table(split).with_row_index("i").select("i", "src", "country", "core", "core_ns", "dom")
    n = n.with_columns(pl.col("core").map_elements(acronym, return_dtype=pl.String).alias("acr"))
    s1 = n.filter(pl.col("src") == 1)
    oth = n.filter(pl.col("src") != 1)
    out = []
    specs = [
        ("acr", "core_ns"),        # S1 acronym equals record core (GAS)
        ("core_ns", "dom"),        # domain stem equals S1 core without spaces
    ]
    for k1, k2 in specs:
        a = s1.filter(pl.col(k1).str.len_chars() >= 3).select(pl.col("i").alias("i1"), "country", pl.col(k1).alias("k"))
        b = oth.filter(pl.col(k2).str.len_chars() >= 3).select(pl.col("i").alias("i2"), "country", pl.col(k2).alias("k"))
        a = a.filter(pl.len().over("country", "k") <= cap)
        b = b.filter(pl.len().over("country", "k") <= cap)
        j = a.join(b, on=["country", "k"]).select("i1", "i2")
        log("key block", k1, k2, j.height)
        out.append(j)
    return pl.concat(out).unique().with_columns(pl.lit(1, pl.Int8).alias("kb"))


def union(tags, split, kf, kr):
    out = None
    for v, t in tags.items():
        k = pl.read_parquet(ART / "knn" / f"{t}_{split}.parquet", columns=["i1", "i2", "rf", "rr"])
        k = k.filter((pl.col("rf") < kf[v]) | (pl.col("rr") < kr[v])).rename({"rf": f"rf_{v}", "rr": f"rr_{v}"})
        out = k if out is None else out.join(k, on=["i1", "i2"], how="full", coalesce=True)
        log("union", v, out.height)
    return out


@torch.no_grad()
def cosines(tags, split, df, chunk=1_000_000):
    """Cosine of every pair in every view. Embeddings stay in CPU memory; chunks go to the GPU."""
    i1 = df["i1"].to_numpy().astype(np.int64)
    i2 = df["i2"].to_numpy().astype(np.int64)
    res = {}
    for v, t in tags.items():
        e = np.load(ART / "emb" / f"{t}_{split}.npy")
        out = np.empty(len(i1), dtype=np.float32)
        for s in range(0, len(i1), chunk):
            a = torch.from_numpy(e[i1[s : s + chunk]]).cuda().float()
            b = torch.from_numpy(e[i2[s : s + chunk]]).cuda().float()
            out[s : s + chunk] = (a * b).sum(1).cpu().numpy()
        res[f"cos_{v}"] = out
        del e
        torch.cuda.empty_cache()
    return df.with_columns([pl.Series(k, v) for k, v in res.items()])


def cheap_features(split, df):
    n = norm_table(split)
    A = n[df["i1"].to_numpy()]
    B = n[df["i2"].to_numpy()]
    f = {
        "q_n_tsort": cp(A["name_n"].to_list(), B["name_n"].to_list(), fuzz.token_sort_ratio),
        "q_c_ratio": cp(A["core"].to_list(), B["core"].to_list(), fuzz.ratio),
        "q_a_tset": cp(A["addr_n"].to_list(), B["addr_n"].to_list(), fuzz.token_set_ratio),
        "q_num_pset": cp(A["nums"].to_list(), B["nums"].to_list(), fuzz.token_set_ratio),
        "q_b_addr_empty": (B["addr_n"] == "").cast(pl.Float32).to_numpy(),
        "q_src": B["src"].cast(pl.Float32).to_numpy(),
    }
    df = df.with_columns([pl.Series(k, v) for k, v in f.items()])
    ctx = []
    for c in ["cos_c", "cos_n", "cos_a", "q_n_tsort", "q_a_tset"]:
        ctx += [
            pl.col(c).rank("ordinal", descending=True).over("i1").cast(pl.Float32).alias(f"{c}_r1"),
            (pl.col(c).max().over("i1") - pl.col(c)).alias(f"{c}_g1"),
            pl.col(c).rank("ordinal", descending=True).over("i2").cast(pl.Float32).alias(f"{c}_r2"),
            (pl.col(c).max().over("i2") - pl.col(c)).alias(f"{c}_g2"),
        ]
    df = df.with_columns(ctx).with_columns(
        pl.len().over("i1").cast(pl.Float32).alias("n1"), pl.len().over("i2").cast(pl.Float32).alias("n2")
    )
    return df


def build_pre(tags, split, kf, kr, force):
    path = ART / "cand" / f"{OUT}_pre_{split}.parquet"
    if done(path, force):
        return pl.read_parquet(path)
    u = union(tags, split, kf, kr)
    kb = key_blocks(split)
    u = u.join(kb, on=["i1", "i2"], how="full", coalesce=True)
    u = u.with_columns([pl.col(c).fill_null(255) for c in u.columns if c.startswith(("rf_", "rr_"))]).with_columns(
        pl.col("kb").fill_null(0)
    )
    log(split, "union + key blocks", u.height)
    u = cosines(tags, split, u)
    u = cheap_features(split, u)
    u.write_parquet(path)
    return u


OUT = "v2"


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="c:r2c,n:r2n,a:r2a,o:r1")
    ap.add_argument("--kf", default="c:20,n:10,a:10,o:20")
    ap.add_argument("--kr", default="c:3,n:2,a:2,o:3")
    ap.add_argument("--out", default="v2")
    ap.add_argument("--recall", type=float, default=0.998, help="target pair recall on C1 (of union)")
    ap.add_argument("--cap", type=int, default=15)
    ap.add_argument("--test-only", action="store_true", help="reuse the saved pruning model for test")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    OUT = a.out
    (ART / "cand").mkdir(exist_ok=True)
    tags = dict(x.split(":") for x in a.tags.split(","))
    kf = {k: int(v) for k, v in (x.split(":") for x in a.kf.split(","))}
    kr = {k: int(v) for k, v in (x.split(":") for x in a.kr.split(","))}

    from stack import labels
    from select_sets import to_ids

    if not a.test_only and done(ART / "cand" / f"{OUT}_train.parquet", a.force) and done(ART / "cand" / f"{OUT}_test.parquet", a.force):
        log("skip, pruned candidates exist:", OUT)
        return
    if a.test_only:
        mdir = ART / "models" / f"prune_{OUT}"
        m = lgb.Booster(model_file=str(mdir / "m.txt"))
        cfg = json.loads((mdir / "curve.json").read_text())
        te = build_pre(tags, "test", kf, kr, a.force)
        te = te.with_columns(pl.Series("ps", m.predict(te.select(cfg["feats"]).to_numpy().astype(np.float32)).astype(np.float32)))
        te = te.with_columns(pl.col("ps").rank("ordinal", descending=True).over("i1").alias("ps_rank"))
        kte = te.filter((pl.col("ps") >= cfg["thr"]) & (pl.col("ps_rank") <= cfg["cap"]))
        kte.write_parquet(ART / "cand" / f"{OUT}_test.parquet")
        log("test pruned pairs", kte.height)
        return

    tr = labels(build_pre(tags, "train", kf, kr, a.force))
    feats = [c for c in tr.columns if c not in {"i1", "i2", "y", "grp"}]
    c1 = s1_group(["C1"])
    blocking(to_ids(tr, "train"), c1, "union C1")
    fit = tr.filter(pl.col("grp") == "B")
    log("prune fit rows", fit.height, "features", feats)
    m = lgb.train(
        dict(objective="binary", learning_rate=0.1, num_leaves=127, min_data_in_leaf=100, num_threads=int(os.environ.get("BER_THREADS", max(1, os.cpu_count() - 2))),
             verbose=-1, feature_fraction=0.9),
        lgb.Dataset(fit.select(feats).to_numpy().astype(np.float32), fit["y"].to_numpy()),
        300,
    )
    mdir = ART / "models" / f"prune_{OUT}"
    mdir.mkdir(parents=True, exist_ok=True)
    m.save_model(str(mdir / "m.txt"))
    tr = tr.with_columns(pl.Series("ps", m.predict(tr.select(feats).to_numpy().astype(np.float32)).astype(np.float32)))
    tr = tr.with_columns(pl.col("ps").rank("ordinal", descending=True).over("i1").alias("ps_rank"))

    # recall / size curve on C1
    cc = tr.filter(pl.col("grp") == "C1")
    n_true = s1_group(["C1"]).pipe(lambda s: pl.read_parquet(ART / "gt_pairs.parquet").join(s.select("s1"), on="s1").height)
    curve = []
    for t in [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]:
        k = cc.filter((pl.col("ps") >= t) & (pl.col("ps_rank") <= a.cap))
        curve.append((t, k["y"].sum() / n_true, k.height / c1.height))
    for t, r, n in curve:
        log(f"prune thr={t:<7} recall={r:.5f} avg_cand={n:.2f}")
    ok = [c for c in curve if c[1] >= a.recall]
    thr = ok[0][0] if ok else curve[-1][0]
    log("chosen threshold", thr)
    (mdir / "curve.json").write_text(json.dumps({"curve": curve, "thr": thr, "cap": a.cap, "feats": feats}))

    keep = lambda d: d.filter((pl.col("ps") >= thr) & (pl.col("ps_rank") <= a.cap))
    ktr = keep(tr).drop("y", "grp")
    ktr.write_parquet(ART / "cand" / f"{OUT}_train.parquet")
    blocking(to_ids(ktr, "train"), c1, "pruned C1")
    blocking(to_ids(ktr, "train"), s1_group(["C2"]), "pruned C2")
    del tr
    te = build_pre(tags, "test", kf, kr, a.force)
    te = te.with_columns(pl.Series("ps", m.predict(te.select(feats).to_numpy().astype(np.float32)).astype(np.float32)))
    te = te.with_columns(pl.col("ps").rank("ordinal", descending=True).over("i1").alias("ps_rank"))
    kte = keep(te)
    kte.write_parquet(ART / "cand" / f"{OUT}_test.parquet")
    log("test pruned pairs", kte.height, "per S1", kte.height / te["i1"].n_unique())


if __name__ == "__main__":
    main()
