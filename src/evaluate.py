"""Official metric: F0.5 per S1 entity, macro-averaged, singletons included.

Also blocking recall and average candidates per S1.
Predictions and candidates are pair tables with columns (s1, rec).
"""
import argparse

import polars as pl

from common import ART


def _gt(s1_ids):
    gt = pl.read_parquet(ART / "gt_pairs.parquet").join(s1_ids.select("s1"), on="s1")
    return gt


def per_entity(pred, s1_ids):
    """pred: (s1, rec). s1_ids: (s1, country) for the evaluated entities. Returns per-S1 frame."""
    gt = _gt(s1_ids)
    pred = pred.select("s1", "rec").unique().join(s1_ids.select("s1"), on="s1")
    tp = pred.join(gt, on=["s1", "rec"]).group_by("s1").len("tp")
    npred = pred.group_by("s1").len("npred")
    ntrue = gt.group_by("s1").len("ntrue")
    df = (
        s1_ids.select("s1", "country")
        .join(tp, on="s1", how="left")
        .join(npred, on="s1", how="left")
        .join(ntrue, on="s1", how="left")
        .with_columns(pl.col("tp", "npred", "ntrue").fill_null(0).cast(pl.Float64))
    )
    p = pl.col("tp") / pl.col("npred")
    r = pl.col("tp") / pl.col("ntrue")
    f = (
        pl.when(pl.col("ntrue") == 0)
        .then((pl.col("npred") == 0).cast(pl.Float64))
        .when(pl.col("tp") == 0)
        .then(0.0)
        .otherwise(1.25 * p * r / (0.25 * p + r))
    )
    return df.with_columns(f.alias("f05"))


def score(pred, s1_ids, name="", verbose=True):
    df = per_entity(pred, s1_ids)
    out = {"all": df["f05"].mean()}
    for c, g in df.group_by("country"):
        out[c[0]] = g["f05"].mean()
    tot = df.select(pl.col("tp").sum(), pl.col("npred").sum(), pl.col("ntrue").sum()).row(0)
    out["micro_P"] = tot[0] / max(tot[1], 1)
    out["micro_R"] = tot[0] / max(tot[2], 1)
    sing = df.filter(pl.col("ntrue") == 0)
    out["singleton_acc"] = sing["f05"].mean() if sing.height else float("nan")
    if verbose:
        print(f"[{name}] F0.5 " + " ".join(f"{k}={v:.5f}" for k, v in out.items()), flush=True)
    return out


def blocking(cand, s1_ids, name="", verbose=True):
    """Pair recall of candidates and average candidates per S1."""
    gt = _gt(s1_ids)
    cand = cand.select("s1", "rec").unique().join(s1_ids.select("s1"), on="s1")
    hit = cand.join(gt, on=["s1", "rec"]).height
    out = {"recall": hit / gt.height, "avg_cand": cand.height / s1_ids.height, "n_pairs": cand.height}
    by = {}
    for c, ids in s1_ids.group_by("country"):
        g = gt.join(ids.select("s1"), on="s1")
        cc = cand.join(ids.select("s1"), on="s1")
        by[c[0]] = (cc.join(g, on=["s1", "rec"]).height / max(g.height, 1), cc.height / ids.height)
    out["by_country"] = by
    # upper bound on F0.5 if the matcher were perfect on this candidate set
    ub = per_entity(cand.join(gt, on=["s1", "rec"]), s1_ids)["f05"].mean()
    out["f05_ceiling"] = ub
    if verbose:
        bc = " ".join(f"{k}:rec={v[0]:.5f},avg={v[1]:.2f}" for k, v in by.items())
        print(f"[{name}] blocking recall={out['recall']:.5f} avg_cand={out['avg_cand']:.2f} ceiling={ub:.5f} {bc}", flush=True)
    return out


def s1_group(groups):
    s = pl.read_parquet(ART / "split_s1.parquet")
    return s.filter(pl.col("grp").is_in(groups))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pred", help="parquet with s1, rec")
    ap.add_argument("--groups", default="C1,C2")
    a = ap.parse_args()
    score(pl.read_parquet(a.pred), s1_group(a.groups.split(",")), a.pred)
