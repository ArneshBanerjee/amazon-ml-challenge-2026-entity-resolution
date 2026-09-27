"""Turn pair probabilities into match sets.

1. One-to-one: every record goes only to the S1 where its probability is highest.
2. Isotonic calibration fitted on C1.
3. Per S1, either a global threshold or the subset that maximizes expected F0.5, tuned on C1.
Reports C1 (tuning) and C2 (untouched) scores, then writes the test submission.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression

from common import ART, ROOT, log
from evaluate import blocking, score, s1_group
from stack import labels


def one_to_one(df):
    return df.sort(["i2", "p", "i1"], descending=[False, True, False]).unique("i2", keep="first", maintain_order=True)


def pick_threshold(df, t):
    return df.filter(pl.col("q") >= t)


def pick_expected(df, bias=0.0):
    """Choose per-S1 top-k maximizing an approximation of expected F0.5. bias shifts toward
    smaller sets when positive."""
    d = df.sort(["i1", "q"], descending=[False, True]).with_columns(
        pl.col("q").cum_sum().over("i1").alias("cs"),
        pl.col("q").cum_count().over("i1").alias("k"),
        pl.col("q").sum().over("i1").alias("tot"),
        (1 - pl.col("q")).log().sum().over("i1").exp().alias("p_empty"),
    )
    d = d.with_columns((1.25 * pl.col("cs") / (pl.col("k") + 0.25 * pl.col("tot"))).alias("ef"))
    best = d.group_by("i1").agg(
        pl.col("ef").max().alias("ef_best"),
        pl.col("k").get(pl.col("ef").arg_max()).alias("k_best"),
        pl.col("p_empty").first(),
    )
    best = best.with_columns(
        pl.when(pl.col("p_empty") + bias >= pl.col("ef_best")).then(0).otherwise(pl.col("k_best")).alias("k_sel")
    )
    return d.join(best.select("i1", "k_sel"), on="i1").filter(pl.col("k") <= pl.col("k_sel"))


def pick_expected_exact(df, bias=0.0, K=16, chunk=200_000):
    """Exact expected F0.5 of each top-k set under independent calibrated probabilities q.

    For S1 i with sorted q_1..q_n, TP of the top-k set and the positives among the rest are
    independent Poisson-binomial variables. E[F(k)] = sum_a,b P(a) P(b) 1.25a / (k + 0.25(a+b)),
    and E[F(0)] = P(no positives at all). The best k is kept (bias favours smaller sets)."""
    import torch

    d = df.sort(["i1", "q"], descending=[False, True]).with_columns(
        pl.col("q").cum_count().over("i1").alias("k"))
    d = d.filter(pl.col("k") <= K)
    s1 = d["i1"].unique(maintain_order=True)
    pos = {v: j for j, v in enumerate(s1.to_list())}
    Q = np.zeros((len(s1), K), dtype=np.float32)
    Q[np.array([pos[v] for v in d["i1"].to_list()]), d["k"].to_numpy() - 1] = d["q"].to_numpy()
    best_k = np.zeros(len(s1), dtype=np.int64)
    a = torch.arange(K + 1, device="cuda", dtype=torch.float64)
    for s in range(0, len(s1), chunk):
        q = torch.from_numpy(Q[s : s + chunk]).cuda().double()
        n = q.shape[0]
        # prefix distributions: top[k][:, a] = P(TP of first k = a)
        top = [torch.zeros(n, K + 1, device="cuda", dtype=torch.float64)]
        top[0][:, 0] = 1
        for k in range(K):
            prev = top[-1]
            nxt = prev * (1 - q[:, k : k + 1])
            nxt[:, 1:] += prev[:, :-1] * q[:, k : k + 1]
            top.append(nxt)
        # suffix distributions: rest[k][:, b] = P(positives among positions k..K-1 = b)
        rest = [None] * (K + 1)
        rest[K] = torch.zeros(n, K + 1, device="cuda", dtype=torch.float64)
        rest[K][:, 0] = 1
        for k in range(K - 1, -1, -1):
            prev = rest[k + 1]
            nxt = prev * (1 - q[:, k : k + 1])
            nxt[:, 1:] += prev[:, :-1] * q[:, k : k + 1]
            rest[k] = nxt
        ef = torch.empty(n, K + 1, device="cuda", dtype=torch.float64)
        ef[:, 0] = rest[0][:, 0] + bias
        for k in range(1, K + 1):
            f = 1.25 * a[:, None] / (k + 0.25 * (a[:, None] + a[None, :]))  # (A, B)
            ef[:, k] = torch.einsum("na,nb,ab->n", top[k], rest[k], f)
        # a set larger than the number of real candidates is not allowed
        ncand = (q > 0).sum(1)
        mask = torch.arange(K + 1, device="cuda")[None, :] > ncand[:, None]
        ef[mask] = -1
        best_k[s : s + n] = ef.argmax(1).cpu().numpy()
    kk = pl.DataFrame({"i1": s1, "k_sel": best_k})
    return d.join(kk, on="i1").filter(pl.col("k") <= pl.col("k_sel"))


def unlabeled_shift(te, iso, apply, tol=0.002, fixed=None):
    """Label-shift correction for test countries that never appear in train.

    The stacker is calibrated on labelled countries and is under-confident on a new one. For each
    unlabelled country we add one logit offset to its pair scores, found by bisection so that its
    predicted matches per S1 equal the average of the labelled countries on the same test set.
    Returns the shifted pair table and the offsets."""
    train_c = set(pl.read_parquet(ART / "records_train.parquet", columns=["country"])["country"].unique().to_list())
    n = pl.read_parquet(ART / "norm_test.parquet", columns=["src", "country"]).with_row_index("i")
    s1 = n.filter(pl.col("src") == 1).select(pl.col("i").alias("i1"), "country")
    te = te.join(s1, on="i1")
    new_c = sorted(set(s1["country"].unique().to_list()) - train_c)
    logit = np.log(np.clip(te["p"].to_numpy(), 1e-7, 1 - 1e-7) / np.clip(1 - te["p"].to_numpy(), 1e-7, 1))

    def rate(df, countries):
        k = one_to_one(df)
        k = k.with_columns(pl.Series("q", iso.predict(k["p"].to_numpy()).astype(np.float32)))
        sel = apply(k)
        c = s1.filter(pl.col("country").is_in(countries))
        return c.join(sel.group_by("i1").len("n"), on="i1", how="left")["n"].fill_null(0).mean()

    def with_offsets(off):
        d = np.zeros(len(logit))
        cc = te["country"].to_numpy()
        for c, v in off.items():
            d[cc == c] = v
        return te.with_columns(pl.Series("p", (1 / (1 + np.exp(-(logit + d)))).astype(np.float32))).drop("country")

    if fixed is not None:
        off = {c: fixed for c in new_c}
        log("fixed offsets", off)
        return with_offsets(off), off
    labelled = [c for c in s1["country"].unique().to_list() if c in train_c]
    target = rate(te.drop("country"), labelled)
    off = {}
    for c in new_c:
        lo, hi = 0.0, 6.0
        if rate(with_offsets({c: 0.0}), [c]) >= target:
            off[c] = 0.0
            continue
        for _ in range(12):
            mid = (lo + hi) / 2
            r = rate(with_offsets({c: mid}), [c])
            if abs(r - target) < tol:
                lo = hi = mid
                break
            lo, hi = (mid, hi) if r < target else (lo, mid)
        off[c] = (lo + hi) / 2
        log(f"unlabelled country {c}: logit offset {off[c]:.3f} (target {target:.4f} matches per S1)")
    return with_offsets(off), off


def to_ids(df, split):
    ids = pl.read_parquet(ART / f"norm_{split}.parquet", columns=["entity_id"]).with_row_index("i")
    return (
        df.join(ids.rename({"i": "i1", "entity_id": "s1"}), on="i1")
        .join(ids.rename({"i": "i2", "entity_id": "rec"}), on="i2")
        .select("s1", "rec")
    )


def write_tsv(pairs, s1_all, col, path):
    g = pairs.group_by("s1").agg(pl.col("rec").sort().str.join(",").alias(col))
    out = s1_all.join(g, on="s1", how="left").with_columns(pl.col(col).fill_null(""))
    out = out.rename({"s1": "source1_entity_id"}).select("source1_entity_id", col)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(f"source1_entity_id\t{col}\n")
        for a, b in out.iter_rows():
            f.write(f"{a}\t{b}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--cand", required=True, help="candidate table name the stacker ran on")
    ap.add_argument("--sub", default="", help="submission folder name under <root>/submissions")
    ap.add_argument("--out-dir", default="", help="write the two output files to this folder instead")
    ap.add_argument("--shift", action="store_true", help="label-shift offset for unlabelled countries (tested, hurt the leaderboard; off)")
    ap.add_argument("--no-shift", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--offset", type=float, default=None, help="fixed logit offset for unlabelled countries (probing only)")
    a = ap.parse_args()

    tr = labels(pl.read_parquet(ART / "pred" / f"{a.pred}_train.parquet"))
    c1 = s1_group(["C1"])
    c2 = s1_group(["C2"])
    s1_ids = pl.read_parquet(ART / "norm_train.parquet", columns=["entity_id", "src"]).with_row_index("i")

    cand = to_ids(pl.read_parquet(ART / "cand" / f"{a.cand}_train.parquet", columns=["i1", "i2"]), "train")
    blocking(cand, c1, "cand C1")
    blocking(cand, c2, "cand C2")

    kept = one_to_one(tr)
    fit = kept.filter(pl.col("grp") == "C1")
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(fit["p"].to_numpy(), fit["y"].to_numpy())
    kept = kept.with_columns(pl.Series("q", iso.predict(kept["p"].to_numpy()).astype(np.float32)))

    results = {}
    best = None
    for t in [0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8]:
        s = score(to_ids(pick_threshold(kept, t), "train"), c1, f"C1 thr={t}", verbose=False)["all"]
        results[f"thr_{t}"] = s
        if best is None or s > best[0]:
            best = (s, "thr", t)
    for b in [-0.1, -0.05, 0.0, 0.05, 0.1]:
        s = score(to_ids(pick_expected(kept, b), "train"), c1, f"C1 expF bias={b}", verbose=False)["all"]
        results[f"expF_{b}"] = s
        if s > best[0]:
            best = (s, "expF", b)
    for b in [-0.05, 0.0, 0.02, 0.05, 0.1]:
        s = score(to_ids(pick_expected_exact(kept, b), "train"), c1, f"C1 exact bias={b}", verbose=False)["all"]
        results[f"exact_{b}"] = s
        if s > best[0]:
            best = (s, "exact", b)
    log("C1 grid", json.dumps({k: round(v, 5) for k, v in results.items()}))
    log("best", best)

    def apply(df):
        if best[1] == "thr":
            return pick_threshold(df, best[2])
        if best[1] == "exact":
            return pick_expected_exact(df, best[2])
        return pick_expected(df, best[2])

    r1 = score(to_ids(apply(kept), "train"), c1, "C1 best")
    r2 = score(to_ids(apply(kept), "train"), c2, "C2 best")
    # no one-to-one, for reference
    raw = tr.with_columns(pl.Series("q", iso.predict(tr["p"].to_numpy()).astype(np.float32)))
    score(to_ids(apply(raw), "train"), c2, "C2 without one-to-one")

    if a.sub or a.out_dir:
        te = pl.read_parquet(ART / "pred" / f"{a.pred}_test.parquet")
        offsets = {}
        if a.offset is not None:
            te, offsets = unlabeled_shift(te, iso, apply, fixed=a.offset)
        elif a.shift:
            te, offsets = unlabeled_shift(te, iso, apply)
        kt = one_to_one(te)
        kt = kt.with_columns(pl.Series("q", iso.predict(kt["p"].to_numpy()).astype(np.float32)))
        sel = to_ids(apply(kt), "test")
        ctest = to_ids(pl.read_parquet(ART / "cand" / f"{a.cand}_test.parquet", columns=["i1", "i2"]), "test")
        s1_all = pl.read_parquet(ART / "norm_test.parquet", columns=["entity_id", "src", "country"]).filter(
            pl.col("src") == 1
        ).select(pl.col("entity_id").alias("s1"), "country")
        sub = Path(a.out_dir) if a.out_dir else ROOT / "submissions" / a.sub
        write_tsv(sel, s1_all, "matched_entity_ids", sub / "matching_results.tsv")
        write_tsv(ctest, s1_all, "candidate_entity_ids", sub / "candidate_pairs.tsv")
        stats = s1_all.join(sel.group_by("s1").len("n"), on="s1", how="left").with_columns(pl.col("n").fill_null(0))
        cstats = s1_all.join(ctest.group_by("s1").len("c"), on="s1", how="left").with_columns(pl.col("c").fill_null(0))
        summ = stats.join(cstats.select("s1", "c"), on="s1").group_by("country").agg(
            pl.col("n").mean().alias("pred_per_s1"), (pl.col("n") == 0).mean().alias("pred_singleton"),
            pl.col("c").mean().alias("cand_per_s1"),
        )
        log("test summary\n", summ)
        (ART / "final_info.json" if a.out_dir else sub / "info.json").write_text(json.dumps({"C1": r1, "C2": r2, "method": best[1:], "pred": a.pred, "cand": a.cand, "offsets": offsets}, indent=1))
        log("submission written to", sub)


if __name__ == "__main__":
    main()
