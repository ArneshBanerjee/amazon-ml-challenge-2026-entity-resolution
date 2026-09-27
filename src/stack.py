"""GBDT stacker. Trains on candidates of split-B S1 entities, predicts every train and test candidate.

Input:  artifacts/feat/<feat>_{train,test}.parquet (plus optional extra score tables joined on i1, i2)
Output: artifacts/pred/<out>_{train,test}.parquet with i1, i2, p
"""
import os
import argparse
import json

import lightgbm as lgb
import numpy as np
import polars as pl

from common import ART, SEED, done, log
from features import context_features

DROP = {"i1", "i2", "y", "grp"}


def labels(split_df):
    """Attach y (true pair) and grp (A/B/C1/C2 of the S1) to a train pair frame."""
    ids = pl.read_parquet(ART / "norm_train.parquet", columns=["entity_id"]).with_row_index("i")
    gt = pl.read_parquet(ART / "gt_pairs.parquet")
    sp = pl.read_parquet(ART / "split_s1.parquet").select("s1", "grp")
    d = (
        split_df.join(ids.rename({"i": "i1", "entity_id": "s1"}), on="i1", how="left")
        .join(ids.rename({"i": "i2", "entity_id": "rec"}), on="i2", how="left")
        .join(gt.with_columns(pl.lit(1, pl.Int8).alias("y")), on=["s1", "rec"], how="left")
        .join(sp, on="s1", how="left")
        .with_columns(pl.col("y").fill_null(0))
        .drop("s1", "rec")
    )
    return d


def load(feat, extra, split, gfeat=()):
    df = pl.read_parquet(ART / "feat" / f"{feat}_{split}.parquet")
    for e in extra:
        x = pl.read_parquet(ART / "score" / f"{e}_{split}.parquet")
        df = df.join(x, on=["i1", "i2"], how="left")
        df = context_features(df, e, f"x_{e}")
    for g in gfeat:
        df = df.join(pl.read_parquet(ART / "group" / f"{g}_{split}.parquet"), on=["i1", "i2"], how="left")
    return df


PARAMS = dict(
    objective="binary", learning_rate=0.05, num_leaves=255, min_data_in_leaf=50,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
    num_threads=int(os.environ.get("BER_THREADS", max(1, os.cpu_count() - 2))), verbose=-1, max_bin=255,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", required=True)
    ap.add_argument("--extra", default="", help="comma separated score tables in artifacts/score")
    ap.add_argument("--gfeat", default="", help="comma separated feature tables in artifacts/group")
    ap.add_argument("--drop", default="", help="comma separated feature names to leave out")
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-groups", default="B")
    ap.add_argument("--kfold", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=3000)
    ap.add_argument("--test-only", action="store_true", help="reuse saved fold models, predict test only")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    (ART / "pred").mkdir(exist_ok=True)
    out_tr = ART / "pred" / f"{a.out}_train.parquet"
    if a.test_only:
        return predict_test(a)
    if done(out_tr, a.force):
        log("skip", out_tr)
        return
    extra = [e for e in a.extra.split(",") if e]
    gfeat = [g for g in a.gfeat.split(",") if g]
    tr = labels(load(a.feat, extra, "train", gfeat))
    drop = DROP | {d for d in a.drop.split(",") if d}
    feats = [c for c in tr.columns if c not in drop]
    log("features", len(feats))
    groups = a.train_groups.split(",")
    is_fit = tr["grp"].is_in(groups).to_numpy()
    # folds by S1 entity; out-of-fold predictions for the training groups, fold average elsewhere
    s1 = np.unique(tr["i1"].to_numpy()[is_fit])
    fold_of = dict(zip(s1.tolist(), np.random.default_rng(SEED).integers(0, a.kfold, len(s1)).tolist()))
    fold = np.array([fold_of.get(i, -1) for i in tr["i1"].to_list()], dtype=np.int8)
    X = tr.select(feats).to_numpy().astype(np.float32)
    y = tr["y"].to_numpy()
    p = np.zeros(len(y), dtype=np.float64)
    models = []
    mdir = ART / "models" / f"lgb_{a.out}"
    mdir.mkdir(parents=True, exist_ok=True)
    for k in range(a.kfold):
        trn, val = (fold != k) & (fold >= 0), fold == k
        dtr = lgb.Dataset(X[trn], y[trn], feature_name=feats)
        dva = lgb.Dataset(X[val], y[val], reference=dtr)
        m = lgb.train(dict(PARAMS, seed=k + 100 * SEED), dtr, a.rounds, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(500)])
        log("fold", k, "best iter", m.best_iteration)
        m.save_model(str(mdir / f"m{k}.txt"))
        models.append(m)
        p[val] = m.predict(X[val], num_iteration=m.best_iteration)
        p[~is_fit] += m.predict(X[~is_fit], num_iteration=m.best_iteration) / a.kfold
    imp = sorted(zip(feats, models[0].feature_importance("gain")), key=lambda x: -x[1])
    (mdir / "importance.json").write_text(json.dumps([(f, float(g)) for f, g in imp], indent=0))
    log("top features", [f for f, _ in imp[:25]])
    tr.select("i1", "i2").with_columns(pl.Series("p", p.astype(np.float32))).write_parquet(out_tr)
    del tr, X
    te = load(a.feat, extra, "test", gfeat)
    Xt = te.select(feats).to_numpy().astype(np.float32)
    pt = np.mean([m.predict(Xt, num_iteration=m.best_iteration) for m in models], axis=0)
    te.select("i1", "i2").with_columns(pl.Series("p", pt.astype(np.float32))).write_parquet(
        ART / "pred" / f"{a.out}_test.parquet")
    log("wrote preds", a.out)


def predict_test(a):
    mdir = ART / "models" / f"lgb_{a.out}"
    models = [lgb.Booster(model_file=str(p)) for p in sorted(mdir.glob("m*.txt"))]
    feats = models[0].feature_name()
    extra = [e for e in a.extra.split(",") if e]
    gfeat = [g for g in a.gfeat.split(",") if g]
    te = load(a.feat, extra, "test", gfeat)
    Xt = te.select(feats).to_numpy().astype(np.float32)
    pt = np.mean([m.predict(Xt) for m in models], axis=0)
    te.select("i1", "i2").with_columns(pl.Series("p", pt.astype(np.float32))).write_parquet(
        ART / "pred" / f"{a.out}_test.parquet")
    log("wrote test preds", a.out, "with", len(models), "saved models")


if __name__ == "__main__":
    main()
