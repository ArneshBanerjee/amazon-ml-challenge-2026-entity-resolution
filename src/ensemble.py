"""Average two (or more) independent training runs of the pipeline.

Each run lives in its own artifacts folder (same raw data, same splits, different random state).
For every candidate pair the stage-2 logits are averaged; a pair missing from one run's candidate
set counts as a very low probability there (that run's pruning model dropped it). The candidate set
of the ensemble is the union of the runs' candidate sets, which is exactly the set it scores.

Output (in this run's artifacts): pred/<out>_{train,test}.parquet and cand/<out>_{train,test}.parquet,
then run select_sets.py --pred <out> --cand <out>.
"""
import argparse
from pathlib import Path

import numpy as np
import polars as pl

from common import ART, log

FLOOR = 1e-4  # probability for a pair that a run did not keep as a candidate


def logit(x):
    x = np.clip(x, 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="comma separated artifacts folders; the first can be this run")
    ap.add_argument("--pred", default="s2_v2")
    ap.add_argument("--cand", default="v2")
    ap.add_argument("--out", default="ens")
    a = ap.parse_args()
    runs = [Path(r) for r in a.runs.split(",")]
    (ART / "pred").mkdir(exist_ok=True)
    (ART / "cand").mkdir(exist_ok=True)
    for split in ("train", "test"):
        tabs = []
        for k, r in enumerate(runs):
            t = pl.read_parquet(r / "pred" / f"{a.pred}_{split}.parquet").rename({"p": f"p{k}"})
            tabs.append(t)
        j = tabs[0]
        for t in tabs[1:]:
            j = j.join(t, on=["i1", "i2"], how="full", coalesce=True)
        cols = [f"p{k}" for k in range(len(runs))]
        L = np.stack([logit(j[c].fill_null(FLOOR).to_numpy()) for c in cols]).mean(0)
        p = (1 / (1 + np.exp(-L))).astype(np.float32)
        out = j.select("i1", "i2").with_columns(pl.Series("p", p))
        out.write_parquet(ART / "pred" / f"{a.out}_{split}.parquet")
        cands = [pl.read_parquet(r / "cand" / f"{a.cand}_{split}.parquet", columns=["i1", "i2"]) for r in runs]
        pl.concat(cands).unique().write_parquet(ART / "cand" / f"{a.out}_{split}.parquet")
        log(split, "ensemble pairs", out.height, "runs", len(runs))


if __name__ == "__main__":
    main()
