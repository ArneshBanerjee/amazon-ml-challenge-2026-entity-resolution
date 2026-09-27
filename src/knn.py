"""Exact GPU kNN between S1 and S2/S3 records, within each exact country string.

Both directions: every S1 gets its top-kf records, every record gets its top-kr S1.
Output artifacts/knn/<tag>_<split>.parquet with columns:
  i1 (row index of S1 in norm_<split>), i2 (row index of record), cos, rf (forward rank, 0 based,
  255 when absent), rr (reverse rank, 255 when absent)
"""
import argparse

import numpy as np
import polars as pl
import torch

from common import ART, done, log


@torch.no_grad()
def topk_chunked(q, x, k, chunk):
    """For each row of q, top-k rows of x by inner product. Returns (vals, idx) on CPU."""
    vals, idxs = [], []
    for s in range(0, q.shape[0], chunk):
        sc = q[s : s + chunk] @ x.T
        v, i = torch.topk(sc, min(k, x.shape[0]), dim=1)
        vals.append(v.float().cpu())
        idxs.append(i.int().cpu())
    return torch.cat(vals).numpy(), torch.cat(idxs).numpy()


def run(emb, meta, kf, kr):
    out = []
    for country in meta["country"].unique().sort().to_list():
        m = meta.with_row_index("row").filter(pl.col("country") == country)
        i1 = m.filter(pl.col("src") == 1)["row"].to_numpy()
        i2 = m.filter(pl.col("src") != 1)["row"].to_numpy()
        if len(i1) == 0 or len(i2) == 0:
            continue
        a = torch.from_numpy(np.ascontiguousarray(emb[i1])).cuda()
        b = torch.from_numpy(np.ascontiguousarray(emb[i2])).cuda()
        log(country, "S1", len(i1), "records", len(i2))
        vf, jf = topk_chunked(a, b, kf, max(256, int(2e9 // (2 * len(i2)))))
        vr, jr = topk_chunked(b, a, kr, max(256, int(2e9 // (2 * len(i1)))))
        del a, b
        torch.cuda.empty_cache()
        fwd = pl.DataFrame({
            "i1": np.repeat(i1, jf.shape[1]).astype(np.int32),
            "i2": i2[jf.ravel()].astype(np.int32),
            "cos": vf.ravel().astype(np.float32),
            "rf": np.tile(np.arange(jf.shape[1], dtype=np.uint8), len(i1)),
        })
        rev = pl.DataFrame({
            "i1": i1[jr.ravel()].astype(np.int32),
            "i2": np.repeat(i2, jr.shape[1]).astype(np.int32),
            "cos_r": vr.ravel().astype(np.float32),
            "rr": np.tile(np.arange(jr.shape[1], dtype=np.uint8), len(i2)),
        })
        j = fwd.join(rev, on=["i1", "i2"], how="full", coalesce=True).with_columns(
            pl.coalesce("cos", "cos_r").alias("cos"),
            pl.col("rf").fill_null(255),
            pl.col("rr").fill_null(255),
        ).drop("cos_r")
        log(country, "pairs", j.height)
        out.append(j)
    return pl.concat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--splits", default="train,test")
    ap.add_argument("--kf", type=int, default=30)
    ap.add_argument("--kr", type=int, default=10)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    (ART / "knn").mkdir(exist_ok=True)
    for split in a.splits.split(","):
        path = ART / "knn" / f"{a.tag}_{split}.parquet"
        if done(path, a.force):
            log("skip", path)
            continue
        meta = pl.read_parquet(ART / f"norm_{split}.parquet", columns=["src", "country"])
        emb = np.load(ART / "emb" / f"{a.tag}_{split}.npy", mmap_mode="r")
        res = run(emb, meta, a.kf, a.kr)
        res.write_parquet(path)
        log("wrote", path, res.height)


if __name__ == "__main__":
    main()
