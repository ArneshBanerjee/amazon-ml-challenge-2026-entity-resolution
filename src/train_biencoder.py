"""Fine-tune a bi-encoder on split A pairs with in-batch negatives plus mined hard negatives.

Loss: symmetric InfoNCE (MultipleNegativesRankingLoss style) with a learned-free fixed scale.
Batches hold one country and unique S1 entities, so in-batch negatives are never true matches.
Output: artifacts/models/<out>/ (HF format, used by encode.py with mean pooling)
"""
import argparse
import json
import math
import random

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from common import ART, SEED, done, log
from encode import build_texts


def mean_pool(model, enc):
    h = model(**enc).last_hidden_state
    m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
    return F.normalize(((h * m).sum(1) / m.sum(1)).float(), dim=-1)


def make_batches(pairs, bs, seed):
    """pairs: frame with i1, i2, hn (hard negative row or -1), country. Returns list of index arrays."""
    rng = np.random.default_rng(seed)
    p = pairs.with_columns(pl.Series("r", rng.random(pairs.height)))
    p = p.with_columns(pl.col("r").rank("ordinal").over("i1").alias("round"))
    batches = []
    for (country, rnd), g in p.with_row_index("row").group_by(["country", "round"]):
        rows = g["row"].to_numpy()
        rows = rows[rng.permutation(len(rows))]
        for s in range(0, len(rows), bs):
            b = rows[s : s + bs]
            if len(b) >= bs // 4:
                batches.append(b)
    random.Random(seed).shuffle(batches)
    return batches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="intfloat/multilingual-e5-small")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix", default="query: ")
    ap.add_argument("--hardneg", default="", help="knn tag used to mine hard negatives")
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--scale", type=float, default=20.0)
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--max-pairs", type=int, default=0)
    ap.add_argument("--init", default="", help="start from this model dir instead of --base")
    ap.add_argument("--views", default="combined:1.0", help="view:weight list, views are combined,name,addr")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    out_dir = ART / "models" / a.out
    if done(out_dir / "config.json", a.force):
        log("skip", out_dir)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)

    norm = pl.read_parquet(ART / "norm_train.parquet", columns=["entity_id", "country", "name_n", "addr_n"])
    views = [(v.split(":")[0], float(v.split(":")[1])) for v in a.views.split(",")]
    vtexts = {v: build_texts(norm, v, a.prefix) for v, _ in views}
    addr_empty = (norm["addr_n"] == "").to_numpy()
    idx = norm.select("entity_id", "country").with_row_index("i")
    split = pl.read_parquet(ART / "split_s1.parquet").filter(pl.col("grp") == "A")
    gt = pl.read_parquet(ART / "gt_pairs.parquet").join(split.select("s1"), on="s1")
    pairs = (
        gt.join(idx.rename({"entity_id": "s1", "i": "i1"}), on="s1")
        .join(idx.select(pl.col("entity_id").alias("rec"), pl.col("i").alias("i2")), on="rec")
        .select("i1", "i2", "country")
    )
    if a.hardneg:
        # hardest non-matching record in the S1 forward list (from an earlier retrieval run)
        knn = pl.read_parquet(ART / "knn" / f"{a.hardneg}_train.parquet", columns=["i1", "i2", "cos", "rf"])
        truth = pairs.select("i1", "i2").with_columns(pl.lit(True).alias("t"))
        # only records whose own S1 is in A, or distractors: B/C pairs must never be seen in training
        rg = pl.read_parquet(ART / "split_rec.parquet").select(pl.col("rec").alias("entity_id"), pl.col("grp").alias("rgrp"))
        ok = idx.join(rg, on="entity_id").filter(pl.col("rgrp").is_in(["A", "X"])).select(pl.col("i").alias("i2"))
        knn = knn.join(ok, on="i2")
        hn = (
            knn.filter(pl.col("rf") < 255)
            .join(pairs.select("i1").unique(), on="i1")
            .join(truth, on=["i1", "i2"], how="left")
            .filter(pl.col("t").is_null())
            .sort("cos", descending=True)
            .group_by("i1")
            .agg(pl.col("i2").head(3).alias("hns"))
        )
        pairs = pairs.join(hn, on="i1", how="left")
    if a.max_pairs:
        pairs = pairs.sample(a.max_pairs, seed=0)
    log("train pairs", pairs.height)

    src = a.init or a.base
    tok = AutoTokenizer.from_pretrained(src)
    model = AutoModel.from_pretrained(src, dtype=torch.float32).cuda()
    model.gradient_checkpointing_enable() if a.bs > 2048 else None
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    n_epochs = math.ceil(a.epochs)
    batches_per_epoch = len(make_batches(pairs, a.bs, 0))
    total = int(batches_per_epoch * a.epochs)
    sched = get_linear_schedule_with_warmup(opt, int(0.05 * total), total)
    i1s = pairs["i1"].to_numpy()
    i2s = pairs["i2"].to_numpy()
    hns = pairs["hns"].to_list() if "hns" in pairs.columns else None
    rng = random.Random(1)

    vnames = [v for v, _ in views]
    vw = np.array([w for _, w in views]) / sum(w for _, w in views)
    vrng = np.random.default_rng(7)

    def enc(ids, texts):
        e = tok([texts[j] for j in ids], padding=True, truncation=True, max_length=a.max_len, return_tensors="pt")
        return {k: v.cuda(non_blocking=True) for k, v in e.items()}

    step = 0
    model.train()
    for ep in range(n_epochs):
        for b in make_batches(pairs, a.bs, ep + 100 * SEED):
            if step >= total:
                break
            view = vnames[vrng.choice(len(vnames), p=vw)]
            if view == "addr":
                b = b[~addr_empty[i2s[b]]]
            texts = vtexts[view]
            q_ids = i1s[b]
            d_ids = list(i2s[b])
            if hns is not None:
                neg = []
                for r in b:
                    h = hns[r]
                    if h:
                        c = rng.choice(h)
                        if not (view == "addr" and addr_empty[c]):
                            neg.append(c)
                d_ids = d_ids + neg
            with torch.autocast("cuda", dtype=torch.bfloat16):
                q = mean_pool(model, enc(q_ids, texts))
                d = mean_pool(model, enc(d_ids, texts))
            sim = q @ d.T * a.scale
            lab = torch.arange(len(b), device=sim.device)
            loss = F.cross_entropy(sim, lab) + F.cross_entropy(sim[:, : len(b)].T, lab)
            loss = loss / 2
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 200 == 0:
                log(f"ep {ep} step {step}/{total} loss {loss.item():.4f}")
            if step % 2000 == 0:
                model.save_pretrained(out_dir / "ckpt")
                tok.save_pretrained(out_dir / "ckpt")
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    (out_dir / "train_args.json").write_text(json.dumps(vars(a)))
    log("saved", out_dir)


if __name__ == "__main__":
    main()
