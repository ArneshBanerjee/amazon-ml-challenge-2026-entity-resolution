"""Cross-encoder pair classifier.

train: fine-tune on candidate pairs of split-A S1 entities (positives plus the hard negatives that
       the candidate generator produced). Negatives whose record belongs to a B/C entity are left out,
       so no B/C pair is ever seen in training.
infer: score every candidate pair of train and test, write artifacts/score/<out>_<split>.parquet
"""
import argparse
import json
import math

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from common import ART, SEED, done, log, prefetch


class PairBatches(torch.utils.data.Dataset):
    """Tokenize pre-planned batches of (i1, i2) pairs inside DataLoader workers."""

    def __init__(self, T, i1, i2, batches, tok, max_len):
        self.T, self.i1, self.i2, self.batches, self.tok, self.max_len = T, i1, i2, batches, tok, max_len

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, k):
        b = self.batches[k]
        enc = self.tok([self.T[j] for j in self.i1[b]], [self.T[j] for j in self.i2[b]], padding=True,
                       truncation=True, max_length=self.max_len, return_tensors="pt")
        return b, dict(enc)


def loader(T, i1, i2, batches, tok, max_len):
    ds = PairBatches(T, i1, i2, batches, tok, max_len)
    return prefetch(ds.__getitem__, range(len(ds)))


TEXT = "norm"


def texts(split):
    if TEXT == "raw":
        n = pl.read_parquet(ART / f"records_{split}.parquet", columns=["name", "addr"])
        return n.select(pl.concat_str([pl.col("name"), pl.lit(" | "), pl.col("addr")]).alias("t"))["t"].to_list()
    n = pl.read_parquet(ART / f"norm_{split}.parquet", columns=["name_n", "addr_n"])
    return n.select(pl.concat_str([pl.col("name_n"), pl.lit(" | "), pl.col("addr_n")]).alias("t"))["t"].to_list()


def train(a):
    out_dir = ART / "models" / a.out
    if done(out_dir / "config.json", a.force):
        log("skip", out_dir)
        return
    from stack import labels

    cand = pl.read_parquet(ART / "cand" / f"{a.cand}_train.parquet", columns=["i1", "i2"])
    d = labels(cand).filter(pl.col("grp") == "A")
    # group of the record's own S1 (X = distractor)
    ids = pl.read_parquet(ART / "norm_train.parquet", columns=["entity_id"]).with_row_index("i2")
    rg = pl.read_parquet(ART / "split_rec.parquet").select(pl.col("rec").alias("entity_id"), pl.col("grp").alias("rgrp"))
    d = d.join(ids.join(rg, on="entity_id").drop("entity_id"), on="i2", how="left")
    d = d.filter((pl.col("y") == 1) | pl.col("rgrp").is_in(["A", "X"]))
    pos = d.filter(pl.col("y") == 1)
    neg = d.filter(pl.col("y") == 0)
    n_pos = min(pos.height, int(a.n_pairs * a.pos_frac))
    n_neg = min(neg.height, a.n_pairs - n_pos)
    d = pl.concat([pos.sample(n_pos, seed=a.seed + 100 * SEED), neg.sample(n_neg, seed=a.seed + 100 * SEED)]).sample(fraction=1.0, shuffle=True, seed=1)
    log("train pairs", d.height, "pos", n_pos, "neg", n_neg, "available neg", neg.height)
    T = texts("train")
    i1, i2, y = d["i1"].to_numpy(), d["i2"].to_numpy(), d["y"].to_numpy().astype(np.float32)
    if a.pseudo:
        # pseudo-labelled test pairs (countries without labels), indices shifted past the train texts
        ps = pl.read_parquet(ART / f"{a.pseudo}.parquet")
        off = len(T)
        T = T + texts("test")
        i1 = np.concatenate([i1, ps["i1"].to_numpy() + off])
        i2 = np.concatenate([i2, ps["i2"].to_numpy() + off])
        y = np.concatenate([y, ps["y"].to_numpy().astype(np.float32)])
        log("added pseudo pairs", ps.height, "pos", int(ps["y"].sum()))

    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(a.init or a.base)
    model = AutoModelForSequenceClassification.from_pretrained(a.init or a.base, num_labels=1, dtype=torch.float32).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = math.ceil(len(y) / a.bs) * a.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
    model.train()
    step = 0
    for ep in range(a.epochs):
        perm = np.random.default_rng(ep + 100 * SEED).permutation(len(y))
        batches = [perm[s : s + a.bs] for s in range(0, len(y), a.bs)]
        for b, enc in loader(T, i1, i2, batches, tok, a.max_len):
            enc = {k: v.cuda(non_blocking=True) for k, v in enc.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = model(**enc).logits.squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logit.float(), torch.from_numpy(y[b]).cuda())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % a.log_every == 0:
                log(f"ep {ep} step {step}/{steps} loss {loss.item():.4f}")
            if step % 5000 == 0:
                model.save_pretrained(out_dir / "ckpt")
                tok.save_pretrained(out_dir / "ckpt")
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    (out_dir / "train_args.json").write_text(json.dumps(vars(a)))
    log("saved", out_dir)


@torch.no_grad()
def infer(a):
    (ART / "score").mkdir(exist_ok=True)
    mdir = ART / "models" / a.out
    global TEXT
    ta = mdir / "train_args.json"
    if ta.exists():
        TEXT = json.loads(ta.read_text()).get("text", "norm")
    tok = AutoTokenizer.from_pretrained(mdir)
    model = AutoModelForSequenceClassification.from_pretrained(mdir, dtype=torch.bfloat16).cuda().eval()
    for split in a.splits.split(","):
        path = ART / "score" / f"{a.out}_{split}.parquet"
        if done(path, a.force):
            log("skip", path)
            continue
        cand = pl.read_parquet(ART / "cand" / f"{a.cand}_{split}.parquet", columns=["i1", "i2"])
        T = texts(split)
        i1, i2 = cand["i1"].to_numpy(), cand["i2"].to_numpy()
        lens = np.fromiter((len(T[x]) + len(T[z]) for x, z in zip(i1, i2)), dtype=np.int32, count=len(i1))
        order = np.argsort(-lens, kind="stable")
        out = np.zeros(len(i1), dtype=np.float32)
        parts_dir = ART / "score" / f"{a.out}_{split}_parts"
        parts_dir.mkdir(exist_ok=True)
        chunk = 1_000_000
        for c, s in enumerate(range(0, len(order), chunk)):
            p = parts_dir / f"{c:04d}.npy"
            idx = order[s : s + chunk]
            if p.exists():
                out[idx] = np.load(p)
                continue
            res = np.zeros(len(idx), dtype=np.float32)
            batches = [np.arange(t, min(t + a.infer_bs, len(idx))) for t in range(0, len(idx), a.infer_bs)]
            for pos, enc in loader(T, i1[idx], i2[idx], batches, tok, a.max_len):
                enc = {k: v.cuda(non_blocking=True) for k, v in enc.items()}
                res[pos] = model(**enc).logits.squeeze(-1).float().cpu().numpy()
            np.save(p, res)
            out[idx] = res
            log(split, f"{s + len(idx)}/{len(order)}")
        cand.with_columns(pl.Series(a.out, out)).write_parquet(path)
        for p in parts_dir.iterdir():
            p.unlink()
        parts_dir.rmdir()
        log("wrote", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "infer"])
    ap.add_argument("--cand", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="microsoft/mdeberta-v3-base")
    ap.add_argument("--n-pairs", type=int, default=3_000_000)
    ap.add_argument("--pos-frac", type=float, default=0.4)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--infer-bs", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--splits", default="train,test")
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--text", default="norm", choices=["norm", "raw"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", default="", help="start from this model dir")
    ap.add_argument("--pseudo", default="", help="artifacts/<name>.parquet with test i1, i2, y")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    TEXT = a.text
    train(a) if a.mode == "train" else infer(a)
