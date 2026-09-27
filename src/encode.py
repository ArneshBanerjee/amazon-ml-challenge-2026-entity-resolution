"""Encode records with a transformer (mean pooling, L2 normalized, fp16 output).

Writes artifacts/emb/<tag>_<split>.npy (N x D float16), row order = norm_<split>.parquet.
Resumable: progress is stored next to the output every chunk.
"""
import argparse
import json

import numpy as np
import polars as pl
import torch
from transformers import AutoModel, AutoTokenizer

from common import ART, done, log, prefetch


def build_texts(df, field, prefix=""):
    if field == "combined":
        t = pl.concat_str([pl.col("name_n"), pl.lit(" | "), pl.col("addr_n")])
    elif field == "name":
        t = pl.col("name_n")
    elif field == "addr":
        t = pl.col("addr_n")
    else:
        raise ValueError(field)
    return df.select((pl.lit(prefix) + t).alias("t"))["t"].to_list()


class _Batches(torch.utils.data.Dataset):
    """Pre-planned batches of text indices; tokenization happens in DataLoader workers."""

    def __init__(self, texts, batches, tok, max_len):
        self.texts, self.batches, self.tok, self.max_len = texts, batches, tok, max_len

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, k):
        idx = self.batches[k]
        enc = self.tok([self.texts[j] for j in idx], padding=True, truncation=True,
                       max_length=self.max_len, return_tensors="pt")
        return idx, dict(enc)


@torch.no_grad()
def encode_texts(texts, model, tok, out, max_len, tokens_per_batch=65536, prog_path=None, workers=8):
    n = len(texts)
    lens = np.fromiter((len(t) for t in texts), dtype=np.int32, count=n)
    order = np.argsort(-lens, kind="stable")
    batches = []
    i = 0
    while i < n:
        # rough tokens per text from its character length; cap batch by a token budget
        est = min(max_len, int(lens[order[i]] * 0.5) + 8)
        bs = max(64, min(4096, tokens_per_batch // est))
        batches.append(order[i : i + bs])
        i += bs
    start = 0
    if prog_path is not None and prog_path.exists():
        start = json.loads(prog_path.read_text())["batch"]
        log("resuming at batch", start)
    ds = _Batches(texts, batches[start:], tok, max_len)
    dl = prefetch(ds.__getitem__, range(len(ds)), threads=workers)
    done_n = sum(len(b) for b in batches[:start])
    last_ck = done_n
    for k, (idx, enc) in enumerate(dl, start=start):
        enc = {kk: v.cuda(non_blocking=True) for kk, v in enc.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = model(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
        e = (h * m).sum(1) / m.sum(1)
        e = torch.nn.functional.normalize(e.float(), dim=-1)
        out[idx] = e.half().cpu().numpy()
        done_n += len(idx)
        if prog_path is not None and done_n - last_ck > 1_000_000:
            out.flush()
            prog_path.write_text(json.dumps({"batch": k + 1}))
            last_ck = done_n
            log(f"{done_n}/{n}")
    if hasattr(out, "flush"):
        out.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--field", default="combined")
    ap.add_argument("--prefix", default="")
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--splits", default="train,test")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    (ART / "emb").mkdir(exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModel.from_pretrained(a.model, dtype=torch.bfloat16).cuda().eval()
    dim = model.config.hidden_size
    for split in a.splits.split(","):
        path = ART / "emb" / f"{a.tag}_{split}.npy"
        prog = ART / "emb" / f"{a.tag}_{split}.progress"
        if done(path, a.force) and not prog.exists():
            log("skip", path)
            continue
        df = pl.read_parquet(ART / f"norm_{split}.parquet", columns=["name_n", "addr_n"])
        texts = build_texts(df, a.field, a.prefix)
        mode = "r+" if path.exists() and prog.exists() and not a.force else "w+"
        if mode == "w+" and prog.exists():
            prog.unlink()
        out = np.lib.format.open_memmap(path, mode=mode, dtype=np.float16, shape=(len(texts), dim))
        log("encoding", split, len(texts), "->", path)
        encode_texts(texts, model, tok, out, a.max_len, prog_path=prog)
        del out
        prog.unlink(missing_ok=True)
        log("done", path)


if __name__ == "__main__":
    main()
