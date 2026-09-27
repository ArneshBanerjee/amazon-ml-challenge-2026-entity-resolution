"""Pairwise string features for (S1, record) pairs.

pair_features(split, pairs) takes a frame with i1, i2 (row indices into norm_<split>.parquet)
and returns it with feature columns added. Uses rapidfuzz cpdist on all cores.
"""
import math
import os
from multiprocessing import Pool

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

from common import ART, log

W = os.cpu_count()
_CACHE = {}


def norm_table(split):
    if split not in _CACHE:
        n = pl.read_parquet(ART / f"norm_{split}.parquet")
        raw = pl.read_parquet(ART / f"records_{split}.parquet", columns=["name", "addr"])
        n = n.with_columns(
            raw["name"].str.to_lowercase().alias("name_raw"),
            pl.col("core").str.replace_all(" ", "").alias("core_ns"),
            pl.col("alt").str.replace_all(" ", "").alias("alt_ns"),
        )
        _CACHE[split] = n
    return _CACHE[split]


def idf_table(split):
    key = ("idf", split)
    if key not in _CACHE:
        n = norm_table(split)
        out = {}
        for field in ("core", "addr_n"):
            t = (
                n.select(pl.col(field).str.replace_all(",", " ").str.split(" ").alias("t"))
                .explode("t")
                .filter(pl.col("t") != "")
                .group_by("t")
                .len()
            )
            N = n.height
            out[field] = dict(zip(t["t"].to_list(), (np.log(N / t["len"].to_numpy())).tolist()))
        _CACHE[key] = out
    return _CACHE[key]


def cp(a, b, scorer):
    return process.cpdist(a, b, scorer=scorer, workers=W, dtype=np.float32).astype(np.float32)


_IDF = {}


def _idf_overlap(args):
    field, pairs = args
    idf = _IDF[field]
    res = np.empty((len(pairs), 3), dtype=np.float32)
    for k, (a, b) in enumerate(pairs):
        ta = set(a.replace(",", " ").split())
        tb = set(b.replace(",", " ").split())
        if not ta or not tb:
            res[k] = (np.nan, np.nan, np.nan)
            continue
        inter = ta & tb
        wi = sum(idf.get(t, 12.0) for t in inter)
        wa = sum(idf.get(t, 12.0) for t in ta)
        wb = sum(idf.get(t, 12.0) for t in tb)
        res[k] = (wi / (wa + wb - wi), wi / wa, max((idf.get(t, 12.0) for t in inter), default=0.0))
    return res


def _tok_diff(args):
    """Substitution vs typo signals. For record tokens with no fuzzy partner in the S1 core, the
    smallest idf (a common real word suggests a swapped word, a rare string suggests a typo).
    Also first-number perturbation signals (nearby but different house numbers)."""
    pairs = args
    idf = _IDF["core"]
    res = np.full((len(pairs), 8), np.nan, dtype=np.float32)
    for k, (c1, c2, n1, n2) in enumerate(pairs):
        t1, t2 = c1.split(), c2.split()
        if t1 and t2:
            u2 = [t for t in t2 if t not in t1 and max(fuzz.ratio(t, u) for u in t1) < 75]
            u1 = [t for t in t1 if t not in t2 and max(fuzz.ratio(t, u) for u in t2) < 75]
            res[k, 0] = len(u2)
            res[k, 1] = len(u1)
            res[k, 2] = min((idf.get(t, 16.0) for t in u2), default=20.0)
            res[k, 3] = min((idf.get(t, 16.0) for t in u1), default=20.0)
        a, b = n1.split(), n2.split()
        if a and b:
            x, y = a[0], b[0]
            if x != y:
                res[k, 4] = math.log1p(abs(int(x[:12]) - int(y[:12])))
                res[k, 5] = float(len(x) == len(y))
                res[k, 6] = float(x.endswith(y) or y.endswith(x) or x.startswith(y) or y.startswith(x))
            else:
                res[k, 4], res[k, 5], res[k, 6] = 0.0, 1.0, 1.0
            sa = set(a)
            res[k, 7] = sum(1 for t in b if t not in sa and not any(u.endswith(t) or t.endswith(u) for u in sa))
    return res


def tok_diff(ca, cb, na, nb, idf):
    items = list(zip(ca, cb, na, nb))
    step = max(1, math.ceil(len(items) / (W * 8)))
    chunks = [items[i : i + step] for i in range(0, len(items), step)]
    with Pool(W, initializer=_init_idf, initargs=(idf,)) as pool:
        res = pool.map(_tok_diff, chunks)
    return np.concatenate(res) if res else np.zeros((0, 8), np.float32)


def _init_idf(idf):
    _IDF.update(idf)


def idf_overlap(field, a, b, idf):
    items = list(zip(a, b))
    step = max(1, math.ceil(len(items) / (W * 8)))
    chunks = [(field, items[i : i + step]) for i in range(0, len(items), step)]
    with Pool(W, initializer=_init_idf, initargs=(idf,)) as pool:
        res = pool.map(_idf_overlap, chunks)
    return np.concatenate(res) if res else np.zeros((0, 3), np.float32)


def num_feats(n1, n2):
    """Numbers in address: first equal, jaccard, suffix match of first numbers."""
    a = n1.str.split(" ")
    b = n2.str.split(" ")
    df = pl.DataFrame({"a": a, "b": b})
    first_a = pl.col("a").list.first()
    first_b = pl.col("b").list.first()
    inter = pl.col("a").list.set_intersection(pl.col("b")).list.len()
    union = pl.col("a").list.set_union(pl.col("b")).list.len()
    empty = (pl.col("a").list.first() == "") | (pl.col("b").list.first() == "")
    out = df.select(
        pl.when(empty).then(None).otherwise((first_a == first_b).cast(pl.Float32)).alias("num_first_eq"),
        pl.when(empty).then(None).otherwise((inter / union).cast(pl.Float32)).alias("num_jac"),
        pl.when(empty).then(None).otherwise(inter.cast(pl.Float32)).alias("num_inter"),
        pl.when(empty)
        .then(None)
        .otherwise((first_a.str.ends_with(first_b) | first_b.str.ends_with(first_a)).cast(pl.Float32))
        .alias("num_first_suffix"),
        pl.when(empty).then(None).otherwise(
            (pl.col("a").list.set_difference(pl.col("b")).list.len() == 0).cast(pl.Float32)
        ).alias("num_a_in_b"),
    )
    return out


def pair_features(split, pairs):
    n = norm_table(split)
    i1 = pairs["i1"].to_numpy()
    i2 = pairs["i2"].to_numpy()
    A = n[i1]
    B = n[i2]
    f = {}
    log("features for", len(i1), "pairs")
    na, nb = A["name_n"].to_list(), B["name_n"].to_list()
    f["n_ratio"] = cp(na, nb, fuzz.ratio)
    f["n_partial"] = cp(na, nb, fuzz.partial_ratio)
    f["n_tset"] = cp(na, nb, fuzz.token_set_ratio)
    f["n_tsort"] = cp(na, nb, fuzz.token_sort_ratio)
    f["n_jw"] = cp(na, nb, JaroWinkler.normalized_similarity)
    ca, cb = A["core"].to_list(), B["core"].to_list()
    f["c_ratio"] = cp(ca, cb, fuzz.ratio)
    f["c_tset"] = cp(ca, cb, fuzz.token_set_ratio)
    f["c_partial"] = cp(ca, cb, fuzz.partial_ratio)
    f["c_jw"] = cp(ca, cb, JaroWinkler.normalized_similarity)
    ra, rb = A["name_raw"].to_list(), B["name_raw"].to_list()
    f["raw_ratio"] = cp(ra, rb, fuzz.ratio)
    # alternate name (text before dba / t/a / formerly) and domain stems
    alt = B["alt"].to_list()
    has_alt = np.array([bool(x) for x in alt])
    f["alt_ratio"] = np.where(has_alt, cp(ca, alt, fuzz.ratio), np.nan).astype(np.float32)
    cna, cnb = A["core_ns"].to_list(), B["core_ns"].to_list()
    f["ns_ratio"] = cp(cna, cnb, fuzz.ratio)
    dom = B["dom"].to_list()
    has_dom = np.array([bool(x) for x in dom])
    f["dom_ratio"] = np.where(has_dom, cp(cna, dom, fuzz.ratio), np.nan).astype(np.float32)
    f["dom_partial"] = np.where(has_dom, cp(cna, dom, fuzz.partial_ratio), np.nan).astype(np.float32)
    f["dom_prefix"] = np.where(
        has_dom, np.array([a.startswith(d) or d.startswith(a) for a, d in zip(cna, dom)], dtype=np.float32), np.nan
    ).astype(np.float32)
    # legal forms
    la, lb = A["legal"].to_numpy(), B["legal"].to_numpy()
    f["legal_eq"] = (la == lb).astype(np.float32)
    f["legal_both_empty"] = ((la == "") & (lb == "")).astype(np.float32)
    f["legal_b_empty"] = (lb == "").astype(np.float32)
    # addresses
    aa, ab = A["addr_n"].to_list(), B["addr_n"].to_list()
    b_empty = np.array([not x for x in ab])
    for nm, sc in (("a_ratio", fuzz.ratio), ("a_tset", fuzz.token_set_ratio), ("a_tsort", fuzz.token_sort_ratio),
                   ("a_partial", fuzz.partial_ratio), ("a_pset", fuzz.partial_token_set_ratio)):
        f[nm] = np.where(b_empty, np.nan, cp(aa, ab, sc)).astype(np.float32)
    f["addr_b_empty"] = b_empty.astype(np.float32)
    f["name_native"] = B["name_native"].cast(pl.Float32).to_numpy()
    f["addr_native"] = B["addr_native"].cast(pl.Float32).to_numpy()
    f["src"] = B["src"].cast(pl.Float32).to_numpy()
    f["len_a"] = A["name_n"].str.len_chars().cast(pl.Float32).to_numpy()
    f["len_b"] = B["name_n"].str.len_chars().cast(pl.Float32).to_numpy()
    f["alen_a"] = A["addr_n"].str.len_chars().cast(pl.Float32).to_numpy()
    f["alen_b"] = B["addr_n"].str.len_chars().cast(pl.Float32).to_numpy()
    idf = idf_table(split)
    o = idf_overlap("core", ca, cb, idf)
    f["c_idf_jac"], f["c_idf_cov"], f["c_idf_max"] = o[:, 0], o[:, 1], o[:, 2]
    o = idf_overlap("addr_n", aa, ab, idf)
    f["a_idf_jac"], f["a_idf_cov"], f["a_idf_max"] = o[:, 0], o[:, 1], o[:, 2]
    o = tok_diff(ca, cb, A["nums"].to_list(), B["nums"].to_list(), idf)
    for j, nm in enumerate(["td_extra_n", "td_miss_n", "td_extra_minidf", "td_miss_minidf",
                            "td_num_absdiff", "td_num_samelen", "td_num_affix", "td_num_unexpl"]):
        f[nm] = o[:, j]
    nf = num_feats(A["nums"], B["nums"])
    ph = (A["phone"] != "") & (B["phone"] != "")
    f["phone_eq"] = np.where(ph.to_numpy(), (A["phone"] == B["phone"]).cast(pl.Float32).to_numpy(), np.nan).astype(np.float32)
    out = pairs.with_columns([pl.Series(k, v) for k, v in f.items()]).hstack(nf)
    log("features done")
    return out


def context_features(df, score="cos", prefix="ctx"):
    """Rank and gap features of a score within each S1 and within each record."""
    s = pl.col(score)
    return df.with_columns(
        s.rank("ordinal", descending=True).over("i1").cast(pl.Float32).alias(f"{prefix}_rank1"),
        (s.max().over("i1") - s).alias(f"{prefix}_gap1"),
        pl.len().over("i1").cast(pl.Float32).alias(f"{prefix}_n1"),
        s.rank("ordinal", descending=True).over("i2").cast(pl.Float32).alias(f"{prefix}_rank2"),
        (s.max().over("i2") - s).alias(f"{prefix}_gap2"),
        pl.len().over("i2").cast(pl.Float32).alias(f"{prefix}_n2"),
        # margin to the second best S1 for this record (only meaningful for the best one)
        (s - s.top_k(2).over("i2", mapping_strategy="join").list.get(1, null_on_oob=True)).alias(f"{prefix}_margin2"),
    )
