"""Normalize names and addresses.

Two steps:
  mine   learn token and address-component alias maps from split A train pairs
         (native script words to Latin words, state and street abbreviations).
  apply  normalize every train and test record with the hand rules plus the mined maps.

Outputs:
  artifacts/maps.json
  artifacts/norm_{train,test}.parquet with columns
    entity_id, src, country, name_n, core, alt, legal, dom, addr_n, nums, phone,
    name_native, addr_native
"""
import argparse
import json
import os
import re
import unicodedata
from collections import Counter, defaultdict
from functools import lru_cache
from multiprocessing import Pool

import polars as pl
from anyascii import anyascii
from rapidfuzz import fuzz

from common import ART, done, log

# ---------------------------------------------------------------- hand rules
# Legal forms, mapped to one canonical token. Language knowledge, not data lookup.
LEGAL = {
    "pvt": "pvt", "private": "pvt", "pte": "pvt",
    "ltd": "ltd", "limited": "ltd", "ltda": "ltd",
    "inc": "inc", "incorporated": "inc",
    "corp": "corp", "corporation": "corp",
    "co": "co", "company": "co", "cie": "co", "compagnie": "co",
    "llc": "llc", "llp": "llp", "lp": "lp", "plc": "plc", "pllc": "pllc",
    "pc": "pc", "pa": "pa", "opc": "opc", "gmbh": "gmbh", "ag": "ag",
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "eurl": "eurl", "sci": "sci",
    "sa": "sa", "snc": "snc", "scop": "scop", "sca": "sca", "ei": "ei", "eirl": "eirl",
    "bv": "bv", "nv": "nv", "srl": "srl", "spa": "spa", "oy": "oy", "ab": "ab",
}
# Generic French business words mapped to their English equivalents, so patterns learned on English
# names (for example "& Sons" or "Group" added to a name) also apply to French names. Language knowledge.
NAME_GENERIC = {
    "associes": "associates", "associe": "associates", "fils": "sons", "freres": "brothers",
    "groupe": "group", "developpement": "development", "entreprises": "enterprises",
    "entreprise": "enterprises", "et": "and", "centre": "center", "holding": "holdings",
    "partenaires": "partners", "internationale": "international",
}
HONORIFIC = {"sri", "shri", "shree", "smt", "ms", "m/s", "mr", "mrs", "dr", "the"}
ALIAS_KW = re.compile(
    r"\b(?:doing business as|d/b/a|dba|t/a|a/k/a|aka|trading as|formerly known as|formerly|f/k/a|fka)\b:?"
)
# Generic street and address abbreviations (English and French), used for every country.
ADDR_GENERIC = {
    "r": "rue", "av": "avenue", "ave": "avenue", "bd": "boulevard", "bld": "boulevard",
    "blvd": "boulevard", "pl": "place", "che": "chemin", "ch": "chemin", "imp": "impasse",
    "all": "allee", "rte": "route", "qu": "quai", "q": "quai", "sq": "square",
    "fg": "faubourg", "fbg": "faubourg", "psg": "passage", "crs": "cours", "pass": "passage", "res": "residence",
    "apt": "apartment", "appt": "appartement", "ste": "suite", "rd": "road", "ln": "lane",
    "dr": "drive", "ct": "court", "hwy": "highway", "pkwy": "parkway", "cir": "circle",
    "ter": "terrace", "trl": "trail", "sts": "streets", "mt": "mount",
}
FILLER = {"null", "<null>", "n/a", "na", "none", "nan", "nil", "-", ""}
URL = re.compile(r"(?:https?://)?(?:www\.)?([a-z0-9][a-z0-9\-]*)\.(?:com|in|net|org|co|fr|biz|info|us|io)\b[^\s]*")
EMAIL = re.compile(r"[\w.\-]+@([\w\-]+)\.[\w.]+")
DOTTED = re.compile(r"\b[a-z](?:\.[a-z])+\b\.?")
ORD = re.compile(r"\b(\d+)(?:st|nd|rd|th)\b")
DIGITS = re.compile(r"\d+")
NONWORD = re.compile(r"[^a-z0-9/ ]+")
SPACES = re.compile(r"\s+")
NUM_SIGN = re.compile(r"\b[nN]\s*°")

MAPS = {"name_tok": {}, "addr_comp": {}, "addr_tok": {}, "drop_comp": {}}


def is_nonlatin(tok):
    for ch in tok:
        if ch.isalpha() and ord(ch) > 0x024F:
            return True
    return False


@lru_cache(maxsize=2_000_000)
def translit(tok):
    return anyascii(tok).lower()


def base_tokens(text):
    """NFKC, split on whitespace, transliterate each token. Returns [(tok, was_native)]."""
    text = NUM_SIGN.sub("no ", unicodedata.normalize("NFKC", text)).replace("°", " ")
    out = []
    for t in text.split():
        nat = is_nonlatin(t)
        out.append((translit(t) if (nat or not t.isascii()) else t.lower(), nat))
    return out


def clean_punct(s):
    s = s.replace("&", " and ").replace("+", " and ")
    s = DOTTED.sub(lambda m: m.group(0).replace(".", ""), s)
    s = s.replace("m/s", "ms")
    s = NONWORD.sub(" ", s)
    s = s.replace("/", " ")
    return SPACES.sub(" ", s).strip()


def dedupe(toks):
    out = []
    for t in toks:
        if not out or out[-1] != t:
            out.append(t)
    return out


# ---------------------------------------------------------------- name
def norm_name(raw, country):
    ntm = MAPS["name_tok"].get(country, {})
    toks = base_tokens(raw)
    native = any(n for _, n in toks)
    toks = [ntm.get(t, t) if n else t for t, n in toks]
    s = " ".join(toks)
    dom = ""
    m = EMAIL.search(s)
    if m:
        dom = m.group(1)
        s = EMAIL.sub(" ", s)
    if "|" in s:
        s, tail = s.split("|", 1)
        m = URL.search(tail)
        if m and not dom:
            dom = m.group(1)
    m = URL.search(s)
    if m:
        if not dom:
            dom = m.group(1)
        s = URL.sub(lambda mm: " " + mm.group(1) + " ", s)
    s = s.strip()
    # social handles like @name or #name
    hm = re.match(r"^[@#]([a-z0-9_]+)$", s)
    if hm and not dom:
        dom = hm.group(1)
    dom = dom.replace("-", "").replace("_", "")
    parts = ALIAS_KW.split(s, maxsplit=1)
    main, alt = (parts[1], parts[0]) if len(parts) == 2 else (s, "")
    main = " ".join(NAME_GENERIC.get(t, t) for t in clean_punct(main).split())
    alt = " ".join(NAME_GENERIC.get(t, t) for t in clean_punct(alt).split())
    legal = set()

    def core_of(x):
        out = []
        for t in x.split():
            if t in LEGAL:
                legal.add(LEGAL[t])
            elif t in HONORIFIC:
                continue
            else:
                out.append(t)
        return " ".join(dedupe(out))

    core = core_of(main)
    alt_core = core_of(alt)
    name_n = " ".join(dedupe([LEGAL.get(t, t) for t in main.split()]))
    if not core:
        core = name_n
    return name_n, core, alt_core, " ".join(sorted(legal)), dom, native


# ---------------------------------------------------------------- address
def addr_components(raw):
    """Split into comma components, transliterate, clean, drop fillers."""
    comps = []
    native = False
    for c in raw.split(","):
        toks = base_tokens(c)
        native |= any(n for _, n in toks)
        s = " ".join(t for t, _ in toks)
        if s.strip() in FILLER:
            continue
        comps.append(s)
    return comps, native


def norm_addr(raw, country):
    acm = MAPS["addr_comp"].get(country, {})
    atm = MAPS["addr_tok"].get(country, {})
    drop = set(MAPS["drop_comp"].get(country, []))
    comps, native = addr_components(raw)
    out = []
    nums, phone = [], []
    for c in comps:
        c = ORD.sub(r"\1", c)
        c = clean_punct(c)
        if c in FILLER or c in drop:
            continue
        c = acm.get(c, c)
        toks = []
        for t in c.split():
            if t.isdigit():
                t = t.lstrip("0") or "0"
            t = atm.get(t, ADDR_GENERIC.get(t, t))
            toks.append(t)
        c = " ".join(dedupe(toks))
        for d in DIGITS.findall(c):
            d = d.lstrip("0") or "0"
            (phone if len(d) >= 7 else nums).append(d)
        out.append(c)
    return ", ".join(out), " ".join(nums), " ".join(phone), native


def norm_row(args):
    name, addr, country = args
    name_n, core, alt, legal, dom, nn = norm_name(name, country)
    addr_n, nums, phone, an = norm_addr(addr, country)
    return name_n, core, alt, legal, dom, addr_n, nums, phone, nn, an


def _init(maps):
    MAPS.update(maps)


def run_rows(names, addrs, countries, maps, procs=None):
    with Pool(procs, initializer=_init, initargs=(maps,)) as pool:
        return pool.map(norm_row, zip(names, addrs, countries), chunksize=20000)


# ---------------------------------------------------------------- mining
def _mine_pair(args):
    """Return candidate (country, kind, variant, canonical) tuples from one true pair."""
    s1_name, s1_addr, o_name, o_addr, country = args
    out = []
    a = base_tokens(s1_name)
    b = base_tokens(o_name)
    if len(a) == len(b) and any(n for _, n in b):
        for (ta, _), (tb, nb) in zip(a, b):
            if nb:
                out.append((country, "name_tok", tb, clean_punct(ta)))
    ca, _ = addr_components(s1_addr)
    cb, _ = addr_components(o_addr)
    drop = set(MAPS["drop_comp"].get(country, []))
    ca = [x for x in (clean_punct(ORD.sub(r"\1", c)) for c in ca) if x not in drop]
    cb = [x for x in (clean_punct(ORD.sub(r"\1", c)) for c in cb) if x not in drop]
    sa, sb = set(ca), set(cb)
    da, db = sa - sb, sb - sa
    if len(da) == 1 and len(db) == 1:
        x, y = db.pop(), da.pop()
        if not any(ch.isdigit() for ch in x + y):
            out.append((country, "addr_comp", x, y))
        tx, ty = x.split(), y.split()
        if len(tx) == len(ty):
            diff = [(p, q) for p, q in zip(tx, ty) if p != q]
            if len(diff) == 1 and not diff[0][0].isdigit() and not diff[0][1].isdigit():
                out.append((country, "addr_tok", diff[0][0], diff[0][1]))
    return out


def is_subseq(a, b):
    it = iter(b)
    return all(ch in it for ch in a)


def plausible(kind, var, can):
    """Address token maps must look like an abbreviation or a typo, not a substitution."""
    if kind == "name_tok":
        return True
    if len(var) < 2:
        return False
    if kind == "addr_comp":
        return True
    return (var[0] == can[0] and is_subseq(var, can)) or fuzz.ratio(var, can) >= 75


def mine_pairs(pairs, min_n=None):
    """pairs: frame with n1, a1 (S1 raw name/address), n2, a2 (record raw), country."""
    log("mining from", pairs.height, "pairs")
    cnt = Counter()
    with Pool(os.cpu_count()) as pool:
        for res in pool.imap_unordered(
            _mine_pair,
            zip(pairs["n1"], pairs["a1"], pairs["n2"], pairs["a2"], pairs["country"]),
            chunksize=20000,
        ):
            cnt.update(res)
    # keep consistent, frequent mappings
    by_var = defaultdict(Counter)
    for (country, kind, var, can), n in cnt.items():
        if var != can:
            by_var[(country, kind, var)][can] += n
    min_n = min_n or {"name_tok": 3, "addr_comp": 20, "addr_tok": 20}
    maps = {"name_tok": defaultdict(dict), "addr_comp": defaultdict(dict), "addr_tok": defaultdict(dict)}
    min_n = dict(min_n)
    for (country, kind, var), c in by_var.items():
        can, n = min(c.items(), key=lambda kv: (-kv[1], kv[0]))  # deterministic tie-break
        tot = sum(c.values())
        if n >= min_n[kind] and n / tot >= 0.6 and can and plausible(kind, var, can):
            maps[kind][country][var] = can
    for kind in maps:
        for country in maps[kind]:
            m = maps[kind][country]
            # resolve chains a->b->c
            for k in list(m):
                v, seen = m[k], {k}
                while v in m and v not in seen:
                    seen.add(v)
                    v = m[v]
                m[k] = v
            for k in [k for k, v in m.items() if k == v]:
                del m[k]
    return {k: {c: dict(v) for c, v in d.items()} for k, d in maps.items()}


def mine(force=False):
    out_path = ART / "maps.json"
    if done(out_path, force):
        return json.loads(out_path.read_text())
    rec = pl.read_parquet(ART / "records_train.parquet")
    split = pl.read_parquet(ART / "split_s1.parquet").filter(pl.col("grp") == "A")
    gt = pl.read_parquet(ART / "gt_pairs.parquet").join(split.select("s1"), on="s1")
    s1 = rec.filter(pl.col("src") == 1).select(
        pl.col("entity_id").alias("s1"), pl.col("name").alias("n1"), pl.col("addr").alias("a1"), "country"
    )
    oth = rec.filter(pl.col("src") != 1).select(pl.col("entity_id").alias("rec"), pl.col("name").alias("n2"), pl.col("addr").alias("a2"))
    pairs = gt.join(s1, on="s1").join(oth, on="rec")
    maps = mine_pairs(pairs)
    maps["drop_comp"] = {}
    for k, d in maps.items():
        for c, v in d.items():
            log("mined", k, c, len(v))
    out_path.write_text(json.dumps(maps, ensure_ascii=False, indent=0, sort_keys=True))
    return maps


def apply(maps, force=False, splits=("train", "test")):
    for split in splits:
        out = ART / f"norm_{split}.parquet"
        if done(out, force):
            continue
        rec = pl.read_parquet(ART / f"records_{split}.parquet")
        log("normalizing", split, rec.height)
        res = run_rows(rec["name"].to_list(), rec["addr"].to_list(), rec["country"].to_list(), maps)
        cols = list(zip(*res))
        names = ["name_n", "core", "alt", "legal", "dom", "addr_n", "nums", "phone", "name_native", "addr_native"]
        df = rec.select("entity_id", "src", "country").with_columns(
            [pl.Series(n, list(c)) for n, c in zip(names, cols)]
        )
        df.write_parquet(out)
        log("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--force-mine", action="store_true")
    ap.add_argument("--test-only", action="store_true", help="re-normalize only the test split with the current maps")
    a = ap.parse_args()
    maps = mine(a.force_mine)
    if a.test_only:
        apply(maps, True, ("test",))
    else:
        apply(maps, a.force or a.force_mine)
