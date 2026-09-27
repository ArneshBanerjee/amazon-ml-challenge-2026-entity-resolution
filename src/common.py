"""Shared paths and small helpers."""
import os
import time
from pathlib import Path

import polars as pl

ROOT = Path(os.environ.get("BER_ROOT", Path(__file__).resolve().parents[1]))
DATA = Path(os.environ.get("BER_DATA", ROOT.parent / "student_resource" / "dataset"))
ART = Path(os.environ.get("BER_ART", ROOT / "artifacts"))
ART.mkdir(parents=True, exist_ok=True)
SEED = int(os.environ.get("BER_SEED", 0))  # run seed; 0 is the default single run


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def read_tsv(path):
    """Read a challenge TSV with every column as a string and empty strings kept."""
    return pl.read_csv(
        path,
        separator="\t",
        quote_char=None,
        infer_schema=False,
        missing_utf8_is_empty_string=True,
    )


def done(path, force=False):
    """True when an output already exists and we are not forcing a rebuild."""
    return Path(path).exists() and not force


def prefetch(fn, items, threads=6, depth=16):
    """Yield fn(item) for items in order, computed ahead in a thread pool.

    Used for tokenization: HF fast tokenizers release the GIL, so threads overlap with GPU work
    without the fork problems of multiprocess DataLoader workers.
    """
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(threads) as ex:
        q = deque()
        it = iter(items)
        for x in it:
            q.append(ex.submit(fn, x))
            if len(q) >= depth:
                break
        while q:
            yield q.popleft().result()
            for x in it:
                q.append(ex.submit(fn, x))
                break
