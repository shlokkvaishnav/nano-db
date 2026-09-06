#!/usr/bin/env python3
"""Verify SIFT1M is usable before a sweep runs against it (#53).

A truncated or partial download silently shrinks the corpus, and every number
downstream changes with it — quietly, because nothing in the harness checks.
That is the same failure shape this project keeps finding in its own
instruments (#26, #38, #46, and #48's 20,000-id probe), so the corpus gets an
explicit check rather than an assumption.

WHAT "COMPLETE" MEANS HERE, and why the first version of this file was wrong.
It initially asserted exactly 1,000,000 base vectors and failed a perfectly
good corpus at 350,000. `sift.py` fetches a **prefix** on purpose — it sends an
HTTP `Range: bytes=0-N` header for `n_base * record_bytes(dim)` bytes, because
the experiment only needs `--sift-vectors` of them (default 200,000) and the
whole file is 516 MB. So the corpus is not required to be complete; it is
required to be **well-formed and long enough for the run about to use it**.

That distinction is the whole check. Asserting the stronger property would have
blocked a valid corpus, which is the same mistake as a checker that cries wolf.

Reads only the `.fvecs` header: each record is a 4-byte little-endian dimension
followed by `dim` float32s, so file size plus the first dimension give the
vector count without reading hundreds of MB.

Usage:
    python research/layer1_reproduction/verify_corpus.py
    python research/layer1_reproduction/verify_corpus.py --vectors 200000
"""
from __future__ import annotations

import argparse
import os
import struct
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", ".."))

BASE = "research/replica_recall/data/sift_base.fvecs"
QUERY = "research/replica_recall/data/sift_query.fvecs"
DIM = 128            # matches config::VECTOR_DIM; the reason SIFT was chosen
QUERY_VECTORS = 10_000   # the query set is taken whole, so this is exact


def describe(rel):
    """(dim, n, size) from the header, or None if unreadable/misaligned."""
    path = os.path.join(ROOT, rel)
    if not os.path.exists(path):
        print(f"  MISSING   {rel}")
        return None
    size = os.path.getsize(path)
    with open(path, "rb") as fh:
        head = fh.read(4)
    if len(head) < 4:
        print(f"  CORRUPT   {rel}: shorter than one header")
        return None
    dim = struct.unpack("<i", head)[0]
    if dim <= 0:
        print(f"  CORRUPT   {rel}: header dim={dim}")
        return None
    rec = 4 + 4 * dim
    if size % rec:
        print(f"  CORRUPT   {rel}: {size:,} bytes is not a whole number of "
              f"{rec}-byte records (a partial write, not a prefix)")
        return None
    return dim, size // rec, size


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors", type=int, default=200_000,
                    help="how many base vectors the sweep will use "
                         "(run_experiment.py's --sift-vectors default)")
    a = ap.parse_args()

    print(f"corpus verification (need >= {a.vectors:,} base vectors at {DIM}-d)")
    ok = True

    b = describe(BASE)
    if b is None:
        ok = False
    else:
        dim, n, size = b
        good = dim == DIM and n >= a.vectors
        ok = ok and good
        note = ("" if good else
                f"  <- need dim={DIM} and >= {a.vectors:,} vectors")
        print(f"  {'OK       ' if good else 'INSUFFICIENT'} {BASE}: dim={dim} "
              f"vectors={n:,} bytes={size:,}{note}")
        if good and n < 1_000_000:
            print(f"             (a {n:,}-vector prefix, which is how sift.py "
                  f"fetches it — not truncation)")

    q = describe(QUERY)
    if q is None:
        ok = False
    else:
        dim, n, size = q
        good = dim == DIM and n == QUERY_VECTORS
        ok = ok and good
        print(f"  {'OK       ' if good else 'MISMATCH  '} {QUERY}: dim={dim} "
              f"vectors={n:,} bytes={size:,}"
              + ("" if good else f"  <- expected exactly {QUERY_VECTORS:,}"))

    if ok:
        print("corpus OK")
        return 0
    print("\ncorpus FAILED verification -- do not run the sweep against it. "
          "Re-fetch with:\n"
          f"  python research/replica_recall/sift.py --vectors {a.vectors}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
