#!/usr/bin/env python3
"""
Step 2c of issue #48: WHAT selects the repair path?

Step 2b established that Weaviate's repair latency is bimodal -- sub-0.2 s or
36-50 s, with nothing in between across 18 runs -- and that the delay before
the restart is associated with which path is taken (exact Mann-Whitney
p = 0.0190, near-separated at a ~13-16 s threshold). That association is
post-hoc, and it confounds two quantities that step 2b varied together:

    the victim was ABSENT longer   AND   the divergence was OLDER

because the delay sat between the write and the restart. No amount of extra
data in that design separates them. Three conditions do, at a fixed
divergence of 50 objects:

    A  short          absent ~6 s   divergence age ~6 s
    B  long           absent ~40 s  divergence age ~40 s
    C  long, young    absent ~40 s  divergence age ~6 s   <- discriminating

C writes the objects in the LAST few seconds of a long outage. If ABSENCE
selects the path, C is slow like B. If DIVERGENCE AGE selects it, C is fast
like A. Conditions are interleaved in randomized order so repetition index
cannot align with condition (the confound step 2b was built to break).

Pre-registered in SPEC.md step 2c before this ran. Outcomes (a) absence,
(b) age, (c) neither -- all conditions mixed at similar rates, (d) mixed at
different rates.

Usage:
    python research/weaviate_repair_window/path_selection.py [--reps 6]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
for _p in ("weaviate_probe", "weaviate_nonperturbing_probe", "weaviate_probe_per_id"):
    sys.path.insert(0, os.path.join(ROOT, "research", _p))

import weaviate_topology as t                      # noqa: E402
import internal_api as ia                          # noqa: E402
from per_id import objects_present_ids             # noqa: E402
from characterize import write_objects, wait_ready, VICTIM  # noqa: E402

SLOW_S = 1.0        # the 0.2-36 s gap makes this threshold unambiguous
GAP_LO, GAP_HI = 0.2, 36.0


def log(m):
    print(m, flush=True)


def one_run(shard, cond, size, seed, absent_s, age_s):
    """`absent_s` is the victim's total downtime; `age_s` is how long the
    divergence has existed when the victim restarts. The write is placed
    (absent_s - age_s) into the outage to hit both targets at once."""
    ids = [str(uuid.UUID(int=seed * 10_000_000 + i)) for i in range(size)]
    t_stop = time.time()
    subprocess.run(["docker", "stop", t.container_name(VICTIM)], capture_output=True)

    before = max(0.0, absent_s - age_s - (time.time() - t_stop))
    time.sleep(before)

    t_write = time.time()
    ok, _ = write_objects(ids, seed, consistency="ONE")
    if not ok:
        subprocess.run(["docker", "start", t.container_name(VICTIM)], capture_output=True)
        return {"cond": cond, "error": "write failed"}

    time.sleep(max(0.0, absent_s - (time.time() - t_stop)))
    t_start = time.time()
    subprocess.run(["docker", "start", t.container_name(VICTIM)], capture_output=True)

    first_t = first_n = None
    last = None
    deadline = time.time() + 300
    while time.time() < deadline:
        okp, got = objects_present_ids(VICTIM, shard, ids)
        if okp:
            if first_t is None:
                first_t, first_n = time.time(), len(got)
            last = (round(time.time() - first_t, 3), len(got))
            if len(got) == size:
                break
    conv = bool(last and last[1] == size)
    return {
        "cond": cond, "seed": seed,
        "absent_s": round(t_start - t_stop, 1),      # realized, not requested
        "age_s": round(t_start - t_write, 1),
        "repair_s": last[0] if conv else None,
        "first": first_n, "converged": conv,
    }


def fisher_2x2(a, b, c, d):
    """Two-sided Fisher exact. Small counts only; no scipy dependency."""
    from math import comb
    n = a + b + c + d
    r1, c1 = a + b, a + c

    def p(x):
        return comb(r1, x) * comb(n - r1, c1 - x) / comb(n, c1)
    p0 = p(a)
    lo = max(0, c1 - (n - r1))
    return sum(p(x) for x in range(lo, min(r1, c1) + 1) if p(x) <= p0 + 1e-12)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--size", type=int, default=50)
    ap.add_argument("--short", type=float, default=6.0)
    ap.add_argument("--long", type=float, default=40.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    conds = {
        "A_short":      (a.short, a.short),
        "B_long":       (a.long,  a.long),
        "C_long_young": (a.long,  a.short),
    }
    order = [c for c in conds for _ in range(a.reps)]
    random.Random(a.seed).shuffle(order)          # interleaved, not blocked

    shard = ia.shard_name(0)
    log(f"shard: {shard}   size: {a.size}   reps/cond: {a.reps}")
    log("order: " + " ".join(c[0] for c in order))
    log("\n  i  cond            absent_s  age_s  repair_s   first")

    rows = []
    for i, cond in enumerate(order):
        absent_s, age_s = conds[cond]
        r = one_run(shard, cond, a.size, seed=2000 + i, absent_s=absent_s, age_s=age_s)
        r["i"] = i
        rows.append(r)
        log(f"  {i:>2}  {cond:<14} {r.get('absent_s'):>8} {r.get('age_s'):>6} "
            f"{str(r.get('repair_s')):>9}   {r.get('first')}/{a.size}")
        with open(os.path.join(a.out, "path_selection.json"), "w") as f:
            json.dump(rows, f, indent=1)
        wait_ready(VICTIM)

    log("\n=== result ===")
    ok = [r for r in rows if r.get("converged")]
    in_gap = [r for r in ok if GAP_LO < r["repair_s"] < GAP_HI]
    counts = {}
    for c in conds:
        cr = [r for r in ok if r["cond"] == c]
        s = sum(1 for r in cr if r["repair_s"] > SLOW_S)
        counts[c] = (s, len(cr))
        times = sorted(round(r["repair_s"], 3) for r in cr)
        log(f"  {c:<14} slow {s}/{len(cr)}   {times}")

    sA, nA = counts["A_short"]
    sB, nB = counts["B_long"]
    sC, nC = counts["C_long_young"]
    pAC = fisher_2x2(sA, nA - sA, sC, nC - sC)
    pAB = fisher_2x2(sA, nA - sA, sB, nB - sB)
    log(f"\n  Fisher exact A vs C: p = {pAC:.4f}   (the discriminating comparison)")
    log(f"  Fisher exact A vs B: p = {pAB:.4f}   (the step 2b association)")
    if in_gap:
        log(f"\n  !! {len(in_gap)} run(s) landed INSIDE the 0.2-36 s gap: "
            f"{[r['repair_s'] for r in in_gap]} -- this weakens the two-path reading")

    verdict = "(d) mixed -- report rates, name no mechanism"
    if sC == sB and sA != sB:
        verdict = "(a) ABSENCE selects the path -- C behaves like B"
    elif sC == sA and sA != sB:
        verdict = "(b) DIVERGENCE AGE selects the path -- C behaves like A"
    elif sA == sB == sC:
        verdict = "(c) neither -- step 2b's association does not replicate"
    log(f"\n  pre-registered outcome: {verdict}")

    with open(os.path.join(a.out, "path_selection.json"), "w") as f:
        json.dump(rows, f, indent=1)
    log(f"\nwrote {os.path.join(a.out, 'path_selection.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
