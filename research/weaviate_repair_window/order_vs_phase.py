#!/usr/bin/env python3
"""
Step 2b of issue #48: is the bimodal repair latency an ORDER effect or a
TIMING effect?

PROVENANCE -- read this before trusting the file. The run that produced
`results/order_vs_phase.json` was executed from an inline scratch script that
was never committed, which is a reproducibility defect in PR #49. This file is
a FAITHFUL RECONSTRUCTION of that protocol from the committed log's columns
(`rep`, `pre_delay_s`, `repair_s`, `first_answer`) and the spec's description,
not the byte-identical original. Re-running it should reproduce the
distribution, not the individual numbers -- the delays are random and the
effect under study is a coin flip weighted by timing.

The design: repeat a FIXED divergence (50 objects) 10 times, with a random
0-50 s delay before each restart. Fixing the size removes size as an
explanation; randomizing the delay scrambles timing while leaving repetition
index intact, so the two explanations separate:

  warming  -- repair gets faster with repetition => strongly NEGATIVE
              correlation(repair_s, rep)
  timing   -- repetition index carries no information => correlation near zero,
              and the delay predicts the outcome instead

Usage:
    python research/weaviate_repair_window/order_vs_phase.py [--reps 10]
                                                             [--size 50]
                                                             [--max-delay 50]
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
from characterize import write_objects, wait_ready, VICTIM, PEERS  # noqa: E402


def log(m):
    print(m, flush=True)


def one_run(shard, size, seed, pre_delay):
    """Stop the victim, write `size` objects it cannot see, wait `pre_delay`
    seconds, restart it, and time the repair from its FIRST successful probe
    response (so restart latency is excluded, per #48's Confounds)."""
    ids = [str(uuid.UUID(int=seed * 10_000_000 + i)) for i in range(size)]
    subprocess.run(["docker", "stop", t.container_name(VICTIM)], capture_output=True)
    time.sleep(3)
    ok, _ = write_objects(ids, seed, consistency="ONE")
    if not ok:
        subprocess.run(["docker", "start", t.container_name(VICTIM)], capture_output=True)
        return None

    time.sleep(pre_delay)
    subprocess.run(["docker", "start", t.container_name(VICTIM)], capture_output=True)

    first_t = first_n = None
    last = None
    deadline = time.time() + 300
    while time.time() < deadline:
        ok, got = objects_present_ids(VICTIM, shard, ids)
        if ok:
            if first_t is None:
                first_t, first_n = time.time(), len(got)
            last = (round(time.time() - first_t, 3), len(got))
            if len(got) == size:
                break
    return {"repair_s": last[0] if last and last[1] == size else None,
            "first": first_n, "converged": bool(last and last[1] == size)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--size", type=int, default=50)
    ap.add_argument("--max-delay", type=float, default=50.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(a.seed)

    shard = ia.shard_name(0)
    log(f"shard: {shard}   size: {a.size}   reps: {a.reps}")
    log("rep  pre_delay_s  repair_s   first_answer")

    rows = []
    for rep in range(a.reps):
        pre = round(rng.uniform(0, a.max_delay), 1)
        r = one_run(shard, a.size, seed=1000 + rep, pre_delay=pre)
        if r is None:
            log(f"  {rep:>2}  {pre:>10}   WRITE FAILED")
            continue
        rows.append({"rep": rep, "delay": pre, **r})
        log(f"  {rep:>2}  {pre:>10}  {r['repair_s']:>8}   {r['first']}/{a.size}")
        wait_ready(VICTIM)

    ok = [r for r in rows if r["converged"]]
    slow = [r["rep"] for r in ok if r["repair_s"] > 1.0]
    fast = [r["rep"] for r in ok if r["repair_s"] <= 1.0]
    log(f"\nslow (>1s) at reps: {slow}\nfast (<=1s) at reps: {fast}")
    if len(ok) > 2:
        xs = [r["rep"] for r in ok]
        ys = [r["repair_s"] for r in ok]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        vx = sum((x - mx) ** 2 for x in xs) ** 0.5
        vy = sum((y - my) ** 2 for y in ys) ** 0.5
        log(f"\ncorrelation(repair_time, repetition_index) = {cov / (vx * vy):+.3f}")
        log("  strongly negative => WARMING (later runs faster)")
        log("  near zero         => PHASE (order carries no information)")

    p = os.path.join(a.out, "order_vs_phase.json")
    with open(p, "w") as f:
        json.dump(rows, f, indent=1)
    log(f"\nwrote {p}\ndone")
    return 0


if __name__ == "__main__":
    sys.exit(main())
