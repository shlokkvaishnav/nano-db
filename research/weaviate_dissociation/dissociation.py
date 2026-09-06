#!/usr/bin/env python3
"""
Does `completeness` heal on Weaviate while `index_recall` does not? (issue #54)

The experiment the four Weaviate method studies (#41, #43, #46, #48) were built
to make runnable, and the only one that tests this project's field-level claim:
object-level anti-entropy repairs the DATA and cannot repair the GRAPH, because
two correct HNSW graphs over identical data differ bit-for-bit.

THE INSTRUMENT IS ASYMMETRIC AND THAT IS PERMANENT. `_search` is unavailable on
the cluster-internal port (#43, 415 across 12 content types), so:

    completeness   sampled every 1 s, peers up, non-perturbing      (#43/#46)
    index_recall   two snapshots, via isolation probing, ~10 min
                   of node health each                              (#41)

So this compares a time series against a pair of endpoints. Any writeup must
carry that in the claim rather than implying both axes were watched.

GROUND TRUTH NEEDS NO SERVER SUPPORT. The corpus comes from a seeded RNG, so
the exact top-k over *the subset a replica actually holds* -- which
objects_present_ids reports -- is computable locally with numpy. index_recall is
the replica's ANN answer against exact search over its own data, holding data
constant. Same definition as nano-db and Qdrant.

Usage:
    python research/weaviate_dissociation/dissociation.py --seeds 5 --dry-run
    python research/weaviate_dissociation/dissociation.py --seed 20260900
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for _p in ("weaviate_probe", "weaviate_nonperturbing_probe", "weaviate_probe_per_id",
           "weaviate_repair_window"):
    sys.path.insert(0, os.path.join(ROOT, "research", _p))

import weaviate_topology as t                      # noqa: E402
import internal_api as ia                          # noqa: E402
from per_id import objects_present_ids             # noqa: E402
from characterize import write_objects, wait_ready  # noqa: E402

VICTIM = 2
PEERS = (0, 1)

# From #48, all measured rather than chosen. See SPEC.md.
DIVERGENCE = 5_000      # size-independent repair, but 5k gives a resolvable ramp
CADENCE_S = 1.0         # #43: 5 s proven; 1 s resolves the ~6 s ramp at this size
OBSERVE_S = 60.0        # young-regime wait is ~32 s (#56); the slowest repair
                        # seen anywhere is 53.3 s, so the margin is 6.7 s, not
                        # comfortable. Amendment 1: a series that never completes
                        # is RIGHT-censored, not a failure to heal.
ID_CAP = 15_000         # base64 ids in the URL fail SILENTLY above this
K = 10


def log(m):
    print(m, flush=True)


def search_full_ids(node, vec, k):
    """k-NN against one node, returning FULL object UUIDs.

    Deliberately not `feasibility_check.search()`, which returns `(ok, [vid])`
    where `vid` is the last 8 characters of the uuid, set as a property at write
    time. Comparing truncated ids against a ground-truth set of full uuids would
    silently score ~0 -- and would look like a real finding rather than a key
    mismatch. This asks for `_additional { id }` instead.
    """
    q = {"query": ("{ Get { %s(nearVector: {vector: %s}, limit: %d) "
                   "{ _additional { id } } } }")
                  % (t.CLASS_NAME, json.dumps([float(x) for x in vec]), k)}
    st, body = t.http_request(t.http_port(node), "POST", "/v1/graphql", q, timeout=30)
    if st != 200 or not isinstance(body, dict) or body.get("errors"):
        return False, []
    got = (body.get("data") or {}).get("Get", {}).get(t.CLASS_NAME) or []
    return True, [(o.get("_additional") or {}).get("id") for o in got]


def corpus(seed, n, offset=0):
    """Deterministic corpus: ids and vectors regenerate from the seed, which is
    what makes local exact ground truth possible.

    `offset` separates the BASE set from the DIVERGENCE set. They must be
    disjoint id ranges: re-writing the same ids while the victim is down would
    leave it missing only the UPDATES, and objects_present_ids checks presence
    rather than version -- so the victim would read as complete and the
    experiment would report a false negative on its primary metric.
    """
    rng = np.random.default_rng(seed + offset)
    ids = [str(uuid.UUID(int=seed * 10_000_000 + offset + i)) for i in range(n)]
    vecs = rng.standard_normal((n, t.VECTOR_DIM)).astype(np.float32)
    return ids, vecs


def exact_topk(query, vecs, held_idx, k):
    """Exact top-k over ONLY the vectors this replica holds -- the 'data held
    constant' half of index_recall's definition."""
    if len(held_idx) == 0:
        return []
    sub = vecs[held_idx]
    d = np.linalg.norm(sub - query, axis=1)
    order = np.argsort(d)[:k]
    return [held_idx[i] for i in order]


def index_recall_snapshot(node, shard, ids, vecs, queries, dry):
    """Isolate `node`, ask it k-NN for each query at consistency ONE, compare
    against exact search over the ids it actually holds, restore peers.

    COSTS ~10 MINUTES OF NODE HEALTH (#41). That is why this is a snapshot and
    not a series, and why the experiment is asymmetric.
    """
    if dry:
        log(f"    [dry] isolate node{node}, {len(queries)} queries, restore")
        return None
    for p in PEERS:
        if p != node:
            subprocess.run(["docker", "stop", t.container_name(p)], capture_output=True)
    time.sleep(3)
    try:
        ok, held = objects_present_ids(node, shard, ids[:ID_CAP])
        if not ok:
            return None
        held_idx = np.array([i for i, x in enumerate(ids[:ID_CAP]) if x in held])
        hits = tot = 0
        for q in queries:
            okq, got = search_full_ids(node, q, K)
            truth = set(ids[i] for i in exact_topk(q, vecs, held_idx, K))
            if not okq or not truth:
                continue
            got_ids = {g for g in got if g}
            hits += len(got_ids & truth)
            tot += len(truth)
        return {"index_recall": (hits / tot) if tot else None,
                "held": int(len(held_idx)), "queries": len(queries)}
    finally:
        for p in PEERS:
            if p != node:
                subprocess.run(["docker", "start", t.container_name(p)],
                               capture_output=True)
        for p in PEERS:
            if p != node:
                wait_ready(p)


def recovery_with_censoring(series):
    """Time-to-completeness-recovery, with censoring recorded, per Amendment 1.

    A 1 s cadence cannot tell "repair was instantaneous" from "the first sample
    landed after repair finished" -- #56's probe-perturbation check was voided by
    exactly that, reporting repair_s = 0.000 with the victim already holding
    50/50 ids on its first sample. So a t=0 completion is CENSORED, not instant.

      left   -- the first sample was already complete. Recovery happened at or
                before that sample's offset. The bound is real; the time is not.
      right  -- the last sample was still incomplete. Recovery did not occur
                within the window. NOT evidence that it never would.
      none   -- an incomplete first sample and a complete later one. This is the
                only case that yields a recovery TIME.

    Censored runs are kept. For the primary metric a left-censored run still
    answers "did completeness return to 1.0 within the window" with yes -- the
    censoring bounds *when*, not *whether*.
    """
    out = {"first_sample_offset_s": None, "recovery_s": None,
           "recovery_bound_s": None, "censored": None}
    ok = [s for s in series if s["ok"] and s["n"] is not None]
    if not ok:
        out["censored"] = "no-data"
        return out

    out["first_sample_offset_s"] = ok[0]["t"]
    complete = [s for s in ok if s["n"] >= DIVERGENCE]

    if not complete:
        out["censored"] = "right"
        out["recovery_bound_s"] = ok[-1]["t"]
        return out

    if ok[0]["n"] >= DIVERGENCE:
        # Already complete on the first look: repair fell inside the blind spot
        # between the restart and the first probe.
        out["censored"] = "left"
        out["recovery_bound_s"] = ok[0]["t"]
        return out

    out["censored"] = "none"
    out["recovery_s"] = complete[0]["t"]
    return out


def one_run(seed, chaos, shard, dry):
    """One seed, one condition.

    Two disjoint id sets, and the distinction is load-bearing:

      BASE       written at consistency ALL, present on every replica.
                 index_recall is measured over this -- the graph axis.
      DIVERGENCE written at ONE while the victim is down, so they are ids the
                 victim has NEVER SEEN. completeness is measured over this --
                 the data axis.

    Re-writing the base ids instead would leave the victim missing only updates,
    which a presence probe cannot see.
    """
    base_ids, base_vecs = corpus(seed, DIVERGENCE, offset=0)
    div_ids, _ = corpus(seed, DIVERGENCE, offset=DIVERGENCE)
    rng = np.random.default_rng(seed + 1)
    queries = rng.standard_normal((20, t.VECTOR_DIM)).astype(np.float32)
    rec = {"seed": seed, "chaos": chaos, "divergence": DIVERGENCE}

    log("")
    log(f"--- seed {seed} chaos={chaos} ---")
    log(f"  writing BASE corpus ({DIVERGENCE} ids) at consistency ALL")
    if not dry:
        write_objects(base_ids, seed, consistency="ALL")

    log("  index_recall snapshot BEFORE (isolation probe, ~10 min node health)")
    rec["index_recall_before"] = index_recall_snapshot(
        VICTIM, shard, base_ids, base_vecs, queries, dry)

    if chaos:
        log(f"  divergence: stop victim, write {DIVERGENCE} NEW ids it cannot see")
        if not dry:
            subprocess.run(["docker", "stop", t.container_name(VICTIM)],
                           capture_output=True)
            time.sleep(3)
            t_write0 = time.time()
            write_objects(div_ids, seed + DIVERGENCE, consistency="ONE")
            t_write = time.time()
            subprocess.run(["docker", "start", t.container_name(VICTIM)],
                           capture_output=True)
            t_restart = time.time()
            # SPEC Amendment 1 (2026-09-06). #56 reported: within a regime the
            # repair clock runs from the RESTART, and divergence age selects the
            # regime. So the origin is the restart, not max(write, restart) --
            # under the old rule a write that takes minutes opens the window
            # AFTER repair has already fired.
            rec["write_s"] = round(t_write - t_write0, 2)
            # Realized divergence age. ~0 here by construction, which puts this
            # run in #56's young regime (expect a ~32 s wait). Recorded, not
            # assumed: a run above 15 s is out of that regime and is reported
            # separately rather than pooled -- pooling across the step is what
            # gave #56 its wrong aggregate answer.
            rec["age_s"] = round(t_restart - t_write, 3)
            rec["regime"] = "young" if rec["age_s"] < 15.0 else "out-of-regime"
            origin = t_restart
        else:
            origin = time.time()

        log(f"  sampling completeness over the DIVERGENCE set every "
            f"{CADENCE_S}s for {OBSERVE_S}s from the restart")
        series = []
        if not dry:
            end = origin + OBSERVE_S
            while time.time() < end:
                ok, got = objects_present_ids(VICTIM, shard, div_ids[:ID_CAP])
                series.append({"t": round(time.time() - origin, 2),
                               "n": len(got) if ok else None, "ok": ok})
                time.sleep(CADENCE_S)
        rec["completeness_series"] = series
        last = series[-1]["n"] if series and series[-1]["n"] is not None else None
        rec["completeness_end"] = (last / DIVERGENCE) if last is not None else None
        rec.update(recovery_with_censoring(series))

    log("  index_recall snapshot AFTER")
    rec["index_recall_after"] = index_recall_snapshot(
        VICTIM, shard, base_ids, base_vecs, queries, dry)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, action="append",
                    help="repeatable; default is 5 pre-registered seeds")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    a = ap.parse_args()
    seeds = a.seed or [20260900, 20260901, 20260902, 20260903, 20260904]
    os.makedirs(a.out, exist_ok=True)

    if not a.dry_run:
        okc, info = t.verify_class(0)
        log(f"topology check (#46's hazard): {okc}  {info}")
        if not okc:
            log("REFUSING TO RUN: the class is not factor 3 / 1 shard. A stale "
                "auto-schema class would give per-replica numbers from a "
                "topology with no replicas.")
            return 2

    shard = ia.shard_name(0) if not a.dry_run else "DRYSHARD"
    log(f"shard: {shard}   divergence: {DIVERGENCE}   cadence: {CADENCE_S}s   "
        f"observe: {OBSERVE_S}s")

    rows = []
    for seed in seeds:
        for chaos in (False, True):        # the no-chaos control is REQUIRED
            rows.append(one_run(seed, chaos, shard, a.dry_run))
            with open(os.path.join(a.out, "dissociation.json"), "w") as f:
                json.dump(rows, f, indent=1)
    log(f"\nwrote {os.path.join(a.out, 'dissociation.json')}")
    log("\nReminder for the writeup: completeness is a 1 s series, index_recall "
        "is two endpoints.\nThe dissociation, if found, is between a series and "
        "a pair of snapshots -- say so.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
