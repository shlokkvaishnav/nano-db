#!/usr/bin/env python3
"""
Instrument characterization for Weaviate repair (issue #48).

`SPEC_TEMPLATE.md` requires this before a confirmatory sweep, because three
of this project's sweeps were voided by apparatus properties that were
computable in advance. Two quantities decide whether Weaviate's async repair
can be measured at all:

  step 1  probe cost      -- the sampling floor: how fast can objects_present_ids
                             be called, as a function of id-set size
  step 2  repair duration -- as a function of divergence size, measured from
                             the victim's FIRST successful probe response (not
                             from `docker start`, which would count restart
                             latency as repair)

and the decision they produce:

  step 3  samples per repair window = duration / achievable interval,
          smallest divergence size where that is >= 10

Usage:
    python research/weaviate_repair_window/characterize.py [--sizes 50,500,5000]
                                                           [--reps 3] [--out results]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import uuid

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in ("weaviate_probe", "weaviate_nonperturbing_probe", "weaviate_probe_per_id"):
    sys.path.insert(0, os.path.join(ROOT, "research", p))

import weaviate_topology as t              # noqa: E402
import internal_api as ia                  # noqa: E402
from per_id import objects_present_ids     # noqa: E402

VICTIM = 2
PEERS = [0, 1]


def log(m):
    print(m, flush=True)


def write_objects(ids, seed, consistency="ALL", batch=500):
    """Batched writes; returns (ok, seconds). Large divergences need minutes,
    which is itself a finding (#48 Confounds: the outage lasts as long as the
    write, so a big divergence is also a long outage)."""
    rng = np.random.default_rng(seed)
    t0 = time.time()
    for i in range(0, len(ids), batch):
        chunk = ids[i:i + batch]
        objs = [{"class": t.CLASS_NAME, "id": x, "properties": {"vid": x[-8:]},
                 "vector": [float(v) for v in rng.standard_normal(t.VECTOR_DIM)]}
                for x in chunk]
        st, b = t.http_request(t.http_port(0), "POST",
                               f"/v1/batch/objects?consistency_level={consistency}",
                               {"objects": objs}, timeout=300)
        if st != 200:
            return False, time.time() - t0
    return True, time.time() - t0


def probe_cost(shard, sizes=(10, 100, 1000, 10000), reps=20):
    """Step 1: the sampling floor."""
    log("\n=== step 1: probe cost (the sampling floor) ===")
    out = {}
    for n in sizes:
        ids = [str(uuid.UUID(int=i)) for i in range(n)]
        times = []
        for _ in range(reps):
            t0 = time.time()
            objects_present_ids(0, shard, ids)
            times.append((time.time() - t0) * 1000)
        med, lo, hi = statistics.median(times), min(times), max(times)
        out[n] = {"median_ms": round(med, 1), "min_ms": round(lo, 1), "max_ms": round(hi, 1)}
        log(f"  {n:>6} ids: median {med:7.1f} ms  (min {lo:.1f}, max {hi:.1f})"
            + ("   <- under 100ms" if med < 100 else ""))
    return out


def wait_ready(node, timeout_s=180):
    end = time.time() + timeout_s
    while time.time() < end:
        st, _ = t.http_request(t.http_port(node), "GET", "/v1/.well-known/ready", timeout=3)
        if st == 200:
            return True
        time.sleep(1)
    return False


def repair_once(shard, size, seed, poll_s=0.0):
    """One divergence-and-repair cycle. Time is measured from the victim's
    FIRST successful probe response, so restart latency is excluded (#48
    Confounds). Returns a record."""
    ids = [str(uuid.UUID(int=seed * 10_000_000 + i)) for i in range(size)]
    subprocess.run(["docker", "stop", t.container_name(VICTIM)], capture_output=True)
    time.sleep(3)
    ok, write_s = write_objects(ids, seed, consistency="ONE")
    if not ok:
        subprocess.run(["docker", "start", t.container_name(VICTIM)], capture_output=True)
        return {"size": size, "error": "write failed", "write_s": round(write_s, 1)}

    okp, got_peer = objects_present_ids(PEERS[0], shard, ids)
    subprocess.run(["docker", "start", t.container_name(VICTIM)], capture_output=True)

    first_t = None
    first_n = None
    traj = []
    t_start = time.time()
    deadline = t_start + 900
    while time.time() < deadline:
        ok, got = objects_present_ids(VICTIM, shard, ids)
        now = time.time()
        if ok:
            if first_t is None:
                first_t, first_n = now, len(got)
            traj.append((round(now - first_t, 3), len(got)))
            if len(got) == size:
                break
        if poll_s:
            time.sleep(poll_s)
    converged = bool(traj) and traj[-1][1] == size
    return {
        "size": size, "seed": seed,
        "write_s": round(write_s, 1),
        "peer_had": len(got_peer) if okp else None,
        "first_answer_n": first_n,
        "repair_s": (traj[-1][0] if converged else None),
        "samples_in_window": len(traj) if converged else None,
        "converged": converged,
        "trajectory_head": traj[:6],
        "trajectory_tail": traj[-3:],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", default="50,500,5000")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    a = ap.parse_args()
    sizes = [int(x) for x in a.sizes.split(",")]
    os.makedirs(a.out, exist_ok=True)

    shard = ia.shard_name(0)
    log(f"shard: {shard}   image: {t.WEAVIATE_IMAGE.split('@')[-1][:23]}...")

    record = {"probe_cost": probe_cost(shard), "repairs": [], "control": None}

    log("\n=== control: no divergence -- the victim must already be complete ===")
    ctl_ids = [str(uuid.UUID(int=i)) for i in range(50)]
    write_objects(ctl_ids, 99, consistency="ALL")
    subprocess.run(["docker", "restart", t.container_name(VICTIM)], capture_output=True)
    wait_ready(VICTIM)
    ok, got = objects_present_ids(VICTIM, shard, ctl_ids)
    record["control"] = {"ok": ok, "complete_on_first_answer": ok and len(got) == len(ctl_ids),
                         "n": len(got)}
    log(f"  complete on first answer after restart: {record['control']['complete_on_first_answer']} "
        f"({len(got)}/{len(ctl_ids)})  <- any window below is repair, not restart latency")

    log("\n=== step 2: repair duration vs divergence size ===")
    for size in sizes:
        for rep in range(a.reps):
            r = repair_once(shard, size, seed=size + rep)
            record["repairs"].append(r)
            log(f"  size {size:>6} rep {rep}: write {r.get('write_s')}s  "
                f"first_answer {r.get('first_answer_n')}/{size}  "
                f"repair {r.get('repair_s')}s  samples {r.get('samples_in_window')}  "
                f"converged {r.get('converged')}")
            wait_ready(VICTIM)

    log("\n=== step 3: decision ===")
    by_size = {}
    for r in record["repairs"]:
        if r.get("converged"):
            by_size.setdefault(r["size"], []).append((r["repair_s"], r["samples_in_window"]))
    floor_ms = record["probe_cost"].get(100, {}).get("median_ms")
    for size in sorted(by_size):
        durs = [d for d, _ in by_size[size]]
        samps = [s for _, s in by_size[size]]
        log(f"  size {size:>6}: repair {min(durs):.2f}-{max(durs):.2f}s   "
            f"samples in window {min(samps)}-{max(samps)}"
            + ("   <- >=10 samples" if min(samps) >= 10 else ""))
    ok_sizes = [s for s in sorted(by_size) if min(x[1] for x in by_size[s]) >= 10]
    record["decision"] = {"probe_floor_ms_at_100_ids": floor_ms,
                          "smallest_size_with_10_samples": ok_sizes[0] if ok_sizes else None}
    log(f"\n  probe floor (100 ids): {floor_ms} ms")
    log(f"  smallest divergence with >=10 samples per window: "
        f"{ok_sizes[0] if ok_sizes else 'NONE at these sizes -- outcome (i)/(c)'}")

    p = os.path.join(a.out, "characterization.json")
    with open(p, "w") as f:
        json.dump(record, f, indent=2)
    log(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
