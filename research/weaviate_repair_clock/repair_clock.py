#!/usr/bin/env python3
"""
Is Weaviate's repair clock anchored to the write or the restart? (issue #56)

#48 measured `repair_s` from the victim's first probe response after restart.
Under that origin, two conditions differing only in how old the divergence was
separated 30x (p = 0.0022). Measured from the WRITE they are 38.9 s and 36.9 s
-- a ~5% difference. So "an older divergence repairs faster" was withdrawn in
review, because "convergence happens at a fixed latency after the write" fits
the same numbers and would be near-definitional.

This separates them. The key move is step 1b: hold the outage at 40 s and vary
where inside it the write lands.

    anchored to the WRITE    ->  age + repair is flat, repair falls as age rises
    anchored to the RESTART  ->  repair is flat, age + repair rises with age

Opposite predictions on the same runs, so one of them dies.

NO THRESHOLD IS REGISTERED ANYWHERE IN THIS FILE. The decision statistic is the
coefficient of variation of one continuous quantity against another. #48's step
2c pre-registered a binary fast/slow cut at 1 s, the gap justifying it was
falsified by that very experiment, and the statistic went void -- which is now
a rule in SPEC_TEMPLATE.md. Where "fast" and "slow" appear below they are
descriptive labels on a distribution whose modes are ~50 s apart, and nothing
inferential depends on where a line would be drawn.

Usage:
    python research/weaviate_repair_clock/repair_clock.py                  # all steps
    python research/weaviate_repair_clock/repair_clock.py --steps 1b       # one step
    python research/weaviate_repair_clock/repair_clock.py --reps 3         # quicker
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import time
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for _p in ("weaviate_probe", "weaviate_nonperturbing_probe", "weaviate_probe_per_id",
           "weaviate_repair_window"):
    sys.path.insert(0, os.path.join(ROOT, "research", _p))

import weaviate_topology as t                      # noqa: E402
import internal_api as ia                          # noqa: E402
from per_id import objects_present_ids             # noqa: E402
from characterize import write_objects, wait_ready, VICTIM   # noqa: E402

SIZE = 50           # #48: repair is size-independent; small keeps the write ~0.1 s
POLL_CAP_S = 300


def log(m):
    print(m, flush=True)


def one_run(shard, label, size, seed, absent_s, age_s, poll_sleep=0.0):
    """Stop the victim, place the write `absent_s - age_s` into the outage,
    restart at `absent_s`, and time convergence from the victim's FIRST
    successful probe response (so restart latency is excluded, #48 Confounds).

    Records wall-clock start, which #48 did not -- hypothesis (ii) needs it.
    """
    ids = [str(uuid.UUID(int=seed * 10_000_000 + i)) for i in range(size)]
    wall_start = time.time()
    t_stop = time.time()
    subprocess.run(["docker", "stop", t.container_name(VICTIM)], capture_output=True)

    before = max(0.0, absent_s - age_s - (time.time() - t_stop))
    time.sleep(before)

    t_write = time.time()
    ok, _ = write_objects(ids, seed, consistency="ONE")
    if not ok:
        subprocess.run(["docker", "start", t.container_name(VICTIM)], capture_output=True)
        return {"label": label, "error": "write failed"}

    time.sleep(max(0.0, absent_s - (time.time() - t_stop)))
    t_start = time.time()
    subprocess.run(["docker", "start", t.container_name(VICTIM)], capture_output=True)

    first_t = first_n = None
    last = None
    deadline = time.time() + POLL_CAP_S
    polls = 0
    while time.time() < deadline:
        okp, got = objects_present_ids(VICTIM, shard, ids)
        polls += 1
        if okp:
            if first_t is None:
                first_t, first_n = time.time(), len(got)
            last = (round(time.time() - first_t, 3), len(got))
            if len(got) == size:
                break
        if poll_sleep:
            time.sleep(poll_sleep)
    conv = bool(last and last[1] == size)
    repair = last[0] if conv else None
    return {
        "label": label, "seed": seed,
        "absent_s": round(t_start - t_stop, 1),      # realized, not requested
        "age_s": round(t_start - t_write, 1),
        "repair_s": repair,
        "since_write_s": (round((t_start - t_write) + repair, 3)
                          if repair is not None else None),
        "first": first_n,
        "polls": polls,
        "poll_sleep": poll_sleep,
        "wall_start": round(wall_start, 3),          # for hypothesis (ii)
        "converged": conv,
    }


def cv(xs):
    """Coefficient of variation. Dimensionless, so it compares two quantities
    measured in the same unit but on different origins -- which is the whole
    question here."""
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return float("nan")
    m = statistics.mean(xs)
    return statistics.pstdev(xs) / m if m else float("nan")


def summarize(rows, key="label"):
    by = {}
    for r in rows:
        if r.get("converged"):
            by.setdefault(r[key], []).append(r)
    return by


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=6, help="reps for step 1b")
    ap.add_argument("--reps-step1", type=int, default=3)
    ap.add_argument("--reps-step2", type=int, default=10)
    ap.add_argument("--steps", default="1,1b,2")
    ap.add_argument("--seed", type=int, default=56)
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    want = {s.strip() for s in a.steps.split(",")}
    rng = random.Random(a.seed)

    shard = ia.shard_name(0)
    log(f"shard: {shard}   size: {SIZE}   steps: {sorted(want)}")
    rows = []
    outp = os.path.join(a.out, "repair_clock.json")

    def record(r):
        rows.append(r)
        with open(outp, "w") as f:
            json.dump(rows, f, indent=1)
        log(f"  {r.get('label'):<16} absent {str(r.get('absent_s')):>5}  "
            f"age {str(r.get('age_s')):>5}  repair {str(r.get('repair_s')):>8}  "
            f"since_write {str(r.get('since_write_s')):>8}  first {r.get('first')}/{SIZE}")
        wait_ready(VICTIM)

    # ---- control: no divergence -------------------------------------------
    log("\n=== control: no divergence, victim must be complete on first answer ===")
    ctl = [str(uuid.UUID(int=990000 + i)) for i in range(SIZE)]
    write_objects(ctl, 99, consistency="ALL")
    subprocess.run(["docker", "restart", t.container_name(VICTIM)], capture_output=True)
    wait_ready(VICTIM)
    okc, gotc = objects_present_ids(VICTIM, shard, ctl)
    log(f"  complete on first answer: {okc and len(gotc) == SIZE} ({len(gotc)}/{SIZE})"
        "   <- any window below is repair, not restart latency")

    # ---- step 1: vary absence, hold age -----------------------------------
    if "1" in want:
        log("\n=== step 1: vary absence at fixed age (~6 s) -- consistency check, "
            "NOT discriminating ===")
        plan = [("abs%d" % A, A, 6.0) for A in (10, 25, 40, 70)] * a.reps_step1
        rng.shuffle(plan)
        for i, (lab, A, g) in enumerate(plan):
            record(one_run(shard, lab, SIZE, 3000 + i, A, g))

    # ---- step 1b: the discriminating step ---------------------------------
    if "1b" in want:
        log("\n=== step 1b: absence fixed at 40 s, vary write-to-restart "
            "-- THE discriminating step ===")
        plan = [("age%d" % g, 40.0, float(g)) for g in (2, 6, 15, 30, 38)] * a.reps
        rng.shuffle(plan)
        for i, (lab, A, g) in enumerate(plan):
            record(one_run(shard, lab, SIZE, 4000 + i, A, g))

    # ---- step 2: the short-outage distribution, with wall clock -----------
    if "2" in want:
        log("\n=== step 2: 6 s outage, wall-clock recorded (hypothesis (ii)) ===")
        for i in range(a.reps_step2):
            record(one_run(shard, "short6", SIZE, 5000 + i, 6.0, 4.0))

        log("\n=== probe-perturbation check: same cell, slow polling ===")
        for i in range(3):
            record(one_run(shard, "short6_slowpoll", SIZE, 5500 + i, 6.0, 4.0,
                           poll_sleep=1.0))

    # ---- decision ---------------------------------------------------------
    log("\n=== the decision statistic ===")
    b = summarize([r for r in rows if str(r.get("label", "")).startswith("age")])
    if b:
        log(f"  {'cell':<10}{'n':>3}{'repair_s median':>18}{'since_write median':>21}")
        for lab in sorted(b, key=lambda x: float(x[3:])):
            rs = [r["repair_s"] for r in b[lab]]
            sw = [r["since_write_s"] for r in b[lab]]
            log(f"  {lab:<10}{len(rs):>3}{statistics.median(rs):>18.3f}"
                f"{statistics.median(sw):>21.3f}")
        all_r = [r["repair_s"] for lab in b for r in b[lab]]
        all_s = [r["since_write_s"] for lab in b for r in b[lab]]
        cv_r, cv_s = cv(all_r), cv(all_s)
        log(f"\n  CV(repair_s)      = {cv_r:.3f}")
        log(f"  CV(since_write_s) = {cv_s:.3f}")
        if cv_s == cv_s and cv_r == cv_r:
            if cv_s < cv_r:
                log("  -> since_write is the steadier quantity: ANCHORED TO THE WRITE "
                    "(outcome (a))")
            elif cv_r < cv_s:
                log("  -> repair_s is the steadier quantity: ANCHORED TO THE RESTART "
                    "(outcome (b))")
            else:
                log("  -> indistinguishable (outcome (c))")

    s2 = [r for r in rows if r.get("label") == "short6" and r.get("converged")]
    if s2:
        vals = sorted(r["repair_s"] for r in s2)
        log(f"\n  6 s outage, n={len(vals)}: {[round(v, 3) for v in vals]}")
        log(f"  spread {min(vals):.3f} .. {max(vals):.3f}  "
            f"(wall-clock starts recorded for phase analysis)")

    log(f"\nwrote {outp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
