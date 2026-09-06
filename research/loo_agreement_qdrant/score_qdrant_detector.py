#!/usr/bin/env python3
"""
Score the ground-truth-free detector on already-committed Qdrant runs (#52).

No new data. `loo_agreement` and `shard_agreement` are columns in every
`samples.csv` this project writes, including all the Qdrant studies, so the
detector has been recording its answers throughout the cross-system work
without anyone scoring them.

WHAT THIS DELIBERATELY DOES NOT DO: define a metric. The scoring is
`replica_recall/analyze.py::_detection_stats()` and `resolution_eps()`,
imported unmodified. Both were written for nano-db before any Qdrant run
existed, so they cannot have been tuned to this data -- which is the whole
reason a post-hoc analysis is admissible here. Reimplementing them, even
faithfully, would throw that away.

`_detection_stats()` takes a truth axis implicitly: it compares the detector's
argmin against the argmin of `e2e_recall`. To score the other two axes
(#52's outcome (iii)) the rows are rewritten so the chosen axis occupies the
`e2e_recall` field, rather than by copying the function and changing a key.
That keeps one implementation of the metric.

Usage:
    python research/loo_agreement_qdrant/score_qdrant_detector.py
    python research/loo_agreement_qdrant/score_qdrant_detector.py --json out.json
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "research", "replica_recall"))

from analyze import _detection_stats, resolution_eps   # noqa: E402

AXES = ("e2e_recall", "index_recall", "completeness")

# Every directory holding committed Qdrant / cross-system runs. Fixed here
# rather than discovered by outcome: #52 pre-registered that every run
# carrying the columns is scored, with no filtering.
PATTERNS = (
    "research/qdrant_*/results/*/samples.csv",
    "research/cross_system_replication/results_sweep/*/samples.csv",
)


def condition_of(path: str, meta: dict | None = None) -> str:
    """baseline / chaos / quiesce, from `run_meta.json` where possible.

    Review round 1 caught the reason this does not read directory names: five
    `qdrant_index_gate` runs are named `..._nochaos_...`, and `"chaos" in
    "nochaos"` is True, so a substring match silently filed five NO-CHAOS runs
    into the chaos group. That contaminated the very comparison the study
    turns on.

    `run_meta.json` carries the run's own `chaos` and `quiesce` flags, written
    by the harness at run time. That is what the run *was*, not what someone
    called its directory, so it is preferred and the name is only a fallback.
    The fallback now tests `nochaos` first and is asserted against in
    `_assert_no_nochaos_in_chaos()`.
    """
    if meta:
        chaos = meta.get("chaos")
        if chaos is not None:
            if not chaos:
                return "baseline"
            return "quiesce" if meta.get("quiesce") else "chaos"

    d = os.path.basename(os.path.dirname(path)).lower()
    if "nochaos" in d or "no_chaos" in d or "no-chaos" in d:
        return "baseline"
    for c in ("baseline", "quiesce", "chaos"):
        if c in d:
            return c
    return "other"


def _assert_no_nochaos_in_chaos(records) -> None:
    """The guard for the bug review round 1 found. Cheap, and it fails loudly."""
    bad = [r["run"] for r in records
           if r["condition"] == "chaos"
           and any(t in r["run"].lower() for t in ("nochaos", "no_chaos", "no-chaos"))]
    if bad:
        raise AssertionError(
            "runs named 'nochaos' were classified as chaos -- the review-round-1 "
            "bug has recurred:\n  " + "\n  ".join(bad))


def load(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def meta_for(path):
    m = os.path.join(os.path.dirname(path), "run_meta.json")
    if os.path.exists(m):
        try:
            return json.load(open(m))
        except Exception:
            return {}
    return {}


def rows_for_axis(rows, axis):
    """Put `axis` where _detection_stats() looks for the truth column.

    Copies each row so the caller's data is untouched; the detector column
    is never modified.
    """
    if axis == "e2e_recall":
        return rows
    out = []
    for r in rows:
        r2 = dict(r)
        r2["e2e_recall"] = r.get(axis, "")
        out.append(r2)
    return out


EXACT_LIMIT = 200_000
RESAMPLES = 200_000
PERM_SEED = 20260906


def mann_whitney(a, b):
    """Two-sided Mann-Whitney U with an exact p where that is feasible.

    This project's other studies enumerate every split, because at 5 vs 5 there
    are only 252. Here the groups are 12-15 runs, where C(25,12) is 5.2 million
    -- so enumeration is used when the split count is at most EXACT_LIMIT and a
    seeded permutation test is used otherwise. Which one ran is returned, so a
    reader is never left guessing whether a p is exact.

    No scipy either way, per project practice. The permutation test makes the
    same null-hypothesis assumption the exact test does (exchangeability under
    the null) and differs only in resolution: with 200,000 resamples the
    smallest reportable p is 5e-6, far below anything this study will claim.
    """
    import itertools
    import math
    import random

    def U(x, y):
        return sum((p > q) + 0.5 * (p == q) for p in x for q in y)

    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan"), "none"
    u = U(a, b)
    obs = min(u, n1 * n2 - u)
    pool = list(a) + list(b)

    if math.comb(n1 + n2, n1) <= EXACT_LIMIT:
        tot = ex = 0
        for idx in itertools.combinations(range(n1 + n2), n1):
            g1 = [pool[i] for i in idx]
            g2 = [pool[i] for i in range(n1 + n2) if i not in idx]
            uu = U(g1, g2)
            tot += 1
            if min(uu, n1 * n2 - uu) <= obs + 1e-9:
                ex += 1
        return u, ex / tot, "exact"

    rng = random.Random(PERM_SEED)
    ex = 0
    for _ in range(RESAMPLES):
        rng.shuffle(pool)
        uu = U(pool[:n1], pool[n1:])
        if min(uu, n1 * n2 - uu) <= obs + 1e-9:
            ex += 1
    # +1/+1 so a p of exactly zero is never reported from a finite resample.
    return u, (ex + 1) / (RESAMPLES + 1), f"perm({RESAMPLES})"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", help="also write the per-run records here")
    a = ap.parse_args()

    paths = []
    for pat in PATTERNS:
        paths.extend(sorted(glob.glob(os.path.join(ROOT, pat))))
    if not paths:
        print("no committed runs found -- nothing to score")
        return 1

    records = []
    for p in paths:
        rows = load(p)
        meta = meta_for(p)
        eps = resolution_eps(meta)
        rec = {
            "run": os.path.relpath(p, ROOT).replace("\\", "/"),
            "condition": condition_of(p, meta),
            "rows": len(rows),
            "eps": eps,
            "axes": {},
        }
        for axis in AXES:
            st = _detection_stats(rows_for_axis(rows, axis), eps)
            rec["axes"][axis] = st
        records.append(rec)

    _assert_no_nochaos_in_chaos(records)

    print(f"scored {len(records)} committed runs "
          f"({sum(r['rows'] for r in records)} sample rows)\n")

    conds = sorted({r["condition"] for r in records})
    for axis in AXES:
        print(f"=== truth axis: {axis} ===")
        print(f"  {'condition':<10}{'runs':>6}{'scoreable':>11}{'hit rate':>11}"
              f"{'chance':>9}{'groups':>9}{'tie-excluded':>14}")
        for c in conds:
            rs = [r for r in records if r["condition"] == c]
            stats = [r["axes"][axis] for r in rs if r["axes"][axis]]
            usable = [s for s in stats
                      if s.get("groups", 0) >= 5 and s.get("hit_rate") == s.get("hit_rate")]
            if not usable:
                cand = sum(s.get("candidates", 0) or 0 for s in stats)
                tied = sum(s.get("tied_excluded", 0) or 0 for s in stats)
                frac = (tied / cand) if cand else float("nan")
                print(f"  {c:<10}{len(rs):>6}{0:>11}{'--':>11}{'--':>9}{0:>9}"
                      f"{frac:>13.1%}")
                continue
            hr = statistics.mean(s["hit_rate"] for s in usable)
            ch = statistics.mean(s["chance"] for s in usable)
            gr = sum(s["groups"] for s in usable)
            cand = sum((s.get("candidates") or 0) for s in stats)
            tied = sum((s.get("tied_excluded") or 0) for s in stats)
            frac = (tied / cand) if cand else float("nan")
            print(f"  {c:<10}{len(rs):>6}{len(usable):>11}{hr:>11.3f}{ch:>9.3f}"
                  f"{gr:>9}{frac:>13.1%}")
        print()

    print("Reading this table:")
    print("  * 'scoreable' counts runs with >=5 non-tied groups. A run below that")
    print("    is reported by _detection_stats() as NaN, not as a low score, and")
    print("    is excluded rather than averaged in.")
    print("  * 'tie-excluded' is the share of candidate groups dropped because the")
    print("    top-two replicas were within the run's own resolution floor. A high")
    print("    share is #52's outcome (ii): the data cannot decide the question.")
    print("  * The baseline row is the control. If baseline and chaos score alike,")
    print("    the detector is detecting nothing -- a healthy cluster scored 0.6409")
    print("    against a 0.333 chance line on nano-db purely from ties.")

    print("\n=== the pre-registered control comparison: chaos vs baseline ===")
    print("  Per-run hit rate is the unit (rounds within a run are serially")
    print("  correlated). Two-sided Mann-Whitney; the 'method' column says")
    print("  whether the p is exact or from a seeded permutation test.\n")
    print(f"  {'axis':<15}{'baseline':>20}{'chaos':>20}{'U':>7}{'p':>9}  {'method':<14}")
    for axis in AXES:
        def hrs(cond):
            return [r["axes"][axis]["hit_rate"] for r in records
                    if r["condition"] == cond and r["axes"][axis]
                    and (r["axes"][axis].get("groups", 0) >= 5)
                    and r["axes"][axis]["hit_rate"] == r["axes"][axis]["hit_rate"]]
        b, c = hrs("baseline"), hrs("chaos")
        if not b or not c:
            print(f"  {axis:<15}{('n=%d' % len(b)):>20}{('n=%d' % len(c)):>20}"
                  f"{'--':>7}{'--':>9}   (a condition has no scoreable run)")
            continue
        u, p, how = mann_whitney(c, b)
        bs = f"{statistics.mean(b):.3f} (n={len(b)})"
        cs = f"{statistics.mean(c):.3f} (n={len(c)})"
        flag = ""
        if p == p and p < 0.05:
            flag = ("  <- separates"
                    if statistics.mean(c) > statistics.mean(b)
                    else "  <- separates, WRONG DIRECTION")
        print(f"  {axis:<15}{bs:>20}{cs:>20}{u:>7.1f}{p:>9.4f}  {how:<14}{flag}")

    if a.json:
        with open(a.json, "w") as fh:
            json.dump(records, fh, indent=1)
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
