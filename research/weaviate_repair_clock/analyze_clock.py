#!/usr/bin/env python3
"""
Which origin is Weaviate's repair clock anchored to? (issue #56)

Reads `results/repair_clock.json` and nothing else. Per Amendment 1 this works
from **realized** `absent_s` / `age_s`, never from the requested cell labels:
two step-1 runs overran badly (a 40 s cell realized 137.4 s at age 0.2 s), and
`qdrant_kill_scheduler` already established that requested and realized offsets
are different quantities on Docker.

The decision statistic, pre-registered:

    CV(repair_s)  vs  CV(age_s + repair_s)     over the step-1b runs

    smaller CV(since_write) -> anchored to the WRITE      (outcome a)
    smaller CV(repair_s)    -> anchored to the RESTART    (outcome b)
    neither stable          -> free-running cycle          (outcome c)

No threshold is applied to anything. Both quantities are continuous and the
comparison is between their dispersions, which is the rule SPEC_TEMPLATE.md
adopted after #48 step 2c's fast/slow cut went void with the gap that justified
it.

Usage:
    python research/weaviate_repair_clock/analyze_clock.py
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))


def cv(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return float("nan")
    m = statistics.mean(xs)
    return statistics.pstdev(xs) / m if m else float("nan")


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (dx * dy) if dx and dy else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=os.path.join(HERE, "results", "repair_clock.json"))
    a = ap.parse_args()
    rows = [r for r in json.load(open(a.json)) if r.get("converged")]
    print(f"converged runs: {len(rows)}\n")

    # ---- step 1: does absence matter once age is held? --------------------
    s1 = [r for r in rows if str(r.get("label", "")).startswith("abs")]
    if s1:
        print("=== step 1 — vary absence, age held (~6 s). Consistency check. ===")
        print(f"  {'realized absent':>16}{'realized age':>14}{'repair_s':>11}{'since_write':>13}")
        for r in sorted(s1, key=lambda x: x["absent_s"]):
            print(f"  {r['absent_s']:>16.1f}{r['age_s']:>14.1f}"
                  f"{r['repair_s']:>11.3f}{r['since_write_s']:>13.3f}")
        onspec = [r for r in s1 if 4.0 <= r["age_s"] <= 8.0]
        if len(onspec) >= 3:
            ab = [r["absent_s"] for r in onspec]
            rp = [r["repair_s"] for r in onspec]
            print(f"\n  runs on-spec (age 4-8 s): {len(onspec)}")
            print(f"  absent_s spans {max(ab) - min(ab):.1f} s; "
                  f"repair_s spans {max(rp) - min(rp):.2f} s "
                  f"({min(rp):.3f}-{max(rp):.3f})")
            print(f"  corr(absent_s, repair_s) = {pearson(ab, rp):+.3f}")
            print("  Read the SPREAD before the correlation: a correlation over a")
            print("  1 s range against a 45 s manipulation is noise with a sign,")
            print("  not an effect. Absence carries information only if repair_s")
            print("  moves materially.")

    # ---- step 1b: the discriminating comparison ---------------------------
    s1b = [r for r in rows if str(r.get("label", "")).startswith("age")]
    if s1b:
        print("\n=== step 1b — outage held at 40 s, write-to-restart varied. "
              "THE discriminating step. ===")
        print(f"  {'realized age':>14}{'repair_s':>11}{'since_write':>13}")
        for r in sorted(s1b, key=lambda x: x["age_s"]):
            print(f"  {r['age_s']:>14.1f}{r['repair_s']:>11.3f}{r['since_write_s']:>13.3f}")

        ages = [r["age_s"] for r in s1b]
        rep = [r["repair_s"] for r in s1b]
        sw = [r["since_write_s"] for r in s1b]
        cv_r, cv_s = cv(rep), cv(sw)
        print(f"\n  n = {len(s1b)}")
        print(f"  CV(repair_s)      = {cv_r:.4f}")
        print(f"  CV(since_write_s) = {cv_s:.4f}")
        print(f"\n  corr(age, repair_s)      = {pearson(ages, rep):+.3f}"
              "   <- write-anchored predicts strongly NEGATIVE")
        print(f"  corr(age, since_write_s) = {pearson(ages, sw):+.3f}"
              "   <- write-anchored predicts near ZERO")

        print("\n  verdict:", end=" ")
        if cv_r != cv_r or cv_s != cv_s:
            print("insufficient data")
        elif cv_s < cv_r:
            print(f"since_write is {cv_r / cv_s:.1f}x steadier "
                  "-> ANCHORED TO THE WRITE (outcome a)")
        elif cv_r < cv_s:
            print(f"repair_s is {cv_s / cv_r:.1f}x steadier "
                  "-> ANCHORED TO THE RESTART (outcome b)")
        else:
            print("indistinguishable (outcome c)")

    # ---- step 2: the short-outage distribution ----------------------------
    s2 = [r for r in rows if r.get("label") == "short6"]
    if s2:
        print("\n=== step 2 — 6 s outage, wall-clock recorded ===")
        vals = sorted(r["repair_s"] for r in s2)
        print(f"  n = {len(vals)}: {[round(v, 3) for v in vals]}")
        print(f"  since_write: {sorted(round(r['since_write_s'], 2) for r in s2)}")
        # phase test: does outcome track wall-clock position mod a candidate period?
        base = min(r["wall_start"] for r in s2)
        for period in (30.0, 40.0, 50.0, 52.0, 60.0):
            ph = [((r["wall_start"] - base) % period) for r in s2]
            print(f"  corr(phase mod {period:>4.0f}s, repair_s) = "
                  f"{pearson(ph, [r['repair_s'] for r in s2]):+.3f}")
        print("  (a strong correlation at some period supports the free-running-cycle"
              " reading; all near zero does not)")

    slow = [r for r in rows if r.get("label") == "short6_slowpoll"]
    if slow and s2:
        print("\n=== probe-perturbation check ===")
        f = [r["repair_s"] for r in s2]
        g = [r["repair_s"] for r in slow]
        print(f"  fast poll  n={len(f)} median {statistics.median(f):.3f}")
        print(f"  slow poll  n={len(g)} median {statistics.median(g):.3f}")
        print("  (a large gap would mean the probe perturbs the repair it measures)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
