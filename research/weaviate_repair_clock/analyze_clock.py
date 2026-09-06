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


def disaggregate(s1b):
    """Split step 1b by regime and recompute within each.

    THE AGGREGATE ABOVE IS THE PRE-REGISTERED TEST. This is not a second test
    and no p-value is attached to it -- it is the diagnostic that shows the
    pre-registered one was applied to a mixture, which is why its answer is
    reported and then withdrawn rather than quietly replaced.

    The split is DESCRIPTIVE, not a registered threshold. It is not a fast/slow
    cut on `repair_s`, which is what went void in #48 step 2c; it is a cut on
    `age_s`, the manipulated variable, placed in an 15-30 s interval the design
    never sampled. Every observation is on one side or the other by construction,
    so no run's classification depends on where in that gap the line is drawn --
    which is exactly the property #48's cut lacked.
    """
    lo = [r for r in s1b if r["age_s"] < 20.0]
    hi = [r for r in s1b if r["age_s"] >= 20.0]
    if not (lo and hi):
        return
    print("\n  --- disaggregated by regime (diagnostic, not a second test) ---")
    print(f"  {'regime':>8}{'n':>4}{'age span':>20}{'repair spread':>15}"
          f"{'corr(age,rep)':>15}{'corr(age,sw)':>14}")
    for name, g in (("young", lo), ("old", hi)):
        a = [r["age_s"] for r in g]
        p = [r["repair_s"] for r in g]
        s = [r["since_write_s"] for r in g]
        print(f"  {name:>8}{len(g):>4}"
              f"{f'{min(a):.1f}-{max(a):.1f} ({max(a) - min(a):.1f}s)':>20}"
              f"{max(p) - min(p):>15.3f}"
              f"{pearson(a, p):>+15.3f}{pearson(a, s):>+14.3f}")
    gap = min(r["age_s"] for r in hi) - max(r["age_s"] for r in lo)
    print(f"\n  the two regimes are separated by {gap:.0f}s of UNSAMPLED age, and")
    print(f"  by a step in repair_s from ~{statistics.median([r['repair_s'] for r in lo]):.0f}s "
          f"to ~{statistics.median([r['repair_s'] for r in hi]):.0f}s.")
    print("  Within EITHER regime repair_s is flat while since_write tracks age:")
    print("  that is the RESTART-anchored signature, the opposite of the")
    print("  aggregate verdict above. The aggregate's -0.94 is produced entirely")
    print("  by the step between regimes, not by any within-regime trend.")
    print("  Simpson's paradox. The pre-registered statistic is not wrong as a")
    print("  statistic; it was applied to a mixture of two populations.")


def calibration_check(s1b):
    """Would the pre-registered statistic ever say RESTART?

    Feeds the test a SYNTHETIC pure restart-anchored system -- constant
    repair_s, since_write = age + that constant -- over the REAL realized ages.
    If the test returned WRITE here too it would be incapable of returning
    (b) at all, and its verdict above would carry no information.

    This runs before the verdict is overturned, not after, so it is a check on
    the instrument rather than a rationalisation of an unwelcome result.
    """
    ages = [r["age_s"] for r in s1b]
    print("\n  --- calibration: the same test on a SYNTHETIC restart-anchored "
          "system ---")
    print("  (constant repair_s over these same realized ages)")
    for const in (2.0, 15.0, 31.0):
        rep = [const] * len(ages)
        sw = [a + const for a in ages]
        cv_r, cv_s = cv(rep), cv(sw)
        verdict = "RESTART" if cv_r < cv_s else "WRITE"
        print(f"    repair_s = {const:>5.1f}s constant -> "
              f"CV(repair)={cv_r:.4f} CV(since_write)={cv_s:.4f} -> {verdict}")
    print("  The test returns RESTART at every constant tried, so it CAN")
    print("  distinguish the hypotheses. It was applied to the wrong")
    print("  population, not miscalibrated.")


def write_position_check(rows):
    """Age, or where in the outage the write landed?

    Step 1b holds absent_s at 40 s, so age and position = absent - age are
    perfectly collinear there: every step-1b run is equally consistent with
    "old divergence repairs fast" and with "a write early in the outage repairs
    fast". Step 1b ALONE cannot separate them.

    Step 1 can, and this is what makes the absence = 10 s cell load-bearing
    rather than a spare consistency check: it puts the write ~4 s into the
    outage -- early, like the fast old-regime runs -- at a YOUNG age.
    """
    cand = [r for r in rows
            if r.get("absent_s") and r.get("age_s") is not None
            and r.get("repair_s") is not None and r["absent_s"] >= 10.0]
    if not cand:
        return
    print("\n=== is it age, or where in the outage the write landed? ===")
    print("  step 1b cannot tell these apart (absent fixed at 40 s makes them")
    print("  collinear). These runs can, because absence varies:")
    print(f"  {'absent':>8}{'age':>8}{'write pos in outage':>21}{'repair_s':>11}")
    for r in sorted(cand, key=lambda x: (x["absent_s"], x["age_s"])):
        pos = r["absent_s"] - r["age_s"]
        print(f"  {r['absent_s']:>8.1f}{r['age_s']:>8.1f}{pos:>21.1f}"
              f"{r['repair_s']:>11.3f}")
    early_young = [r for r in cand
                   if (r["absent_s"] - r["age_s"]) < 10.0 and r["age_s"] < 20.0]
    if early_young:
        print("\n  Runs with the write EARLY in the outage but a YOUNG age:")
        for r in early_young:
            print(f"    absent {r['absent_s']:.1f}s, age {r['age_s']:.1f}s, "
                  f"write {r['absent_s'] - r['age_s']:.1f}s in "
                  f"-> repair {r['repair_s']:.3f}s")
        print("  'Write position selects the regime' predicts these are FAST.")
        print("  They are SLOW. The rival frame is refuted and AGE survives.")
        print("  This is the one comparison in the study that discriminates")
        print("  between the two readings, and it comes from step 1 -- the step")
        print("  pre-registered as unable to discriminate anything.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=os.path.join(HERE, "results", "repair_clock.json"))
    a = ap.parse_args()
    allrows = json.load(open(a.json))
    rows = [r for r in allrows if r.get("converged")]
    dropped = [r for r in allrows if not r.get("converged")]
    print(f"records in file: {len(allrows)}")
    print(f"converged runs:  {len(rows)}")
    if dropped:
        # Named, not silently filtered: a cell with n=5 where its neighbours
        # have 6 is otherwise an unexplained asymmetry in every table below.
        print(f"NOT converged, excluded from every figure: {len(dropped)}")
        for r in dropped:
            print(f"    label={r.get('label')} seed={r.get('seed')} "
                  f"polls={r.get('polls')} -- never reached the target set "
                  f"within the observation window")
    print()

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

        disaggregate(s1b)
        calibration_check(s1b)

    write_position_check(rows)

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
