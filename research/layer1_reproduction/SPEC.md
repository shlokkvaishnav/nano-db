# Spec: reproduce Layer 1 and commit its raw data

**Branch:** `reproduction/layer1-data`
**Date opened:** 2026-09-06
**Status:** IN PROGRESS — harness written, no runs yet

Issue: closes #53. Body copied verbatim below (per `research/AGENT_PIPELINE.md`'s implementer instructions — this is the issue text unmodified, not a paraphrase).

---

## Research question

Can the Layer-1 nano-db result be reproduced and its **raw per-seed data committed**, so the project's foundational claim stops being reported-but-not-checkable?

## Hypothesis

The published Layer-1 numbers reproduce within their stated variability on a fresh 5-seed sweep, and the resulting per-seed data can be committed at a size the repository can carry.

## Null / alternative hypothesis

(i) **They do not reproduce.** Then that is a far more important finding than anything else queued, and everything downstream — including `RELATED_WORK.md`'s novelty verdict — is affected. This possibility is the reason the issue is worth doing rather than assuming.

(ii) **They reproduce but the data is too large to commit.** Then commit the per-seed aggregates plus one full sample series, and say exactly what was omitted and why.

(iii) **The harness no longer runs.** `src/`, `cluster/` and `proto/` have not been touched since June 2026 while the research moved to Python-on-Docker; the build is exercised by CI but the *experiment path* is not. A silent bit-rot finding here is itself worth recording.

## Motivation

This is the single largest hole in the work, and it is not a conceptual one.

`research/replica_recall/` produces the numbers the top-level README's ESTABLISHED box opens with, and those numbers are what `research/RELATED_WORK.md` positions the entire contribution around. **None of the raw data behind them is in version control.** `research/replica_recall/RESULTS.md` states this itself:

> **No raw experiment output is currently committed to this repository.** … Anyone auditing this project's claims should treat those numbers as reported-but-not-independently-checkable.

Every *other* study commits its evidence — roughly 13,400 sample rows across 90+ runs. The foundational one does not. A reader auditing this project finds the strongest claim resting on the only data they cannot inspect, which is the worst possible distribution of trust.

## Experimental design

Run the **pre-existing** 5-seed protocol unchanged — baseline and chaos per seed — analysed with the committed `analyze.py` / `aggregate.py`, unmodified. Any change to the protocol makes this a new experiment rather than a reproduction and is refused here.

`run_reproduction.py` drives it inside a Linux container, because the development host is Windows/MinGW with no `g++` and no `protoc`, and the harness launches cluster binaries directly (`chaos_harness.py`, whose own docstring says "Run (Linux, after building the binaries)").

**Deviation from the issue's step 5, with the reason.** The issue proposed narrowing `.gitignore`'s `research/replica_recall/results*/` rule so evidence becomes tracked. This spec keeps that rule and commits the reproduction's output under `research/layer1_reproduction/results_sweep/` instead. Two reasons: the ignore rule exists because scratch output was committed by accident twice (#18, #24 — see `.gitignore`'s own comment), and loosening it re-opens that hazard; and a *reproduction on a different host* is not the same artifact as the original run, so filing it separately keeps the distinction visible rather than blurring the two in one directory.

## Metrics

The study's own pre-existing metrics, unchanged: `index_recall`, `completeness`, `e2e_recall`, within-shard spread, `miss@stop` vs `miss@end`, and the exact two-sided Mann-Whitney across 5 seeds per condition.

## Instrument characterization

- **The build is exercised, the experiment path is not.** CI compiles the cluster and runs `ctest` on every push, so a total build failure would already be visible. What is *not* exercised is `run_experiment.py` → `chaos_harness.py` → live binaries; outcome (iii) is live and is the first thing the run tests.
- **The container is a new variable and is treated as one.** Process-launch and I/O timing inside a container differ from bare metal — exactly the class of confound `qdrant_kill_scheduler` had to characterize when `docker start` cost a near-constant ~3.3 s of every requested kill gap. `run_reproduction.py` runs `ctest` first as a smoke test, and the writeup must present this as a reproduction on a **new host**, not a re-run under identical conditions.
- **Corpus integrity is checkable before use.** SIFT1M is ~186 MB and gitignored; `sift.py` re-downloads it. Vector count and dimensionality are verified before the sweep, because a truncated download would silently shrink the corpus and change every number.
- **Data volume is computable in advance.** Comparable committed sweeps run 54–426 sample rows per run; 10 runs at this protocol's duration should land in the low hundreds of KB — well inside what the repository already carries (~1.6 MB of results). Outcome (ii) is therefore unlikely, and if it fires the fallback is stated above.
- **Known blind spot:** the original runs' host is unknown, so "within stated variability" is judged against the published spread, not against a matched machine.

## Baselines / controls

The published numbers are the comparison. Reproduction is judged by whether the new per-seed values fall within the published spread — **not** by whether p is again 0.0079, which is the floor at n=5 and would be reached by any complete separation. `ctest` passing is the precondition; if it fails, that is outcome (iii) and the sweep does not run.

## Expected outcomes

(a) Reproduces, data committed → the biggest audit gap closes. (b) Reproduces qualitatively with different magnitudes → report both, and treat the published figures as one draw rather than the value. (c) Does not reproduce → highest-priority finding in the project; stop and investigate before any other queued work. (d) Harness bit-rot → fix or record, and state that Layer 1 was unreproducible for a period.

## Interpretation plan

Outcome (a) or (b) updates `RESULTS.md`, the README's Raw data status section, and `research/README.md`'s Layer 1 row. Outcome (c) blocks #54 and the README's ESTABLISHED box until understood, because both build on Layer 1's framing.

**This is not independent confirmation.** Same protocol, same system, same code — agreement is evidence of *reproducibility*, not a second confirmation of the effect, and must not be written up as one.

## Confounds considered

**A different machine is a different measurement**, so timing-sensitive quantities (kill spacing, replication lag, settle-window adequacy) may differ. **Docker adds a layer** the original did not have; if results differ, the container is a live hypothesis and must be named as one. **Corpus download integrity** — verified before use. **Not a fresh test of the hypothesis** — see the interpretation plan.

## Results

*(no runs yet)*

## Interpretation

*(to be filled)*

## Decision

*(to be filled)*
