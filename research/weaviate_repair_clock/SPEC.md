# Spec: is Weaviate's repair clock anchored to the write or the restart?

**Branch:** `method/weaviate-repair-clock`
**Date opened:** 2026-09-06
**Status:** IN PROGRESS — pre-registered before any run

Issue: closes #56. Body copied verbatim below (per `research/AGENT_PIPELINE.md`'s implementer instructions — this is the issue text unmodified, not a paraphrase).

---

## Research question

Is Weaviate's repair clock anchored to the **write** or to the **restart** — and what selects between a millisecond repair and a ~52 s one at a short outage?

## Hypothesis

Convergence occurs at an approximately fixed latency measured **from the write**, and `repair_s` measured from the restart is that latency minus however much of it elapsed while the node was still down.

## Null / alternative hypothesis

(i) **Anchored to the restart.** Then divergence age genuinely changes repair duration, the sentence withdrawn in #51's review was right, and it should be reinstated with the evidence that was missing.

(ii) **Neither — anchored to a free-running cluster clock.** Repair fires on a periodic cycle independent of both events, so latency depends on the *phase* at which the run lands. This predicts within-condition variance that fixed offsets cannot remove, which is testable directly.

(iii) **A is a different path entirely.** A 6 s outage may be short enough that the victim is never marked dead, so a different code path runs; then A is not comparable to B and C and its bimodality is a separate question.

## Motivation

#48 step 2c left two things unresolved, and #51's review turned one of them from a curiosity into a blocker for interpretation.

**The origin problem.** `repair_s` is measured from the victim's first probe response after restart. Under that origin, conditions B (40 s outage, divergence ~38 s old) and C (40 s outage, divergence ~6 s old) are disjoint — 1.17 s vs 30.89 s medians, exact p = 0.0022. Measured from the **write** instead (`age_s + repair_s`) they are 38.9 s and 36.9 s: a 30× gap becomes ~5%. So the tidy sentence "an older divergence repairs faster" was withdrawn in review, because convergence at a roughly fixed latency after the write fits the same data and would be near-definitional.

Neither reading is established, and one cell rules out the simple version of each: condition A's slow runs sit at **~56 s** since-write, not ~37 s.

**Why it matters beyond tidiness.** #54's observation window has to be timed from *something*. If the clock starts at the write, a window timed from the restart can close before repair fires — the exact failure that voided #24 and #9, arriving from a new direction.

**The bimodality.** Condition A (6 s outage) split 2 runs at ~0.023 s against 4 at ~51.6–52.3 s, at parameters differing only by ~1.6 s of realized jitter. Neither manipulated variable accounts for it.

## Experimental design

Fixed divergence size (50 objects — small and fast, since #48 showed size does not matter), the same digest-pinned 3-replica cluster, conditions interleaved in randomized order.

**Step 1 — vary absence at fixed age.** Write-to-restart held at ~6 s; absence 10, 25, 40, 70 s. Both hypotheses predict `repair_s` is constant here, so this step **cannot discriminate** — it is a consistency check and is labelled as one. It does test whether absence has any effect of its own once age is held fixed.

**Step 1b — the discriminating step.** Absence held at 40 s; write-to-restart across 2, 6, 15, 30, 38 s.
- Anchored to the **write** ⇒ `age_s + repair_s` is roughly constant while `repair_s` falls as age rises.
- Anchored to the **restart** ⇒ `repair_s` is roughly constant while `age_s + repair_s` rises.

These predictions are opposite, and the existing #48 data already spans part of this axis (C at ~6 s, B at ~38 s), so this extends a line rather than starting one.

**Step 2 — A's bimodality.** Runs at a 6 s outage recording **wall-clock start time**, to test (ii): if latency depends on phase against a free-running cycle, the fast/slow split should correlate with start time modulo a period, and the observed ~52 s gives a candidate period.

## Metrics

**Primary: the coefficient of variation of `repair_s` versus that of `age_s + repair_s`, across the step-1b cells.** The smaller one names the origin. This is a comparison between two *continuous* summaries of the same runs — deliberately **not** a threshold applied to either.

That choice is the direct consequence of #48's failure, now written into `SPEC_TEMPLATE.md`: step 2c pre-registered a binary fast/slow split at 1 s, the gap justifying that cut was falsified by the same experiment, and the pre-registered statistic went void. **No threshold is registered here.** Runs are never classified as fast or slow for any inferential purpose; where "fast" and "slow" appear in step 2 they are descriptive labels on a bimodal distribution whose modes are separated by ~50 s, and no test depends on where the line is drawn.

Secondary: within-cell spread (bearing on (ii)); the empirical distribution at a 6 s outage and any correlation between outcome and wall-clock start.

## Instrument characterization

Required by `SPEC_TEMPLATE.md`, and every quantity below is measured, not assumed — all from #48's committed artifacts:

- **Probe cost / sampling floor.** `objects_present_ids` at 50 ids is well under 100 ms, against repairs of 0.02–52 s. Sampling is not the limit here, which is why the polling loop runs flat out rather than at a cadence.
- **The effect size is huge relative to the noise.** #48's within-condition spreads are tight (C spans 0.81 s across six runs at ~31 s), so 6 repetitions per cell is enough to see a trend across cells. This is not a small-effect design.
- **Restart latency is excluded by construction.** Time is measured from the victim's *first successful probe response*, not from `docker start` — #48's Confounds, and its no-divergence control confirmed the victim is already complete on its first answer, so a measured window is repair and not startup.
- **Write cost is negligible at this size.** 50 objects write in ~0.1 s (#48 `sizes.json`), so the write does not eat into the age offsets the design sets. This is why the size is 50 here and 5,000 in #54, which wants a resolvable ramp instead.
- **Known blind spot, stated in advance:** #48 never recorded wall-clock start times, so hypothesis (ii) **cannot** be tested from existing data and step 2 exists to capture it. The probe-perturbs-repair question (#48 flagged, never tested) is also unresolved and is checked here at one cell.

## Baselines / controls

#48's committed 18 runs are the anchor: the new 40 s / ~6 s cell must reproduce ~30.3–31.1 s and the 40 s / ~38 s cell ~0.7–2.2 s, or something has changed and nothing further is trustworthy. A **no-divergence control** must show the victim complete on its first answer, so a measured window is repair and not restart latency. A **fast-vs-slow polling comparison** at one cell tests whether the probe perturbs the repair it measures.

## Expected outcomes

(a) `age_s + repair_s` near-constant while `repair_s` varies → anchored to the write; #54 times its window from the write, and #48's withdrawn sentence stays withdrawn as near-definitional. (b) The reverse → anchored to the restart; the withdrawn sentence is reinstated **with** the evidence it lacked. (c) Neither is stable → outcome (ii); report the phase dependence and give #54 a window longer than a full period. (d) A behaves unlike B and C on every measure → outcome (iii); exclude short outages from the comparison and say why.

## Interpretation plan

Any outcome fixes #54's window timing, which is the practical purpose. (a) or (b) resolves an interpretation currently reported without a mechanism, and updates `weaviate_repair_window/SPEC.md` step 2c plus the `claim_corrections/` entry. (c) is the least convenient and most likely to matter for the eventual write-up, because a phase-dependent repair means single-run healing observations are not comparable at all.

## Confounds considered

**Absence and age cannot both be free** — a 6 s outage cannot hold a 38 s age, so the design's cells are bounded by construction and the corners are unreachable; the analysis must not extrapolate into them. **Write duration is inside the outage**, negligible at 50 objects. **Wall-clock recording is essential** for step 2 and absent from #48's runs, so that step cannot be answered from existing data. **The probe may perturb the repair it measures** — polling flat out loads the node doing the reconciliation; tested here at one cell rather than assumed. **Order effects** — cells are interleaved in randomized order, since #48 step 2b showed repetition index carries no information but only after testing it. **One host, one build, one topology.**

## Amendment 1 (2026-09-06, mid-run, before any result is analysed): analyse realized offsets, not requested labels

Two runs in step 1 missed their requested timings badly — a cell requesting a
40 s absence realized **137.4 s** with an age of 0.2 s instead of 6 s, and a
10 s cell realized 18.0 s with an age of 1.8 s. The cause is structural, not a
bug: the write is placed `absent_s - age_s` into the outage, so when
`docker stop` plus the write take longer than that gap, the write lands late,
the age collapses, and the outage overruns. Short absences have the least slack
and fail first.

`repair_clock.py` already records **realized** `absent_s` and `age_s` per run
rather than the requested values — the same lesson `qdrant_kill_scheduler`
learned when `docker start` cost a near-constant ~3.3 s out of every requested
gap. So the data is not lost; only the cell labels are unreliable.

**The analysis therefore uses realized offsets and ignores the labels.** The
decision statistic is unchanged in kind — it is still CV of `repair_s` against
CV of `age_s + repair_s` — but computed over runs binned by *realized* age
rather than by requested cell, and the relationship is also reported as a plain
scatter of `repair_s` and `since_write_s` against realized `age_s`, which needs
no binning at all.

This is written before any step 1b result has been read. It is recorded here
rather than applied silently because switching from requested to realized
offsets after seeing data is exactly the kind of choice that has to be dated to
be trustworthy — and because the anomalous runs are kept, not dropped: a run
that overran is a valid observation at its realized offsets.

## Results

*(to be filled after the runs)*

## Interpretation

*(to be filled)*

## Decision

*(to be filled)*
