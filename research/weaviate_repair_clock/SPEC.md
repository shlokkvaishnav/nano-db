# Spec: is Weaviate's repair clock anchored to the write or the restart?

**Branch:** `method/weaviate-repair-clock`
**Date opened:** 2026-09-06
**Status:** COMPLETE — outcome (d). The pre-registered statistic returned (a) and disaggregation reverses it.

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

**55 records in `results/repair_clock.json`; 54 converged; every figure below is over the 54.**

The one excluded run is `age15`, seed 4004: `repair_s` null, `converged` false, 38 polls — it never reached the target set within its observation window. That is why the 15 s cell carries n = 5 where every other cell carries 6, and why the young regime is n = 17 rather than 18. *(Corrected in review round 1, which previously said "53 runs" — a count matching neither the file nor the analyzer — and did not disclose the exclusion.)*

That a run failed to converge is itself a result worth #54's attention: it is a direct observation that repair can exceed the window you allot it, which is why #54 now records right-censoring rather than reading a non-recovery as a failure to heal.

Analysis uses **realized** offsets per Amendment 1.

### Step 1 — absence carries no information (consistency check, as pre-registered)

Holding write-to-restart at ~6 s and varying the outage across **10 / 25 / 40 / 70 s**: absence spans 60 s while `repair_s` spans **2.23 s** (30.171–32.400), over 9 on-spec runs.

*(Corrected in review round 1. Both figures previously read "1.35 s (30.171–31.517) across 25/40/70", which silently dropped the pre-registered absence = 10 s cell — a range 40% narrower than `results/analysis_output.txt`, produced by the same script, printed three lines above it. The conclusion is unchanged: 2.23 s of spread against a ~30 s regime step is still no effect. The number now matches the artifact.)*

Both hypotheses predicted a flat `repair_s` here, so this step discriminates *between them* nothing — as pre-registered. It does rule out absence as a variable in its own right **over 10–70 s at age ≈ 6 s**, which is the range actually tested; see "absence is not irrelevant everywhere" below. And it turns out to discriminate something else entirely, which the pre-registration did not anticipate:

#### The absence = 10 s cell is the study's only discriminating control

Step 1b holds the outage at 40 s. That makes realized age and **where in the outage the write landed** (`position = absent − age`) perfectly collinear, so *every step-1b observation is equally consistent with two readings*:

- **age selects the regime** — an older divergence repairs fast; and
- **write position selects the regime** — a write early in the outage repairs fast.

Old-regime runs have the write ~2 s into the outage; young-regime runs ~38 s in. Step 1b **cannot** tell these apart, and nothing in the pre-registration noticed.

The absence = 10 s cell can, because it varies absence: it places the write **4 s into the outage** — early, exactly like the fast old-regime runs — at a **young** age of 6 s.

| absent | age | write position in outage | `repair_s` |
|---|---|---|---|
| 10 s | 6.0 s | **4.0 s (early)** | **31.306** |
| 10 s | 6.0 s | **4.0 s (early)** | **32.400** |
| 40 s | 38.0 s | 2.0 s (early) | 1.733–2.188 |

Write-position predicts the first two rows are fast. They are slow, and by a factor of ~16. **The rival frame is refuted and age survives.**

So the step pre-registered as *"cannot discriminate — it is a consistency check and is labelled as one"* contains the only comparison in the experiment that separates the two readings of the main result. That is an argument for keeping consistency checks that a design expects to be uninformative, and against reporting a narrower range than the data holds: the dropped cell was the load-bearing one.

`write_position_check()` in `analyze_clock.py` computes this.

### Step 1b — the discriminating step

| realized age | n | `repair_s` median | `age + repair` median |
|---|---|---|---|
| 2 s | 6 | 31.81 | 33.81 |
| 6 s | 6 | 32.29 | 38.29 |
| 15 s | 5 | 31.20 | 46.20 |
| 30 s | 6 | 1.95 | 31.95 |
| 37–38 s | 6 | 1.95 | 39.95 |

**The pre-registered statistic says outcome (a):** CV(`repair_s`) = 0.766 against CV(`age + repair`) = 0.132, so time-from-the-write is 5.8× steadier. Aggregate correlations agree — corr(age, repair) = **−0.942**, corr(age, since_write) = **−0.012**, exactly the write-anchored signature.

**The statistic is not mechanically biased.** Simulating a pure restart-anchored system (constant `repair_s`) over these same realized ages returns "RESTART" at every constant tried (2 s, 15 s, 31 s), so the test can distinguish the hypotheses in principle.

**And the conclusion is still wrong.** Disaggregating by regime reverses it:

| regime | n | age span | `repair_s` spread | corr(age, repair) | corr(age, since_write) |
|---|---|---|---|---|---|
| young, age 2–15 s | 17 | 13 s | **2.34 s** | **−0.267** | **+0.993** |
| old, age 30–38 s | 12 | 8 s | **1.45 s** | **−0.103** | **+0.991** |

Within either regime, `repair_s` is **flat** across 13 s and 8 s of age variation, while `age + repair` tracks age almost perfectly. **That is the restart-anchored signature, not the write-anchored one** — the opposite of what the aggregate reported.

This is Simpson's paradox. The aggregate correlation of −0.942 is produced entirely by the **step between the two regimes** (~32 s → ~2 s), not by any within-regime trend. Pooling across a step and reading the slope is exactly the error.

### Step 2 — the short-outage bimodality is unexplained, and the perturbation check failed

At a 6 s outage, n = 10: **0.011, 0.013, 0.018, 0.024, 0.026, 0.028** and **43.126, 43.733, 50.779, 53.331**. Still sharply bimodal, six fast and four slow, at identical parameters.

Wall-clock phase was recorded for the first time (#48 never captured it). Correlations against candidate periods are all weak — 30 s: −0.101, 40 s: +0.429, 50 s: −0.399, 52 s: −0.466, 60 s: +0.556. At n = 10 none of these is meaningful. **Hypothesis (ii) is neither supported nor excluded.**

**The probe-perturbation check is void, and the reason matters for #54.** The three slow-poll runs report `repair_s = 0.000` with `first = 50/50` — the victim already held everything on its *first* sample. That does not show the probe leaves repair alone; it shows the first sample landed *after* convergence. The check cannot distinguish perturbation from missing the window, so it answers nothing. **What it does establish is that a 1 s cadence can miss a fast repair entirely** — a direct constraint on #54, which planned exactly that cadence.

## Interpretation

**Outcome (d): both quantities matter, in different ways, and neither pre-registered model is right.**

- **Divergence age selects the *regime*.** Below ~15 s the victim waits ~32 s; above ~30 s it reconciles in ~2 s. The threshold is somewhere in between and was not sampled.
- **Within a regime the clock is anchored to the *restart*.** `repair_s` is flat over 13 s of age variation; `age + repair` is not.
- **Absence is irrelevant for outages of 10–70 s** once age is fixed (step 1). It is **not** irrelevant in general — see immediately below.

#### Absence is not irrelevant everywhere, and this study contains the counterexample

Corrected in review round 1; the bullet above previously read "absence is irrelevant" without qualification, in this document, `README.md` and the PR body. At a realized age inside the young band, this experiment holds three different behaviours:

| absence | realized age | `repair_s` |
|---|---|---|
| 40 s (step 1b) | 2.0 s ×6 | 31.08 – 32.71 |
| 6 s (`short6`) | 1.95 – 4.03 | 0.011 – 0.028 ×6, 43.1 – 53.3 ×4 |
| 18 s (step 1, overran) | 1.83 s | 3.330 |

Age is held inside the young band across all three rows and `repair_s` moves over three orders of magnitude. The honest statement is that absence carries no information **over the 10–70 s range step 1 tested, at age ≈ 6 s**.

This is not a new concession — outcome (d) is *defined* in the Expected outcomes as "A behaves unlike B and C on every measure → outcome (iii); exclude short outages from the comparison and say why". The Interpretation simply failed to carry that exclusion into a bullet stated as general.

The **18 s / age 1.83 s run at 3.330 s** deserves naming rather than leaving in a table. It is an overrun step-1 run, so it was never a planned cell; it sits at a young age with a repair that is neither the ~32 s young-regime wait nor the ~0.02 s `short6` fast mode nor the ~2 s old-regime value. It is a **third behaviour at a fourth absence**, from n = 1, and it is the clearest single indication that the two-regime picture is a description of the 40 s outage rather than a general law. It is reported, not explained.

So the answer to "write or restart" is **neither, cleanly**: age acts as a *selector* between two behaviours rather than as an offset against a countdown, and within each behaviour the latency is measured from the restart.

**The methodological finding is the more transferable one.** The pre-registered statistic was well chosen — it registers an estimand rather than a threshold, exactly as `SPEC_TEMPLATE.md` now requires after #48 step 2c — and it *still* gave the wrong answer, because it aggregated across a step. Registering the right kind of statistic does not protect against pooling heterogeneous regimes. The defence that worked was disaggregation, prompted by the reviewer step (`AGENT_PIPELINE.md` step 1c) that asks whether some other cut of the same data collapses or reverses the effect.

**What is not established.** The mechanism — no Weaviate-internal scheduler was observed, only its effect on a probe. The threshold's location, sampled nowhere between 15 s and 30 s. Why a 6 s outage is bimodal. Whether the probe perturbs repair, which this design failed to test.

## Decision

**MERGE**, as outcome (d), with the pre-registered statistic reported *and* overturned in the same section.

**Consequences to file:**
1. `experiment/*` (#54) — **time the observation window from the restart**, and make it long enough to contain the ~32 s regime plus margin; ≥60 s stands. Also: **a 1 s cadence can miss a fast repair entirely**, so #54 must record the first sample's offset and treat a `t=0` completion as censored rather than instant.
2. `method/*` — locate the threshold between 15 s and 30 s, which this design bracketed but never sampled.
3. `analysis/*` — `AGENT_PIPELINE.md` step 1c currently asks whether a derived quantity *collapses* an effect. It should also ask whether **disaggregation reverses** it, which is what worked here.

**What must not be claimed.** That repair is write-anchored — the aggregate says so and the within-regime data contradicts it. That repair is restart-anchored *simpliciter* — true within a regime, false across the step. That the threshold is at any particular value; it is bracketed to (15 s, 30 s) and no finer.
