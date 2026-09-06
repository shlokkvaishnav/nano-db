# Spec: method/weaviate-repair-window

**Branch:** `method/weaviate-repair-window` (from #47's head; stacked behind it until it lands on `main`)
**Issue:** #48 (body copied verbatim below, per `AGENT_PIPELINE.md`)
**Date opened:** 2026-09-05
**Status:** COMPLETE — amended 2026-09-06 (step 2c), and step 2b's "two discrete paths" reading **withdrawn** on that amendment's data

### Type

method (a new methodological component — metric, detector, protocol)

### Research question

What combination of **divergence size** and **sampling rate** makes Weaviate's async repair measurable — i.e. produces a repair window long enough to contain enough samples to describe it?

PR #44 recorded repair converging in **~0.3s** for 50 objects and named "sub-second sampling" as the remaining prerequisite. This issue deliberately reopens that framing: sampling faster is only one of the two levers, and it is the one with a hard floor.

### Hypothesis

**Repair duration scales with the amount of divergence, and that is the cheaper lever.** 0.3s was measured for 50 missing objects; if repair time grows roughly with the number of objects to reconcile, then 50,000 missing objects should give a window of seconds to minutes — measurable at the ~1–5s cadence this project already uses on nano-db and Qdrant, with no new sampling machinery at all.

Secondary: the probe's own cost bounds how fast sampling can go regardless. One `objects_present_ids` call is a single HTTP GET returning a few tens of KB, so a floor in the tens of milliseconds is plausible — but it has never been measured, and #25's whole lesson was that an unmeasured probe cost is how a sweep ends up void.

### Null / alternative hypothesis

(i) **Repair time is roughly constant in divergence size** — a fixed hash-tree comparison dominates and 50,000 objects still reconcile in well under a second. Then the only lever is sampling rate, and if the probe floor is above the repair window, Weaviate's repair is **not measurable by this instrument** and the healing half of the dissociation prediction has to be dropped or pursued another way (server-side metrics, logs).

(ii) Repair time grows but so unevenly (bursty, or dominated by a fixed poll interval inside Weaviate) that a "duration" is not a well-defined quantity to sample.

(iii) The probe floor is much worse than expected — e.g. tens of KB per call at 3 nodes makes 5s the practical cadence — in which case even a long window is coarsely sampled and the achievable resolution must be stated with any result.

### Motivation

The Weaviate leg exists to test one prediction: `completeness` heals (hash-tree anti-entropy) while `index_recall` does not. The healing half is the more convincing form, and it is only claimable if the healing is *observed over time* rather than inferred from before/after. #43 showed the probe can run at 5s cadence indefinitely without perturbing the cluster; what is unknown is whether there is anything left to see at that cadence. Answering this decides whether the experiment measures a curve or two endpoints — and `SPEC_TEMPLATE.md`'s instrument-characterization section exists precisely to force this question before compute is spent, after three sweeps (#11/#16, #20, #25) failed on apparatus properties that were computable in advance.

### Experimental design

No experiment; instrument characterization on `method/weaviate-repair-window`, reusing the digest-pinned cluster and `objects_present_ids` from #46.

1. **Probe cost floor.** Time `objects_present_ids` for id-set sizes {10, 100, 1,000, 10,000} against one node, 20 repetitions each. Report median and spread, and the largest set that stays under 100ms. This is the sampling floor and must be known before any cadence is chosen.
2. **Repair duration vs divergence size.** For divergence sizes {50, 500, 5,000, 50,000}: stop node2, write that many objects at `consistency_level=ONE`, restart node2 with peers up, and poll `objects_present_ids(2, …)` as fast as step 1 allows until the set is complete. Record time-to-convergence and the trajectory. Three repetitions per size — enough to see whether the relationship is even monotone, not enough to fit a curve, and the writeup must say so.
3. **The decision.** From (1) and (2): the smallest divergence size whose repair window contains **≥10 samples** at the achievable cadence. That is the parameter the experiment should use.
4. If no size satisfies (3) within a corpus this harness can write in reasonable time, that is outcome (i) and the honest conclusion is that repair is not measurable here.

### Metrics

Primary: **samples per repair window** = (time to convergence) ÷ (achievable probe interval), as a function of divergence size. The decision is the smallest size where this is ≥10. Secondary: probe latency vs id-set size (the floor); whether time-to-convergence is monotone in divergence size across the three repetitions.

### Baselines / controls

#44's single observation — 50 objects, ~0.3s — is the anchor and the smallest point on the curve; if step 2 does not reproduce roughly that at n=50, something has changed and nothing further is trustworthy. Control for the timing itself: a run with **no** divergence must show the set already complete on the first poll, confirming the measured window is repair and not restart latency.

### Expected outcomes

(a) Repair grows with divergence and 5,000–50,000 objects give a multi-second window → the experiment samples at the existing cadence, no new machinery, and the healing curve is measurable. (b) Repair grows but only into the sub-second range → sub-second sampling is genuinely required; report the achievable resolution and design the experiment around few samples. (c) Repair is constant and sub-second at every size → outcome (i): not measurable by this instrument; the dissociation experiment reports endpoints, not a curve, and says why. (d) Time-to-convergence is not monotone → characterize before designing anything.

### Interpretation plan

(a) → the experiment issue specifies the divergence size and cadence from this study's numbers. (b) → same, with the sampling work scoped and the resolution stated as a limitation. (c) → the healing half of the dissociation prediction is dropped from the Weaviate experiment, `DECISION_LOG` records why, and the README's eventual Weaviate sentence claims the dissociation at two endpoints rather than as a curve. (d) → no experiment is designed until the relationship is understood. In every branch the experiment's parameters come from measurement rather than from a guess, which is the point of doing this first.

### Confounds considered

**Restart latency is not repair.** A node coming back spends time loading before it can serve; the control in Baselines separates the two, and time-to-convergence must be measured from the node's *first successful probe response*, not from `docker start`. **Write time is not free:** writing 50,000 objects takes minutes and the node must be down for all of it, so the divergence is also an outage of that length — long outages may trigger different repair paths than short ones, and the writeup must not present them as the same mechanism. **Probe cost changes with id-set size**, so a fast cadence and a large id set are in tension; step 1 exists to make that tradeoff explicit rather than discovered. **The measurement can perturb the thing measured** — polling at maximum rate puts load on the very node doing the repair, which could itself slow convergence; worth a sanity check comparing convergence time under fast and slow polling at one size, and stating the result either way.

### Before submitting

- [x] I checked README.md's "Open research questions" and research/DECISION_LOG.md and this isn't a duplicate or already-ruled-out question.
- [x] This is one answerable question, not a broad restatement of the whole research thesis.


---


## Instrument characterization

*Section added 2026-09-06. `SPEC_TEMPLATE.md:43` made this required on 2026-09-03; these five SPECs were opened after that date without it. The text below records what the study actually established about its apparatus — it is not back-filled content invented after the fact.*

This study **is** an instrument characterization, and was opened for exactly the reason `SPEC_TEMPLATE.md` requires one: PR #44's single ~0.3 s observation had become a design constraint. Properties surfaced: the probe-cost floor by id-set size; repair duration independent of divergence size across 50→5,000; the base64-in-URL bound (works at 15,000 ids, fails silently at 20,000); and — via steps 2b and 2c — that repair latency is timing-determined, which cancelled the sub-second sampling work entirely.

## Results

Digest-pinned Weaviate 1.29.0, 3 nodes, factor 3, 1 shard (verified via `verify_class`). Script: `characterize.py`; raw records in `results/`.

### Step 1 — probe cost, and a hard bound nobody had hit

| id-set size | median | notes |
|---|---|---|
| 10 | 31.7 ms | |
| 100 | 54.0 ms | |
| 1,000 | 40.4 ms | |
| 10,000 | 451.0 ms | |
| 15,000 | — | works; URL param 800 KB |
| **20,000** | — | **fails** (`ok=False`); URL param 1.07 MB |

The `ids` parameter is base64 in the **URL**, so the probe has a **hard ceiling near 1 MB of encoded ids — about 15,000 per call**. Beyond it the request does not fail slowly, it fails. Any future use at larger corpora must page the id set across calls. This was found by a run that reported `first_answer None`, i.e. the probe silently never succeeded, which is the same silent-failure shape as #26 and #38 and is why it is recorded here rather than left as an anomaly.

Sampling floor: **~50 ms** for id sets up to 1,000.

### Step 2 — repair duration is bimodal, and independent of divergence size

| divergence | repair (s), per repetition |
|---|---|
| 50 | 44.725, 0.008, 0.010 |
| 500 | 0.169, 50.234, 50.047 |
| 5,000 | 38.286, 40.787 |

**A 100× increase in divergence (50 → 5,000) does not increase repair time.** Every observation is either **sub-0.2 s** or **38–50 s**; nothing lands in between. Control: with no divergence, the victim was already complete on its first answer, so these windows are repair and not restart latency.

That distribution points at a **timing-determined mechanism** rather than work proportional to divergence. Two candidate readings were then tested directly (see "Step 2b"), because the obvious one is wrong in a checkable way. **Stated as consistent-with, not established** — Weaviate's sync scheduler was not observed directly, only its effect. At 5,000 the victim's first answer already held 13–140 objects, so streaming catch-up had begun before the probe could see it, which is why "duration" is only well defined once the divergence outlasts startup.

### Step 2b — order effect or timing? (added after review, prompted by the question "is it remembering something?")

The back-to-back repetitions in step 2 confound two explanations, because each run's duration set the next run's timing:

- **warming / "remembering"** — repair gets faster with repetition index, mediated by persistent state
- **timing** — latency depends on *when* the restart lands relative to some periodic process, and index carries no information

They were separated by repeating a fixed divergence (50 objects) **10 times with a randomized 0–50 s delay before each restart**, which scrambles timing while leaving index intact (`results/order_vs_phase.json`).

| rep | pre-delay (s) | repair (s) |
|---|---|---|
| 0 | 6.7 | 0.018 |
| 1 | 19.2 | 0.000 |
| 2 | 44.3 | 36.908 |
| 3 | 16.4 | 36.516 |
| 4 | 4.3 | 0.008 |
| 5 | 11.7 | 0.030 |
| 6 | 7.7 | 0.008 |
| 7 | 13.0 | 0.045 |
| 8 | 29.2 | 48.207 |
| 9 | 39.4 | 43.988 |

**Warming is refuted.** correlation(repair, repetition index) = **+0.373** — warming predicts a strongly *negative* correlation — and the first run of the series was among the fastest (0.018 s). Slow runs fall at reps 2, 3, 8, 9, not at the start.

**Timing is what differs**: fast runs had mean pre-delay 10.4 s, slow runs 32.3 s.

**But the latency is not a countdown to a tick.** Pooling all 18 observations from steps 2 and 2b:

```
0.000 0.008 0.008 0.008 0.010 0.018 0.030 0.045 0.169 |gap| 36.516 36.908 38.286 40.787 43.988 44.725 48.207 50.047 50.234
```

**Nothing falls between 0.2 s and 36 s.** A uniform wait-for-next-tick, with restart phase sampled roughly uniformly by the randomized delay, would populate that interval; it is empty across 18 runs. So the mechanism is better described as **two discrete paths** — an immediate catch-up that either does or does not capture the restarting node, and, when it does not, a wait of ~36–50 s — rather than a continuous countdown.

> **WITHDRAWN 2026-09-06 by step 2c.** The paragraph above is false. Step 2c ran 18 more repairs at fixed timing offsets and **12 of them land between 0.2 s and 36 s**. The interval was empty in step 2b because randomizing the delay sampled the latency surface sparsely, not because Weaviate has two paths. The withdrawn text is kept in place, as `GIT_WORKFLOW.md` requires. See step 2c below.

**A claim in an earlier draft of this spec is withdrawn on this evidence:** it described the wait as "time-until-the-next-tick, uniform-ish over an interval of roughly 40–50 s". The uniform-interval half is falsified by the gap. The timing-determined half survives.


### Step 2c — what selects the path? (amendment, 2026-09-06, pre-registered before the run)

Step 2b's own limitations section named this as the next question: *"Why a longer pre-delay makes the slow path likelier (10.4 s vs 32.3 s mean) is unexplained."* Re-analysing the step 2b data (`analyze_path_selection.py`, no new runs) shows the association is stronger than that sentence suggests:

| | pre-delays (s) | mean |
|---|---|---|
| fast (≤1 s), n=6 | 4.3, 6.7, 7.7, 11.7, 13.0, 19.2 | 10.4 |
| slow (>1 s), n=4 | 16.4, 29.2, 39.4, 44.3 | 32.3 |

Exact two-sided Mann-Whitney U = 23 of a possible 24, **p = 0.0190** (4 of 210 permutations; the floor at n=4 vs 6 is 0.0095). The groups are nearly separated: a single threshold at ~13–16 s misclassifies **1 of 10** runs, and the only overlap is fast-at-19.2 s against slow-at-16.4 s.

This is post-hoc on data collected to answer a different question, so it is a hypothesis, not a result. It is worth one confirmatory run because it converts "timing" from a placeholder into something mechanical and checkable.

**Hypothesis.** Path selection is decided by **how long the victim is absent before it restarts**, not by the phase at which the restart lands. A short absence takes the fast path; a long absence takes the slow one.

**The confound this design exists to break.** In step 2b the delay was inserted *between the write and the restart*, so a longer delay made the node absent longer **and** made the divergence older by exactly the same interval. Those are one quantity in that data and cannot be told apart by collecting more of it. Three conditions separate them, at a fixed divergence of 50 objects:

| | victim absent | divergence age at restart |
|---|---|---|
| **A** short | ~6 s | ~6 s |
| **B** long | ~40 s | ~40 s |
| **C** long, young | ~40 s | ~6 s (objects written in the last ~6 s of the outage) |

C is the discriminating cell. If **absence** selects the path, C behaves like B (slow). If **divergence age** selects it, C behaves like A (fast). Conditions are interleaved in randomized order so repetition index cannot align with condition.

**Metric.** Proportion of runs taking the slow path (>1 s) per condition, 6 repetitions each. The 0.2 s / 36 s gap established in step 2b makes this classification unambiguous; any run landing inside the gap is reported separately and weakens the two-path reading.

**Pre-registered outcomes.**
- **(a) absence decides** — A fast, B and C slow. Fisher exact on A vs C, complete separation at 6 v 6 gives p = 0.0022.
- **(b) divergence age decides** — A and C fast, B slow.
- **(c) neither** — every condition mixed at similar rates; then step 2b's association was a small-n accident and "phase" stands as the honest description.
- **(d) mixed within a condition but at different rates** — both quantities contribute, or the threshold sits inside the range sampled; report rates and do not name a mechanism.

**Interpretation plan.** (a) or (b) names the controlled variable and the dissociation experiment can *choose* its path rather than accept whichever it draws — worth having, because a fast-path run has no observable healing window at all. (c) retracts the step 2c hypothesis in this spec and leaves step 2b's conclusion exactly as it stands. In no branch does this change step 2b's finding or the ≥60 s / 1–5 s decision, both of which are derived from the slow path's duration.

**Results (2026-09-06, 18 runs, 6 per condition).**

| condition | absent | divergence age | repair (s), sorted | median |
|---|---|---|---|---|
| **A** short | 6 s | ~4 s | 0.021, 0.025, 51.578, 51.663, 52.281, 52.335 | 51.62 |
| **B** long | 40 s | ~38 s | 0.735, 0.995, 1.125, 1.223, 1.288, 2.245 | 1.17 |
| **C** long, young | 40 s | 6 s | 30.334, 30.353, 30.688, 31.096, 31.099, 31.146 | 30.89 |

**Step 2b's gap claim is withdrawn.** It said: *"Nothing falls between 0.2 s and 36 s. … it is empty across 18 runs. So the mechanism is better described as two discrete paths."* **Twelve of these eighteen runs fall inside that interval.** The claim is false and the "two discrete paths" reading built on it does not survive.

Why it looked true: step 2b randomized the pre-restart delay, which scrambles phase but samples the latency surface sparsely and unevenly. Eighteen draws that way populated two regions and missed the middle. The emptiness was a property of the sampling, not of Weaviate.

**The pre-registered statistic is void, and this must be said plainly.** The metric above fixes a binary fast/slow split at 1 s *because the gap was believed to exist*. With the gap gone that threshold falls inside B's tight 0.735–2.245 s cluster and splits it arbitrarily — which is exactly what `path_selection.py` printed: "B_long slow 4/6", and Fisher A vs C p = 0.4545. Those numbers are artifacts of a dichotomization whose premise failed; they are reported here only so the record shows what the pre-registered analysis produced. **The continuous data below is the result.** Pre-registration protected the design from motivated reading, and it still encoded an assumption that the data destroyed. That is the lesson worth keeping.

**What the continuous data shows.** Each condition is internally tight — C spans 0.81 s across six runs — and the conditions separate from each other:

- **B vs C isolates divergence age at fixed absence** (both 40 s). They are **disjoint**: max(B) = 2.245 s < min(C) = 30.334 s. Exact two-sided Mann-Whitney at 6 v 6: **U = 0, p = 0.0022**, the floor for this design. B's data had existed ~38 s before the node returned and reconciled in ~1 s; C's had existed ~6 s and took ~31 s.

  **Two readings survive this, and the data does not choose between them** (added in review round 1). `repair_s` is measured from the victim's first probe response *after restart*, and the conditions differ in when the write happened relative to that origin. Measuring from the **write** instead — `age_s + repair_s` — collapses most of the difference:

  | condition | `repair_s` median | `age_s + repair_s` median | range from the write |
  |---|---|---|---|
  | **B** long | 1.17 s | **38.91 s** | 37.84 – 41.35 |
  | **C** long, young | 30.89 s | **36.89 s** | 36.33 – 37.15 |
  | **A** short, fast (n=2) | 0.023 s | 4.42 s | 3.83 – 5.02 |
  | **A** short, slow (n=4) | 51.97 s | 56.29 s | 55.74 – 56.56 |

  B and C differ by **30×** on `repair_s` and by **~5%** on time-since-write. So *"an older divergence repairs faster"* — the sentence an earlier draft of this section asserted — is **withdrawn as an explanation**. It is equally consistent with convergence occurring at a roughly fixed latency **after the write**, in which case a divergence already ~38 s old when the node returns simply has almost none of its wait left, and the same process is being observed from a different origin. That version would be close to definitionally true and would say nothing about Weaviate.

  Neither reading is established. A's slow runs sit at ~56 s since-write rather than ~37 s, so a single global constant does not fit either. **What stands is the measurement, not a mechanism:** at a fixed 40 s outage, divergence age and time-to-convergence-after-restart trade off almost exactly, and whether the repair clock is anchored to the write or to the restart is the open question this study leaves. It is cheap to test with the existing harness — hold the write-to-restart interval fixed and vary absence — and is the next step rather than an assumption.
- **A vs C** — the comparison this amendment nominated as discriminating — gives p = 0.3939 and is uninformative, because A is not unimodal. Nominating it was a design error on my part: **B vs C is the clean contrast**, since it holds absence fixed and varies only age. A vs C varies absence but leaves A's own bimodality in the comparison.

**A is bimodal and unexplained.** Two runs at ~0.023 s, four at ~52 s, at parameters that differ only in the ~1.6 s of jitter in realized age (3.4–5.0 s). Whatever selects between those outcomes is not captured by the two variables this design manipulates. No mechanism is named here.

**A caution about the shape.** Pooled across conditions the 18 values still cluster, with empty stretches at 2.2–30.3 s and 31.1–51.6 s. That is **not** evidence of discrete mechanisms: only three parameter combinations were sampled, so the pooled distribution inherits the design's discreteness. Reading multimodality out of a pooled sample over a handful of hand-picked cells is the same error step 2b made, one level up.

**Pre-registered outcome: (d).** Not (a) — C does not behave like B. Not (b) — C does not behave like A. Both quantities appear to matter, and per the interpretation plan for (d), rates are reported and no mechanism is named.

**What this does not change.** The ≥60 s observation window and 1–5 s cadence from step 3 were derived from the slow path's *duration*, which is unaffected — indeed C's ~31 s and A's ~52 s sit inside the range those parameters were chosen to cover. The step 2 finding that repair is independent of divergence size across 50→5,000 is untouched; this amendment varies timing at a fixed size.

### Step 3 — the decision

At a 40–50 s window: **1 s cadence gives 40–50 samples; 5 s gives 8–10.** The ≥10-sample bar is met at the cadence this project already uses on nano-db and Qdrant, at every divergence size tested, with a 50 ms probe floor two orders of magnitude below what is needed.

**Sub-second sampling is not required, and the premise that it was is wrong.**

## Interpretation

**The follow-on this issue was created to scope should not be built.** PR #44 recorded repair converging in "~0.3 s" and named sub-second sampling as the last prerequisite. That figure is now visible as **one draw from a bimodal distribution** — a run that restarted just before a sync tick. Three repetitions at the same divergence size give 44.7 s, 0.008 s, 0.010 s. A single observation was generalised into a bound on the experiment's design, and it was wrong in the direction that would have cost the most work: it implied machinery nobody needs.

**The healing signal is a step at small divergences, and a short ramp at larger ones — re-argued 2026-09-06.** This conclusion originally leaned on the "two discrete paths" reading, which step 2c withdrew, so it is restated here from the trajectory data alone, which is independent of that reading. In the committed trajectories the victim sits at 0 held objects and then completes over a span of **at most 8–16 ms at 50 objects, at most 106–167 ms at 500, but 5.8–6.3 s at 5,000**. The small-size figures are **upper bounds, not measured durations**: `trajectory_tail` holds at most the last three points, and at 50 objects it is two samples with the count going 0 → 50 between them, so the transition was faster than the probe could resolve rather than 8 ms long. At 5,000 the tail spans 5.8–6.3 s with intermediate values (2,500 → 5,000), so that ramp is genuinely observed and the size-dependence conclusion rests on it. Note also that `samples_in_window` reports up to 8,532 samples while only ≤9 points are persisted — the committed trajectories are truncated by construction, not lost. So at 50–500 objects the transition is a step at any cadence this project can achieve; at 5,000 it is a ramp that a 1 s cadence would resolve into roughly **5–6 points**. The step-versus-curve question is therefore **decided by divergence size**, not settled in general — an experiment wanting a curve should use 5,000, one wanting an unambiguous step should use 50–500. Either way the robust comparison is **whether the transition happens for `completeness` and does not for `index_recall`** within the same window, observed for ≥60 s so the run outlasts the wait that precedes it.

**What is not established.** The mechanism behind the two paths — neither the fast catch-up nor the slow sync was observed in Weaviate itself, only their effect on a probe. The slow path's ~36–50 s spread is not resolved into a period; that would need many more restarts and a way to observe the scheduler. Why a longer pre-delay makes the slow path likelier (10.4 s vs 32.3 s mean) is unexplained and is the obvious next question if this ever matters. Whether divergence size matters *above* 5,000 — the 20,000 cell failed on the probe's URL ceiling, not on repair. Whether repair behaves differently after a long outage: writing 20,000 objects took 48.6 s, so big divergences are also long outages, and the two are confounded by construction here. n = 2–3 per cell, one host, one build.

## Decision

**MERGE**, and **do not open the sub-second sampling issue**. The characterization did its job: it prevented work that a single unreplicated number had justified, and it produced the parameter the experiment actually needs (observe ≥60 s per run at 1–5 s cadence).

**Consequences to file:**
1. `analysis/*` — correct the "~0.3 s" claim where it now sits on `main` (`research/weaviate_nonperturbing_probe/SPEC.md` and its README, from PR #45), replacing it with the bimodal distribution and the step-not-curve reading. The original text stays, struck, per `GIT_WORKFLOW.md`.
2. `experiment/*` — the dissociation experiment, now fully specified on the instrument side: 1–5 s cadence, ≥60 s observation, divergence anywhere in 50–5,000, id sets ≤15,000 per probe call.
3. `method/*` — page the id set across calls if a future corpus exceeds ~15,000 ids.
