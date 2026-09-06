# Spec: method/weaviate-repair-window

**Branch:** `method/weaviate-repair-window` (from #47's head; stacked behind it until it lands on `main`)
**Issue:** #48 (body copied verbatim below, per `AGENT_PIPELINE.md`)
**Date opened:** 2026-09-05
**Status:** COMPLETE

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

**A claim in an earlier draft of this spec is withdrawn on this evidence:** it described the wait as "time-until-the-next-tick, uniform-ish over an interval of roughly 40–50 s". The uniform-interval half is falsified by the gap. The timing-determined half survives.

### Step 3 — the decision

At a 40–50 s window: **1 s cadence gives 40–50 samples; 5 s gives 8–10.** The ≥10-sample bar is met at the cadence this project already uses on nano-db and Qdrant, at every divergence size tested, with a 50 ms probe floor two orders of magnitude below what is needed.

**Sub-second sampling is not required, and the premise that it was is wrong.**

## Interpretation

**The follow-on this issue was created to scope should not be built.** PR #44 recorded repair converging in "~0.3 s" and named sub-second sampling as the last prerequisite. That figure is now visible as **one draw from a bimodal distribution** — a run that restarted just before a sync tick. Three repetitions at the same divergence size give 44.7 s, 0.008 s, 0.010 s. A single observation was generalised into a bound on the experiment's design, and it was wrong in the direction that would have cost the most work: it implied machinery nobody needs.

**The healing signal is a step, not a decay.** `completeness` on a diverged replica stays flat and then jumps to complete at the next sync tick. That changes what the dissociation experiment should measure: not the *shape* of a repair curve, but **whether the step happens for `completeness` and does not happen for `index_recall`** within the same window. That is an easier and more robust claim than a curve comparison, and it is unaffected by where in the sync cycle a run happens to start — provided each run is observed for longer than one full interval (≥60 s to be safe at 40–50 s observed).

**What is not established.** The mechanism behind the two paths — neither the fast catch-up nor the slow sync was observed in Weaviate itself, only their effect on a probe. The slow path's ~36–50 s spread is not resolved into a period; that would need many more restarts and a way to observe the scheduler. Why a longer pre-delay makes the slow path likelier (10.4 s vs 32.3 s mean) is unexplained and is the obvious next question if this ever matters. Whether divergence size matters *above* 5,000 — the 20,000 cell failed on the probe's URL ceiling, not on repair. Whether repair behaves differently after a long outage: writing 20,000 objects took 48.6 s, so big divergences are also long outages, and the two are confounded by construction here. n = 2–3 per cell, one host, one build.

## Decision

**MERGE**, and **do not open the sub-second sampling issue**. The characterization did its job: it prevented work that a single unreplicated number had justified, and it produced the parameter the experiment actually needs (observe ≥60 s per run at 1–5 s cadence).

**Consequences to file:**
1. `analysis/*` — correct the "~0.3 s" claim where it now sits on `main` (`research/weaviate_nonperturbing_probe/SPEC.md` and its README, from PR #45), replacing it with the bimodal distribution and the step-not-curve reading. The original text stays, struck, per `GIT_WORKFLOW.md`.
2. `experiment/*` — the dissociation experiment, now fully specified on the instrument side: 1–5 s cadence, ≥60 s observation, divergence anywhere in 50–5,000, id sets ≤15,000 per probe call.
3. `method/*` — page the id set across calls if a future corpus exceeds ~15,000 ids.
