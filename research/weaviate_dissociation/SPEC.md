# Spec: does `completeness` heal on Weaviate while `index_recall` does not?

**Branch:** `experiment/weaviate-dissociation`
**Date opened:** 2026-09-06
**Status:** COMPLETE — outcome (a). The dissociation holds in 4 of 5 seeds (the fifth right-censored, not negative): `completeness` returns to exactly 1.0 while `index_recall` falls ~0.5, disjoint at p = 0.0079, against a no-chaos control with 0.0000 drift. **A claim about a 60 s horizon, not about permanence** — see Interpretation.

Issue: closes #54. Body copied verbatim below (per `research/AGENT_PIPELINE.md`'s implementer instructions — this is the issue text unmodified, not a paraphrase).

---

## Research question

On a system with **real anti-entropy**, does `completeness` heal after node-kill chaos while `index_recall` does not — the dissociation this project's central argument predicts?

## Hypothesis

Within one observation window after chaos stops, a diverged Weaviate replica's `completeness` returns to 1.0 while its `index_recall` does not return to its own pre-chaos baseline.

## Null / alternative hypothesis

(i) **Both heal.** Weaviate's repair rebuilds enough of the index as a side effect of re-inserting objects that graph quality recovers too. Then the dissociation is false on the one system that could show it, and the project's structural argument is substantially weakened — this must be reported at least as prominently as a positive result.

(ii) **Neither heals** within the window — repair did not fire, or the window was too short. Distinguishable from (i) by the `completeness` series itself; if the step never happens, this is an instrument or timing failure, not a finding.

(iii) **`index_recall` cannot be compared** across the two snapshot moments at useful precision. If snapshot noise exceeds the effect, the design cannot answer the question and says so.

## Motivation

This is the experiment the project has been building toward, and the only one that tests its field-level claim rather than a within-system one.

`RELATED_WORK.md` §4 argues structurally: every production anti-entropy mechanism operates on **exact object identity**, and two correct HNSW graphs over identical data differ bit-for-bit, so object-level repair *cannot* be pointed at the index. The prediction is sharp: after chaos, a replica's missing objects come back and its graph quality does not.

Neither system tested so far can decide it. nano-db has **no** anti-entropy by design, so its 0% healing is close to definitional — which is why `RELATED_WORK.md`'s novelty verdict demoted that result in #55. Qdrant heals both axes at 180 s. Weaviate is the only one of the three where object-level repair exists and can be watched separately from graph quality.

Four method studies (#41, #43, #46, #48) exist solely to make this runnable, and all four are closed.

## Experimental design

**Every parameter comes from a measurement, not a guess.**

- **Topology:** 1 shard × 3 replicas, `weaviate@sha256:4d2eceef…` (digest-pinned, #46), with `verify_class()` asserting factor 3 / 1 shard so a stale auto-schema class cannot silently make it sharded-not-replicated (#46's hazard).
- **Divergence size: 5,000 objects.** #48 found repair is size-independent, so size is free to choose on other grounds — and at 5,000 the completeness transition is a genuine ~6 s ramp rather than a sub-200 ms step, which a 1 s cadence resolves into 5–6 points. Below 500 the transition is faster than the probe and only endpoints are observable.
- **Cadence: 1 s**, sampling `objects_present_ids` per replica (#43: 20/20 probes at 5 s with peers up, proven local; #46: per-id set, 0 false positives).
- **Observation: ≥60 s** after the last kill, which outlasts the longest repair latency seen in #48 (~52 s).
- **Id set ≤15,000 per probe call** — #48 found base64 ids in the URL work at 15,000 and fail *silently* at 20,000. At 5,000 there is ample margin.
- **`index_recall`:** snapshot before chaos and after the observation window, via the isolation probe (#41), accepting ~10 min of node health per snapshot and therefore no sampling.
- **Seeds: 5**, with the exact two-sided Mann-Whitney at the 5v5 floor.

**Ground truth for `index_recall` is computable locally and needs no server support.** The corpus is generated from a seeded RNG, so the exact top-k over *the subset the replica actually holds* — which `objects_present_ids` reports — can be computed with numpy. `index_recall` is then the replica's ANN result against exact search over its own data, holding data constant, which is the definition this project already uses on nano-db and Qdrant.

~~**One parameter is blocked.** The observation window must be timed from the right origin, and whether Weaviate's repair clock is anchored to the write or the restart is exactly what **#56** is measuring right now. Until it reports, this spec fixes the window as *"≥60 s measured from the later of (write, restart)"*, which is safe under either answer at the cost of a longer run. If #56 settles the anchor, this is tightened and the amendment dated.~~

**Unblocked by #56 — see Amendment 1 (2026-09-06) below.** The window is now timed from the **restart**, and a `t = 0` completion is recorded as **censored**, not as an instant recovery.

## Metrics

**Primary: the dissociation** — per seed, whether `completeness` on the victim returns to 1.0 within the window **and** post-chaos `index_recall` is outside its own pre-chaos range.

Secondary: time-to-completeness-recovery; the `index_recall` delta with its snapshot-to-snapshot noise stated.

No threshold is registered on a continuous quantity. "Returns to 1.0" is exact set equality on ids, not a cutoff, and the `index_recall` comparison is against a measured no-chaos noise floor rather than a fixed number — per `SPEC_TEMPLATE.md`'s rule after #48 step 2c.

## Instrument characterization

Everything below is measured, from the four closed method studies:

- **Per-replica reads work without perturbing peers** (#43): 20/20 probes at 5 s cadence, provably local — refuses when the node is down, returned 0 bytes while peers returned 31,190.
- **Presence is a set, not a byte count** (#46): 8/8 checks against a constructed expectation, 0 false positives on never-written ids.
- **Repair is timing-determined and size-independent** (#48) across 50→5,000, so size is free and 5,000 is chosen for the resolvable ramp.
- **The probe's hard ceiling is ~15,000 ids**, found by hitting it — it fails *silently*.
- **`_search` is unavailable on the internal port** (#43, 415 across 12 content types), which is why `index_recall` is snapshot-only and the experiment is asymmetric by construction.
- **Writes at `consistency_level=ONE` with a node down occasionally take minutes** (#56, observed mid-run). The harness must time the outage from realized timestamps, not requested ones, and must not assume the write returns promptly.

## Baselines / controls

- **A no-chaos run per seed** is required and is what makes the `index_recall` comparison interpretable: two isolation-probe snapshots the same interval apart, with no chaos, give the snapshot-to-snapshot noise floor. Without it any post-chaos difference is uninterpretable, since the snapshots are separated in time on a live corpus.
- **A healthy-replica control** — a node never killed, probed alongside the victim in the same runs.
- **The #46 topology check** must pass before any run counts.

## Expected outcomes

(a) `completeness` recovers, `index_recall` does not → the dissociation holds on the one system with real anti-entropy; the project's strongest possible result. (b) Both recover → outcome (i); report as prominently, and revise `RELATED_WORK.md` §4. (c) The `index_recall` delta sits inside the no-chaos noise floor → outcome (iii); the instrument bounds the question rather than answering it. (d) Repair never fires → outcome (ii), an apparatus failure, recorded as such.

## Interpretation plan

Whatever the outcome, the write-up carries **the asymmetry of the instrument in the claim itself**: the data axis is a 1 s time series, the graph axis is two endpoints. A dissociation shown between a series and a pair of snapshots is weaker than one between two series, and must never be written as though both were watched continuously.

(a) updates the README's ESTABLISHED box with that qualifier attached. (b) or (c) updates `RELATED_WORK.md` §4 and the DO-NOT-CLAIM list, and is recorded in `DECISION_LOG.md` as a negative result with its design intact.

## Confounds considered

**The snapshot cost perturbs the thing measured.** The isolation probe leaves a node UNHEALTHY for ~10 minutes (#41), so the post-chaos `index_recall` snapshot cannot be taken until the completeness window has closed — the two axes are measured at *different moments*. This is the central design compromise and must be stated in the result, not buried.

**Write duration is outage duration.** Writing 5,000 objects while the victim is down makes the divergence and the outage the same interval (#48), and #56 found the write can occasionally take minutes.

~~**The repair clock's origin is unresolved** (#56, running) — hence the conservative window above.~~ **Resolved, partially, by #56 — see Amendment 1.** Within a regime the clock runs from the restart; *which* regime is selected by divergence age. The window is anchored accordingly and the realized age is recorded per run so the regime is known rather than assumed.

**Undocumented API.** The probe depends on an internal endpoint of one pinned build; a Weaviate upgrade invalidates the instrument, not just the numbers.

**One host, one topology, 5 seeds** — the same scale limit every study here carries.

## Amendment 1 (2026-09-06, before any run): the window is anchored to the restart, and `t = 0` is censored

The blocked parameter above is unblocked. #56 (PR #59) has reported, and this
amendment applies its answer. **No run of this spec has happened yet** — the
harness has never been executed against a cluster and `results/` holds no data —
so nothing here is a post-hoc choice. It is dated anyway, because a window
origin changed after a dependency reported is exactly the kind of edit that has
to be legible later.

### What #56 found, and the part of it this spec depends on

#56 asked whether repair is anchored to the write or the restart and returned
**neither, cleanly**. Its aggregate statistic said write-anchored — CV(`repair_s`)
0.766 against CV(`age + repair`) 0.132, corr(age, `repair_s`) = −0.942 — and
disaggregating by divergence age **reverses** it: within either regime `repair_s`
is flat (spreads of 2.34 s and 1.45 s across 13 s and 8 s of age variation) while
`age + repair` tracks age at +0.99. The aggregate slope is produced entirely by a
step *between* regimes. Simpson's paradox.

Two consequences, and this spec needs both:

1. **Within a regime, the clock runs from the restart.** So the window is timed
   from the restart.
2. **Divergence age selects the regime.** Below ~15 s the victim waits ~32 s;
   above ~30 s it reconciles in ~2 s. The threshold is bracketed to (15 s, 30 s)
   and was never sampled.

**This experiment sits in the young regime by construction.** The harness restarts
the victim immediately after the divergence write returns, so realized age is
≈ 0 s — younger than any age #56 sampled, and its nearest observations (realized
ages 0.126 s and 0.186 s) gave `repair_s` of 30.4 s and 19.4 s. Expect the ~32 s
wait, not the ~2 s one.

Provenance note, so this is not read as firmer than it is: PR #59 is at
`stage:changes-requested` as of this writing. Its five review findings concern a
step-1 number that disagrees with its own analyzer output, an unscoped "absence is
irrelevant" claim, run accounting, and the disaggregation not being reproducible
from committed code. **None of them bear on the two consequences above** — review
round 1 recomputed the regime split independently from the committed raw JSON and
reproduced it exactly, with the two clusters disjoint (31.08–33.43 s and
0.82–2.27 s, nothing between). If a later round overturns the split itself, this
amendment is void and the window returns to `max(write, restart)`.

### The three changes

**(1) Origin: the restart, not `max(write, restart)`.** `origin = t_restart`.
Under the old rule a write that takes minutes pushes the origin past the restart
and the window opens *after* repair has already fired — the failure that voided
#24 and #9, arriving from the direction #56 warned about. The realized divergence
age `age_s = t_restart − t_write` is recorded per run, and a run whose age lands
above 15 s is reported as **out of the young regime** rather than pooled with the
others. #56's own methodological finding is that pooling across a step is what
produced its wrong answer; this spec will not repeat it at n = 5.

**(2) A `t = 0` completion is censored, not instant.** #56's probe-perturbation
check was voided by exactly this: its three slow-poll runs report `repair_s = 0.000`
with `first = 50/50` — the victim already held every id on its *first* sample. That
does not show repair was instantaneous. It shows the first sample landed after
convergence, and at a 1 s cadence the two are indistinguishable.

This spec plans that same 1 s cadence, so it inherits the same blind spot. The
harness therefore records the first sample's offset from the origin
(`first_sample_offset_s`), and:

- if the **first** sample is already complete, `recovery_s` is **left-censored**:
  recorded as `censored: "left"` with the observed bound, and reported as
  *"recovered at or before t = <offset>"*, never as a recovery time;
- if the **last** sample is still incomplete, it is **right-censored**
  (`censored: "right"`) — recovery did not occur within the window;
- only a run with an incomplete first sample and a complete later one yields an
  uncensored `recovery_s`.

Censored runs are **kept, not dropped**, and the censoring status is reported
alongside every recovery figure. This matters for the primary metric: the
dissociation asks whether `completeness` returns to 1.0 within the window, and a
left-censored run answers that **yes** — the censoring bounds *when*, not
*whether*. Secondary time-to-recovery is the quantity that degrades to a bound.

**(3) The ≥60 s window stands, and here is its actual margin.** It is not
generous. The slowest repair observed anywhere in this project is **53.3 s**
(#56, `short6`), against a 60 s window — 6.7 s of margin, not the comfortable
outlasting the original line implied. It is kept rather than lengthened because
the expected young-regime wait is ~32 s and every young-regime observation across
#48 and #56 falls in 19.4–33.4 s, so 60 s covers the expected behaviour with
roughly 2× margin and the tail case with little. Right-censoring (change 2) is
what makes that honest: a run that does not converge by t = 60 s is recorded as
censored rather than as a failure to heal, and outcome (ii) — "neither heals" —
may not be concluded from a right-censored series.

### What this does not fix

The threshold between the regimes is unlocated, so "this experiment sits in the
young regime" rests on realized age ≈ 0 s being far below a bracket of (15 s, 30 s),
not on knowing where the boundary is. And #56's short-outage bimodality (six runs
at ~0.02 s against four at 43–53 s at identical parameters) is unexplained; if it
reaches this experiment it will appear as a mixture across seeds, which n = 5
cannot resolve. Both are stated here so a mixed result is not later reframed as
a surprise.

## Amendment 2 (2026-09-06, after a discarded partial run): the graph axis was not measuring graph quality

The first attempt at the sweep was **stopped 12 minutes in and its data
discarded**, because the very first number it produced was impossible.

**What was seen.** The no-chaos control for seed 20260900 — an undisturbed,
healthy replica, nothing killed, nothing diverged — reported
`index_recall = 0.23`, identical before and after. A healthy replica scored
against exact search over its own data must sit near 1.0; that is what
"the graph is intact" means. 0.23 is not a finding about Weaviate, it is an
instrument returning noise.

**Why this is not an outcome-dependent amendment.** What was read was a
*control*, not a comparison: no chaos condition had completed, and nothing about
`completeness`, the dissociation, or the difference between conditions was
observed. The amendment changes how the instrument computes a number, on
evidence that the number was invalid on its own terms — a healthy replica cannot
score 0.23 — not on evidence about which way the result came out. The run's
partial output is discarded rather than kept, so nothing measured under the
broken instrument reaches the analysis.

### Defect 1 — ground truth used the wrong distance metric

`exact_topk()` ranked neighbours by **L2** (`np.linalg.norm(sub - query)`) while
the class's HNSW index is configured with **`distance: cosine`**, read from the
live schema:

```
vectorIndexType : hnsw
distance        : cosine
```

So `index_recall` compared Weaviate's cosine-nearest-10 against exact
L2-nearest-10 — two different neighbour sets. The metric was never a measure of
graph quality; it measured how often the cosine and L2 orderings happen to
agree on random-normal vectors.

Fixed by reading `distance` from the live schema and computing ground truth with
it, refusing to run on a metric it cannot reproduce. Derived from the artifact
rather than duplicated, which is the fix #17's review applied to the kill
scheduler's mirrored `FIXED_DOWN_S`: a constant that can silently disagree with
reality is a defect even when it is currently correct.

### Defect 2 — the class was shared scratch and had never been cleared

The decisive one. `RrdVector` is written by **every** Weaviate study — #41, #43,
#46, #48, #56 — and nothing has ever emptied it. At the start of the run it held:

```
count: 14,200        against a per-run corpus of 5,000
```

`index_recall` scores the replica's ANN answer against exact search over *the
ids this run wrote*. With 9,200 objects from older studies also in the index,
Weaviate's nearest ten are drawn from the superset while the ground truth is
drawn from the subset. The two disagree for reasons that have nothing to do with
the graph, and the disagreement grows with every study that ever ran.

Fixed by deleting and recreating the class at the start of **each run**, so it
holds exactly that run's corpus. Recreating is safe: the class is scratch
infrastructure and every study that wrote into it has its own committed
`results/`, so no evidence lives there.

### The fixes, verified against the live cluster before restarting

| condition | `index_recall`, healthy undisturbed replica |
|---|---|
| as originally written (14,200-object class, L2 truth) | **0.23** |
| class reset to 5,000, still L2 truth | 0.32 |
| class reset to 5,000, cosine truth — **both fixes** | **0.9750** |

Both fixes are necessary; neither alone is sufficient. 0.9750 is what an intact
HNSW graph is supposed to look like, and it is now the baseline against which
any chaos-induced drop is measured.

### Defect 3, the one that let the other two through — there was no positive control

This is the part worth carrying to other specs.

The Baselines/controls section required a **no-divergence control** — and got
one, for `completeness`: "the victim is complete on its first answer." Nothing
anywhere required the **graph** axis to be shown reading ~1.0 when nothing is
wrong. The Instrument characterization section characterizes the completeness
probe in detail (cadence, locality, per-id decoding, the 15,000-id bound) and
inherits `index_recall` from #41 without ever asking what it returns on a
healthy replica.

So the spec could pass its own pre-flight checks — topology verified, probe
proven local, ids decoded per-id — with the graph axis measuring nothing at all.
Every check was on the axis that was already working.

The harness now asserts `index_recall >= 0.90` on the no-chaos control and
**aborts the run** if it fails, naming the class count and the distance metric
in the message, so the same failure can never again be discovered by reading a
result. A measurement axis with no positive control is an axis that cannot fail
visibly, and this experiment's central claim is a *dissociation* — one axis
moving while the other does not — which a silently dead axis would have produced
for free.

**This is why the run is worth its four hours only now.** Had it completed as
written, it would have reported `index_recall` failing to heal — the
pre-registered hypothesis — from an instrument that could not have shown healing
under any circumstances.

## Results

**Outcome (a). 10 of 10 runs, no aborts. The dissociation holds in 4 of 5 seeds; the fifth is censored, not negative.**

### The control first

A graph axis that drifts on its own has nothing to say about a graph axis that drops under chaos. Across all five no-chaos seeds, `index_recall` before and after the identical pair of snapshots:

| seed | before | after | drift |
|---|---|---|---|
| 20260900 | 0.980 | 0.980 | **0.000** |
| 20260901 | 0.980 | 0.980 | **0.000** |
| 20260902 | 0.965 | 0.965 | **0.000** |
| 20260903 | 0.975 | 0.975 | **0.000** |
| 20260904 | 0.990 | 0.990 | **0.000** |

Max |drift| = **0.0000**. The instrument that Amendment 2 had to rebuild is now exactly stable when nothing happens, which is what makes any drop attributable to the chaos rather than to the measurement.

### The two axes

| seed | IR before | IR after | IR delta | `completeness` end | censored | recovery |
|---|---|---|---|---|---|---|
| 20260900 | 0.980 | 0.520 | **−0.460** | **1.00** | none | 38.19 s |
| 20260901 | 0.975 | 0.460 | **−0.515** | **1.00** | none | 38.21 s |
| 20260902 | 0.965 | 0.460 | **−0.505** | **1.00** | none | 36.79 s |
| 20260903 | 0.975 | 0.515 | **−0.460** | **1.00** | none | 40.94 s |
| 20260904 | 0.990 | 0.790 | −0.200 | 0.30 | **right** | — |

`index_recall` after: baseline **0.9780 ± 0.0091** against chaos **0.5490 ± 0.1378**, disjoint, exact two-sided Mann-Whitney **p = 0.0079** — the floor at 5 v 5, meaning complete separation, not a large effect. The effect size is carried by the deltas, which are around **−0.5 on a 0–1 metric**.

**The pre-registered primary metric — completeness returns to 1.0 *and* post-chaos `index_recall` falls outside its own pre-chaos range — is met in 4 of 5 seeds.** The fifth, 20260904, is **right-censored**: its completeness series had reached only 30% when the 60 s window closed, so it answers "did it heal within 60 s" with *unknown*, not with *no*. Amendment 1 is what makes that distinction available; without it the run would have been recorded as a failure to heal.

### The window was set correctly

Realized divergence age was **0.277–0.329 s** across all five chaos runs — every one in #56's *young* regime, as Amendment 1 predicted by construction. Observed recovery: **36.79–40.94 s**, against #56's young-regime prediction of ~32 s and its observed 19.4–33.4 s range. Slightly slower than predicted and comfortably inside the 60 s window, with the one exception being the censored seed.

That is a cross-study prediction made in advance from a different experiment's data and borne out.

## Interpretation

**On the one system in this project with real anti-entropy, object-level repair restores the data and does not restore the graph — at a 60 second horizon.**

That is the field-level claim `RELATED_WORK.md` positions the contribution around, and it is the first direct test of it. Qdrant and nano-db both lack graph-level repair to observe, so neither could answer it.

**The strongest form of the evidence is the conjunction within a single run.** In four seeds the victim ends the window holding **every** object it missed — `completeness` exactly 1.0, an id-set equality, not a threshold — while its `index_recall` sits at roughly half its own pre-chaos value. Same replica, same window, same instrument. The data came back; the graph did not.

### What this does NOT establish, and the precedent that says so

**It is a claim about a horizon, not about permanence.** `index_recall_after` is snapshotted when the 60 s completeness window closes. Nothing here shows the graph damage is permanent, and this project has already been burned by exactly this inference: #37 found that Qdrant graph damage which a 50 s window called permanent was **gone by 180 s**, and the claim had to be withdrawn. The supported sentence is *"at ~60 s, completeness has healed and index_recall has not."* Anything stronger is unsupported by this design.

**It is one host, one topology, five seeds, one Weaviate build**, on an undocumented internal API, at the n = 5 statistical floor.

**The mechanism is not observed.** No Weaviate-internal repair process was instrumented; only its effect on a probe.

**The instrument is asymmetric**, and that asymmetry is now weaker than the spec claimed rather than permanent — see Amendment 2c. `completeness` is a 1 s series; `index_recall` is two endpoints. A dissociation between a series and a pair of snapshots is weaker evidence than one between two series, and must never be written as though both axes were watched continuously.

### One exploratory observation, flagged as such

The censored seed is the only one that did not fully repair — it reached 30% — and it is also the one with the **smallest** graph damage: −0.200 against a mean of −0.485 across the four that fully repaired.

Less repair, less graph damage. That is the shape a mechanism would have **if the repair itself is what damages the graph, rather than the outage**. It would also explain why the damage is so much larger here (−0.5) than on Qdrant (−0.012), which has no graph-level anti-entropy to run.

This rests on **one** partially-repaired seed, was not pre-registered, and is a hypothesis for a new specification rather than a finding. It is recorded because it is the most interesting thing in the data and because filing it now, before any follow-up, is what keeps it from being reported later as though it had been predicted.

## Decision

**MERGE**, as outcome (a).

**What must not be claimed.** That the graph damage is permanent — measured at one 60 s horizon, and #37 is the standing counterexample. That repair *causes* the damage — one seed, exploratory. That this generalizes beyond one host, one build, one topology, five seeds. That both axes were watched continuously — they were not, and the writeup must carry the asymmetry in the claim itself.

**Consequences to file.**

1. `experiment/*` — **the long-quiesce re-run.** Hold the observation open for 180 s or more after repair completes and re-snapshot `index_recall`. This is the single most valuable follow-up in the project: it converts "has not healed at 60 s" into either "does not heal" or a second withdrawn claim, and #37 says the difference is real.
2. `experiment/*` — **does repair cause the damage?** Vary the fraction of the divergence set that is allowed to repair before snapshotting, and see whether graph damage tracks it. Pre-register before looking again at the seed that suggested it.
3. `method/*` — **`index_recall` may not have to be snapshot-only.** Amendment 2c showed isolation is reversible in seconds via `docker pause`, where #41 measured ~10 minutes with `docker stop` and this spec inherited that as permanent. If it holds, the Weaviate leg could report two series instead of a series and two endpoints, which is a materially stronger design.
4. `method/*` — **`characterize.write_objects()` reports success on a fully failed batch.** It checks only the batch's HTTP status, and Weaviate returns 200 with per-object `result.errors`. Observed here writing nothing while reporting success. Shared code, also used by #48 and #56.
