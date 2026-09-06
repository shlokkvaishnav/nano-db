# Spec: does `completeness` heal on Weaviate while `index_recall` does not?

**Branch:** `experiment/weaviate-dissociation`
**Date opened:** 2026-09-06
**Status:** IN PROGRESS — pre-registered, no runs yet; one parameter is blocked on #56

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

## Results

*(no runs yet)*

## Interpretation

*(to be filled)*

## Decision

*(to be filled)*
