# Spec: does `loo_agreement` detect the degraded replica on Qdrant?

**Branch:** `analysis/loo-agreement-qdrant`
**Date opened:** 2026-09-06
**Status:** COMPLETE — outcome (a) on end-to-end quality, outcome (iii) on graph quality

Issue: closes #52. Body copied verbatim below (per `research/AGENT_PIPELINE.md`'s implementer instructions — this is the issue text unmodified, not a paraphrase).

---

## Research question

Does the ground-truth-free detector `loo_agreement` identify the degraded replica **on Qdrant**, at the same above-chance rate it reaches on nano-db?

## Hypothesis

`loo_agreement` identifies the worst replica within a shard-round on Qdrant at a rate above the 1/3 chance line, at a rate not obviously different from nano-db's 0.87.

## Null / alternative hypothesis

(i) **At chance on Qdrant.** The detector's nano-db performance depends on something nano-db-specific — its replication path, its probe, or the fact that the same codebase computes both the signal and the truth. Then Layer 3 is a single-system result and must be stated that way permanently.

(ii) **Not scoreable.** Too few rounds may have a *non-tied* worst replica. `_detection_stats()` excludes tied groups by design, and Qdrant's `index_recall` has only ~1% of headroom, so ties may dominate. Then the answer is "this data cannot decide it" and the reason is recorded.

(iii) **Above chance, but for the wrong reason** — tracking `completeness` (measured directly by the probe, trivially detectable) rather than graph quality. Distinguishable by scoring each truth axis separately.

## Motivation

Layer 3 — production detection without ground truth — is the project's only layer with a practical claim attached, and it is currently scoped to **one system**.

That scoping may be unnecessary. `loo_agreement` and `shard_agreement` are columns in *every* `samples.csv` this project has written, including all the Qdrant studies. A scan of the committed tree finds **51 Qdrant / cross-system runs carrying 3,594 populated `loo_agreement` rows, none zero**. The detector has been recording its answers throughout the Qdrant work and nobody has scored them.

Everything needed is committed: the detector signal per replica per round, the ground-truth columns in the same row, and `events.json` naming the killed node. **Zero compute.**

## Experimental design

No new runs. Analysis only, over committed data.

1. **Reuse the existing metric; do not invent one.** `research/replica_recall/analyze.py::_detection_stats()` defines the scoring: group by `(t_rel, shard)`, require ≥3 reachable replicas, exclude groups whose top-two truth values are within `resolution_eps()` (derived from the run's own `k` and `queries`, not tuned), then ask whether `argmin(loo_agreement) == argmin(truth)`. **Import it.** Reimplementing is how a metric quietly changes.
2. Score every committed Qdrant / cross-system run, per condition (baseline / chaos / quiesce).
3. Score against all three truth axes separately — `e2e_recall`, `index_recall`, `completeness` — because outcome (iii) is invisible if they are pooled.
4. **Report the tie-exclusion rate per condition**, prominently.
5. Compare to nano-db's 0.87 / 0.86 / 0.81 as context, not as a between-system significance test.

## Metrics

Primary: **hit rate** against the **1/3 chance line**, per condition, per truth axis. Secondary: tie-exclusion rate; mean rank correlation between detector ordering and truth ordering; number of scored groups.

**The unit of analysis is the run, not the round.** Rounds within a run are serially correlated, exactly as `aggregate.py` already treats seeds.

## Instrument characterization

Required by `SPEC_TEMPLATE.md:43`, and the properties that decide whether this is answerable were computable in advance and were computed before implementation:

- **The signal exists in the data.** 51 runs, 3,594 populated `loo_agreement` rows, none zero. Verified by scanning committed CSVs before this spec was written.
- **The scoring code already exists and is pre-specified.** `_detection_stats()` and `resolution_eps()` were written for nano-db before any Qdrant run existed, so the metric cannot be tuned to this data. This is the strongest form of pre-registration available without new compute, and it is the reason this study is worth doing as an analysis rather than a fresh experiment.
- **The known failure mode is ties, and it is quantified.** `_detection_stats()`'s docstring records that on realistic data a healthy baseline scored 0.6409 against a 0.333 chance line *purely from ties*. Qdrant's `index_recall` sits in ~1% of headroom, so the tie rate is expected to be worse here than on nano-db, and it is reported as a first-class result rather than a footnote.
- **`_detection_stats()` returns a NaN-filled record when fewer than 5 groups survive**, so under-powered runs are distinguishable from failed ones and must not be averaged in.

## Baselines / controls

The **baseline (no-chaos) runs are the control and are load-bearing.** A chaos hit rate is only meaningful beside the baseline hit rate on the same data with the same tie handling. If baseline and chaos score alike, the detector is detecting nothing.

## Expected outcomes

(a) Above chance on chaos and near chance on baseline, on ≥1 truth axis → Layer 3 replicates on a second system at zero compute. (b) At chance on both → outcome (i). (c) Too few scoreable groups → outcome (ii), recorded with the tie rate. (d) Above chance only against `completeness` → outcome (iii): a data-loss detector, not a graph-quality detector — narrower, still useful, a different claim.

## Interpretation plan

This is a **replication of a pre-specified metric on independent data**, not a new estimator, and is written up as such. Any outcome updates the README's HYPOTHESIS box and `research/README.md`'s Layer 3 row. Outcome (b) or (c) is reported as prominently as (a); a null is genuinely informative because the detector is the project's only practical claim.

## Confounds considered

**Ties** may remove most groups; the rate is reported, not just the surviving hit rate. **Different probes** — Qdrant's `loo_agreement` is the same *definition* over a different code path, so the writeup must not imply one implementation was validated twice. **Post-hoc selection** — runs were collected for other questions; every run carrying the columns is scored, with no filtering by outcome, and that rule is fixed here before looking. **Non-independence** — the run is the unit, not the round. **Chaos runs contain healthy periods** — a chaos run's rounds before the first kill and after recovery have nothing to detect, which depresses the chaos hit rate toward baseline and makes any positive result conservative.

## Results

51 committed runs, 4,644 sample rows, no new compute. Hit rate is the per-run mean over scoreable runs; chance is 1/3 throughout.

**Condition is read from each run's own `run_meta.json` (`chaos`, `quiesce`), not from its directory name** — review round 1 found that a substring match filed five `..._nochaos_...` runs into the chaos group, because `"chaos" in "nochaos"`. All numbers below are post-fix.

| truth axis | condition | runs | scoreable | hit rate | groups | tie-excluded |
|---|---|---|---|---|---|---|
| `e2e_recall` | baseline | 22 | 16 | **0.635** | 229 | 57.3% |
| | chaos | 10 | 9 | **0.908** | 80 | 25.0% |
| | quiesce | 19 | 13 | 0.620 | 214 | 58.0% |
| `index_recall` | baseline | 22 | 20 | **0.670** | 323 | 40.4% |
| | chaos | 10 | 9 | **0.348** | 84 | 21.4% |
| | quiesce | 19 | 16 | 0.594 | 247 | 53.9% |
| `completeness` | baseline | 22 | **0** | — | 0 | **100.0%** |
| | chaos | 10 | 8 | **0.929** | 65 | 35.7% |
| | quiesce | 19 | 3 | 0.725 | 25 | 88.1% |

The pre-registered control comparison, per-run hit rate as the unit:

| axis | baseline | chaos | U | p | |
|---|---|---|---|---|---|
| `e2e_recall` | 0.635 (n=16) | **0.908** (n=9) | 123.0 | **0.0023** | separates |
| `index_recall` | 0.670 (n=20) | **0.348** (n=9) | 13.0 | **0.0001** | separates, **wrong direction** |
| `completeness` | — (n=0) | 0.929 (n=8) | — | — | no control available |

**A note on the test.** This project's other studies enumerate every split exactly, because at 5v5 there are 252. Here the groups are 9–20 runs and C(25,12) is 5.2 million, so a seeded permutation test (200,000 resamples, seed 20260906) is used and the output labels which method produced each p. Nothing is claimed below p = 5e-6, the resolution floor.

## Interpretation

**The detector replicates on a second system for end-to-end search quality — outcome (a).** Chaos runs score 0.908 against a 0.635 no-chaos control, p = 0.0023.

**The baseline is doing exactly the work it was pre-registered to do.** At 0.635 against a 0.333 chance line it confirms the artifact `_detection_stats()`'s docstring warns about: a healthy cluster scores well above chance purely from near-ties surviving the exclusion rule. Reading the chaos number against *chance* rather than against *baseline* would overstate the detector by a wide margin.

**On graph quality the detector collapses to chance — outcome (iii), and this is the more important result.** Against `index_recall`, chaos runs score **0.348** — the chance line is 0.333 — against a 0.670 baseline, and the comparison separates at **p = 0.0001 in the wrong direction**. This is not a weak null. It is a significant, sizeable *drop to chance* in exactly the condition where the detector is supposed to work.

**Why that matters here specifically.** The established Qdrant finding is replica-level `index_recall` divergence (p = 0.0079, cluster mean a null at p = 0.31). The ground-truth-free detector, applied to those same runs, is **at chance** at identifying the replica that finding is about. Layer 3 does not cover Layer 1's strongest result.

The mechanism is worth stating as a hypothesis, not a conclusion: `loo_agreement` measures how far a replica's *returned results* diverge from its peers'. A replica missing objects returns visibly different results and is easy to flag — hence 0.908 on `e2e_recall` and 0.929 on `completeness`. A replica holding all the data with a slightly worse graph returns **nearly the same results**, which is precisely what "approximation converts damage into silence" means. **The detector is subject to the very effect this project exists to study**, and that is a sharper statement of the problem than the project has had so far.

An honest note on why the baseline is *higher* than chaos on this axis: under chaos, the replica the detector flags (the data-poor one) and the replica with the worst `index_recall` are frequently different, so the detector is actively pointed at the wrong replica rather than merely uninformed. Under no chaos there is no such competing signal and near-ties break together, producing the inflated 0.670.

**A limitation of the tie threshold on one axis, found in review.** `resolution_eps()` returns `0.5/(k*queries)`, derived from how mean recall@k over nq queries quantises. That is correct for `e2e_recall` and `index_recall`. `completeness` is a fraction of ids, quantised by the id count, so the same epsilon is *not* derived for it — the SPEC's claim that the threshold comes from the run's own parameters holds for two axes of three. Since the completeness column is already uncontrolled and unused for any comparison, it is reported with this stated rather than re-derived.

**`completeness` cannot be tested, and the reason is informative.** Every baseline run is 100% tie-excluded: on a healthy cluster all replicas hold every object, so no "worst" replica exists. The chaos figure of 0.929 is therefore **uncontrolled** and must not be quoted as though it had a baseline. It is also not independent of `e2e_recall` — a replica missing objects scores worse on both.

## Decision

**MERGE.** Layer 3 moves from one system to two, with a hard boundary: **`loo_agreement` detects data-loss-driven quality divergence on Qdrant, and is at chance on graph-quality divergence.**

Consequences to file:
1. `analysis/*` — the top-level README's HYPOTHESIS box says the detector "still rests on one system and one implementation." Now wrong in one direction and incomplete in another: it replicates for `e2e_recall`, and it is *at chance* for `index_recall`. The DO-NOT-CLAIM list needs a line forbidding "`loo_agreement` detects the degraded replica" without naming the axis.
2. `experiment/*` — whether a *different* ground-truth-free statistic can see graph-quality divergence is now the project's most consequential open question, because the current one demonstrably cannot. `shard_agreement` is recorded in every run and was not scored here.
3. `method/*` — a detector study designed from scratch would choose a truth axis with more headroom than `index_recall`'s ~1%, or a larger `k`.

**What must not be claimed.** That the detector was "validated on Qdrant" without naming the axis. That 0.908 vs nano-db's 0.87 is agreement between systems — the runs differ in corpus, topology, chaos mechanism and duration. That the `completeness` 0.929 is controlled. That `index_recall` at 0.348 proves the detector *cannot ever* work on graph quality — it is 9 chaos runs, on one system, against an effect living in ~1% of headroom; what is shown is that **this** statistic, on **this** data, is at chance.
