# Spec: does `loo_agreement` detect the degraded replica on Qdrant?

**Branch:** `analysis/loo-agreement-qdrant`
**Date opened:** 2026-09-06
**Status:** IN PROGRESS — pre-registered before any Qdrant detector number was computed

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

*(to be filled after the analysis runs)*

## Interpretation

*(to be filled)*

## Decision

*(to be filled)*
