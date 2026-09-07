# The Weaviate dissociation experiment

Issue #54 · branch `experiment/weaviate-dissociation` · **Outcome (a): the dissociation holds, at a 60 s horizon.**

## The result

| seed | `index_recall` before → after | `completeness` end | censored | recovery |
|---|---|---|---|---|
| 20260900 | 0.980 → 0.520 | **1.00** | none | 38.19 s |
| 20260901 | 0.975 → 0.460 | **1.00** | none | 38.21 s |
| 20260902 | 0.965 → 0.460 | **1.00** | none | 36.79 s |
| 20260903 | 0.975 → 0.515 | **1.00** | none | 40.94 s |
| 20260904 | 0.990 → 0.790 | 0.30 | **right** | — |

**The control drifts by exactly 0.0000** across all five no-chaos seeds — same two snapshots, same interval, no chaos. That is what makes the drop attributable to the chaos rather than to the measurement.

`index_recall` after: baseline **0.9780 ± 0.0091** vs chaos **0.5490 ± 0.1378**, disjoint, **p = 0.0079** (the floor at 5 v 5 — complete separation, not a large effect; the effect size is the ~**−0.5** delta on a 0–1 metric).

**The pre-registered dissociation holds in 4 of 5 seeds.** The fifth is **right-censored**, not negative: its completeness series had reached 30% when the window closed, so it says *unknown*, not *did not heal*. Amendment 1 is what makes that distinction available.

In those four seeds the victim ends the window holding **every** object it missed — exact id-set equality — while its graph quality sits at about half its own pre-chaos value. Same replica, same window, same instrument. The data came back; the graph did not.

**The window was set from another study's prediction and it held.** Realized divergence age was 0.277–0.329 s, placing every run in #56's *young* regime as Amendment 1 predicted by construction; observed recovery was 36.79–40.94 s against #56's ~32 s prediction.

## What this is not

**It is a claim about a 60 s horizon, not about permanence.** `index_recall` is snapshotted when the completeness window closes. #37 is the standing precedent: Qdrant graph damage that a 50 s window called permanent was **gone by 180 s**, and that claim had to be withdrawn. The long-quiesce re-run is the most valuable follow-up in the project.

One host, one topology, one build, five seeds, an undocumented internal API, and the n = 5 floor. The mechanism is not observed — only its effect on a probe.

## The most interesting thing in the data, and it is exploratory

The censored seed is the only one that did not fully repair (30%) and also the one with the **smallest** graph damage: −0.200 against a mean of −0.485 across the four that fully repaired.

Less repair, less damage. That is the shape a mechanism would have **if the repair itself damages the graph rather than the outage** — which would also explain why the damage here (−0.5) dwarfs Qdrant's (−0.012), a system with no graph-level anti-entropy to run. It rests on **one** partially-repaired seed, was not pre-registered, and is filed as a hypothesis for a new spec rather than a finding.

The experiment the four Weaviate method studies (#41, #43, #46, #48) were built to make runnable, and the only one that tests this project's **field-level** claim rather than a within-system one.

## The prediction

`RELATED_WORK.md` §4 argues structurally: every production anti-entropy mechanism works on **exact object identity**, and two correct HNSW graphs over identical data differ bit-for-bit — so object-level repair *cannot* be pointed at the index. Therefore, after chaos: **the missing objects come back and the graph quality does not.**

Neither system tested so far can decide it. nano-db has no anti-entropy by design, so its 0% healing is near-definitional (which is why #55 demoted that result). Qdrant heals both axes at 180s. Weaviate is the only one of the three where object-level repair exists and can be watched separately from graph quality.

## The instrument is asymmetric — but "permanently" no longer holds (Amendment 2c)

| axis | how it is measured | cost |
|---|---|---|
| `completeness` | sampled every **1 s**, peers up, non-perturbing | cheap (#43, #46) |
| `index_recall` | **two snapshots**, via isolation probing | ~10 min of node health *each* (#41) |

`_search` returns 415 on the cluster-internal port across 12 content types (#43), so this compares **a time series against a pair of endpoints**. Any writeup must carry that in the claim.

**The ~10 min figure came from `docker stop` and no longer applies.** Amendment 2c switched isolation to `docker pause`: the victim is just as isolated, nothing terminates, and unpausing restores identical Raft membership in seconds. So the *cost* justification for `index_recall` being snapshot-only is materially weaker than this table says. Not acted on here — making it a series is a different experiment needing its own pre-registration — but the writeup must not repeat "permanent" unqualified.

## Ground truth needs no server support

The corpus comes from a seeded RNG, so the exact top-k over *the subset a replica actually holds* — which `objects_present_ids` reports — is computable locally with numpy. `index_recall` is the replica's ANN answer against exact search over its own data, holding data constant: the same definition used on nano-db and Qdrant.

## Two bugs caught before running

**`feasibility_check.search()` returns `(ok, [vid])`** where `vid` is the last 8 characters of the uuid, set as a property at write time. Comparing those against full-uuid ground truth would have scored ~0 and **looked like a finding** rather than a key mismatch. `search_full_ids()` asks for `_additional { id }` instead.

**The base and divergence id sets must be disjoint.** Re-writing the same ids while the victim is down leaves it missing only the *updates*, and a presence probe cannot see versions — so the victim would read as complete and the primary metric would report a false negative. Verified disjoint.

## The graph axis was dead, and nothing would have said so (Amendment 2)

The first attempt at the sweep was **stopped 12 minutes in and its data discarded.** Its very first number was impossible: the no-chaos control — an undisturbed replica, nothing killed — reported `index_recall = 0.23`. A healthy replica scored against exact search over its own data must sit near 1.0. That is an instrument returning noise, not a fact about Weaviate.

Two defects, both fatal to the graph axis:

- **Ground truth used the wrong metric.** `exact_topk()` ranked by L2 while the class indexes with `distance: cosine`. `index_recall` was comparing Weaviate's cosine-nearest-10 against exact L2-nearest-10 — two different neighbour sets. It now reads the metric from the live schema and refuses to run on one it cannot reproduce.
- **The class was shared scratch that had never been cleared.** `RrdVector` is written by #41, #43, #46, #48 and #56, and held **14,200 objects** against a per-run corpus of 5,000. The ANN drew its nearest ten from the superset while ground truth drew from the subset. The corpus is now cleared between runs — by **deleting the objects, not the class** (Amendment 2b): recreating the class mints a new shard and re-places replicas, which twice cost the cluster its Raft quorum and forced a rebuild.

Verified on the live cluster before restarting:

| condition | `index_recall`, healthy replica |
|---|---|
| as written (14,200-object class, L2 truth) | **0.23** |
| class reset, still L2 truth | 0.32 |
| class reset + cosine truth | **0.9750** |

**The third defect is the one worth carrying elsewhere: there was no positive control.** The spec required a no-divergence control and got one — for `completeness`. Nothing required the *graph* axis to be shown reading ~1.0 when nothing is wrong. So every pre-flight check passed on the axis that already worked, while the axis carrying the hypothesis measured nothing.

That matters more than usual here, because the claim under test is a **dissociation** — one axis moving while the other does not. A silently dead `index_recall` would have produced exactly that result for free, and it would have confirmed the pre-registered hypothesis. The harness now aborts if the no-chaos control scores below 0.90.

## The window parameter — unblocked by #56 (Amendment 1, 2026-09-06)

It was blocked: the window has to be timed from the right origin, and whether Weaviate's repair clock is anchored to the write or the restart was exactly what #56 was measuring. It has reported, and the answer is **neither, cleanly** — within a regime the clock runs from the **restart**, and divergence *age* selects which regime you are in (below ~15 s the victim waits ~32 s; above ~30 s it reconciles in ~2 s).

Applied here, before any run:

- **Origin: the restart**, no longer `max(write, restart)`. Under the old rule a slow write — and #56 saw writes at `consistency_level=ONE` take minutes with a node down — pushes the origin past the restart, so the window opens *after* repair has already fired. That is the failure that voided #24 and #9.
- **A `t = 0` completion is censored, not instant.** #56's own probe-perturbation check was voided by this: it reported `repair_s = 0.000` with the victim already holding 50/50 ids on its *first* sample. At a 1 s cadence — which this experiment plans — "repaired instantly" and "the first sample landed after repair" are indistinguishable. Recovery is therefore recorded as left-censored (complete on first look), right-censored (never complete in the window), or an actual time. Censored runs are kept, and a left-censored run still answers the primary question — completeness *did* return within the window; only *when* is bounded rather than measured.
- **≥60 s stands, with its real margin stated.** The expected young-regime wait is ~32 s, but the slowest repair seen anywhere in this project is 53.3 s — 6.7 s of margin, not the comfortable outlasting the earlier wording implied. Right-censoring is what keeps that honest.
- **Realized divergence age is recorded per run.** It should be ≈ 0 s here (the victim restarts as soon as the divergence write returns), placing every run in the young regime. Recorded rather than assumed: pooling across the regime step is precisely what gave #56 its wrong aggregate answer.

Full reasoning, including why this is not void if PR #59 changes under review: [`SPEC.md`](SPEC.md) § Amendment 1.

## Running it

```bash
python research/weaviate_dissociation/dissociation.py --dry-run
python research/weaviate_dissociation/dissociation.py
```

It **refuses to run** unless `verify_class()` reports factor 3 / 1 shard — #46's silent-topology hazard, where a stale auto-schema class would give per-replica numbers from a topology with no replicas.

Full pre-registration, outcomes and confounds: [`SPEC.md`](SPEC.md).
