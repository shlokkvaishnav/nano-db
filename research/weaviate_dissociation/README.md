# The Weaviate dissociation experiment

Issue #54 · branch `experiment/weaviate-dissociation` · **Outcome (i): both axes heal. The dissociation does NOT hold.**

## The result

| seed | control drift | chaos delta | `completeness` | censored | recovery |
|---|---|---|---|---|---|
| 20260900 | −0.040 | **−0.020** | 0.60 | right | — |
| 20260901 | −0.035 | **−0.035** | **1.00** | none | 56.95 s |
| 20260902 | −0.035 | **−0.035** | **1.00** | none | 60.55 s |
| 20260903 | −0.065 | **−0.050** | **1.00** | none | 40.08 s |
| 20260904 | −0.050 | **−0.020** | 0.17 | right | — |

**In no seed does chaos cost more `index_recall` than the corpus-matched control.** The dissociation is met in **0 of 5** seeds.

The comparison has to be **paired**, and only three seeds qualify: the two censored seeds ended with fewer objects (9,500 and 6,748 vs 10,000), and a smaller corpus scores *higher* recall — the very effect the control exists to neutralise, so including them biases the test toward the conclusion. Paired differences on the three matched seeds: **0.000, 0.000, +0.015** (positive = chaos lost *less*). **Two of three are identical.**

In the three seeds where repair completed, the chaos arm's `index_recall` landed on its control's value — twice to the third decimal. Same corpus, same size: the graph is as good after chaos-plus-repair as after the same writes with no chaos.

The residual −0.05 is **corpus size, not damage**: doubling the index 5,000 → 10,000 costs that much with no chaos anywhere.

## Why the first answer was wrong

The first sweep reported `index_recall` collapsing 0.975 → 0.49 and called it outcome (a). It was **dilution**: ground truth covered 5,000 objects while the replica answered from 10,000 after repair. Verified on one replica at one moment — **0.495 → 0.935** once ground truth covered the full corpus. A parameter-free model predicted all five original values to within 0.027.

That sweep's control drifted **0.0000**, which looked like strong evidence and was actually the tell: it was not corpus-matched, so it could not see the effect that produced the entire result.

## What this does not say

Two of five seeds are **right-censored** (repair at 60% and 17% when the window closed), and recovery now runs 40.08–60.55 s, so **the 60 s window is marginal** — the negative rests on **three** matched seeds, not five. A **60 s horizon**, one host, one build.

The precision is exact rather than hand-waved: 20 queries × top-10 = 200 truth items, so `index_recall` moves in steps of **0.005**. Any real chaos-specific deficit is **under ~0.02**; anything smaller is below the quantisation floor and is *not* excluded. More queries per snapshot lowers that floor at no cluster cost.

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
