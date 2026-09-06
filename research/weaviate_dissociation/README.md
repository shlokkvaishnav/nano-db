# The Weaviate dissociation experiment

Issue #54 · branch `experiment/weaviate-dissociation` · **pre-registered, no runs yet.**

The experiment the four Weaviate method studies (#41, #43, #46, #48) were built to make runnable, and the only one that tests this project's **field-level** claim rather than a within-system one.

## The prediction

`RELATED_WORK.md` §4 argues structurally: every production anti-entropy mechanism works on **exact object identity**, and two correct HNSW graphs over identical data differ bit-for-bit — so object-level repair *cannot* be pointed at the index. Therefore, after chaos: **the missing objects come back and the graph quality does not.**

Neither system tested so far can decide it. nano-db has no anti-entropy by design, so its 0% healing is near-definitional (which is why #55 demoted that result). Qdrant heals both axes at 180s. Weaviate is the only one of the three where object-level repair exists and can be watched separately from graph quality.

## The instrument is asymmetric, permanently

| axis | how it is measured | cost |
|---|---|---|
| `completeness` | sampled every **1 s**, peers up, non-perturbing | cheap (#43, #46) |
| `index_recall` | **two snapshots**, via isolation probing | ~10 min of node health *each* (#41) |

`_search` returns 415 on the cluster-internal port across 12 content types (#43), so this compares **a time series against a pair of endpoints**. Any writeup must carry that in the claim.

## Ground truth needs no server support

The corpus comes from a seeded RNG, so the exact top-k over *the subset a replica actually holds* — which `objects_present_ids` reports — is computable locally with numpy. `index_recall` is the replica's ANN answer against exact search over its own data, holding data constant: the same definition used on nano-db and Qdrant.

## Two bugs caught before running

**`feasibility_check.search()` returns `(ok, [vid])`** where `vid` is the last 8 characters of the uuid, set as a property at write time. Comparing those against full-uuid ground truth would have scored ~0 and **looked like a finding** rather than a key mismatch. `search_full_ids()` asks for `_additional { id }` instead.

**The base and divergence id sets must be disjoint.** Re-writing the same ids while the victim is down leaves it missing only the *updates*, and a presence probe cannot see versions — so the victim would read as complete and the primary metric would report a false negative. Verified disjoint.

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
