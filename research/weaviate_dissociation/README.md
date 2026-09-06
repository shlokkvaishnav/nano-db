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

## One parameter is blocked on #56

The observation window must be timed from the right origin, and whether Weaviate's repair clock is anchored to the write or the restart is exactly what #56 is measuring. Until it reports, the window is **≥60 s from the later of (write, restart)** — safe under either answer, at the cost of a longer run. When #56 lands, this gets tightened and the amendment dated.

## Running it

```bash
python research/weaviate_dissociation/dissociation.py --dry-run
python research/weaviate_dissociation/dissociation.py
```

It **refuses to run** unless `verify_class()` reports factor 3 / 1 shard — #46's silent-topology hazard, where a stale auto-schema class would give per-replica numbers from a topology with no replicas.

Full pre-registration, outcomes and confounds: [`SPEC.md`](SPEC.md).
