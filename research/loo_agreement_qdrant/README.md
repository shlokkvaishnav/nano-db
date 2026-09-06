# The detector replicates on Qdrant — for data loss, not for graph quality

Issue #52 · branch `analysis/loo-agreement-qdrant` · **no new compute.**

`loo_agreement` and `shard_agreement` are columns in every `samples.csv` this project writes, including all the Qdrant studies. The detector had been recording its answers throughout the cross-system work — **51 runs, 4,644 sample rows** — and nobody had ever scored them.

## Result

| truth axis | baseline (control) | chaos | p | verdict |
|---|---|---|---|---|
| `e2e_recall` | 0.635 | **0.908** | **0.0023** | **detects it** |
| `index_recall` | 0.670 | **0.348** | **0.0001** | **at chance — significantly worse than baseline** |
| `completeness` | — (100% tied) | 0.929 | no control | uncontrolled |

Chance is 1/3. The unit is the per-run hit rate; two-sided Mann-Whitney by seeded permutation, since the groups run 9–20 runs and exact enumeration would be 5.2M splits.

Condition comes from each run's own `run_meta.json`, **not** from its directory name — review round 1 caught that `"chaos" in "nochaos"` had filed five no-chaos runs into the chaos group.

**Layer 3 now holds on two systems** — for end-to-end search quality. Qdrant's 0.908 is close to nano-db's 0.87, but that agreement should not be over-read: the runs differ in corpus, topology, chaos mechanism and duration.

## The negative is the important half

**Against `index_recall`, the detector scores 0.348 under chaos — the chance line is 0.333 — against a 0.670 baseline, separating at p = 0.0001 in the wrong direction.** This is not a weak null: it is a significant, sizeable collapse *to chance* in exactly the condition where the detector is meant to work.

That is a limit on this project's own headline. The established Qdrant finding is **replica-level `index_recall` divergence** (p = 0.0079, cluster mean a null at p = 0.31). The ground-truth-free detector, run over those same runs, does **not** identify the replica that finding is about.

A plausible mechanism, offered as hypothesis and not conclusion: `loo_agreement` measures how far one replica's *returned results* diverge from its peers'. A replica missing objects returns visibly different results and is easy to flag. A replica holding all the data with a slightly worse graph returns **nearly the same results** — which is exactly what "approximation converts damage into silence" means. The detector is subject to the very effect the project exists to study.

## Why the baseline row is not decoration

At 0.635 against a 0.333 chance line, the healthy-cluster control scores well above chance **with nothing to detect**. This is the artifact `_detection_stats()`'s own docstring warns about: near-ties that survive the exclusion rule break toward the same replica for both the detector and the truth.

Reading the chaos number against *chance* rather than against *baseline* would overstate the detector badly. Against baseline it is smaller and still clearly there.

It also explains why the `index_recall` baseline scores *higher* than chaos: with no competing signal, near-ties break the same way for detector and truth, while under chaos the detector is actively pointed at the data-poor replica rather than the graph-degraded one.

`completeness` cannot be tested at all: every baseline run is **100% tie-excluded**, because on a healthy cluster all replicas hold every object and no "worst" replica exists. A second caveat on that column, from review: `resolution_eps()` derives its tie threshold as `0.5/(k*queries)` from the way *recall@k* quantises, which is right for `e2e_recall` and `index_recall` and is **not derived from anything** for `completeness`, a fraction of ids. That axis is uncontrolled and its threshold is un-derived, so it is reported and not used. The 0.929 chaos figure has no control and must not be quoted as though it did — nor is it independent of the `e2e_recall` result, since a replica missing objects scores worse on both.

## What makes this admissible as a post-hoc analysis

The metric was not chosen after seeing the data. `_detection_stats()` and `resolution_eps()` live in `replica_recall/analyze.py`, were written for nano-db **before any Qdrant run existed**, and are *imported unmodified* here rather than reimplemented. The tie threshold is derived from each run's own `k` and `queries`, not tuned. `SPEC.md` was committed before any Qdrant detection number was computed, and fixed the rule that every run carrying the columns is scored with no filtering by outcome.

To score the other two truth axes without a second copy of the metric, rows are rewritten so the chosen axis occupies the `e2e_recall` field. One implementation, three axes.

## Reproducing

```bash
python research/loo_agreement_qdrant/score_qdrant_detector.py
```

No cluster, no Docker, no network — it reads only committed CSVs. Takes about a minute, most of it in the permutation tests. Output is committed at `results/analysis_output.txt`, per-run records at `results/detector_scores.json`.

## What must not be claimed

That the detector was "validated on Qdrant" without naming the axis. That 0.908 vs nano-db's 0.87 is agreement *between systems*. That the `completeness` 0.929 is controlled. And **not** that the `index_recall` result proves the detector *cannot ever* work on graph quality — it is 9 chaos runs, one system, against an effect living in ~1% of headroom. What is shown is that **this** statistic, on **this** data, is at chance.

Full pre-registration, results and decision: [`SPEC.md`](SPEC.md).
