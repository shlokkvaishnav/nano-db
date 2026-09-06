# What makes Weaviate's async repair measurable?

Issue #48 · branch `method/weaviate-repair-window` · the last instrument prerequisite before the dissociation experiment.

**Answer: it already is — and the follow-on this study was meant to scope should not be built.**

PR #44 measured repair at **~0.3 s** once and named "sub-second sampling" as the remaining prerequisite. Three repetitions at that same divergence size give **44.7 s, 0.008 s, 0.010 s**. The 0.3 s was one draw from a **bimodal** distribution.

| divergence | repair (s) |
|---|---|
| 50 | 44.725, 0.008, 0.010 |
| 500 | 0.169, 50.234, 50.047 |
| 5,000 | 38.286, 40.787 |

Every observation is either **sub-0.2 s** or **38–50 s**, and a **100× increase in divergence does not increase repair time** — the signature of a fixed periodic sync where the wait is time-until-the-next-tick. (Consistent with, not established: the scheduler was not observed, only its effect.)

**Consequences.**

- **Sub-second sampling is not needed.** A 40–50 s window holds 40–50 samples at 1 s cadence, 8–10 at 5 s — the cadence this project already uses. The probe floor is ~50 ms, two orders of magnitude below what's required.
- **The healing signal is a step, not a decay curve.** The experiment should ask whether the step happens for `completeness` and does not for `index_recall` in the same window — a more robust claim than comparing curve shapes, and insensitive to where in the sync cycle a run starts, provided each run is observed for **≥60 s**.
- **A hard probe bound, found by hitting it.** `ids` is base64 in the URL: the probe works to **15,000 ids (800 KB)** and **fails at 20,000 (1.07 MB)**. It failed silently — the run reported `first_answer None` — the same shape as #26 and #38, so it is recorded rather than filed as an anomaly.

Full numbers, the control that separates repair from restart latency, and what is not established: [`SPEC.md`](SPEC.md).

## Reproducing

```bash
python research/weaviate_repair_window/characterize.py --sizes 50,500,5000 --reps 3
```

Digest-pinned Weaviate 1.29.0, 3 nodes, factor 3, 1 shard, one host. n = 2–3 per cell; the sync interval itself was not measured, only the waits it produces.
