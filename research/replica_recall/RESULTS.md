# Results

**Partly superseded 2026-09-06 by [`../layer1_reproduction/`](../layer1_reproduction/) (#53).** A fresh 5-seed sweep of this exact protocol, unmodified, now commits 1,632 sample rows under `research/layer1_reproduction/results_sweep/`, and it reproduces: `index_recall` 0.9973 → 0.9709, `completeness` 1.0000 → 0.9580, `e2e_recall` 0.9994 → 0.9581, within-shard spread 0.0000 → 0.0534, each at p = 0.0079.

Read as **reproducibility, not confirmation** — same code, new host (a container, on the container's own filesystem, on a memory-constrained machine). And note what it could *not* check: the published magnitudes were never recorded, so only the claim as written could be tested. The **quiesce protocol was not re-run**, so the non-recovery claim below is still uncommitted.

The statement below remains true of *this directory*, which is why it is kept rather than deleted — the reproduction was deliberately filed separately, since a run on a different host is a different artifact.

---

**No raw experiment output is currently committed to this repository.**

The measurement harness (`run_experiment.py`, `sweep.py`, `forensics_experiment.py`) requires Linux with the cluster binaries built — it launches processes directly, no Docker. It has not been (re-)run in every environment this project has been developed in, so there is no `samples.csv`, sweep directory, or forensics output sitting here to inspect.

This is a documented gap, not a silent one: the numbers quoted in the top-level `README.md` and in `../README.md`'s ESTABLISHED box come from prior runs whose raw per-seed data is not currently in version control. Anyone auditing this project's claims should treat those numbers as reported-but-not-independently-checkable until this directory is populated.

## Regenerating

```bash
pip install grpcio grpcio-tools numpy
cmake -B build -DCMAKE_BUILD_TYPE=Release -DNANODB_BUILD_CLUSTER=ON
cmake --build build -j$(nproc)

python research/replica_recall/sift.py --vectors 200000     # pre-warm the corpus

python research/replica_recall/sweep.py --seeds 5 --out-dir results_sweep_sift --dist sift
python research/replica_recall/aggregate.py --sweep-dir research/replica_recall/results_sweep_sift
```

For the healing/quiesce protocol, add `--with-quiesce` to the sweep. Full detail on every flag and what it controls: [`../README.md`](../README.md).

Once regenerated, commit the sweep directory's `samples.csv` and `aggregate.py`'s summary output as a new file alongside this one (the generated `results*/` directories themselves stay gitignored, per `.gitignore` — this file lives next to them, not inside one, specifically so it isn't swept up by that pattern) so the headline numbers are independently checkable rather than only reviewable as a pipeline.
