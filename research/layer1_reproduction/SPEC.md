# Spec: reproduce Layer 1 and commit its raw data

**Branch:** `reproduction/layer1-data`
**Date opened:** 2026-09-06
**Status:** IN PROGRESS — harness written, no runs yet

Issue: closes #53. Body copied verbatim below (per `research/AGENT_PIPELINE.md`'s implementer instructions — this is the issue text unmodified, not a paraphrase).

---

## Research question

Can the Layer-1 nano-db result be reproduced and its **raw per-seed data committed**, so the project's foundational claim stops being reported-but-not-checkable?

## Hypothesis

The published Layer-1 numbers reproduce within their stated variability on a fresh 5-seed sweep, and the resulting per-seed data can be committed at a size the repository can carry.

## Null / alternative hypothesis

(i) **They do not reproduce.** Then that is a far more important finding than anything else queued, and everything downstream — including `RELATED_WORK.md`'s novelty verdict — is affected. This possibility is the reason the issue is worth doing rather than assuming.

(ii) **They reproduce but the data is too large to commit.** Then commit the per-seed aggregates plus one full sample series, and say exactly what was omitted and why.

(iii) **The harness no longer runs.** `src/`, `cluster/` and `proto/` have not been touched since June 2026 while the research moved to Python-on-Docker; the build is exercised by CI but the *experiment path* is not. A silent bit-rot finding here is itself worth recording.

## Motivation

This is the single largest hole in the work, and it is not a conceptual one.

`research/replica_recall/` produces the numbers the top-level README's ESTABLISHED box opens with, and those numbers are what `research/RELATED_WORK.md` positions the entire contribution around. **None of the raw data behind them is in version control.** `research/replica_recall/RESULTS.md` states this itself:

> **No raw experiment output is currently committed to this repository.** … Anyone auditing this project's claims should treat those numbers as reported-but-not-independently-checkable.

Every *other* study commits its evidence — roughly 13,400 sample rows across 90+ runs. The foundational one does not. A reader auditing this project finds the strongest claim resting on the only data they cannot inspect, which is the worst possible distribution of trust.

## Experimental design

Run the **pre-existing** 5-seed protocol unchanged — baseline and chaos per seed — analysed with the committed `analyze.py` / `aggregate.py`, unmodified. Any change to the protocol makes this a new experiment rather than a reproduction and is refused here.

`run_reproduction.py` drives it inside a Linux container, because the development host is Windows/MinGW with no `g++` and no `protoc`, and the harness launches cluster binaries directly (`chaos_harness.py`, whose own docstring says "Run (Linux, after building the binaries)").

**Deviation from the issue's step 5, with the reason.** The issue proposed narrowing `.gitignore`'s `research/replica_recall/results*/` rule so evidence becomes tracked. This spec keeps that rule and commits the reproduction's output under `research/layer1_reproduction/results_sweep/` instead. Two reasons: the ignore rule exists because scratch output was committed by accident twice (#18, #24 — see `.gitignore`'s own comment), and loosening it re-opens that hazard; and a *reproduction on a different host* is not the same artifact as the original run, so filing it separately keeps the distinction visible rather than blurring the two in one directory.

## Metrics

The study's own pre-existing metrics, unchanged: `index_recall`, `completeness`, `e2e_recall`, within-shard spread, `miss@stop` vs `miss@end`, and the exact two-sided Mann-Whitney across 5 seeds per condition.

## Instrument characterization

- **The build is exercised, the experiment path is not.** CI compiles the cluster and runs `ctest` on every push, so a total build failure would already be visible. What is *not* exercised is `run_experiment.py` → `chaos_harness.py` → live binaries; outcome (iii) is live and is the first thing the run tests.
- **The container is a new variable and is treated as one.** Process-launch and I/O timing inside a container differ from bare metal — exactly the class of confound `qdrant_kill_scheduler` had to characterize when `docker start` cost a near-constant ~3.3 s of every requested kill gap. `run_reproduction.py` runs `ctest` first as a smoke test, and the writeup must present this as a reproduction on a **new host**, not a re-run under identical conditions.
- **Corpus integrity is checkable before use.** SIFT1M is ~186 MB and gitignored; `sift.py` re-downloads it. Vector count and dimensionality are verified before the sweep, because a truncated download would silently shrink the corpus and change every number.
- **Data volume is computable in advance.** Comparable committed sweeps run 54–426 sample rows per run; 10 runs at this protocol's duration should land in the low hundreds of KB — well inside what the repository already carries (~1.6 MB of results). Outcome (ii) is therefore unlikely, and if it fires the fallback is stated above.
- **Known blind spot:** the original runs' host is unknown, so "within stated variability" is judged against the published spread, not against a matched machine.

## Baselines / controls

The published numbers are the comparison. Reproduction is judged by whether the new per-seed values fall within the published spread — **not** by whether p is again 0.0079, which is the floor at n=5 and would be reached by any complete separation. `ctest` passing is the precondition; if it fails, that is outcome (iii) and the sweep does not run.

## Expected outcomes

(a) Reproduces, data committed → the biggest audit gap closes. (b) Reproduces qualitatively with different magnitudes → report both, and treat the published figures as one draw rather than the value. (c) Does not reproduce → highest-priority finding in the project; stop and investigate before any other queued work. (d) Harness bit-rot → fix or record, and state that Layer 1 was unreproducible for a period.

## Interpretation plan

Outcome (a) or (b) updates `RESULTS.md`, the README's Raw data status section, and `research/README.md`'s Layer 1 row. Outcome (c) blocks #54 and the README's ESTABLISHED box until understood, because both build on Layer 1's framing.

**This is not independent confirmation.** Same protocol, same system, same code — agreement is evidence of *reproducibility*, not a second confirmation of the effect, and must not be written up as one.

## Confounds considered

**A different machine is a different measurement**, so timing-sensitive quantities (kill spacing, replication lag, settle-window adequacy) may differ. **Docker adds a layer** the original did not have; if results differ, the container is a live hypothesis and must be named as one. **Corpus download integrity** — verified before use. **Not a fresh test of the hypothesis** — see the interpretation plan.

## Amendment 1 (2026-09-06, before the sweep): the experiment runs off the bind mount

Written after `deps`, `build` and `corpus` and **before any sweep run**, so no
result influenced it. Four defects in the harness, all found by trying to run it.

### (a) The harness could not run on the only host it exists for

`sh()` built shell command strings and quoted them with `shlex.quote()` under
`subprocess.run(shell=True)`. On Windows `shell=True` runs `cmd.exe`, which does
not treat single quotes as quoting at all. So the very first command died:

```
-v 'C:\...\Replica-Recall-Divergence':/repo   ->  invalid mode: /repo
```

Docker split on the colon in `C:` because the quotes were literal characters.
`bash -lc 'apt-get ...'` was broken the same way — bash would have received
`'apt-get` as its entire command.

This is worth stating plainly because of what this study's README argues: the
container exists *because* the dev host is Windows with no `g++` and no
`protoc`. A harness whose command construction only works under a POSIX shell
cannot run anywhere it is needed. Fixed by passing **argument lists** to
`subprocess.run`, which removes the quoting layer rather than making it
platform-conditional. Only the string handed to `bash -lc` inside the container
stays a string, where POSIX quoting is correct by definition.

### (b) The bind mount is not a filesystem the substrate's assumptions hold on

With the build running against the mounted repo, **`ctest` failed the go/no-go
gate** — 8 of 9 passing, `ClusterConfigRace` failing:

```
[ClusterConfigRace] writes=213 reads=920 read_failures=2
FAILED: save_cluster_config is not atomic with respect to concurrent
        load_cluster_config calls.
```

Under this spec's own rules that is outcome (iii): the substrate no longer
passes its tests, and the sweep does not run. **It would have been the wrong
call**, and the reason it is wrong is measurable.

Deterministic, not flaky: **10 of 10 runs failed** on the mount. Running the same
binary with its working directory on the container's own filesystem: **5 of 5
passed, with 0 read failures out of ~90,000 concurrent reads per run.** The
difference is not marginal — the on-mount runs completed 920 reads where the
off-mount runs completed 75,766–93,228 in the same second.

Measured directly, mount versus container filesystem:

| | bind mount (`/repo`) | container fs (`/tmp`) | penalty |
|---|---|---|---|
| bulk write, 200 MiB `conv=fsync` | 118 MB/s | 933 MB/s | **7.9×** |
| 300 × create + rename + unlink | 4.29 s | 0.87 s | **4.9×** |
| `ClusterConfigRace` | 10/10 fail | 5/5 pass | — |

CI runs the identical `ctest` on `ubuntu-latest` and is green, which agrees:
the substrate is fine and the mount is the variable.

So the gate failure was an **instrument artifact**, and the instrument was the
filesystem. Two consequences:

1. **Outcome (iii) is not triggered.** After the fix the gate passes **9/9** in
   the container. The substrate builds and its tests hold.
2. **The sweep must not run on the mount either**, and this is the more
   important half. Layer 1 measures recall under chaos across 180 s and 300 s
   windows, with nodes writing and renaming state throughout. A filesystem that
   is 5–8× slower and does not give rename the atomicity the code assumes would
   not merely add noise — it would change kill spacing, replication lag and
   settle-window adequacy, which are exactly the quantities the Confounds
   section already flags as machine-sensitive. A reproduction run there would
   not be a reproduction of the published protocol.

**The change.** `deps` now `rsync`s the repo from the mount to `/work` on the
container's own filesystem, excluding `build/` (CMake caches absolute paths) and
`results_sweep/`. Build, corpus and every run happen in `/work`. Results are
synced back to the mount after **each sweep cell**, not only at the end, so an
interrupted sweep does not lose completed runs. Submodule init still happens on
the mount, since that is where the git directory lives.

The resumability check moved into the container with it: it previously tested a
host path that runs no longer write to, so every resume would have silently
redone the entire sweep.

### (c) "Exactly CI's toolchain list" was not enough, and a whole failed sweep exited 0

With (a) and (b) fixed, the sweep ran — and **all ten cells produced no data**.
Every one died in `probe.ensure_stubs()`:

```
RuntimeError: grpcio and grpcio-tools are required for the replica probe.
(import failed: No module named 'grpc')
```

The apt list carried a comment that it was *"exactly CI's toolchain list
(`.github/workflows/ci.yml`), so a build that works in CI works here"*. That
reasoning is sound for the build and wrong for the sweep, structurally rather
than accidentally: **CI builds and runs `ctest`; it never runs
`run_experiment.py`.** So nothing in CI ever imports `grpc`, and CI's list has no
reason to carry the Python gRPC bindings. Inheriting it inherited that gap.

Fixed by adding `python3-grpcio` and `python3-grpc-tools` — apt rather than pip,
since 24.04 marks the system environment externally managed. Verified: stub
generation from `proto/nanodb_cluster.proto` succeeds, and a 20 s smoke run
produced 24 samples and a well-formed `samples.csv`.

**The worse half of this defect is that the stage exited 0.** Each cell's
`run_experiment.py` call is deliberately `check=False`, so that one bad cell does
not cost the others — that part is right. But the stage then reported success
after ten consecutive failures, printing `NO_OUTPUT_..._run_failed` ten times
into a log nobody had a reason to read. Had the analyse stage run, it would have
been over an empty sweep.

This is the exact silent-failure shape #26, #38, #46 and #48's 20,000-id probe
were each caught by, and it appeared here in the harness whose own README cites
that pattern as the reason `verify_corpus.py` exists. Tolerating each failure
individually is correct; reporting the aggregate as success is not. The sweep now
counts produced / skipped / failed cells, prints the tally, and exits non-zero if
any cell produced no `samples.csv`. `deps` additionally asserts `import grpc,
grpc_tools.protoc` immediately, so a missing binding fails at the start rather
than ten cells later.

### (d) A host memory kill destroyed a running cell, because the cell's life belonged to the client

The first full sweep attempt was killed part-way through cell 1 — not by the
container, but by the **host**: 7.7 GB of RAM with ~1.2 GB free. The kill landed
on the harness process, and it took the experiment with it. All nine
`nano_shard_node` / `nano_coordinator` processes died and the cell produced
nothing.

The cause is `docker exec` without `-d`: the exec'd process's lifetime is tied to
the client that started it. So any interruption on the host — an OOM killer, a
closed terminal, a dropped connection — costs whatever run was in flight. For a
sweep of ten cells at 3.5–5.5 minutes each, that is a ~50-minute job that cannot
survive a single bad minute on a machine already short of memory.

Cells now run with `docker exec -d` and signal completion by writing a marker
file as their own last step, which the harness polls. Detached, the run's
lifetime belongs to the container; the marker's presence means the cell reached
its end, which is the completion signal `-d` otherwise gives up by returning no
exit status. Combined with the existing per-cell skip, an interrupted sweep now
resumes at the cell it lost rather than at the beginning.

Headroom was measured rather than assumed before restarting: during a run the
container's memory climbs ~105 MB/min against ~2.4 GB available, so both the
180 s and 300 s cells fit. The binding constraint was host RAM, not the VM.

**This is a host limitation, and it belongs in the write-up.** The reproduction
is running on a machine with 7.7 GB of RAM and a full disk, which is a further
respect in which it is not the original host — alongside the container and the
filesystem. It does not invalidate the runs; it does mean "a different machine is
a different measurement" is now three specific differences, not a generic caveat.

### What this does not change, and what it costs

**No protocol parameter is touched.** Seeds, durations, `--dist sift`,
`--sift-vectors`, `run_experiment.py` and `aggregate.py` are all exactly as
published. What changed is *where the process's working directory points*.

The honest cost: this moves the experiment one step further from the original
host, not closer. The README already says the container is part of the
measurement; now the filesystem is too, and the write-up must say so. The
alternative was worse — running on a filesystem measurably unable to support the
substrate's concurrency assumptions, and reporting whatever came out as a
failure to reproduce.

**It also means the mount is a confound for every other containerised study on
this host**, not just this one. Filed as a consequence rather than fixed here.

## Results

*(no sweep runs yet — `deps`, `build` (9/9) and `corpus` complete)*

## Interpretation

*(to be filled)*

## Decision

*(to be filled)*
