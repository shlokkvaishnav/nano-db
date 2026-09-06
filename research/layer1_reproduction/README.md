# Reproducing Layer 1, and committing its data

Issue #53 · branch `reproduction/layer1-data` · **harness written, no runs yet.**

## Why this exists

`research/replica_recall/` produces the numbers the top-level README opens with, and that `RELATED_WORK.md` positions the whole contribution around. **None of its raw data is in version control.** Its own `RESULTS.md` says so:

> **No raw experiment output is currently committed to this repository.** … Anyone auditing this project's claims should treat those numbers as reported-but-not-independently-checkable.

Every other study commits its evidence — ~13,400 sample rows across 90+ runs. The foundational one does not, so the strongest claim rests on the only data a reader cannot inspect. That is the worst possible distribution of trust, and it is fixable by re-running.

## Why a container

The harness launches `build/nano_shard_node` and `build/nano_coordinator` directly, and its docstring says *"Run (Linux, after building the binaries)"*. This development host is Windows/MinGW with **no `g++` and no `protoc`**, so the experiment cannot run here at all. Docker is the only Linux available — which makes the container **part of the measurement**, not an implementation detail. Any writeup must say so.

## What it does not do

Change the protocol. It runs the committed `run_experiment.py` at the published parameters (`--dist sift`, 180 s baseline / 300 s chaos, default 200,000 vectors) and analyses with the committed `aggregate.py`, unmodified. A reproduction that alters the thing being reproduced is a new experiment.

## Running it

```bash
python research/layer1_reproduction/run_reproduction.py --dry-run   # print every command
python research/layer1_reproduction/run_reproduction.py --stage all
```

Stages are resumable and skip completed work: `deps` → `build` → `corpus` → `sweep` → `analyse`. The build is ~10 minutes and the corpus ~186 MB, so a failure in the sweep should not force either to be redone. **`ctest` is the go/no-go gate** — if the substrate no longer builds, that is outcome (iii) and the sweep does not run.

## The corpus check, and the bug it already caught

`verify_corpus.py` runs before the sweep, because a truncated download silently shrinks the corpus and changes every number downstream — the same silent-failure shape as #26, #38, #46 and #48's 20,000-id probe.

Its first version asserted **exactly 1,000,000 base vectors** and failed a perfectly good corpus at 350,000. `sift.py` fetches a **prefix on purpose**, with an HTTP `Range` header, because the run needs only `--sift-vectors` of them. The check is therefore *well-formed and long enough for the run about to use it* — not *complete*. Asserting the stronger property would have blocked valid data, which is the same mistake as a checker that cries wolf.

It verifies the header dimension, that the file size is a whole number of records (a partial write, as distinct from a deliberate prefix), and that enough vectors are present. Verified in both directions: exit 0 on the real corpus, exit 1 when asked for more than exists.

## The bind mount is part of the measurement too (Amendment 1)

Four defects surfaced on the first real attempts to run this, all fixed before any sweep data existed.

**The harness could not run on Windows** — the host it exists for. It built shell strings and quoted them with `shlex.quote()` under `subprocess.run(shell=True)`, but `shell=True` on Windows is `cmd.exe`, where single quotes are literal characters. Docker saw `-v 'C:\...':/repo`, split on the colon in `C:`, and died with `invalid mode: /repo`. Now every command is passed as an argument list.

**`ctest` then failed the go/no-go gate, and it was the filesystem lying.** `ClusterConfigRace` failed 10/10 with the repo bind-mounted from Windows/OneDrive, reporting that `save_cluster_config` is not atomic against concurrent readers. Run with its working directory on the container's own filesystem, the same binary passed 5/5 with **0 failures in ~90,000 concurrent reads per run** — where the mounted runs managed 920 reads total. Measured: the mount is **7.9× slower** for bulk writes (118 vs 933 MB/s) and **4.9× slower** for create+rename+unlink. CI runs the same `ctest` on `ubuntu-latest` and is green.

So the gate failure was an instrument artifact — outcome (iii) does **not** apply, and the build now passes **9/9**. The larger consequence is that the sweep must not run there either: Layer 1 measures recall under chaos over 180 s and 300 s windows, and a filesystem 5–8× slower without atomic rename would change kill spacing, replication lag and settle-window adequacy — the exact quantities `SPEC.md`'s Confounds section flags as machine-sensitive.

**Then the sweep ran and all ten cells produced nothing — and the stage exited 0.** Every cell died on `No module named 'grpc'`. The apt list was inherited from CI on the reasoning that "a build that works in CI works here", which is sound for the build and wrong for the sweep: CI builds and runs `ctest`, it never runs `run_experiment.py`, so nothing in CI imports `grpc` and its list has no reason to carry the Python gRPC bindings.

The worse half is the exit code. Each cell tolerates its own failure so one bad cell doesn't cost the others — that's right — but the stage then reported success after ten consecutive failures, and the analyse stage would have run over an empty sweep. That is the silent-failure shape this README already cites (#26, #38, #46, #48) as the reason `verify_corpus.py` exists, reappearing one layer up. The sweep now counts produced/skipped/failed cells and exits non-zero if any produced no `samples.csv`, and `deps` asserts the imports up front.

**Finally, a host memory kill destroyed a running cell.** The host has 7.7 GB of RAM with ~1.2 GB free; the OOM pressure killed the harness process, and because `docker exec` without `-d` ties the exec'd process's lifetime to its client, all nine cluster processes died with it. Cells now run detached and signal completion with a marker file, so a run's lifetime belongs to the container and an interrupted sweep resumes at the cell it lost. Headroom was measured before restarting: ~105 MB/min growth against ~2.4 GB available, so both cell durations fit.

The repo is now `rsync`ed to `/work` inside the container and everything runs there; results sync back to the mount after each cell. **No protocol parameter changed** — only where the working directory points. The cost is honest and goes in the write-up: the container, the filesystem, and now the host's memory limits are three specific ways this is not the original machine — not a generic "different machine" caveat.

## Where the data will land

`results_sweep/seed<N>_{baseline,chaos}/`, the layout `aggregate.py --sweep-dir` expects.

**Deliberately not** in `research/replica_recall/results*/`, whose `.gitignore` rule stays. That rule exists because scratch output was committed by accident twice (#18, #24), and loosening it re-opens the hazard — and a reproduction on a different host is a different artifact from the original run, so filing it separately keeps that visible.

Full pre-registration, outcomes and confounds: [`SPEC.md`](SPEC.md).
