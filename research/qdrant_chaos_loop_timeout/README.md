# A chaos loop that died silently, and the seed it cost

Issue #38 · PR #40 · a **method** fix to the instrument, not an experiment.

**Closed — fixed and covered.**

## What went wrong

In [`../qdrant_index_recall_healing/`](../qdrant_index_recall_healing/), seed 20261100's quiesce run fired **zero kills** across a chaos window the log recorded as 127 seconds, and reported nothing wrong. The run completed, exited zero, and produced a `samples.csv` that looked like every other run. It was excluded by hand once someone noticed `events.json` was empty — 5 pre-registered seeds, 4 judged.

## Root cause, narrower than the issue guessed

The issue supposed a Docker call had blocked forever. It had not: `ManagedContainer._docker` already passes `timeout=30` to `subprocess.run`, so a hung daemon **raises** `subprocess.TimeoutExpired` rather than hanging.

The defect is what happens to that raise. Both chaos loops run in **daemon threads with no exception handling**, so the exception ends the loop instantly and invisibly — no event, no log line, no non-zero exit. `chaos_stop_rel` is stamped when the phase timer expires regardless, which is why the window read 127 s against an empty `events.json`.

The fix everyone would have proposed — "add a timeout" — was already half-present. What was missing was catching what the timeout raises.

## What changed

- Both loops wrap each kill/restart in `try/except`, append an event with `failed: true`, `error`, `timed_out`, and **keep going** to the next kill.
- `run_meta.json` gains `kill_count`, `kill_failures`, `chaos_no_kills`, `chaos_requested_s`, `chaos_realized_s`. Successful events gain `failed: false`, so old and new records stay distinguishable.
- `qdrant_sweep.py` **refuses** a chaos run in which no kill completed, the same way it refuses a missing `samples.csv`.

A bug found while writing that guard is worth recording: `json` was not imported in `qdrant_sweep.py`, so the guard as first written would have raised `NameError` **on the exact path it guards**. A silent-failure fix that fails silently. Caught by importing it and asserting the symbol resolves.

## Validation

| check | result |
|---|---|
| `test_chaos_failures.py` (no Docker) | **13/13** — both loops against fakes raising `TimeoutExpired` and `RuntimeError`; asserts the failed event's fields, that `timed_out` separates the two causes, that **the loop survives and keeps killing**, and that no thread is left running |
| `../qdrant_kill_scheduler/test_kill_schedule.py` | **27/27**, unchanged |
| live run (60k vectors, seed 20261200) | `kill_count: 3`, `kill_failures: 0`, `chaos_realized_s: 40.929`, all events `failed: false` |

## What this does not do

It does not make a hung Docker daemon less likely, and it does **not** retro-diagnose seed 20261100 — that run predates the fix, so its `events.json` is empty either way and it stays excluded. It also does not add timeouts to the *sampler* and *validator* threads, which have the same daemon-thread shape and are the obvious next place to look; that is left out of scope deliberately rather than quietly folded in.

## Why this is in the research record at all

This is the third instrument in this project to report success while measuring something else — #26 (stale output looking current), this one, and #46 (a probe running against a sharded, un-replicated class). A fourth followed: #48's 20,000-id probe returned `first_answer None` rather than an error.

That pattern is the research thesis reproduced inside its own tooling. The work argues that approximation converts data loss into silence, so no checker fires; these are the same failure in the measurement apparatus. Recording them is not self-flagellation — it is the reason every result in this repo is now expected to say how it would look if the instrument were broken.

## Reproducing

```bash
python research/qdrant_chaos_loop_timeout/test_chaos_failures.py    # 13/13, no Docker needed
python research/qdrant_kill_scheduler/test_kill_schedule.py         # 27/27, no Docker needed
```

Full pre-registration, root-cause detail and decision: [`SPEC.md`](SPEC.md).
