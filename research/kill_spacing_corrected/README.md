# Kill spacing, attempt two: a void result kept for the methodology

Issue #24 · PR #25 · **Closed — result VOID, and the kill-spacing line of work stopped.**

## What this asked

Does the *spacing* between node kills change how much damage accumulates in a Qdrant cluster? Attempt one ([`experiment/qdrant-kill-spacing`](https://github.com/shlokkvaishnav/Replica-Recall-Divergence/tree/experiment/qdrant-kill-spacing), issue #9) ran 15 runs and produced a degenerate metric — zero damage survived to the measurement point. This branch corrected the measurement and ran 15 more.

## Why the result is void

The comparison **failed its own pre-registered sampling precondition**, and the analyzer caught it before any number was interpreted:

- realized kill spacing was **4.27 s against a required ≤4.0 s**
- **11 of 42** damage episodes were captured by a single sample

An episode seen once has no measurable duration, so the primary metric could not be computed as specified. The branch reports that rather than the secondary peak result, which would have looked like a finding.

## Why it is not too-slow-by-a-bit

The probe cost scales with corpus size, because `completeness` currently ships every id on every replica every round. Lowering the sample interval does not fix it — at these corpus sizes the floor is above the phenomenon. **A third sweep at these parameters would reproduce this outcome exactly**, which is why the line of work stopped instead of being re-run.

The fix is a harness change, not a parameter change: a probe mode that fetches only what `completeness` needs — an id-set digest, or a count-plus-checksum — rather than the full list.

## What is worth keeping

- **The first quantitative description of the damage transient in this project**: lag median **14.1 s**, range **7.5–46.8 s** (Amendment 1), plus the episode-duration data here. Any future design needs these numbers.
- **The probe-cost / corpus-size floor**, now characterized rather than guessed at.
- `analyze_corrected.py`, which evaluates and prints validity preconditions **before** running any comparison. That is what caught this, and unlike #9's analyzer its comparison path did run against real damage — 42 episodes across 15 runs — so it is exercised where it matters.

This study is the direct reason `SPEC_TEMPLATE.md` now requires an **Instrument characterization** section. Three sweeps (#11/#16, #20, #25) were voided by properties of the measuring apparatus that were computable from existing artifacts *before* the compute was spent.

## Recorded against myself

`## Addendum: review round 1` in the SPEC records two overstatements of mine that the reviewer caught by recomputing the primary metric from the raw CSVs without this branch's analyzer (185.3 / 128.3 / 148.7, p = 0.5476 / 0.8413 / 0.6905 — exact match). Kept in place rather than edited away.

## Reproducing

```bash
python research/kill_spacing_corrected/analyze_corrected.py
```

Runs against the committed `results/{spread,short-gap-same-node,long-gap-same-node}_seed2026095{0..4}/` — 15 runs, and the densest per-replica time series in the repo at 52–71 sample rounds each. The preconditions print first; the void verdict is reproduced from the committed data without re-running any cluster.

Full pre-registration, four amendments, results and decision: [`SPEC.md`](SPEC.md).
