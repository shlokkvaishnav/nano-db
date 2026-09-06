# The repair clock: neither write nor restart, but a regime selector

Issue #56 · branch `method/weaviate-repair-clock` · **53 runs. Outcome (d).**

## The headline, and the trap

`repair_s` is timed from the victim's restart; `age + repair` from the write. #51's review showed the same 18 runs tell opposite stories depending on which you use, so this study varied where inside a fixed 40 s outage the write landed.

**The pre-registered statistic said write-anchored.** CV(`repair_s`) = 0.766 against CV(`age + repair`) = 0.132 — 5.8× steadier — with aggregate corr(age, repair) = **−0.942** and corr(age, since_write) = **−0.012**. Textbook write-anchored signature.

**Disaggregating by regime reverses it:**

| regime | n | age span | `repair_s` spread | corr(age, repair) | corr(age, since_write) |
|---|---|---|---|---|---|
| young, 2–15 s | 17 | 13 s | **2.34 s** | **−0.267** | **+0.993** |
| old, 30–38 s | 12 | 8 s | **1.45 s** | **−0.103** | **+0.991** |

Within either regime `repair_s` is **flat** while `age + repair` tracks age almost perfectly — the **restart**-anchored signature. The aggregate −0.942 comes entirely from the step *between* regimes (~32 s → ~2 s), not from any trend inside one.

Simpson's paradox, in an experiment designed to settle a question about origins.

## What the answer actually is

- **Divergence age selects the regime.** Below ~15 s the victim waits ~32 s; above ~30 s it reconciles in ~2 s. The threshold sits between and was never sampled.
- **Within a regime, the clock runs from the restart.**
- **Absence is irrelevant** once age is fixed: across 25/40/70 s outages, `repair_s` spans 1.35 s.

So: age is a **selector between two behaviours**, not an offset against a countdown.

## The methodological finding is the more transferable one

The statistic was well chosen. It registers an estimand rather than a threshold — exactly what `SPEC_TEMPLATE.md` now requires after #48 step 2c's fast/slow cut went void. And it was still wrong, because it pooled across a step.

**Registering the right kind of statistic does not protect against aggregating heterogeneous regimes.** What caught it was disaggregation, prompted by the reviewer step (`AGENT_PIPELINE.md` 1c) that asks whether another cut of the same data collapses the effect. Here it *reversed* it, which the step doesn't currently mention — filed as a consequence.

The statistic was also checked for mechanical bias before being overturned: simulating a pure restart-anchored system over these same realized ages returns "RESTART" at every constant tried, so the test can distinguish the hypotheses in principle. It was applied to the wrong population, not miscalibrated.

## Two open items, and one failed check

**The 6 s outage is still bimodal** — six runs at 0.011–0.028 s and four at 43.1–53.3 s, at identical parameters. Wall-clock phase was recorded for the first time (#48 never captured it); correlations against candidate periods are all weak at n = 10 (|r| ≤ 0.56). Neither supported nor excluded.

**The probe-perturbation check is void**, and the reason matters for #54. The slow-poll runs report `repair_s = 0.000` with `first = 50/50` — the victim already held everything on its *first* sample. That shows the first sample landed after convergence, not that the probe is harmless. **It does establish that a 1 s cadence can miss a fast repair entirely** — a direct constraint on #54, which planned exactly that cadence.

## Reproducing

```bash
python research/weaviate_repair_clock/repair_clock.py     # ~90 min, 53 runs
python research/weaviate_repair_clock/analyze_clock.py    # reads committed JSON only
```

Analysis uses **realized** offsets, never requested labels (Amendment 1): several runs overran badly because `write_objects` at `consistency_level=ONE` occasionally takes minutes with a node down — itself an instrument property #54 needs.

Full pre-registration, amendment, results and decision: [`SPEC.md`](SPEC.md).
