# Is the repair clock anchored to the write, or the restart?

Issue #56 · branch `method/weaviate-repair-clock` · **runs in progress.**

#48 measured Weaviate repair from the victim's restart. Under that origin two conditions differing only in divergence age separated 30× (p = 0.0022). Measured from the **write**, the same runs are 38.9 s and 36.9 s — about 5% apart. #51's review withdrew the causal reading on that basis, leaving the measurement standing and the mechanism open.

This decides it. Hold the outage at 40 s and vary where inside it the write lands:

| | prediction |
|---|---|
| anchored to the **write** | `age + repair` flat; `repair` falls as age rises |
| anchored to the **restart** | `repair` flat; `age + repair` rises with age |

Opposite predictions on the same runs, so one dies.

**It also decides something practical:** #54's observation window has to be timed from *something*. If the clock starts at the write, a window timed from the restart can close before repair fires — the failure that voided #24 and #9, arriving from a new direction.

**No threshold is registered anywhere.** The statistic is the coefficient of variation of one continuous quantity against another. #48's step 2c pre-registered a binary fast/slow cut at 1 s, the gap justifying it was falsified by that same experiment, and the statistic went void — which is now a rule in `SPEC_TEMPLATE.md`.

Results and decision will land in [`SPEC.md`](SPEC.md).
