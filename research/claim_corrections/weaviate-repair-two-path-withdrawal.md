# Weaviate repair: the "~0.3 s bound" and the "two discrete paths" reading, both withdrawn

**Date:** 2026-09-06 · **PR:** #51 · **Issue:** #48 · No new experiment beyond step 2c; this records what was corrected and why.

Two claims about Weaviate's async repair reached `main` and are withdrawn here. They failed in opposite directions, which is the interesting part.

---

## Correction 1 — "any healing measurement must beat 0.3 s"

**What was on `main`.** PR #45 (`weaviate_nonperturbing_probe/`) observed one async repair completing in ~0.3 s and wrote it up as a design constraint: *"Any Weaviate healing measurement must beat 0.3s."* PRs #44 and #46 inherited it, and by #46 sub-second sampling was recorded as **"the only remaining prerequisite"** before the dissociation experiment could run.

**Why it was wrong.** It was **n = 1**, and `weaviate_nonperturbing_probe/SPEC.md:125` said so at the time — *"0.3s is one observation, n = 1"* — in a limitations list two lines below the sentence that had already turned it into a bound. The caveat was written and then not applied.

Characterizing the apparatus (#48) showed the same 50-object divergence gives **44.7 s, 0.008 s, 0.010 s**, and that repair duration is independent of divergence size across 50→5,000. Sampling only ever had to beat the *slow* route.

**Cost of the error.** It nearly bought a piece of engineering nobody needed. The correction's practical content is that **1–5 s cadence over a ≥60 s observation is sufficient** — the cadence this project already uses on nano-db and Qdrant.

**Where the withdrawal is recorded:** `weaviate_nonperturbing_probe/README.md:18` and `SPEC.md:123`, `weaviate_probe_per_id/SPEC.md:58, 110, 120`, `weaviate_nonperturbing_probe/SPEC.md:134`, and `research/README.md`.

A process note on the cleanup itself: #48's decision named only two files as needing correction, so **four further sites survived for a day** after the withdrawal was published. A consequence note that enumerates files is only as good as the grep behind it.

---

## Correction 2 — "repair is two discrete paths"

**What was on `main`.** PR #49 reported that pooling 18 observations, *"nothing falls between 0.2 s and 36 s"*, and read that emptiness as evidence of **two discrete code paths** — an immediate catch-up, or a ~36–50 s wait.

**Why it was wrong.** Step 2c held divergence size fixed and set the timing offsets *deliberately* instead of randomizing the delay. **Twelve of eighteen runs landed inside the supposedly empty interval.**

The gap was an artifact of the sampling design. A uniform random delay samples the latency surface sparsely and unevenly; eighteen draws populated two regions and missed the middle. **Randomization is not neutral** — it distributes observations according to the design, and reading structure out of where they happen to fall is a mistake that a fixed-condition design exposes immediately.

**A second failure, worth more than the first.** The step 2c pre-registration fixed a **binary fast/slow split at 1 s** as its metric — a threshold chosen *because the gap was believed to exist*. With the gap gone, that threshold cuts condition B's tight 0.735–2.245 s cluster in half, so the pre-registered statistic reported "slow 4/6" and Fisher p = 0.4545 on a dichotomization whose premise was dead.

Pre-registration did its job against motivated reading of the *result*, and did nothing against a bad *analysis choice* locked in beforehand. The defence is to pre-register the estimand — here, the latency distribution by condition — and keep the dichotomization out of it.

**What replaced it, and it is stronger.** Holding the outage at 40 s and varying only how old the divergence was when the node returned: ~38 s old reconciles in **~1 s**, ~6 s old takes **~31 s**. Disjoint, exact two-sided Mann-Whitney U = 0 at 6v6, **p = 0.0022**. **Amended in review round 1:** this replacement is itself weaker than written. `repair_s` is measured from the restart, and the conditions differ in when the write happened relative to that origin. From the **write** (`age_s + repair_s`) B and C are 38.9 s and 36.9 s — a 30× gap becomes ~5%. Convergence at a roughly fixed latency after the write fits the B/C contrast equally well and would be near-definitional, so the causal sentence is withdrawn and the measurement reported without a mechanism. A's slow runs at ~56 s since-write rule out a single global constant, so neither reading is established. Whether the repair clock is anchored to the write or the restart is the open question, and it is cheap to test. A third condition (6 s outage) is bimodal at ~0.02 s or ~52 s and is **not explained** by either manipulated variable — recorded as open, with no mechanism named.

**A downstream conclusion re-argued rather than carried.** *"The healing signal is a step, not a decay curve"* had rested on the withdrawn reading. Re-derived from the committed trajectories alone, it holds at 50–500 objects (transition in 8–167 ms) but **not** at 5,000, where it spans 5.8–6.3 s — roughly 5–6 points at 1 s cadence. So divergence size decides whether the experiment measures a step or a curve. That is more useful than the claim it replaces.

---

## What this pair adds to the pattern

`README.md` in this directory records that every overstatement found so far ran in the same direction — toward a tidier, more confident claim than the evidence supported. **These two do as well, but they are not overstatements of a result.** They are overstatements of a *mechanism*: "0.3 s" became a bound, and an empty interval became two code paths. In both cases a description of what was observed was promoted to a claim about how the system works.

The observation survives in each case. The mechanism did not. The rule this suggests: a mechanism claim needs its own evidence, and cannot be inferred from the shape of a measurement made for another purpose.

Both corrections were caught the same way every previous one was — by re-reading raw per-run data against the prose describing it.
