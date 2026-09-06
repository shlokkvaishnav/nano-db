#!/usr/bin/env python3
"""
Re-analysis of step 2b's data to test whether the pre-restart delay predicts
which repair path Weaviate takes (issue #48, step 2c). No new runs: this reads
`results/order_vs_phase.json` and nothing else.

Step 2b reported only the group means (fast 10.4 s, slow 32.3 s). That
understates the separation, and a mean is the wrong summary for n=4 vs 6, so
this computes the exact two-sided Mann-Whitney U by enumerating all 210
permutations -- the same non-parametric test this project uses elsewhere, and
for the same reason: no distributional assumption survives n<10.

The threshold search at the end is DESCRIPTIVE, not a fitted classifier. With
10 points a single cut will always look good; it is reported to show where the
groups touch, and the overlap is reported with it.

Usage:
    python research/weaviate_repair_window/analyze_path_selection.py
"""
import json, itertools
from statistics import mean
d=json.load(open('research/weaviate_repair_window/results/order_vs_phase.json'))
fast=[r for r in d if r['repair_s']<=1.0]; slow=[r for r in d if r['repair_s']>1.0]
F=[r['delay'] for r in fast]; S=[r['delay'] for r in slow]
print("fast n=%d delays=%s mean=%.1f max=%.1f"%(len(F),sorted(F),mean(F),max(F)))
print("slow n=%d delays=%s mean=%.1f min=%.1f"%(len(S),sorted(S),mean(S),min(S)))
# exact two-sided Mann-Whitney U (no scipy dependency, matches project practice)
def U(a,b):
    return sum((x>y)+0.5*(x==y) for x in a for y in b)
u_obs=U(S,F); n1,n2=len(S),len(F)
allv=S+F; cnt=0; tot=0; extreme=0
for idx in itertools.combinations(range(n1+n2),n1):
    g1=[allv[i] for i in idx]; g2=[allv[i] for i in range(n1+n2) if i not in idx]
    u=U(g1,g2); tot+=1
    if min(u,n1*n2-u)<=min(u_obs,n1*n2-u_obs)+1e-9: extreme+=1
print("U=%.1f of max %d ; exact two-sided p=%.4f (%d/%d permutations)"%(u_obs,n1*n2,extreme/tot,extreme,tot))
print("floor for n=4 vs 6:", 2/tot)
# threshold separation
best=None
for th in [x/10 for x in range(30,460)]:
    err=sum(1 for r in d if (r['delay']>th) != (r['repair_s']>1.0))
    if best is None or err<best[1]: best=(th,err)
print("best single-threshold rule: delay > %.1f s -> slow ; misclassifies %d/10"%best)
print("overlap zone: fast max %.1f vs slow min %.1f"%(max(F),min(S)))
