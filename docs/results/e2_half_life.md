# E2 — Half-life recency weighting (RQ1)

Script: `scripts/e2_half_life.py` (~4.5 min). Sample weight 0.5^(age / half-life), age in days
from the newest training article (idea inspired by VCPToolBox's TimeDecay; own implementation).
5 rolling-origin folds; paired per-fold difference against unweighted training, 95% t-interval.
Table `results/tables/e2_half_life.csv`, figure `results/figures/e2_half_life.png`.

| Half-life (days) | HistGB ΔAUC [95% CI] | Logistic regression ΔAUC [95% CI] |
| --- | --- | --- |
| 30 | −0.013 [−0.019, −0.007] | +0.005 [−0.006, +0.015] |
| 90 | −0.002 [−0.008, +0.005] | +0.008 [−0.000, +0.015] |
| 180 | −0.001 [−0.005, +0.002] | +0.006 [+0.002, +0.010] |
| 365 | +0.001 [−0.003, +0.006] | +0.004 [+0.002, +0.006] |
| 730 | +0.000 [−0.002, +0.002] | +0.002 [+0.002, +0.003] |

Unweighted rolling-origin AUC: HistGB 0.730, logistic regression 0.715.

## Reading

- **For the best model, recency weighting does not help** (no half-life gives a detectable gain),
  and a short 30-day half-life hurts (−0.013): it throws away most of the data, which E5 shows
  the tree ensembles still need.
- Logistic regression gains a small but consistent +0.004 to +0.006 with half-lives of 180–365
  days. It is far from closing the 0.015 gap to HistGB.
- This fits E1: ranking is fairly stable over time (the decay is 0.03 over six months), so the
  benefit of emphasising recent articles is small, and only a low-variance model can use it.
- Write it as a mostly negative result in the report; it tests a plausible idea and says why it
  does not pay off for the strongest model.
