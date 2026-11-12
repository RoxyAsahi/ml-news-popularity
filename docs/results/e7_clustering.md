# E7 — Clustering and anomalies in topic + sentiment space (RQ4)

Scripts: `scripts/e7_clustering.py` (~1.5 min), `scripts/e7b_confounding.py` (~10 s; confounding checks). Input: 5 LDA topic weights + 16 sentiment
features, standardised; 1,181 articles with no content words removed (38,462 left). No share
information is used to form clusters. Tables `results/tables/e7_*.csv`, figures `e7_clusters.png`,
`e7_model_selection.png`.

## Findings

1. **There is no strong cluster structure.** KMeans silhouette is 0.09–0.12 for every k from 2 to 10
   (best k = 9, 0.121); GMM BIC keeps decreasing up to k = 10. Clusters are regions of a continuum,
   not natural groups — say so in the report.

2. **Clusters still differ in popularity, but mainly by recovering the editorial channel.**
   k = 9: χ² = 1,360, p < 1e-280, Cramér's V = 0.188 (k = 5: V = 0.134). The channel itself gives
   V = 0.245, so the clusters carry *less* popularity information than the channel label.

   | Cluster (k = 9) | Articles | Popular (≥ 1400) | Median shares | Main channel |
   | --- | --- | --- | --- | --- |
   | tech, topic 4 | 6,314 | 62% | 1,700 | tech 70% |
   | business, topic 0 | 5,180 | 62% | 1,700 | business 72% |
   | mixed, topic 4, positive | 4,330 | 61% | 1,700 | mixed |
   | topic 3 | 4,985 | 58% | 1,600 | mostly no channel |
   | mixed, topic 3 | 1,298 | 53% | 1,400 | mixed |
   | mixed, topic 4, most positive | 3,484 | 52% | 1,400 | mixed |
   | mixed, topic 2, **negative polarity** | 3,330 | 46% | 1,300 | world 32% |
   | entertainment, topic 1 | 3,462 | 42% | 1,200 | entertainment 82% |
   | world, topic 2 | 6,079 | 38% | 1,100 | world 85% |

   The one cluster with negative mean polarity (−0.03) is 6–15 points less often popular than the
   positive mixed clusters — but see point 4: much of that is channel.

3. **Unusual articles look more often viral, but that is mostly a channel effect.** Isolation
   Forest flags the 2% most unusual articles (770). Crude: they reach the top 5% of shares 9.2% of
   the time vs 4.9% (1.9×). The outliers are short (median 173 vs 429 words), with strongly
   subjective, polar titles and unusually positive word polarity, and 43% have no channel
   (vs 14% of inliers) — and the no-channel articles are popular anyway. Stratified by channel
   (Mantel–Haenszel) the odds ratio for reaching the top 5% shrinks to 1.27 [0.98, 1.63] (p = 0.06)
   at 2% contamination, 1.17 [0.82, 1.67] at 1% and 1.21 [1.02, 1.44] (p = 0.03) at 5%.
   **At most a weak effect remains after accounting for channel.**

4. **The sentiment effect survives channel adjustment, but is smaller and channel-specific**
   (`scripts/e7b_confounding.py`, review item R6). Articles with negative overall polarity
   (3,264) are less often popular: crude odds ratio 0.69; within channels (CMH) 0.81 [0.75, 0.88],
   p < 1e-6; logistic regression with channel dummies gives the same 0.81 [0.76, 0.88]. The effect
   differs by channel (Breslow–Day p = 0.001): business 39% vs 54% popular, world 32% vs 39%,
   social media 66% vs 76%, entertainment 38% vs 41%, but none in tech (66% vs 64%) or articles
   without a channel (65% vs 65%). The negative cluster gives the same picture (adjusted OR 0.82).

## For the report

- Lead with the negative result (weak structure) and the comparison against the channel baseline —
  this is the scientifically honest reading.
- Report the anomaly result only together with its channel-adjusted odds ratio; the crude 1.9× is mostly confounding.
- The sentiment result is a good example of checking a finding for confounding: say crude 0.69 → adjusted 0.81, and that it holds in news-like channels but not in tech.
