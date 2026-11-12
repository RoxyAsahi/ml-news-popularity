# E7 — Clustering and anomalies in topic + sentiment space (RQ4)

Script: `scripts/e7_clustering.py` (~1.5 min). Input: 5 LDA topic weights + 16 sentiment
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

   Within the same broad content, sentiment shifts popularity: the one cluster with negative mean
   polarity (−0.03) is 6–15 points less often popular than the positive mixed clusters.

3. **Unusual articles are more often viral.** Isolation Forest flags the 2% most unusual articles
   (770). They reach the top 5% of shares 9.2% of the time vs 4.9% for the rest (≈ 1.9×), and are
   popular (≥ 1400) 58.6% vs 53.1% (χ² p = 0.003).

## For the report

- Lead with the negative result (weak structure) and the comparison against the channel baseline —
  this is the scientifically honest reading.
- The anomaly result is a small, clean positive finding worth one sentence and one number.
