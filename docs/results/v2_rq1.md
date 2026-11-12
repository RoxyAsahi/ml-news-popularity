# v2 RQ1: what is the model's skill made of? (scripts s1, s1b, s2, s3)

## Twins (s1, s1b)
- Slug TF-IDF cosine >= 0.5, <= 7 days apart: 3,382 pairs; the series filter removes recurring columns
  (viral video recap, olympics day N, episode recaps ...) -> 2,941 pairs, 1,278 stories.
- 180 pairs annotated by the authors' assistant (not independent annotators; protocol in
  `scripts/s1b_twin_precision.py`). At >= 0.5 after the filter: same story 0.78 [0.68, 0.85]
  (Wilson), same story or same topic 0.96. Filter: 27/33 series caught, 0 false positives.
- Channel labels agree only weakly with slug topic (e.g. 'game of thrones' slugs: 27% entertainment),
  so 70% of twins sit in different channels. Channel behaves like a placement label, not a topic.

## Pairwise AUC within contexts, past-only monthly rolling origin (s2, `s2_anatomy.csv`)
HistGB, all features, 2013-07..2014-12 scored (week-cluster bootstrap; twins: story-cluster bootstrap)

| stratum | AUC | 95% CI | concordance on >2x share gaps |
| --- | --- | --- | --- |
| all pairs | 0.734 | [0.725, 0.740] | 0.714 |
| same week | 0.734 | [0.726, 0.742] | 0.714 |
| same day | 0.727 | [0.719, 0.735] | 0.711 |
| same channel | 0.690 | [0.682, 0.696] | 0.676 |
| same channel x day | 0.675 | [0.668, 0.683] | 0.667 |
| same-story twins | 0.719 | [0.692, 0.748] | 0.690 [0.656, 0.727] |
| same-story twins, same channel | 0.672 | [0.602, 0.738] | 0.655 [0.589, 0.726] |

Reading: time contributes nothing; the channel label is worth ~0.045 AUC; fixing the story adds no
detectable further drop (same-channel twins overlap same channel x day). Most skill is article-level.
LogReg and HistGB-without-share-features show the same pattern (0.709 -> 0.639 and 0.701 -> 0.628).

## Do same-story articles share their fate? (s3, `s3_story_similarity.csv`)
Residual log shares after channel x week + weekday fixed effects; baseline = j replaced by a random
article from the same channel and day (20 draws); story-cluster bootstrap.
- Twin correlation 0.051 [0.016, 0.086] vs matched random 0.002; same-channel twins 0.103 [0.040, 0.167].
  Story identity explains ~5% (10% within a channel) of what context leaves.
- The earlier twin gets more shares: +0.070 [0.005, 0.132] log points (random same-week pairs: -0.002);
  larger for closer twins (sim >= 0.7: +0.135). Consistent with the follow-up being cannibalised,
  which also means the low correlation could partly be split attention rather than story irrelevance.
- Variance shares of log shares: week 1.5%, channel x week 7.8%, channel x day 9.7%.
