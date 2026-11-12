# v2 RQ2-RQ4 (scripts s4-s7, figures via fig_v2.py)

## RQ2 content effects with the story held fixed (s4, `s4_twin_effects.csv`)
13 standardised content features. Cross-section: channel x week + weekday FE, week-cluster bootstrap.
Twins: earlier-minus-later differences + constant + channel-difference dummies, story-cluster bootstrap;
difference test by combining independent bootstraps; BH across the 13 twin coefficients.
- Body subjectivity is the one effect that survives and grows: cross-section 0.070 [0.057, 0.084],
  twins 0.166 [0.092, 0.239], BH q = 0.013, difference 0.097 [0.018, 0.165]; holds for sim >= 0.6,
  same channel only, and similar-length twins.
- Links (0.067 [0.010, 0.132]) and word length (-0.113 [-0.198, -0.030]) agree in sign with the
  cross-section but do not pass BH (q 0.095 / 0.065).
- Videos: cross-section 0.054 [0.041, 0.068], twins 0.007 [-0.041, 0.059] (difference p = 0.09):
  suggestive of topic confounding, not established.
- Negative tone (share of negative words) is ~0 in both designs with these controls; body length
  is not robust (the earlier 'length doubles' result came from a sparser specification).
- Twins differ a lot in format: within-twin correlation of body length 0.02, subjectivity 0.05.

## RQ3 same-day competition (s5) and semantic memory (s6)
- log shares per log daily volume: -0.286 [-0.372, -0.200] (week, weekday, channel FE, day-clustered);
  + content -0.282; long articles -0.216; without the busiest 5% days -0.309; 2013 -0.481, 2014 -0.115;
  within twins -0.200 [-0.324, -0.076]; placebo (next week's volume) -0.059 [-0.131, 0.014].
  Same-channel -0.108 and other-channel -0.153 per log volume (both matter).
  Doubling volume ~ -18% shares. n_day counts dataset articles, not necessarily Mashable's output.
- Semantic memory (time-decayed slug neighbours, half-life inf/7/30/180 d): Spearman 0.016-0.029;
  past-only AUC change -0.0008 [-0.0019, 0.0004]. Negative result (with old E2 recency weighting).

## RQ4 honest uncertainty (s7)
- 80% CQR interval median 776-5,017 shares (upper/lower ratio 6.2). Monthly coverage of CQR dips to
  0.763 in Sep 2014; ACI keeps 0.796-0.810 (online, assumes shares known before the next article).
- Selective prediction (past-only HistGB): accuracy 0.673 on all, 0.785 on the 40% most confident,
  0.869 on the top 10%, 0.903 on the top 5%.
