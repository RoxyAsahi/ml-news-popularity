# E8 / E4b — Direction of effects and the editor's view (added after review round 2)

## E8 — accumulated local effects
Script `scripts/e8_ale.py` (~1 min, own ALE implementation, Apley & Zhu 2020); table `e8_ale.csv`,
figure `e8_ale.png`. HistGB, threshold 1,400, training block. Effect on P(popular), 10th → 95th percentile:

- `kw_avg_avg`: −0.077 → +0.074 (0.15), steadily increasing: the topic's track record.
- `self_reference_avg_sharess`: −0.029 → +0.023 (0.05).
- `num_hrefs`: flat below 10 links, +0.03 by 31 links (p95).
- `n_tokens_content`: flat up to ~800 words, +0.02 by 1,400 words (p95).
- The last bin (top 5% to max) of each curve rests on few articles; don't interpret its jump.

## E4b — weekly precision of the top 5% at the 95th-percentile threshold
Script `scripts/e4b_precision_at_k.py` (~30 s); tables `e4b_weekly_precision.csv`,
`e4b_precision_summary.csv`, figure `e4b_precision_at_top5.png`.

- Viral = ≥ 11,300 shares (training-block 95th percentile). HistGB trained on the training block.
- 18 full test weeks, ~440 articles and a shortlist of ~22 per week.
- Precision 0.119 [0.090, 0.145] (bootstrap over weeks) vs base rate 0.040: lift 3.0;
  ~2.7 viral hits per week; 1 week with none.
