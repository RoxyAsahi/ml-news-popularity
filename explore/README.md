# Exploration for the v2 research direction (not part of the report yet)

Feasibility scripts run on 2026-11-12. Each prints its numbers; none writes into results/.
Run from the repo root with `python explore/<script>.py` after `pip install -e .`.

| Script | Question | Headline (exploratory) |
| --- | --- | --- |
| f1_slug_memory.py | Do TF-IDF slug neighbours (point-in-time, time-decayed) carry popularity memory? | No: Spearman ~0, AUC unchanged |
| f2_twins.py | Same-story "twin" articles from URL slugs (TF-IDF cos >= 0.5, <= 7 days apart) | 3,382 pairs; log-share r ~0.05, same as random same-channel-week pairs |
| f3_within_auc.py | What is AUC 0.74 made of? Pairwise AUC within day / channel / channel-day | Section explains most of the context signal |
| f4_semantic_memory.py | Same as f1 with word vectors, raw and context-residualised targets | No gain (word vectors in the md wheel looked unreliable) |
| f5_competition.py | Attention competition: same-day publishing volume | -0.29 log shares per log-unit of daily volume (week + weekday + channel FE, day-clustered CI [-0.37, -0.20]); robust to content controls |
| f6_twin_effects.py | Topic-controlled natural experiment: feature effects within twins vs cross-section | negativity/polarity effects vanish, length and subjectivity effects grow |
| f7_rashomon.py | Do near-equally-accurate models agree on feature-group importance? | 30/33 models within 0.01 val AUC; importance varies up to 2x; channel vs LDA ranks swap |
| f8_conformal.py | CQR + adaptive conformal under drift; selective classification | 80% interval spans x6 in shares; ACI fixes monthly under-coverage; top-10% confident -> 0.875 accuracy |
| f9_anatomy_ci.py | f3 with out-of-fold scores and week-cluster bootstrap CIs, plus twin ICC | within channel 0.66, same-story same-channel twins 0.60 [0.54, 0.66]; story ICC 0.04-0.10 |
