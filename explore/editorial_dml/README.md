# What can an editor still change? Adjusted associations of the editorial levers with shares

Exploration, not part of the report (v19 and v18 are untouched). Everything here lives in
`explore/editorial_dml/`; outputs are in `out/`.

**Question.** Once an article's topic is fixed, how much do the things an editor can still change before
publishing (images, videos, length, links, headline length, weekend, how crowded the day is) move shares, and
how does that compare with what a predictive model's feature importance suggests?

## Data

- `OnlineNewsPopularity.csv`, rows in file order. The file's `url` column is shifted against its feature rows
  from June 2013 on, so publication dates come from `timedelta` (2015-01-08 minus `timedelta`; this matches
  the weekday flags on 100% of rows, the url date on 76%) and urls / real headlines come from
  `data/external/realigned.csv` (copy of `/mnt/project-files/kw_rebuild/realigned.csv`, built and verified by
  the main research thread; re-checked here: headline word count vs `n_tokens_title` correlates 0.84 after
  June 2013 with the realigned headline and 0.06 with the headline of the url in the file).
- Sample: 38,302 articles (dropped: 1,181 with unparsed text, 1 corrupt row, the last 152 rows whose true url
  is outside the file, 24 without a sitemap headline). Outcome: log(1 + shares).

## Design

| | |
|---|---|
| Levers T | log(1 + count) of images, videos, body words, headline words, links to other sites, links to Mashable articles; weekend (0/1); log number of same-section and of other-section articles published that day |
| Wording (weaker sense of "changeable") | body and headline subjectivity, headline polarity, share of positive / negative words, average word length (2013 dictionary measures) |
| Confounders W (main set, 88 columns) | publication date, day of week, section, LDA topics, keyword history `kw_*`, number of keywords, 64 principal components of the real headline's bge-base vector |
| Not a lever | `self_reference_*` (shares of the linked articles): used only as a control (`ref`) and as a negative-control outcome |
| Estimator | partially linear double ML (Robinson 1988; Chernozhukov et al. 2018): cross-fitted LightGBM predictions of log shares and of each lever from W, then residual on residual. 5 folds made of whole weeks, 5 repetitions (median), standard errors clustered by week. "Joint" = all levers in the final regression together, so each is read with the other levers held fixed |
| Heterogeneity | lever x section / year interactions on the same residuals; causal forest (Athey, Tibshirani & Wager 2019, `econml.grf`) trained on half of the weeks and scored on the other half |
| Naive comparison | raw slope; a LightGBM on the 58 dataset features + the two volume counts: TreeSHAP importance and the model's own what-if (change the lever for held-out articles, average the change in prediction) |

**Causal assumptions, stated plainly.** (1) *No unobserved confounding*: given W, nothing else drives both a
lever and the shares. This is not testable and almost certainly false to some degree: we do not observe the
article's format (gallery, explainer, sponsored post), its quality, its author, or where it was placed on the
site. (2) Overlap: every kind of article could have had more or fewer images, etc. (3) The levers are measured
on the published page, so they are choices that were made, not interventions; "one more video" is read as
"an otherwise similar article that had one more video". The page is the one crawled in January 2015, not the
one at publication: an article that became popular may have had a video, images or links added afterwards
(reverse timing), which would push the article-lever estimates up. (4) The number of articles per day counts articles in
this dataset, not Mashable's full output. Section D below is about how far assumption (1) can be pushed. Its
verdict, up front: the article-lever numbers are **adjusted associations**, not effects; only the day-level
crowding result is defensible as a causal reading.

## Results

All numbers: % change in shares for doubling (1 + count), or weekend vs weekday, or +1 SD for wording; 95% CI.

### A. The predictive view overstates the levers (`d1`, `d2`, `out/fig_levers.png`)

| lever | raw association | model what-if | double ML (adjusted), one lever | double ML (adjusted), levers jointly (main) |
|---|---|---|---|---|
| images | +7.3 | +2.8 | +2.5 | **+1.8** [0.8, 2.9] |
| videos | +5.7 | +4.9 | +3.0 | **+3.1** [1.8, 4.4] |
| body length | +0.8 | +1.2 | +4.2 | **+0.8** [-0.6, 2.2] |
| headline length | -7.6 | +9.4 | +4.1 | **+3.6** [-0.2, 7.5] |
| links to other sites | +7.0 | +2.8 | +3.0 | **+2.4** [1.5, 3.3] |
| links to Mashable articles | +5.1 | +0.8 | +4.4 | **+3.8** [2.1, 5.5] |
| weekend | +37.5 | +5.1 | +29.1 [25.0, 33.4] | -2.5 [-8.4, 3.8] once the day's volume is held fixed |
| same-section articles that day | -16.5 | -4.8 | -9.1 | **-9.0** [-10.7, -7.3] |
| other-section articles that day | -15.5 | -12.1 | -13.7 | **-13.0** [-16.9, -9.0] |
| body subjectivity (+1 SD) | +13.2 | | +2.2 | **+2.4** [1.4, 3.5] |
| headline subjectivity (+1 SD) | +5.0 | | 0.0 | -0.4 [-1.8, 1.1] |
| headline polarity (+1 SD) | +5.6 | | +0.1 | +0.3 [-1.1, 1.7] |
| share of positive words (+1 SD) | +5.7 | | -1.4 | -1.7 [-2.9, -0.5] |
| share of negative words (+1 SD) | +0.3 | | -0.6 | -0.9 [-1.9, 0.0] |
| average word length (+1 SD) | -3.6 | | -0.3 | -1.1 [-2.2, 0.0] |

- In the predictive model the lever columns carry 24% of total SHAP importance (keyword history 32%, linked
  articles' shares 14%, section + LDA 16%, dictionary sentiment 10%). Once W is held fixed, the six article
  levers together explain **0.56%** of the variance W leaves over, the two volume counts another 0.60%; W
  itself explains 19.2% out of fold. These are two different quantities (a share of importance vs a partial
  R2), so the contrast is illustrative; the like-for-like comparison is the model what-if column above.
- What disappears: the headline's dictionary sentiment (+5% raw, 0 after the headline vector is in W; since
  the sentiment score is close to a function of the headline, read this as "no association beyond what the
  headline's content carries", not as "sentiment has no effect"), three
  quarters of the image association, two thirds of the external-link association, body length once links are
  held fixed, most of body subjectivity (+13% to +2.4%). Positive wording changes sign.
- What stays: videos, images and links at **+2 to +4% per doubling**; day-level crowding at -9% / -13%.
- The model's what-if is not a safe substitute: it is close for videos, several times too small for internal
  links, and for headline length it reports +9% for a change (doubling a headline) that no article in the data
  shows.
- The ladder (`out/d2_ladder.csv`): most of the movement has happened once context, topic and keyword history
  are in W. The headline vector still matters for videos (+4.4 to +3.0) and removes headline sentiment (+2.4 to
  0); adding the linked articles' shares changes little. Random folds or extra trees instead of LightGBM change nothing material (`d5_robustness.csv`).

### B. Thresholds, not slopes (`d3`, `out/fig_dose_response.png`)

Other levers held fixed. One video vs none **+10%** [7, 12], two or more +11% [7, 15]. Images beyond the lead
image: 2-3 **+6%** [3, 9], 4-9 +9% [5, 13], 10+ +7% [3, 11]. No internal link at all: **-17%** [-22, -12] against
1-2, and nothing gained beyond two. Body length is flat except above 1,200 words (+9% [4, 15]). Headline
length is flat.

### C. The weekend premium is accounted for by crowding or exposure on the day (`d7`)

The day's volume counts articles in this dataset, so mechanically it measures exposure (how long an article
stays near the top of the site) as much as competition for attention; "crowding" below means either.

Weekend articles get +29% [25, 33] more shares than comparable weekday articles, but weekends have 0.42x the
same-section and 0.36x the other-section articles. Estimating the volume slopes **from Monday-Friday articles
only** and multiplying by that gap predicts a weekend premium of **+36% [26, 47]**; with the day's volume held
fixed the weekend term is -2.5% [-8, 4]. Volume and weekend are both day-level (720 days, 103 weeks; standard
errors are clustered by week), and nearly collinear, so this says the premium *is accounted for by* volume, not
that weekends have no effect of their own.

| check | result |
|---|---|
| Placebo: same-section volume 7 days later | -1.2% [-2.7, 0.4] (own-day volume -9.4% in the same regression) |
| Identification within week and weekday only | same-section -8.3% [-10.4, -6.2], other-section -11.8% [-16.5, -6.8] (main: -8.9%, -12.3%); predicted weekend premium +34% [23, 46] |
| Log-linearity (the prediction extrapolates) | quieter half of weekdays -7.0% / -12.7%, busier half -8.3% / -9.8% |
| Overlap | weak: only 2 of 515 weekdays are as quiet as a median weekend day (24 articles), 11 are below the weekend's 90th percentile (32), all US holidays. On those 11 days the observed premium is +32% against +30% predicted from the other weekdays' slope, but with 294 articles the interval is [-21, +123] |
| Neighbouring days (t-1, t, t+1 together) | own day unchanged (-8.4%, -12.5%); same-section day before -1.5% [-3.1, 0.1], day after -0.6% [-2.1, 0.9]; other-section neighbours about 0. The association is concentrated on the article's own day; the spillover that exposure competition predicts is at most weak, so this check does not separate crowding from selection of what is published that day |
| Composition (does W predict weaker articles on busy days?) | content-predicted log shares, within week, weekday and section: same-section -2.2% [-2.8, -1.6] against -9.5% for actual shares; other-section +0.8% [-0.5, 2.2] against -13.4%. Observable composition is about a quarter of the same-section association and none of the other-section one |

Reading: the other-section volume result is the cleaner one (no composition, same slope everywhere). Selection on
unobservables on quiet days remains the open caveat.

### D. How much of this is causal? (`d4`, `d5`)

- **Robustness values** (Cinelli & Hazlett 2020): 3-4% for the article levers (1% for headline length), 5-6% for volume,
  10% for weekend. A hidden factor explaining about 3-4% of what is left of both the lever and the shares would erase
  an article-lever effect. For scale: the headline vector explains 5.4% of the remaining variance of shares and
  6-10% of the levers', so a hidden factor **about half as strong as the headline vector** (0.45-0.65x) is
  enough; it would have to be 1.3-4.8x as strong as the whole keyword-history block.
- **Negative control fails.** The shares of the *older* articles an article links to cannot be caused by its
  images or videos, and they share the likely confounder (how hot the story is, what format the piece has).
  Yet, given W and the other levers, doubling images "raises" them by +3.5% [2.2, 4.8], videos +5.9% [4.2, 7.6],
  links to other sites +2.2% [1.2, 3.3]: as large as or larger than the associations with the article's own
  shares (+1.8%, +3.1%, +2.4%; one lever at a time the control gives +2.6%, +4.7%, +2.5%).
  Taken at face value, nothing is left of the image and video estimates after calibrating against the control.
  (The control is not valid for internal links: it only exists when there is one.) Adding the linked
  articles' shares to W (`ref`) moves the estimates only slightly (images 1.8 to 1.6, videos 3.1 to 3.0), so
  the missing factor is not simply that variable.
- **Same-story twins** (2,720 pairs rebuilt on the realigned urls; story held fixed by design, no W needed):
  images **+8.0%** [3.1, 13.1], videos +5.6% [-0.5, 12.1], links to other sites 0.0% [-3.7, 3.7], internal links
  +3.0% [-4.2, 10.8], same-section volume -6.3% [-13.3, 1.3]. Images and videos agree in sign with double ML;
  external links do not replicate. This does not rescue a causal reading: the twin estimate for images is four
  times the double-ML one, and two articles on one story still differ in format (gallery vs news piece).
- **Not stable over time or sections.** Images: +0.9% in 2013, +4.3% in 2014 (p < 0.001); external links +1.3%
  vs +4.3%; videos +9.6% in tech, +6.9% world, +6.6% business, +2% in entertainment, about 0 in social media
  and in articles without a section (p < 0.001). The causal forest finds heterogeneity that holds up on held-out weeks (calibration
  slope 1.1-2.0, all p < 0.001 for article levers; e.g. videos +11% in the third of articles with the highest
  predicted effect, 0 in the lowest), driven mostly by the other levers (internal links, length) and the date,
  i.e. by article format. For the two volume counts the forest finds no reliable heterogeneity: crowding hurts
  everything about equally.

## Bottom line

1. Feature importance is not a to-do list: the lever columns look like a quarter of the model, but holding
   topic and headline fixed they explain about 1% of what is left.
2. Format choices show adjusted associations of 2-4% per doubling, with thresholds (have a video; more than
   the lead image; at least one internal link). Our negative control indicates these are mostly or wholly
   residual confounding by story or format, and a modest hidden factor would erase them (robustness values
   3-4%). They are upper bounds on what an editor could expect, not effects.
3. Dictionary sentiment of the headline shows no association with shares beyond what the headline's content
   carries.
4. The one large result that survives the checks is **when** an article appears: crowding on the day. It passes the
   placebo, holds within week and weekday, is homogeneous across articles, and quantitatively accounts for the
   whole weekend premium. The remaining caveat is selection on unobservables (what gets published on quiet
   days); observable composition explains a quarter of the same-section part and none of the other-section part.
   Wording accepted in review: "Within the same week and weekday, a day with twice as many articles carries
   8-12% fewer shares per article; observable composition explains at most a quarter of this. Extrapolated to
   weekend volumes, this slope is large enough to account for the whole weekend premium (+34% [23, 46] predicted,
   +29% observed), though weekdays almost never reach weekend volumes, and the data cannot separate crowding or
   exposure from unobserved same-day selection."
5. Side observation for the main line: with correctly aligned headlines, the 64-component headline vector
   lifts the out-of-fold R2 of log shares from 0.146 to 0.194 (whole-week folds), so headlines do carry
   cross-article signal in this data.

## Reproduce

```bash
pip install lightgbm econml onnxruntime tokenizers huggingface_hub     # on top of requirements.txt
cp /mnt/project-files/kw_rebuild/realigned.csv data/external/           # file_row -> true url, headline
PY=python explore/editorial_dml/run_all.sh                             # ~1 h on 4 cores; restartable
```

| script | does | outputs |
|---|---|---|
| `d0_title_embeddings.py` | bge vectors of the realigned headlines | `data/interim/dml/title_bge_true.npy` |
| `d1_naive.py` | raw slopes, SHAP, model what-if | `d1_*.csv` |
| `d2_dml_ladder.py` | double ML over the confounder ladder | `d2_ladder.csv`, `d2_nuisance_r2.csv`, `d2_joint_r2.csv` |
| `d3_dose_response.py` | effects by bin | `d3_dose_response.csv` |
| `d4_heterogeneity.py` | section / year / period, causal forest | `d4_*.csv` |
| `d5_sensitivity.py` | robustness values, negative control, placebo, other folds / learner, twins | `d5_*.csv` |
| `d7_weekend_crowding.py` | weekend premium predicted from weekday crowding; within-week, neighbouring-day, composition, overlap checks | `d7_weekend_crowding.csv`, `d7_crowding_checks.csv`, `d7_quiet_weekdays.csv` |
| `d8_cite_tables.py` | compact tables for the report | `cite_levers.csv`, `cite_weekend_crowding.csv` |
| `d6_figures.py` | figures | `fig_levers.png`, `fig_dose_response.png` |

References: Robinson (1988, Econometrica); Chernozhukov, Chetverikov, Demirer, Duflo, Hansen, Newey & Robins
(2018, Econometrics Journal); Athey, Tibshirani & Wager (2019, Annals of Statistics); Cinelli & Hazlett (2020,
JRSS-B); Chernozhukov, Cinelli, Newey, Sharma & Syrgkanis (2022, "Long story short"); Lipsitch, Tchetgen
Tchetgen & Cohen (2010, negative controls).
