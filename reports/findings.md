# Findings: answers to the five research questions

Personalized movie recommendation with collaborative filtering (CSE575). All numbers are test-split results for seed 42 unless stated otherwise, on MovieLens 25M (ML) and a 50,000-user sample of the Netflix Prize data (NF). Ranking metrics are at K = 10 with "relevant" meaning a held-out rating of at least 4 and already-rated movies excluded. Seed means, bootstrap intervals and the source tables are listed in `reports/figures/INDEX.md`. Every method uses the same split, the same users and the shared evaluation module.

## Summary

| Question | Short answer |
|:--|:--|
| 1. Can CF predict preferences accurately? | Yes, clearly better than naive baselines: RMSE 0.81–0.82 (ML) and 0.87–0.88 (NF) vs 0.957 / 1.010 for the item mean, and about 1.7–2× popularity in top-K quality. |
| 2. Does matrix factorization improve quality? | For **rating prediction**, yes (RMSE 0.754 / 0.824, a further 5–7% below the best CF). For **top-K ranking**, no, when it is trained on rating error; trained for ranking (one-class WRMF) it is by far the best method. |
| 3. How does MF compare with simple methods? | It beats the item-mean baseline by 18–21% in RMSE, but its top-K lists score *below* the popularity baseline (NDCG@10 0.058 vs 0.111 on ML, 0.049 vs 0.089 on NF). |
| 4. How well does it recommend unrated movies? | Best system (one-class): about 70% of users get at least one liked held-out movie in their top 10 (NDCG@10 0.29 / 0.27, about 2.6× / 3.0× popularity). Lists remain concentrated on popular movies. |
| 5. Behaviour with little history? | Accuracy degrades for every method. With a single rating nothing beats popularity; from about 3–5 ratings one-class CF and user CF pass it, and one-class CF is best from about 20 ratings (ML) / 5–10 (NF). |

The central finding: **the model with the best rating error is not the best recommender.** Rating-trained MF has the lowest RMSE and the poorest top-K quality; a positives-only model trained for ranking is the opposite.

## 1. Can collaborative filtering predict user preferences accurately?

**Yes, with a large margin over non-personalized baselines.**

| | ML RMSE | NF RMSE | ML NDCG@10 | NF NDCG@10 |
|:--|---:|---:|---:|---:|
| Global mean | 1.058 | 1.081 | – | – |
| Item mean (≥50 ratings) | 0.957 | 1.010 | 0.042 | 0.041 |
| Popularity | – | – | 0.111 | 0.089 |
| Item-based CF | 0.8125 | 0.8723 | 0.184 | 0.171 |
| User-based CF | 0.8229 | 0.8830 | 0.216 | 0.166 |

- Both CF variants cut RMSE by 14–15% (ML) and 12–14% (NF) relative to the item mean, and raise NDCG@10 to 1.7–2.0× popularity.
- The 95% bootstrap intervals are narrow (about ±0.001–0.005) and the ordering of all methods is the same in all three seeds, so these differences are not noise. With 48k–152k users almost any gap is statistically detectable, so the effect sizes matter more than the p-values.
- The better CF variant depends on the dataset and the task: on ML user-based CF ranks better (NDCG +0.032) while item-based CF predicts ratings better (RMSE −0.010); on NF item-based CF wins both (NDCG +0.005, RMSE −0.011).

*Caveats.* "Accurate" is relative: about 46% of users (item CF) and 40% (user CF) have no relevant movie in their top 10 on ML (Hit@10 0.542 / 0.602), and RMSE of 0.81 is still large on a 5-star scale. Neighbourhood parameters (k, significance weighting) were tuned on validation only; user CF was tuned on a 5,000-user sample.

## 2. Does matrix factorization improve recommendation quality?

**It depends on what "quality" means.**

- *Rating prediction: yes.* Biased MF reaches RMSE 0.7539 (ML) and plain MF 0.8241 (NF), 7.2% and 5.5% below the best CF. The paired differences are real (intervals exclude 0). User/item bias terms matter little here (ΔRMSE ≤ 0.003).
- *Top-K ranking: no, when trained on rating error.* NDCG@10 is 0.058 (ML) and 0.049 (NF): about half of popularity and a quarter of the best CF. This held for every seed and every user-history size.
- *Matrix factorization with the right objective is the best ranker.* One-class WRMF (a weighted factorization of the binary "liked" matrix, trained with ALS) reaches NDCG@10 0.285 (ML) and 0.271 (NF), and the dual-regularized variant 0.291 on ML. So factorization does improve ranking when its loss matches the task.

*Why.* Ranking by predicted rating promotes rarely rated movies: MF's recommended movies have a median of 1,000–2,000 training ratings versus about 7,000–39,000 for the other personalized methods, and 16–29% of its slots come from the less-rated half of the catalogue versus at most 2% for the others (`reports/tables/list_diagnostics.md`). The qualitative examples show documentary titles with 500–1,400 ratings in MF's top 5 for four of six users, regardless of taste.

*Training behaviour* (`fig_learning_curves.png`). Biased MF's validation RMSE flattens after about 30 epochs and does not rise again (ML: 0.7534 at the last epoch) while its training RMSE keeps falling (0.54), so the regularization is doing its job; one-class NDCG@10 converges within about 10 ALS iterations.

*Caveats.* The dual-regularized model is our own implementation of the idea in Yao et al. (CIKM 2014), not a faithful reproduction. Its gain over plain WRMF is small but real on ML (+0.0062 NDCG@10, interval [0.0058, 0.0066]; seed means 0.2928 vs 0.2863) and reversed on NF (−0.0018).

## 3. How does MF compare with simple methods?

- vs the **item-mean** rating baseline: RMSE −21.2% (ML) and −18.4% (NF).
- vs **popularity** (the simple ranking baseline): rating-trained MF is *worse* (NDCG@10 0.058 vs 0.111 on ML, 0.047 vs 0.089 on NF). One-class WRMF is 2.6× (ML) and 3.0× (NF) popularity.
- MF with item and user biases gives only a negligible improvement over plain MF in RMSE and none in ranking.

*Caveat.* Popularity is a strong baseline here because the protocol counts any held-out liked movie as a hit, and users are more likely to have rated popular movies. This favors popularity-like recommendations for every method (see the limitations).

## 4. How well does the system recommend movies the user has not rated?

The test protocol is exactly this task: candidates exclude everything the user has already rated, and a hit is a movie the user later rated at least 4 that the model had not seen.

| Best system (one-class: dual-regularized on ML, WRMF on NF) | ML | NF |
|:--|---:|---:|
| Hit rate@10 (at least one liked movie in the top 10) | 0.704 | 0.708 |
| Precision@10 | 0.166 | 0.199 |
| Recall@10 | 0.283 | 0.195 |
| NDCG@10 | 0.291 | 0.271 |

(Popularity: Hit@10 0.406 / 0.378; best CF: 0.602 / 0.559.) In the 5,000-user sample used for the examples, 29% of users have no hit in their top 10 with the one-class model (popularity 59%, user CF 39%, MF 73%).

*Caveats.* (a) Held-out ratings are a partial observation: a recommended movie without a held-out rating is **unknown, not a miss**, so these metrics are a conservative estimate of recommendation quality. The qualitative examples (`reports/examples.md`) show users with zero hits whose lists look reasonable. (b) Even the best system's lists are popularity-heavy: about half or more of its slots (49–60%) come from the 1% most-rated movies, and it covers only 9–13% of the catalogue. (c) The best ranker recommends movies with a lower average crowd rating (about 3.9) than the CF lists (4.0–4.2) or the mean-rating list (4.3–4.6), because it predicts what users will rate, not what the crowd rates highest.

## 5. How does the model behave for users with little rating history?

Two complementary analyses.

**Natural variation** (users grouped by their number of training ratings; every user has at least 20, `history_buckets.md`). All metrics improve with history. Relative to popularity, one-class CF is about 2.9–3.0× for ML users with under 30 ratings, falling to 2.0× at 300+; rating-trained MF is weakest for short histories (0.43× popularity at under 30 ratings on ML, 0.11× on NF). Biased MF has the best RMSE in every bucket and its edge over item CF is largest for short histories. User-based CF predicts ratings better than item-based CF for histories under about 60 ratings, and item-based wins beyond that.

**Simulated cold start** (each user reveals only N = 1, 3, 5, 10, 20 or 50 randomly chosen training ratings; `cold_start.md`). NDCG@10 on ML (NF is similar):

| N | 1 | 3 | 5 | 10 | 20 | 50 | full |
|:--|---:|---:|---:|---:|---:|---:|---:|
| Popularity | 0.110 | 0.110 | 0.110 | 0.110 | 0.110 | 0.110 | 0.110 |
| User CF | 0.110 | 0.142 | 0.158 | 0.176 | 0.192 | 0.205 | 0.216 |
| Item CF | 0.110 | 0.070 | 0.079 | 0.101 | 0.127 | 0.155 | 0.184 |
| One-class (dual) | 0.098 | 0.109 | 0.126 | 0.161 | 0.203 | 0.249 | 0.292 |
| MF (biased) | 0.017 | 0.029 | 0.038 | 0.045 | 0.044 | 0.036 | 0.059 |

- **With one rating, nothing beats popularity.** CF has no usable signal (the single centred rating is 0) and falls back to the popularity list; one-class CF is below it (0.098 / 0.083) because one liked movie is a weaker signal than the crowd.
- **User-based CF is the best choice for the first few ratings** (above popularity from N = 3 on both datasets). **One-class CF overtakes popularity at about N = 3–5 and user CF at about N = 20 on ML (N = 5–10 on NF)**, and its advantage keeps growing with N.
- **Item-based CF is worse than popularity until about N = 20.** MF-by-rating never beats popularity at any N.
- **For rating prediction**, biased MF is the only model that is never worse than the item-mean baseline (ML RMSE 0.936 at N = 1 vs 0.950). CF at N = 1 predicts the single rating as the user's mean (RMSE 1.35–1.41); plain MF has no mean term and starts at 2.09. User CF and plain MF pass the item mean at about N = 10, item CF at about N = 50.

*Caveats.* The smallest natural bucket is 20–29 ratings, not a true cold start; the simulation fills that gap but it truncates the history of existing users rather than observing real new users, and it keeps the item structure learned from all users. Users with no usable signal receive the popularity ranking (the share is reported in `cold_start.md`). ML-25M uses a fixed 30,000-user subset for this analysis. CF predicts the user mean from the revealed ratings without shrinking it toward the global mean, which explains the poor N = 1 RMSE; a production system would shrink.

## Limitations

1. **Random per-user split, not temporal.** Held-out ratings are drawn from the same period as the training ratings, which is easier than predicting the future. Absolute numbers are probably optimistic for a deployed system; the comparison between methods is still fair because all use the same split.
2. **Popularity bias in the evaluation.** Observed ratings reflect what users chose to watch. A held-out "liked" movie is more likely to be a popular one, so popularity-leaning methods are rewarded; unobserved good recommendations count as misses.
3. **Filtered data.** ML-25M is restricted to movies with at least 50 ratings (13,172 of 59,047; 98.6% of ratings) and Netflix Prize to a 50,000-user sample of users with at least 20 ratings, for feasibility. The Kaggle "Netflix dataset" contains titles only, so it is used for metadata and EDA.
4. **Limited tuning budget and one-time model selection.** Hyperparameters were chosen on validation by staged grids (not exhaustive); seeds 43 and 44 reuse the seed-42 settings. MF was tuned separately for RMSE and for ranking; one-class dual regularization is our own interpretation of Yao et al.
5. **Few seeds.** Three seeds and 1,000 user resamples quantify split and sampling variation, not every source of randomness (for example the Netflix user sample is redrawn per seed but only three times).
6. **Qualitative examples** cover ML-25M only and are a handful of users chosen by fixed rules; they illustrate mechanisms rather than prove them.
7. **No explicit-feedback signals beyond star ratings** (no timestamps in the models, no tags, no content features), and no online evaluation.

## Where to find the evidence

| Claim | Source |
|:--|:--|
| Main numbers | `reports/tables/main_results.md`, `bootstrap_ci.md`, `seed_summary.md` |
| Significance of the 17 headline comparisons | `reports/tables/paired_comparisons.md` |
| History and cold start | `history_buckets.md`, `cold_start.md`, figures `history_buckets.png`, `cold_start.png` |
| Why methods differ | `list_diagnostics.md`, `reports/examples.md`, `reports/examples_analysis.md` |
| Method fitting | `fig_sensitivity_*.png`, `fig_learning_curves.png` |
| Deviations from the proposal and why | listed in the README summary; the full log is kept in `DIFFERENCE.md` (not tracked in git) |
