# Web app walkthrough

A run-through of the CineMatch web app (`frontend/` + `src/api.py`) in the order a user would meet it, taken at a normal 1440 px-wide browser window. Run it with `uvicorn src.api:app --port 8000` and `cd frontend && npm run dev`. The recommender shots are a real session on MovieLens 25M: the user searches, adds and rates movies, and the five live models respond in milliseconds. A dark theme exists (toggle in the top bar) but is not shown here.

## 1. Dashboard (landing page)

| File                                    | What it shows                                                                                                  |
| :-------------------------------------- | :------------------------------------------------------------------------------------------------------------- |
| `01_dashboard_hero.jpg`                 | Landing page: hero with a call to action and popular movies.                                                   |
| `02_dashboard_profiles_and_popular.jpg` | Ready-made taste profiles and the most-rated movies; either one opens the recommender with that choice loaded. |
| `03_dashboard_engines.jpg`              | "How it works" and the five recommendation engines, closing call to action.                                    |

## 2. Overview (the results of the project)

| File                                  | What it shows                                                                                                                                         |
| :------------------------------------ | :---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `04_overview_summary_movielens.jpg`   | Key numbers (lowest RMSE, best NDCG@10, share of users with a hit, users evaluated), three takeaways, dataset switch (MovieLens 25M / Netflix Prize). |
| `05_overview_charts.jpg`              | Ranking-quality and rating-accuracy bars with 95% bootstrap confidence intervals.                                                                     |
| `06_overview_accuracy_vs_ranking.jpg` | Accuracy vs. ranking scatter: the best rating predictor (matrix factorization) is not the best recommender.                                           |
| `07_overview_full_results_table.jpg`  | Full test table for all nine methods (RMSE, MAE, Precision/Recall/Hit/NDCG@10); the seed-42 / 3-seed mean ± std switch is at the top right.           |
| `08_overview_head_to_head.jpg`        | Paired bootstrap head-to-head comparisons with confidence intervals and a verdict.                                                                    |
| `09_overview_netflix_dataset.jpg`     | The same overview for the Netflix Prize sample (generalisation check).                                                                                |

## 3. Get Recommendations (trying the system)

| File                                      | What it shows                                                                                                                                                                 |
| :---------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `10_recommend_empty_start.jpg`            | Empty start: search box, sample profiles and popular movies to pick from.                                                                                                     |
| `11_recommend_search_matrix.jpg`          | Typeahead search ("matrix").                                                                                                                                                  |
| `12_recommend_one_rating_fallback.jpg`    | After one rating (The Matrix, 4 stars): the personalised models do not have enough signal and the app says so, showing popular movies instead.                                |
| `13_recommend_five_ratings_consensus.jpg` | Five rated movies (Star Wars, Terminator 2, Blade Runner, Inception, The Matrix); half-star ratings, engine selection, and consensus picks (movies several engines agree on). |
| `14_recommend_popularity_and_item_cf.jpg` | Popularity (same list for everyone) next to item-based CF, with per-engine latency and "similar to ... which you liked" hints.                                                |
| `15_recommend_user_cf_and_mf.jpg`         | User-based CF and matrix factorization lists; MF surfaces less-rated titles.                                                                                                  |
| `16_recommend_one_class.jpg`              | One-class CF (the best ranker) list.                                                                                                                                          |
| `17_recommend_genre_filter_comedy.jpg`    | Genre filter (Comedy): every model re-ranks within the genre.                                                                                                                 |
| `18_recommend_single_model_view.jpg`      | "One model" view that shows a single engine's full list.                                                                                                                      |
| `19_recommend_new_user_profile.jpg`       | The brand-new-user profile (two ratings): how each model copes with almost no history.                                                                                        |

## 4. Explore analyses

| File                             | What it shows                                                                                   |
| :------------------------------- | :---------------------------------------------------------------------------------------------- |
| `20_explore_history_size.jpg`    | Metrics by number of ratings in a user's history.                                               |
| `21_explore_cold_start.jpg`      | Simulated cold start: NDCG@10 against the number of revealed ratings, dashed line = popularity. |
| `22_explore_list_quality.jpg`    | List diagnostics: median popularity of recommended movies per method (why MF ranks badly).      |
| `23_explore_training_curves.jpg` | MF train/validation RMSE per epoch and one-class validation NDCG@10 per ALS iteration.          |
| `24_explore_report_figures.jpg`  | Gallery of the static report figures.                                                           |

---

# Workflow: how the app is navigated and used

The left sidebar is always visible and has four entries in a fixed order: **Dashboard → Overview → Get Recommendations → Explore analyses**. The top bar shows whether the models have finished loading ("Models ready") and holds the light/dark toggle. The sidebar tip (bottom left) repeats the main idea: rate a few movies you love and compare what five engines suggest.

## Step 1. Start on the Dashboard

Open the app; it lands on the Dashboard (`01_dashboard_hero.jpg`). Read what the site does, then choose how to begin:

- Click a **sample taste profile** or a **popular movie** (`02_dashboard_profiles_and_popular.jpg`). The choice is handed to the recommender page, which opens with it already loaded.
- Scroll to see how it works and the five engines: popularity, item-based CF, user-based CF, matrix factorization and one-class CF (`03_dashboard_engines.jpg`).
- Or use the sidebar to go anywhere directly.

## Step 2. Check the evidence on the Overview

Go to **Overview**. The page answers "which recommender works best?":

1. Read the headline numbers and the three takeaways (`04_overview_summary_movielens.jpg`).
2. Switch between **MovieLens 25M** and **Netflix Prize** at the top right (`09_overview_netflix_dataset.jpg`).
3. Compare methods in the ranking-quality and rating-accuracy charts, choosing the metric (NDCG, Hit rate, Precision, Recall; RMSE, MAE) and K (5/10/20) with the toggles (`05_overview_charts.jpg`).
4. Look at the scatter to see that the best rating predictor is not the best recommender (`06_overview_accuracy_vs_ranking.jpg`).
5. Read exact numbers in the full table; switch between seed 42 and the 3-seed mean ± std (`07_overview_full_results_table.jpg`).
6. Check significance in the head-to-head table (NDCG, Hit rate or RMSE tab; difference with its 95% CI and a verdict) (`08_overview_head_to_head.jpg`).

## Step 3. Get recommendations

Go to **Get Recommendations**. This is the interactive demo; the models score every movie in about a millisecond to tens of milliseconds, with no retraining.

1. **Start empty** (`10_recommend_empty_start.jpg`) or load a sample profile with one click.
2. **Add movies:** type in the search box, pick a suggestion or press Enter (`11_recommend_search_matrix.jpg`), or tap a card under _Popular movies_ (it can be filtered by genre). A new movie starts at 4 stars.
3. **Rate:** click the stars (half-star steps) in _Your ratings_; use the × to remove a movie or _Clear_ to start over. Ratings are remembered in the browser, and results update automatically about a third of a second after each change.
4. **Too little signal:** with one rating (`12_recommend_one_rating_fallback.jpg`) the personalised models cannot say much, so the app shows popular movies and labels that engine's column with a notice. The Dashboard-style "brand-new user" profile (`19_recommend_new_user_profile.jpg`) shows the same situation with two ratings.
5. **Read the results** once there are a handful of ratings (`13_recommend_five_ratings_consensus.jpg`):
   - _Consensus picks_ at the top are movies several engines recommend; the coloured dots show which ones.
   - Below, one column per engine with its top-N list, latency, and a "Similar to X, which you liked" hint (`14_recommend_popularity_and_item_cf.jpg`, `15_recommend_user_cf_and_mf.jpg`, `16_recommend_one_class.jpg`). Notice that popularity is the same for everyone, and that matrix factorization surfaces less-rated titles.
6. **Refine:** the chips toggle engines on and off; the selectors set the genre (`17_recommend_genre_filter_comedy.jpg`) and list length (Top 5-30); **One model** shows a single engine's full list with a picker for which one (`18_recommend_single_model_view.jpg`).

## Step 4. Understand the behaviour in Explore analyses

Go to **Explore analyses**, which has five tabs that map to the research questions:

1. **History size:** how each method performs for users with few vs. many ratings (`20_explore_history_size.jpg`).
2. **Cold start:** simulated users who reveal only N ratings; shows when each method starts beating popularity (`21_explore_cold_start.jpg`).
3. **List quality:** why the methods differ — popularity, catalogue coverage and diversity of their top-10 lists (`22_explore_list_quality.jpg`).
4. **Training:** learning curves for matrix factorization and one-class CF (`23_explore_training_curves.jpg`).
5. **Report figures:** the static figures used in the report; click one to enlarge (`24_explore_report_figures.jpg`).

## Suggested path for a demo

Dashboard (`01`) → Overview headline and scatter (`04`, `06`) → Get Recommendations: rate five sci-fi films (`13`), compare engines (`14`–`16`), apply the genre filter (`17`) → try the new-user profile (`19`) → Explore: cold start and list quality (`21`, `22`).

## Behind the scenes

The page talks to a FastAPI backend (`src/api.py`, `src/serving.py`): result pages read the saved tables in `reports/tables/`, and the recommender folds a new user's ratings into the frozen models (closed-form fold-in for MF and one-class CF; neighbour search for the CF models), so no model is retrained. The Netflix Prize data appears only in the results pages because its movie titles are not part of the live demo.
