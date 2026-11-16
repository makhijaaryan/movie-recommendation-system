# DATA_DICTIONARY.md

Raw data lives in `data/raw/` (gitignored). Processed data lives in `data/processed/` (gitignored). Checked 2026-11-16.

## MovieLens (GroupLens) — `data/raw/movielens/`
Source: https://grouplens.org/datasets/movielens/ (ml-25m, ml-latest-small).

| File | Columns | Notes |
|---|---|---|
| `ratings.csv` | userId, movieId, rating, timestamp | rating 0.5–5.0 in 0.5 steps; timestamp = Unix seconds. 25M: 25,000,095 rows, 162,541 users, 59,047 movies. Small: 100,836 rows, 610 users, 9,724 movies. 0 invalid, 0 duplicates. |
| `movies.csv` | movieId, title, genres | genres pipe-separated (`Adventure|Animation|...`); year in title |
| `tags.csv` | userId, movieId, tag, timestamp | free-text user tags |
| `links.csv` | movieId, imdbId, tmdbId | external IDs |
| `genome-scores.csv` (25M only) | movieId, tagId, relevance | tag-relevance 0–1 per movie |
| `genome-tags.csv` (25M only) | tagId, tag | tag vocabulary |

## Netflix Prize — `data/raw/netflix_prize/`
Source: Kaggle `netflix-inc/netflix-prize-data`.

| File | Format | Notes |
|---|---|---|
| `combined_data_1–4.txt` | Movie header line `<movieId>:` then rows `<userId>,<rating>,<YYYY-MM-DD>` | ratings 1–5 integers |
| `movie_titles.csv` | movieId, year, title | title may contain commas |
| `probe.txt`, `qualifying.txt` | competition test sets | Proposed to ignore (DIFFERENCE.md D6) |

Processed: `data/processed/netflix_prize/ratings_part{1-4}.parquet` — movieId (int16), userId (int32), rating (int8), date (datetime). Produced by `src/parse_netflix_prize.py`.

| Statistic | Value |
|---|---|
| Ratings | 100,480,507 |
| Users | 480,189 |
| Movies | 17,770 |
| Dates | 1999-11-11 to 2005-12-31 |
| Density | 1.18% |
| Nulls / ratings outside 1–5 | 0 / 0 |
| Duplicate user-movie pairs | 0 |

## Kaggle Netflix titles — `data/raw/netflix_titles/`
Source: Kaggle `mirajshah07/netflix-dataset`. **No user-level ratings; metadata and EDA only** (DIFFERENCE.md D2).

| File | Columns |
|---|---|
| `netflix_dataset.csv` | show_id, type, title, director, cast, country, date_added, release_year, rating, duration, listed_in, description |
| `IMDb movies.csv` | imdb_title_id, title, original_title, year, date_published, genre, duration, country, language, director, writer, production_company, actors, description, avg_vote, votes, budget, usa_gross_income, worlwide_gross_income, metascore, reviews_from_users, reviews_from_critics |
| `IMDb ratings.csv` | imdb_title_id, weighted_average_vote, total_votes, mean_vote, median_vote, votes_1…votes_10, plus per-age/gender average votes and counts (aggregates, not individual users) |

## Cross-dataset notes
- MovieLens `links.csv` gives IMDb IDs, so MovieLens movies can be joined to `IMDb movies.csv` for extra metadata.
- Netflix Prize movies have only title/year, so joining to other sources needs fuzzy title matching (not planned).
- Rating scales differ: MovieLens 0.5–5 (half steps); Netflix Prize 1–5 (integers). Results are compared per dataset, not pooled.
