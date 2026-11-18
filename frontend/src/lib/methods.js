// Display metadata for every method that appears in the result tables.
export const METHODS = {
  'Global mean': { short: 'Global mean', color: 'var(--m-global)', kind: 'baseline',
    about: 'Predicts the average rating of all training ratings for every movie.' },
  'Mean rating (min 50 ratings)': { short: 'Item mean', color: 'var(--m-mean)', kind: 'baseline',
    about: 'Predicts each movie’s average rating (movies with ≥50 ratings). Ranking = highest average first.' },
  'Popularity (most rated)': { short: 'Popularity', color: 'var(--m-pop)', kind: 'baseline',
    about: 'Recommends the most-rated movies the user has not seen. The same list for everyone.' },
  'Item-based CF': { short: 'Item CF', color: 'var(--m-item)', kind: 'cf',
    about: 'Recommends movies similar to the ones the user rated, using cosine similarity between movies.' },
  'User-based CF': { short: 'User CF', color: 'var(--m-user)', kind: 'cf',
    about: 'Finds users with similar rating patterns and recommends what they liked.' },
  'MF (plain)': { short: 'MF (plain)', color: 'var(--m-mfp)', kind: 'mf',
    about: 'Matrix factorization R ≈ P·Qᵀ trained on rating error with SGD.' },
  'MF (biased)': { short: 'MF (biased)', color: 'var(--m-mf)', kind: 'mf',
    about: 'Matrix factorization with global, user and item bias terms. Best rating predictor.' },
  'One-class WRMF': { short: 'One-class WRMF', color: 'var(--m-wrmf)', kind: 'oc',
    about: 'Treats ratings ≥ 4 as positives and everything else as unknown; weighted ALS factorization for ranking.' },
  'One-class dual-regularized': { short: 'One-class dual', color: 'var(--m-dual)', kind: 'oc',
    about: 'One-class factorization plus user–user and item–item similarity-graph regularizers (own implementation of the idea in Yao et al., CIKM 2014).' },
}

export const METHOD_ORDER = Object.keys(METHODS)
export const short = (m) => METHODS[m]?.short ?? m
export const color = (m) => METHODS[m]?.color ?? 'var(--primary)'

export const METRICS = {
  rmse: { label: 'RMSE', better: 'lower', digits: 3, help: 'Root-mean-square error of predicted star ratings.' },
  mae: { label: 'MAE', better: 'lower', digits: 3, help: 'Mean absolute error of predicted star ratings.' },
  'ndcg@10': { label: 'NDCG@10', better: 'higher', digits: 3, help: 'Ranking quality of the top-10 list; rewards liked movies near the top.' },
  'hit@10': { label: 'Hit rate@10', better: 'higher', digits: 3, help: 'Share of users with at least one liked movie in their top 10.' },
  'precision@10': { label: 'Precision@10', better: 'higher', digits: 3, help: 'Share of the top 10 that the user liked.' },
  'recall@10': { label: 'Recall@10', better: 'higher', digits: 3, help: 'Share of the user’s liked held-out movies that appear in the top 10.' },
}
export const metricLabel = (m) => METRICS[m]?.label ?? m

export const MODEL_KEYS = {
  popularity: { label: 'Popularity', color: 'var(--m-pop)', about: METHODS['Popularity (most rated)'].about },
  item_cf: { label: 'Item CF', color: 'var(--m-item)', about: METHODS['Item-based CF'].about },
  user_cf: { label: 'User CF', color: 'var(--m-user)', about: METHODS['User-based CF'].about },
  mf: { label: 'Matrix factorization', color: 'var(--m-mf)', about: METHODS['MF (biased)'].about },
  one_class: { label: 'One-class CF', color: 'var(--m-dual)', about: METHODS['One-class dual-regularized'].about },
}
