# CineMatch frontend

React (Vite) dashboard for the CSE575 recommender project. Four pages (the sidebar order is Dashboard, Overview, Get Recommendations, Explore analyses):

* **Dashboard** (landing page): a product-style home page with sample taste profiles, popular movies and a short explanation of how it works.
* **Overview**: method comparison with 95% confidence intervals, accuracy-vs-ranking trade-off, full table, head-to-head tests.
* **Get recommendations**: rate movies (search, popular picks or sample profiles) and see five models side by side (live, MovieLens 25M).
* **Explore analyses**: history size, simulated cold start, list quality, training curves, report figures.

Light theme by default, dark theme via the toggle in the top bar (remembered in the browser).

## Run

```bash
# 1. once: export the two factorization models the live page serves (about 3 minutes)
python -m src.export_models

# 2. backend (from the project root)
uvicorn src.api:app --port 8000

# 3. frontend (in a second terminal)
cd frontend && npm install && npm run dev      # http://localhost:5173
```

The dev server proxies `/api` and `/figures` to the backend on port 8000. The result pages read `reports/tables/*.csv`
and work immediately; the recommendation page becomes available once the status pill reads “Models ready”.
Production build: `npm run build` (output in `dist/`).
