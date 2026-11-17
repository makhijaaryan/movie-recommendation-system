Top-10 list diagnostics (seed 42; frozen ranking configurations; fixed random sample of eligible users per dataset; already-rated items excluded). Popularity percentile: 0 = most-rated item, 1 = least-rated; 'Less-rated-half' = items below the median in number of training ratings.

**MovieLens 25M** (10,000 users)

| Method | Coverage | Gini | Popularity pct. | Median #ratings | Top-1% share | Less-rated-half share | Novelty (bits) | Avg item rating | Genres/list | NDCG@10 |
|:--|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Item-based CF | 0.190 | 0.977 | 0.023 | 25,745 | 0.650 | 0.003 | 9.89 | 4.00 | 9.97 | 0.1870 |
| User-based CF | 0.054 | 0.993 | 0.007 | 38,609 | 0.831 | 0.001 | 9.25 | 4.08 | 10.67 | 0.2209 |
| Mean rating (min 50 ratings) | 0.005 | 0.999 | 0.077 | 13,397 | 0.430 | 0.000 | 10.94 | 4.33 | 8.68 | 0.0441 |
| MF (biased) | 0.163 | 0.980 | 0.220 | 2,056 | 0.246 | 0.156 | 12.79 | 4.14 | 8.72 | 0.0603 |
| MF (plain) | 0.167 | 0.980 | 0.331 | 1,118 | 0.187 | 0.290 | 13.76 | 4.11 | 7.99 | 0.0561 |
| One-class dual-regularized | 0.094 | 0.979 | 0.013 | 24,040 | 0.604 | 0.000 | 9.77 | 3.91 | 10.82 | 0.2953 |
| One-class WRMF | 0.092 | 0.978 | 0.014 | 22,609 | 0.576 | 0.000 | 9.83 | 3.89 | 10.85 | 0.2882 |
| Popularity (most rated) | 0.008 | 0.998 | 0.001 | 47,988 | 1.000 | 0.000 | 8.63 | 4.08 | 11.70 | 0.1132 |

**Netflix Prize (50,000-user sample)** (10,000 users)

| Method | Coverage | Gini | Popularity pct. | Median #ratings | Top-1% share | Less-rated-half share | Novelty (bits) | Avg item rating | NDCG@10 |
|:--|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Item-based CF | 0.314 | 0.948 | 0.073 | 7,412 | 0.269 | 0.020 | 10.84 | 4.11 | 0.1728 |
| User-based CF | 0.064 | 0.992 | 0.013 | 12,349 | 0.619 | 0.000 | 9.73 | 4.24 | 0.1690 |
| Mean rating (min 50 ratings) | 0.005 | 0.999 | 0.036 | 7,998 | 0.251 | 0.000 | 10.39 | 4.57 | 0.0411 |
| MF (biased) | 0.136 | 0.985 | 0.234 | 1,133 | 0.106 | 0.187 | 12.81 | 4.43 | 0.0476 |
| MF (plain) | 0.145 | 0.983 | 0.272 | 1,034 | 0.103 | 0.239 | 13.08 | 4.39 | 0.0499 |
| One-class dual-regularized | 0.127 | 0.973 | 0.018 | 10,856 | 0.513 | 0.000 | 9.87 | 3.88 | 0.2705 |
| One-class WRMF | 0.129 | 0.971 | 0.019 | 10,704 | 0.485 | 0.000 | 9.91 | 3.88 | 0.2718 |
| Popularity (most rated) | 0.011 | 0.998 | 0.001 | 16,523 | 1.000 | 0.000 | 9.12 | 3.75 | 0.0889 |
