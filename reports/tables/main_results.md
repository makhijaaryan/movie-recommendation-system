**MovieLens 25M** — ranking metrics over 152,012 users

| Method | RMSE | MAE | P@10 | R@10 | Hit@10 | NDCG@10 |
|:--|---:|---:|---:|---:|---:|---:|
| Global mean | 1.0582 | 0.8397 | – | – | – | – |
| Mean rating (min 50 ratings) | 0.9568 | 0.7397 | 0.0265 | 0.0433 | 0.2075 | 0.0418 |
| Popularity (most rated) | – | – | 0.0676 | 0.1070 | 0.4060 | 0.1105 |
| Item-based CF | 0.8125 | 0.6098 | 0.1084 | 0.1652 | 0.5421 | 0.1842 |
| User-based CF | 0.8229 | 0.6189 | 0.1206 | 0.2100 | 0.6020 | 0.2160 |
| MF (plain) | 0.7568 | 0.5719 | 0.0351 | 0.0484 | 0.2367 | 0.0543 |
| MF (biased) | **0.7539** | **0.5690** | 0.0370 | 0.0542 | 0.2564 | 0.0579 |
| One-class WRMF | – | – | 0.1627 | 0.2783 | 0.6993 | 0.2852 |
| One-class dual-regularized | – | – | **0.1658** | **0.2834** | **0.7041** | **0.2914** |

**Netflix Prize (50,000-user sample)** — ranking metrics over 48,264 users

| Method | RMSE | MAE | P@10 | R@10 | Hit@10 | NDCG@10 |
|:--|---:|---:|---:|---:|---:|---:|
| Global mean | 1.0813 | 0.9060 | – | – | – | – |
| Mean rating (min 50 ratings) | 1.0095 | 0.8098 | 0.0321 | 0.0216 | 0.2197 | 0.0408 |
| Popularity (most rated) | – | – | 0.0678 | 0.0649 | 0.3782 | 0.0889 |
| Item-based CF | 0.8723 | 0.6692 | 0.1246 | 0.1157 | 0.5589 | 0.1711 |
| User-based CF | 0.8830 | 0.6793 | 0.1202 | 0.1100 | 0.5387 | 0.1664 |
| MF (plain) | **0.8241** | 0.6395 | 0.0381 | 0.0265 | 0.2308 | 0.0486 |
| MF (biased) | 0.8253 | **0.6384** | 0.0368 | 0.0246 | 0.2279 | 0.0466 |
| One-class WRMF | – | – | **0.1985** | **0.1954** | **0.7076** | **0.2709** |
| One-class dual-regularized | – | – | 0.1970 | 0.1916 | 0.7012 | 0.2691 |

RMSE and MAE come from each method's configuration with the best *validation* RMSE; ranking metrics (K = 10 here; relevant = held-out rating ≥ 4; already-rated items excluded) come from its configuration with the best *validation* NDCG@10. Matrix factorization and one-class CF are trained for different objectives, hence two settings. One-class CF predicts no star ratings. Best value per column in bold. All numbers are single runs (seed 42) on the held-out test split.
