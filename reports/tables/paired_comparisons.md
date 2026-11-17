Paired differences (A − B); the same resampled users are used for both methods. 'Real' means the 95% interval excludes 0. For RMSE a negative difference favours A.

**MovieLens 25M**

| A | B | Metric | A | B | A − B [95% CI] | Real? |
|:--|:--|:--|---:|---:|---:|:--:|
| User-based CF | Item-based CF | NDCG@10 | 0.2160 | 0.1842 | 0.0318 [0.0308, 0.0328] | yes |
| User-based CF | Item-based CF | Hit@10 | 0.6020 | 0.5421 | 0.0598 [0.0575, 0.0623] | yes |
| User-based CF | Item-based CF | RMSE | 0.8229 | 0.8125 | 0.0105 [0.0098, 0.0112] | yes |
| One-class dual-regularized | One-class WRMF | NDCG@10 | 0.2914 | 0.2852 | 0.0062 [0.0058, 0.0066] | yes |
| One-class WRMF | User-based CF | NDCG@10 | 0.2852 | 0.2160 | 0.0692 [0.0681, 0.0702] | yes |
| User-based CF | Popularity (most rated) | NDCG@10 | 0.2160 | 0.1105 | 0.1055 [0.1045, 0.1065] | yes |
| MF (biased) | MF (plain) | RMSE | 0.7539 | 0.7568 | -0.0029 [-0.0032, -0.0026] | yes |
| MF (biased) | Item-based CF | RMSE | 0.7539 | 0.8125 | -0.0586 [-0.0591, -0.0580] | yes |
| MF (biased) | Popularity (most rated) | NDCG@10 | 0.0579 | 0.1105 | -0.0526 [-0.0536, -0.0517] | yes |

**Netflix Prize (50,000-user sample)**

| A | B | Metric | A | B | A − B [95% CI] | Real? |
|:--|:--|:--|---:|---:|---:|:--:|
| Item-based CF | User-based CF | NDCG@10 | 0.1711 | 0.1664 | 0.0047 [0.0030, 0.0062] | yes |
| Item-based CF | User-based CF | Hit@10 | 0.5589 | 0.5387 | 0.0203 [0.0162, 0.0242] | yes |
| Item-based CF | User-based CF | RMSE | 0.8723 | 0.8830 | -0.0107 [-0.0117, -0.0097] | yes |
| One-class WRMF | One-class dual-regularized | NDCG@10 | 0.2709 | 0.2691 | 0.0018 [0.0012, 0.0025] | yes |
| One-class WRMF | Item-based CF | NDCG@10 | 0.2709 | 0.1711 | 0.0998 [0.0980, 0.1019] | yes |
| MF (plain) | MF (biased) | RMSE | 0.8241 | 0.8253 | -0.0012 [-0.0015, -0.0009] | yes |
| MF (plain) | Item-based CF | RMSE | 0.8241 | 0.8723 | -0.0482 [-0.0491, -0.0474] | yes |
| Item-based CF | Popularity (most rated) | NDCG@10 | 0.1711 | 0.0889 | 0.0821 [0.0801, 0.0840] | yes |
