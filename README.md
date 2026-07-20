# Term Deposit Subscription Prediction (UCI Bank Marketing)

> A seven-model comparison in R predicting which bank customers will subscribe to a term deposit — built around leakage avoidance, disciplined feature engineering, and choosing the simplest model that meets the bar.

![R](https://img.shields.io/badge/R-caret-blue) ![License: MIT](https://img.shields.io/badge/License-MIT-green)

## Overview
A Portuguese bank's telemarketing campaign: predict **before calling** which customers will subscribe. Two traps define the problem — severe class imbalance (~11% "yes") and a leakage variable that makes naive models look artificially strong.

- **Dataset:** [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing) (`bank-additional-full.csv`) — 41,188 records, 20 predictors, ~11.3% positive.

## Headline Finding
A **well-preprocessed logistic regression matched the non-linear models** (neural net, KNN, FDA, …) — so the interpretable, stable model was chosen. Non-linearity bought only marginal gains.

## Key Results (cross-validated ROC-AUC)
| Model | CV ROC-AUC |
|---|---|
| **Logistic Regression** | **0.7905** |
| Penalized Logistic (GLMNet) | 0.7901 |
| Neural Net | 0.7886 |
| LDA | 0.7866 |
| PLSDA | 0.7863 |
| FDA | 0.7747 |
| KNN | 0.7710 |

Final logistic model on the held-out test set: ~82% accuracy, 0.72 balanced accuracy, with probability-threshold tuning mapped to campaign cost trade-offs.

## Approach & Key Decisions
- **Dropped `duration` (leakage control)** — known only after the call ends; useless for pre-call targeting.
- **Engineered `pdays` three ways** (`contacted_before`, `pdays_bucket`, `pdays_recency = 1/(pdays+1)`).
- **Treated `unknown` as its own category** — missingness carries signal in marketing.
- **Spatial-sign transform + center/scale** for outliers; **correlation filter (0.90)** dropped collinear macro indicators.
- **In-fold SMOTE** + **ROC-AUC** as the tuning metric over repeated 10-fold CV.

## Full Report
Complete methodology, all figures (EDA, transforms, model ROC curves, threshold sweep), and results are in **[the project report](report/term_deposit_report.pdf)**.

## Note on Methodology
Preprocessing was fit on the full dataset before the split (a course requirement); the report flags this as a limitation. In production, fit all preprocessing inside the CV pipeline on training folds only.

## Repository Structure
```
.
├── src/        # add the R pipeline script here
├── images/     # add key figures here (see report)
├── report/     # full write-up (PDF)
├── LICENSE
└── README.md
```

## License
MIT — see [LICENSE](LICENSE).
