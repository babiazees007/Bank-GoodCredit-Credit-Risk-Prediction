# Bank GoodCredit – Credit Risk Prediction 

A machine learning project that predicts which credit card customers are likely to fall **30 or more days past due (30 DPD+)**, so the bank can act early and reduce credit losses.

The project is delivered as:

- a single Jupyter notebook containing the full analysis (Sections 1–15), and
- a static project webpage (`index.html`) that presents the methodology and the results.

---

## Contents

```
.
├── Bank_GoodCredit_Credit_Risk_Prediction_PR0015.ipynb   # Full project notebook (Sections 1–15)
├── index.html                                            # Project webpage with embedded results
├── README.md                                             # This file
└── data/                                                 # Input data (not included, see below)
    ├── Cust_Account.csv
    ├── Cust_Enquiry.csv
    └── Cust_Demographics.csv
```

---

## The Business Problem

Bank GoodCredit wants to identify existing credit card customers with a higher chance of poor credit behaviour.

**Target variable: `Bad_label`**

| Value | Meaning |
|---|---|
| 0 | Good credit history – pays on time |
| 1 | Bad credit history – 30+ days past due (30 DPD+) |

**Benchmark:** The business case sets a benchmark **Gini of 37.9**.

**Gini** measures how well a model separates good from bad customers:

```
Gini = (2 × ROC-AUC − 1) × 100
```

0 means random guessing; 100 means perfect separation.

---

## Data

Three datasets, linked by `customer_no`:

| Dataset | Content | Granularity |
|---|---|---|
| `Cust_Account` | Account and payment history | Many rows per customer |
| `Cust_Enquiry` | Credit enquiry history | Many rows per customer |
| `Cust_Demographics` | Anonymised features (`feature_1`–`feature_79`) and `Bad_label` | One row per customer |

The raw CSV files are **not** included in this repository. Place them in a `data/` folder next to the notebook before running it.

---

## Method Overview

| Section | Step | Key decisions |
|---|---|---|
| 1 | Introduction | Explains credit risk, the target, and the Gini metric |
| 2 | Libraries | pandas, numpy, matplotlib, seaborn, scikit-learn, XGBoost, LightGBM |
| 3 | Data loading | Shapes, columns, types, missing-value summaries |
| 4 | Cleaning | Date conversion, numeric conversion, duplicate checks, high-missingness flags (>95%) |
| 5 | EDA | Class imbalance, feature distributions, correlations, missing-value patterns |
| 6 | Integration | Accounts and enquiries summarised to one row per customer, then left-joined; reference date applied to prevent leakage |
| 7 | Feature engineering | Business-case features built where the data supports them (see below) |
| 8 | Feature selection | Mutual information and LightGBM gain, computed on training data only; top 40 kept |
| 9 | Preparation | Stratified 70/30 split; median imputation and scaling inside pipelines |
| 10 | Models | Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM, compared by 5-fold CV ROC-AUC |
| 11 | Tuning | RandomizedSearchCV (20 combinations, 3-fold CV, ROC-AUC) on LightGBM |
| 12 | Evaluation | Accuracy, precision, recall, F1, ROC-AUC, Gini, confusion matrix, ROC curves, benchmark comparison |
| 13 | Rank ordering | Test customers split into 10 deciles by predicted risk (decile 1 = highest risk) |
| 14 | Interpretation | Feature importance, optional SHAP |
| 15 | Conclusion | Summary printed from the run, plus business recommendations and limitations |

### Leakage protection

- Only account and enquiry records on or before a **reference date** are used.
- Feature selection uses the **training split only**; the test set is held back until Section 12.
- Imputation and scaling sit inside `Pipeline` objects, so they are learned from training folds only.

### Features

**Built (9 engineered features):**

- `total_diff_lastpaymt_opened_dt`, `mean_diff_lastpaymt_opened_dt`
- `payment_history_avg_dpd_0_29_bucket`, `payment_history_mean_length`
- `Ratio_currbalance_creditlimit`
- `count_enquiry_recency_90`, `count_enquiry_recency_365`
- `mean_diff_open_enquiry_dt`, `max_freq_enquiry`

Two of these (`mean_diff_open_enquiry_dt`, `max_freq_enquiry`) are our best reading of the business-case definitions and should be checked against the business-case PDF.

**Skipped (cannot be calculated from the data provided):**

- `min_months_last_30_plus` – the month order inside the payment-history string is not confirmed.
- `utilisation_trend` – requires several balance readings over time; the data has one per account.
- `perc_unsecured_others` – requires an account-type to secured/unsecured mapping that was not provided.

---

## Results

Results are from one run of the notebook on the provided data.

| Item | Value |
|---|---|
| Customers | 23,896 |
| Reference date | 2015-12-20 |
| Test bad rate | 4.20% |
| Selected model (highest CV ROC-AUC) | Random Forest |
| Test ROC-AUC (selected model) | 0.672 |
| **Test Gini (selected model)** | **34.34** |
| Benchmark Gini | 37.9 |
| Difference | −3.56 points (below benchmark) |

### Test-set comparison

| Model | Accuracy | Precision (bad) | Recall (bad) | F1 (bad) | ROC-AUC | Gini (%) |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.615 | 0.068 | 0.648 | 0.124 | 0.681 | 36.25 |
| LightGBM (tuned) | 0.731 | 0.079 | 0.508 | 0.137 | 0.676 | 35.20 |
| Random Forest (selected) | 0.950 | 0.108 | 0.027 | 0.043 | 0.672 | 34.34 |
| LightGBM | 0.921 | 0.088 | 0.093 | 0.090 | 0.650 | 30.03 |
| XGBoost | 0.792 | 0.079 | 0.369 | 0.130 | 0.647 | 29.43 |
| Decision Tree | 0.536 | 0.054 | 0.608 | 0.099 | 0.575 | 14.95 |

### Decile analysis (test set)

- Decile 1 contains about **9.5%** bad customers (lift 2.26×), against an overall rate of about 4.2%.
- Deciles 1–3 together capture about **54%** of all bad customers in the test set.
- The bad rate falls across deciles overall, but not strictly: decile 9 (2.23%) is slightly higher than decile 8 (1.67%).

### Important caveats

1. **The model is below the benchmark.** The test Gini of 34.34 is about 3.6 points under 37.9. The benchmark may come from a different population or method, so the comparison is indicative only.
2. **The selected model catches few bad customers.** Random Forest was chosen on cross-validation ROC-AUC. Its test recall for bad customers is only about 2.7%, while Logistic Regression catches about 65%. Since the business case stresses catching risky customers, the selection rule should be reviewed before the results are presented as final.
3. **Feature importance is not causation.** The model finds these features useful for prediction; it does not show that they cause default.

---

## How to Run the Notebook

**1. Install dependencies**

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm
pip install shap   # optional, for SHAP plots in Section 14
```

**2. Add the data**

Place `Cust_Account.csv`, `Cust_Enquiry.csv`, and `Cust_Demographics.csv` in a `data/` folder next to the notebook.

**3. Run the notebook**

Open the notebook in Jupyter and run all cells from the top in a fresh kernel (*Kernel → Restart & Run All*). The notebook uses fixed random seeds (`random_state=42`), so results should repeat on the same data.

**4. (Optional) Export results**

The final cell of Section 15 writes `results.json`. The webpage no longer needs this file, because its results are embedded directly, but it is useful if you want to reuse the numbers elsewhere.

---

## How to View the Webpage

Open `index.html` in any modern browser. No server, internet connection, or external libraries are needed.

The page includes:

- a project overview and pipeline summary,
- one section per notebook stage,
- headline metrics, a Gini gauge against the benchmark, the test-set table, and decile bars.

The results shown on the page are the ones listed above. If you re-run the notebook and get different numbers, update the `RESULTS` object in `index.html` to match.

---

## Limitations

- Most demographic features are **anonymised**, so their real-world meaning is unknown.
- Three business-case features could not be built from the provided data.
- The data is a **single snapshot**, so model stability over time is not tested.
- The split is a stratified random split, not a time-based split, because reliable per-customer observation dates were not available.
- The classification threshold is the default **0.5**, not one chosen from the bank's cost of missed risky customers versus false alarms.
- The benchmark's population and method are not documented in the data we received.

---

## Future Improvements

- Add a time-based validation split once reliable observation dates are available.
- Build the three skipped features if their definitions and data become available.
- Choose the model and classification threshold with the business's recall and cost targets in mind.
- Calibrate predicted probabilities so that a score of 0.10 corresponds to a 10% chance of being bad.
- Add per-customer SHAP explanations for credit staff.
- Monitor performance over time and retrain on new data.

---

## Technologies

Python · pandas · NumPy · scikit-learn · XGBoost · LightGBM · Matplotlib · Seaborn · (optional) SHAP · HTML / CSS / JavaScript

---

## Author

Prepared as part of Intership at Rubixe - AI for solutions company (Bank GoodCredit – Credit Risk Prediction).
