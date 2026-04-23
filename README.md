# Insider Threat Detection Using Machine Learning

Unsupervised insider threat detection system using Isolation Forest on user activity logs. Features personalised user-delta baselines, a 17-feature engineering pipeline, and an interactive Streamlit dashboard.

**Course:** CS 484 — Introduction to Machine Learning

---

## Overview

Insider threats account for roughly 60% of cyber breaches and cost organisations an average of $11.45M per incident. Unlike external attackers, insiders operate with legitimate access — making them extraordinarily difficult to detect with rule-based systems, and impossible to catch with supervised ML because labelled examples of malicious behaviour are almost never available in real deployments.

This project takes an unsupervised approach: it learns what normal user activity looks like from unlabelled logs and flags deviations. The system uses **Isolation Forest** combined with personalised feature engineering that compares each user to their own historical baseline rather than to a global threshold.

---

## Key Features

- **Fully unsupervised detection** — no labels required
- **17 engineered features** across three groups: raw, time-based, and user-delta
- **Personalised baselines** — flags Bob accessing 160 files (avg: 8) while ignoring Alice accessing the same 160 (avg: 150)
- **Three-tier risk classification** — High (≥92nd percentile), Medium (≥75th), Low (below)
- **Dual interface** — command-line for batch processing and Streamlit dashboard for interactive analysis
- **Reproducible synthetic dataset** generator for demos and evaluation

---

## Results

On a synthetic CERT-style dataset of 2,500 records with 8% anomaly ratio:

| Metric | Score |
|---|---|
| Accuracy | 0.964 |
| Precision | 0.775 |
| Recall | 0.775 |
| F1-Score | 0.775 |

**Strongest feature (by Spearman correlation):** `file_access_user_delta`
**Off-hours concentration:** 80% of flagged anomalies occurred outside standard business hours

---

## Installation

```bash
git clone https://github.com/YOUR-USERNAME/insider-threat-detection-ml.git
cd insider-threat-detection-ml
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, numpy, pandas, scikit-learn, matplotlib, streamlit

---

## Usage

### CLI mode (batch processing)

Run with the built-in synthetic dataset:

```bash
python insider_threat_detection.py
```

Run with your own CERT-style CSV:

```bash
python insider_threat_detection.py --dataset path/to/activity_logs.csv
```

Customise the expected anomaly rate:

```bash
python insider_threat_detection.py --contamination 0.05
```

Skip the interactive plot window (useful for CI):

```bash
python insider_threat_detection.py --no-show
```

### Streamlit dashboard

```bash
streamlit run insider_threat_detection.py
```

The dashboard opens in your browser and provides:
- CSV upload with chunked loading for large files
- Adjustable contamination slider
- Risk-level filtering (High / Medium / Low)
- Three live visualisations (histogram, scatter, time trend)
- Top 20 suspicious users table
- Downloadable scored results CSV

---

## Expected Input Format

The system accepts any CSV with user activity records. Expected columns (alternative names are auto-mapped):

| Field | Type | Aliases accepted |
|---|---|---|
| `user_id` | string | user, username, employee, actor |
| `activity_date` | datetime | date, day, timestamp, datetime |
| `login_hour` | int (0–23) | hour, logon_hour |
| `file_access_count` | int | file_count, files_accessed |
| `email_sent_count` | int | emails_sent, email_count |
| `device_used` | string | device, pc, machine, device_type |
| `session_duration_minutes` | float | — |
| `failed_login_attempts` | int | — |
| `anomaly_label` | int (optional) | label, malicious, target |

If `anomaly_label` is present, the system computes precision/recall/F1 against it. If not, it returns detection statistics only.

---

## Outputs

Running the pipeline produces the following files in the working directory:

- `insider_threat_results.csv` — full scored output with anomaly score, predicted label, and risk tier per record
- `top_suspicious_users.csv` — top 10 highest-risk users summary
- `anomaly_scores_histogram.png` — score distribution with threshold
- `normal_vs_anomaly_scatter.png` — login hour vs file access visualisation
- `time_based_anomaly_trend.png` — daily anomaly counts with spike annotations

---

## Project Structure

```
insider-threat-detection-ml/
├── insider_threat_detection.py   # Single-file implementation (CLI + Streamlit)
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── insider_threat_results.csv    # Sample output (scored records)
├── top_suspicious_users.csv      # Sample output (ranked users)
├── anomaly_scores_histogram.png  # Sample visualisation
├── normal_vs_anomaly_scatter.png # Sample visualisation
└── time_based_anomaly_trend.png  # Sample visualisation
```

---

## Methodology Summary

1. **Preprocessing** — Mean imputation for numeric fields, mode for categorical, label encoding for `device_used`
2. **Feature engineering** — Derive 17 features: 6 raw + 3 time-based + 8 user-delta (personalised baselines)
3. **Training** — Isolation Forest with `n_estimators=250`, `contamination=0.08`, `random_state=42`
4. **Scoring** — `anomaly_score = -decision_function(x)` (higher = more anomalous)
5. **Classification** — Threshold at 92nd percentile → High; 75th–92nd → Medium; below → Low

---

## Limitations

- Evaluated on synthetic data only; real-world performance may differ
- Contamination parameter assumed at 8% rather than estimated from data
- Isolation Forest has no native feature importance; Spearman correlation used as a proxy
- User baselines computed on the full dataset, which introduces mild data leakage

---

## Future Work

- Validate on the real CERT r4.2 benchmark (1,000 users, 500 days, 70 confirmed insider threats)
- Add SHAP values for principled, model-agnostic feature attribution
- Explore LSTM or GRU networks to capture sequential behavioural patterns

---

## Author

Dhanalakshmi Sathyanarayanan 

## License

Academic project — for educational and research use.
