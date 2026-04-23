"""
Insider Threat Detection Dashboard

Single-file anomaly detection project using Isolation Forest with:
- CLI mode
- Streamlit dashboard mode
- Synthetic fallback dataset
- Large CERT CSV upload support via chunked aggregation
"""

from __future__ import annotations

import argparse
import io
import os
import site
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

USER_SITE = site.getusersitepackages()
if USER_SITE and USER_SITE not in sys.path:
    sys.path.append(USER_SITE)

try:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import IsolationForest
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
except ImportError as exc:
    missing_module = exc.name or "required package"
    print(
        f"Missing dependency: {missing_module}\n"
        "Install requirements first:\n"
        "  python -m pip install -r requirements.txt\n"
    )
    sys.exit(1)


RANDOM_STATE = 42
RESULTS_CSV = "insider_threat_results.csv"
TOP_USERS_CSV = "top_suspicious_users.csv"
SCORES_HIST_PATH = "anomaly_scores_histogram.png"
SCATTER_PATH = "normal_vs_anomaly_scatter.png"
TREND_PATH = "time_based_anomaly_trend.png"
TOP_USERS_DISPLAY_COUNT = 10
UPLOAD_CHUNK_SIZE = 100000
MAX_CATEGORIES_PER_FEATURE = 25
MAX_SCATTER_POINTS = 15000

NUMERIC_FEATURES = [
    "login_hour",
    "file_access_count",
    "email_sent_count",
    "session_duration_minutes",
    "failed_login_attempts",
    "is_weekend",
    "off_hours_activity",
    "login_hour_deviation",
    "daily_activity_volume",
    "files_per_email_ratio",
    "user_avg_file_access",
    "user_avg_email_sent",
    "user_avg_login_hour",
    "user_activity_days",
    "file_access_user_delta",
    "email_sent_user_delta",
    "login_hour_user_delta",
]
CATEGORICAL_FEATURES = ["device_used"]

CERT_USEFUL_COLUMNS = {
    "user", "username", "employee", "employee_id", "actor", "user_id",
    "date", "day", "timestamp", "datetime", "event_date",
    "hour", "logon_hour", "login_time_hour",
    "file_count", "files_accessed", "file_accesses", "file_access_count",
    "emails_sent", "email_count", "email_activity", "email_sent_count",
    "device", "pc", "machine", "device_type", "device_used",
    "label", "malicious", "insider_threat", "is_malicious", "target",
    "attachments", "size", "content_length",
    "activity",
}


@dataclass
class DetectionOutput:
    """Container for pipeline outputs."""

    raw_df: pd.DataFrame
    processed_df: pd.DataFrame
    predictions_df: pd.DataFrame
    top_users_df: pd.DataFrame
    feature_importance_df: pd.DataFrame
    metrics: Dict[str, float]
    threshold: float
    figures: Dict[str, plt.Figure]


def generate_synthetic_dataset(n_samples: int = 2500, anomaly_ratio: float = 0.08) -> pd.DataFrame:
    """Generate a synthetic insider-threat dataset for demo/testing."""
    rng = np.random.default_rng(RANDOM_STATE)
    n_anomalies = int(round(n_samples * anomaly_ratio))
    n_normal = n_samples - n_anomalies
    assert n_normal + n_anomalies == n_samples, "Synthetic dataset row count mismatch."

    user_ids = [f"U{idx:04d}" for idx in rng.integers(1, 350, size=n_samples)]
    activity_dates = pd.to_datetime("2025-01-01") + pd.to_timedelta(rng.integers(0, 90, size=n_samples), unit="D")

    normal_df = pd.DataFrame(
        {
            "user_id": user_ids[:n_normal],
            "activity_date": activity_dates[:n_normal],
            "login_hour": np.clip(rng.normal(9.5, 2.0, n_normal).round(), 0, 23).astype(int),
            "file_access_count": rng.poisson(22, n_normal),
            "email_sent_count": rng.poisson(14, n_normal),
            "device_used": rng.choice(["laptop", "desktop", "vpn"], size=n_normal, p=[0.55, 0.35, 0.10]),
            "session_duration_minutes": np.clip(rng.normal(180, 45, n_normal), 10, 600).round(1),
            "failed_login_attempts": rng.poisson(0.4, n_normal),
            "anomaly_label": 0,
        }
    )
    noisy_normal_mask = rng.random(n_normal) < 0.14
    normal_df.loc[noisy_normal_mask, "login_hour"] = rng.choice([5, 6, 7, 18, 19, 20, 21], size=noisy_normal_mask.sum())
    normal_df.loc[noisy_normal_mask, "file_access_count"] += rng.integers(12, 42, size=noisy_normal_mask.sum())
    normal_df.loc[noisy_normal_mask, "email_sent_count"] += rng.integers(6, 30, size=noisy_normal_mask.sum())
    normal_df.loc[noisy_normal_mask, "session_duration_minutes"] += rng.integers(50, 210, size=noisy_normal_mask.sum())
    normal_df.loc[noisy_normal_mask, "failed_login_attempts"] += rng.integers(0, 5, size=noisy_normal_mask.sum())

    anomaly_df = pd.DataFrame(
        {
            "user_id": user_ids[n_normal:],
            "activity_date": activity_dates[n_normal:],
            "login_hour": rng.choice([5, 6, 7, 9, 10, 11, 16, 19, 20, 21], size=n_anomalies, p=[0.13, 0.14, 0.14, 0.04, 0.04, 0.04, 0.04, 0.17, 0.17, 0.09]),
            "file_access_count": rng.integers(44, 92, n_anomalies),
            "email_sent_count": rng.integers(26, 70, n_anomalies),
            "device_used": rng.choice(["laptop", "desktop", "vpn", "usb", "unknown"], size=n_anomalies, p=[0.24, 0.12, 0.32, 0.23, 0.09]),
            "session_duration_minutes": rng.integers(260, 620, n_anomalies),
            "failed_login_attempts": rng.integers(1, 7, n_anomalies),
            "anomaly_label": 1,
        }
    )

    df = pd.concat([normal_df, anomaly_df], ignore_index=True).sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    df["is_anomaly"] = df["anomaly_label"]
    print(f"Injected anomalies in dataset: {df['is_anomaly'].sum()}")
    for col in ["device_used", "email_sent_count", "session_duration_minutes"]:
        mask = rng.random(df.shape[0]) < 0.02
        df.loc[mask, col] = np.nan
    return df


def load_dataset(dataset_path: Optional[str] = None) -> pd.DataFrame:
    """Load CSV or return synthetic fallback."""
    if dataset_path:
        try:
            df = pd.read_csv(dataset_path)
            print(f"Loaded dataset from: {dataset_path}")
            return df
        except Exception as exc:
            print(f"Could not load dataset '{dataset_path}' ({exc}). Falling back to synthetic data.")

    print("Using generated synthetic insider-threat dataset.")
    return generate_synthetic_dataset()


def _standardize_input_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map common CERT-style column names to the internal schema."""
    df = df.copy()
    lowered = {col.lower(): col for col in df.columns}
    candidates = {
        "user_id": ["user", "username", "employee", "employee_id", "actor"],
        "activity_date": ["date", "day", "timestamp", "datetime", "event_date"],
        "login_hour": ["hour", "logon_hour", "login_time_hour"],
        "file_access_count": ["file_count", "files_accessed", "file_accesses"],
        "email_sent_count": ["emails_sent", "email_count", "email_activity"],
        "device_used": ["device", "pc", "machine", "device_type"],
        "anomaly_label": ["label", "malicious", "insider_threat", "is_malicious", "target"],
    }

    rename_map = {}
    for canonical, aliases in candidates.items():
        if canonical in df.columns:
            continue
        for alias in aliases:
            if alias in lowered:
                rename_map[lowered[alias]] = canonical
                break
    if rename_map:
        df = df.rename(columns=rename_map)

    if "activity_date" in df.columns:
        df["activity_date"] = pd.to_datetime(df["activity_date"], errors="coerce")
    else:
        df["activity_date"] = pd.date_range("2025-01-01", periods=len(df), freq="h")

    if "login_hour" not in df.columns:
        df["login_hour"] = df["activity_date"].dt.hour.fillna(12)

    if "user_id" not in df.columns:
        df["user_id"] = [f"U{idx:04d}" for idx in range(len(df))]

    defaults = {
        "file_access_count": 0,
        "email_sent_count": 0,
        "device_used": "unknown",
        "session_duration_minutes": np.nan,
        "failed_login_attempts": np.nan,
    }
    for col, default_value in defaults.items():
        if col not in df.columns:
            df[col] = default_value

    return df


def _read_uploaded_header(uploaded_file) -> List[str]:
    """Read the header from a Streamlit uploaded CSV."""
    uploaded_file.seek(0)
    header_bytes = uploaded_file.read(16384)
    uploaded_file.seek(0)
    text = header_bytes.decode("utf-8", errors="ignore") if isinstance(header_bytes, bytes) else str(header_bytes)
    first_line = text.splitlines()[0] if text.splitlines() else ""
    return [column.strip() for column in first_line.split(",") if column.strip()]


def _select_useful_columns(columns: List[str]) -> List[str]:
    """Choose only relevant columns from large CERT CSV uploads."""
    selected = [col for col in columns if col.lower() in CERT_USEFUL_COLUMNS]
    return selected if selected else columns[: min(12, len(columns))]


def _aggregate_activity_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Aggregate event-level logs into user/day activity records."""
    chunk = _standardize_input_columns(chunk)
    chunk = chunk.copy().dropna(subset=["activity_date"])
    if chunk.empty:
        return pd.DataFrame()

    chunk["activity_day"] = chunk["activity_date"].dt.floor("D")
    chunk["device_used"] = chunk["device_used"].fillna("unknown").astype(str)
    chunk["login_hour"] = pd.to_numeric(chunk["login_hour"], errors="coerce").fillna(chunk["activity_date"].dt.hour).astype(int)

    if "email_sent_count" not in chunk.columns or chunk["email_sent_count"].isna().all():
        chunk["email_sent_count"] = 1
    else:
        chunk["email_sent_count"] = pd.to_numeric(chunk["email_sent_count"], errors="coerce").fillna(1)

    if "file_access_count" not in chunk.columns or chunk["file_access_count"].isna().all():
        if "attachments" in chunk.columns:
            chunk["file_access_count"] = pd.to_numeric(chunk["attachments"], errors="coerce").fillna(0)
        else:
            chunk["file_access_count"] = 0
    else:
        chunk["file_access_count"] = pd.to_numeric(chunk["file_access_count"], errors="coerce").fillna(0)

    if "activity" in chunk.columns:
        chunk["failed_login_attempts"] = (chunk["activity"].str.lower() == "failed").astype(int)

    for col in ["session_duration_minutes", "failed_login_attempts", "anomaly_label"]:
        if col not in chunk.columns:
            chunk[col] = np.nan
        chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

    return chunk.groupby(["user_id", "activity_day"], as_index=False).agg(
        activity_date=("activity_day", "first"),
        login_hour=("login_hour", "median"),
        file_access_count=("file_access_count", "sum"),
        email_sent_count=("email_sent_count", "sum"),
        device_used=("device_used", lambda values: values.mode().iloc[0] if not values.mode().empty else "unknown"),
        session_duration_minutes=("session_duration_minutes", "mean"),
        failed_login_attempts=("failed_login_attempts", "sum"),
        anomaly_label=("anomaly_label", "max"),
    )


def load_uploaded_csv_optimized(uploaded_file) -> pd.DataFrame:
    """Read large uploaded CSV files in chunks and aggregate them safely."""
    header_columns = _read_uploaded_header(uploaded_file)
    usecols = _select_useful_columns(header_columns)
    dtype_map = {col: "string" for col in usecols}
    for numeric_name in ["attachments", "size", "label", "target", "malicious"]:
        exact = next((col for col in usecols if col.lower() == numeric_name), None)
        if exact:
            dtype_map[exact] = "float32"

    uploaded_file.seek(0)
    text_stream = io.TextIOWrapper(uploaded_file, encoding="utf-8", errors="ignore", newline="")
    chunk_frames: List[pd.DataFrame] = []
    try:
        for chunk in pd.read_csv(text_stream, usecols=usecols, chunksize=UPLOAD_CHUNK_SIZE, low_memory=True, dtype=dtype_map):
            aggregated_chunk = _aggregate_activity_chunk(chunk)
            if not aggregated_chunk.empty:
                chunk_frames.append(aggregated_chunk)
    finally:
        try:
            text_stream.detach()
        except Exception:
            pass
        uploaded_file.seek(0)

    if not chunk_frames:
        raise ValueError("No usable records were found in the uploaded CSV.")

    combined = pd.concat(chunk_frames, ignore_index=True)
    combined["activity_date"] = pd.to_datetime(combined["activity_date"], errors="coerce")
    combined["activity_day"] = combined["activity_date"].dt.floor("D")
    result = combined.groupby(["user_id", "activity_day"], as_index=False).agg(
        activity_date=("activity_date", "first"),
        login_hour=("login_hour", "median"),
        file_access_count=("file_access_count", "sum"),
        email_sent_count=("email_sent_count", "sum"),
        device_used=("device_used", lambda values: values.mode().iloc[0] if not values.mode().empty else "unknown"),
        session_duration_minutes=("session_duration_minutes", "mean"),
        failed_login_attempts=("failed_login_attempts", "sum"),
        anomaly_label=("anomaly_label", "max"),
    ).reset_index(drop=True)
    result["login_hour"] = result["login_hour"].round().astype(int)
    result = result.drop(columns=["activity_day"], errors="ignore")
    return result


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create behavior features used for anomaly detection."""
    df = _standardize_input_columns(df)
    df = df.copy()

    device_counts = df["device_used"].fillna("unknown").astype(str).value_counts()
    top_devices = set(device_counts.head(MAX_CATEGORIES_PER_FEATURE).index)
    normalized_devices = df["device_used"].fillna("unknown").astype(str)
    df["device_used"] = normalized_devices.where(normalized_devices.isin(top_devices), other="other_device")

    df["is_weekend"] = df["activity_date"].dt.dayofweek.isin([5, 6]).astype(int)
    df["off_hours_activity"] = ((df["login_hour"] < 8) | (df["login_hour"] > 18)).astype(int)
    df["login_hour_deviation"] = np.minimum(np.abs(df["login_hour"] - 9), np.abs(df["login_hour"] - 17))
    df["daily_activity_volume"] = df["file_access_count"].fillna(0) + df["email_sent_count"].fillna(0)
    df["files_per_email_ratio"] = df["file_access_count"].fillna(0) / (df["email_sent_count"].fillna(0) + 1.0)

    user_stats = df.groupby("user_id").agg(
        user_avg_file_access=("file_access_count", "mean"),
        user_avg_email_sent=("email_sent_count", "mean"),
        user_avg_login_hour=("login_hour", "mean"),
        user_activity_days=("activity_date", "nunique"),
    )
    df = df.merge(user_stats, on="user_id", how="left")
    df["file_access_user_delta"] = df["file_access_count"].fillna(0) - df["user_avg_file_access"].fillna(0)
    df["email_sent_user_delta"] = df["email_sent_count"].fillna(0) - df["user_avg_email_sent"].fillna(0)
    df["login_hour_user_delta"] = np.abs(df["login_hour"].fillna(12) - df["user_avg_login_hour"].fillna(12))
    return df


def build_preprocessor(feature_df: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """Build preprocessing pipeline."""
    numeric_features = [col for col in NUMERIC_FEATURES if col in feature_df.columns]
    categorical_features = [col for col in CATEGORICAL_FEATURES if col in feature_df.columns]

    try:
        encoder = OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            max_categories=MAX_CATEGORIES_PER_FEATURE,
            sparse_output=True,
            dtype=np.float32,
        )
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=True)

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", encoder),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
    )
    return preprocessor, numeric_features, categorical_features


def train_isolation_forest(model_df: pd.DataFrame, preprocessor: ColumnTransformer, contamination: float = 0.08):
    """Train Isolation Forest and return scores/predictions."""
    label_available = "anomaly_label" in model_df.columns and model_df["anomaly_label"].notna().any()
    train_df = model_df[model_df["anomaly_label"] == 0].copy() if label_available else model_df.copy()
    if train_df.empty:
        train_df = model_df.copy()

    x_train = preprocessor.fit_transform(train_df)
    x_all = preprocessor.transform(model_df)
    model = IsolationForest(
        n_estimators=250,
        contamination=contamination,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(x_train)
    normality_scores = model.decision_function(x_all)
    threshold = float(np.percentile(normality_scores, contamination * 100))
    predictions = (normality_scores < threshold).astype(int)
    anomaly_scores = -normality_scores
    return model, anomaly_scores, predictions


def build_predictions_table(df: pd.DataFrame, anomaly_scores: np.ndarray, predictions: np.ndarray, threshold: float) -> pd.DataFrame:
    """Attach scores, anomaly labels, and risk level."""
    medium_cutoff = float(np.quantile(anomaly_scores, 0.75))
    result_df = df.copy()
    result_df["anomaly_score"] = anomaly_scores
    result_df["predicted_anomaly"] = predictions
    result_df["risk_level"] = np.select(
        [result_df["anomaly_score"] >= threshold, result_df["anomaly_score"] >= medium_cutoff],
        ["High", "Medium"],
        default="Low",
    )
    return result_df.sort_values("anomaly_score", ascending=False).reset_index(drop=True)


def evaluate_predictions(predictions_df: pd.DataFrame) -> Dict[str, float]:
    """Print and return evaluation metrics."""
    print("\n" + "=" * 72)
    print("Model Performance Summary")
    print("=" * 72)

    if "anomaly_label" not in predictions_df.columns or predictions_df["anomaly_label"].isna().all():
        anomaly_count = int(predictions_df["predicted_anomaly"].sum())
        print(f"Ground-truth labels unavailable. Total anomalies detected: {anomaly_count}")
        print("\nAnomaly Score Distribution Summary")
        print(predictions_df["anomaly_score"].describe().round(4))
        return {
            "detected_anomalies": float(anomaly_count),
            "mean_anomaly_score": float(predictions_df["anomaly_score"].mean()),
            "max_anomaly_score": float(predictions_df["anomaly_score"].max()),
        }

    y_true = predictions_df["anomaly_label"].astype(int)
    y_pred = predictions_df["predicted_anomaly"].astype(int)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }
    print("\nEvaluation Metrics")
    for name, value in metrics.items():
        print(f"{name:>10}: {value:.4f}")
    print("\nClassification Report")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Anomaly"], zero_division=0))
    return metrics


def estimate_feature_importance(feature_df: pd.DataFrame, anomaly_scores: np.ndarray) -> pd.DataFrame:
    """Approximate feature importance via absolute Spearman correlation."""
    rows = []
    for col in [c for c in NUMERIC_FEATURES if c in feature_df.columns]:
        series = pd.to_numeric(feature_df[col], errors="coerce")
        if series.nunique(dropna=True) <= 1:
            continue
        corr_value = pd.Series(anomaly_scores).corr(series, method="spearman")
        if pd.notna(corr_value):
            rows.append(
                {
                    "feature": col,
                    "abs_correlation": abs(float(corr_value)),
                    "correlation": float(corr_value),
                    "variance": float(series.var(skipna=True)),
                }
            )
    importance_df = pd.DataFrame(rows).sort_values(["abs_correlation", "variance"], ascending=False)
    print("\nTop Contributing Features (Approximate)")
    if importance_df.empty:
        print("No numeric feature importance values could be estimated.")
    else:
        print(
            importance_df.head(8).to_string(
                index=False,
                formatters={
                    "abs_correlation": "{:.4f}".format,
                    "correlation": "{:.4f}".format,
                    "variance": "{:.4f}".format,
                },
            )
        )
    return importance_df


def create_visualizations(predictions_df: pd.DataFrame, threshold: float) -> Dict[str, plt.Figure]:
    """Create presentation-ready charts."""
    figures: Dict[str, plt.Figure] = {}
    plt.style.use("seaborn-v0_8-whitegrid")

    score_fig, score_ax = plt.subplots(figsize=(10, 6))
    score_ax.hist(predictions_df["anomaly_score"], bins=35, color="#4C78A8", edgecolor="black", alpha=0.85)
    score_ax.axvline(threshold, color="#D62728", linestyle="--", linewidth=2.5, label=f"Threshold = {threshold:.4f}")
    score_ax.set_title("Distribution of Anomaly Scores", fontsize=15, fontweight="bold")
    score_ax.set_xlabel("Anomaly Score")
    score_ax.set_ylabel("Frequency")
    score_ax.legend()
    score_fig.tight_layout()
    score_fig.savefig(SCORES_HIST_PATH, dpi=140)
    figures["histogram"] = score_fig

    plot_df = predictions_df.sample(n=min(len(predictions_df), MAX_SCATTER_POINTS), random_state=RANDOM_STATE)
    normal_points = plot_df[plot_df["predicted_anomaly"] == 0]
    anomaly_points = plot_df[plot_df["predicted_anomaly"] == 1]
    scatter_fig, scatter_ax = plt.subplots(figsize=(10, 6))
    scatter_ax.scatter(normal_points["login_hour"], normal_points["file_access_count"], c="#2CA02C", alpha=0.6, s=36, label="Normal")
    scatter_ax.scatter(anomaly_points["login_hour"], anomaly_points["file_access_count"], c="#D62728", alpha=0.85, s=44, label="Anomaly", marker="x")
    scatter_ax.set_title("Normal vs Anomalous User Activity", fontsize=15, fontweight="bold")
    scatter_ax.set_xlabel("Login Hour")
    scatter_ax.set_ylabel("File Access Count")
    scatter_ax.legend()
    scatter_fig.tight_layout()
    scatter_fig.savefig(SCATTER_PATH, dpi=140)
    figures["scatter"] = scatter_fig

    trend_df = predictions_df.copy()
    trend_df["activity_date"] = pd.to_datetime(trend_df["activity_date"], errors="coerce")
    trend_df = trend_df.dropna(subset=["activity_date"])
    trend_df["activity_day"] = trend_df["activity_date"].dt.date
    trend_df = trend_df.groupby("activity_day", as_index=False)["predicted_anomaly"].sum().rename(columns={"predicted_anomaly": "anomaly_count"})
    trend_df["activity_day"] = pd.to_datetime(trend_df["activity_day"])

    trend_fig, trend_ax = plt.subplots(figsize=(12, 6))
    trend_ax.plot(trend_df["activity_day"], trend_df["anomaly_count"], color="#FF7F0E", marker="o", linewidth=2.5, label="Daily Anomaly Count")
    if not trend_df.empty:
        spike_cutoff = max(1.0, float(trend_df["anomaly_count"].quantile(0.90)))
        spike_df = trend_df[trend_df["anomaly_count"] >= spike_cutoff]
        trend_ax.scatter(spike_df["activity_day"], spike_df["anomaly_count"], color="#D62728", s=90, zorder=3, label="Spike")
        for _, row in spike_df.iterrows():
            trend_ax.annotate(
                int(row["anomaly_count"]),
                (row["activity_day"], row["anomaly_count"]),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=9,
                color="#D62728",
            )
    trend_ax.set_title("Time-based Anomaly Trend", fontsize=15, fontweight="bold")
    trend_ax.set_xlabel("Activity Date")
    trend_ax.set_ylabel("Detected Anomalies")
    trend_ax.legend()
    trend_fig.autofmt_xdate()
    trend_fig.tight_layout()
    trend_fig.savefig(TREND_PATH, dpi=140)
    figures["trend"] = trend_fig
    return figures


def summarize_top_suspicious_users(predictions_df: pd.DataFrame, top_n: int = TOP_USERS_DISPLAY_COUNT) -> pd.DataFrame:
    """Aggregate and rank suspicious users."""
    user_summary = (
        predictions_df.groupby("user_id", as_index=False)
        .agg(
            anomaly_events=("predicted_anomaly", "sum"),
            avg_anomaly_score=("anomaly_score", "mean"),
            max_anomaly_score=("anomaly_score", "max"),
            total_file_access=("file_access_count", "sum"),
            total_emails_sent=("email_sent_count", "sum"),
            off_hours_activity_count=("off_hours_activity", "sum"),
        )
        .sort_values(["anomaly_events", "max_anomaly_score", "avg_anomaly_score"], ascending=False)
        .head(top_n)
    )
    print(f"\nTop {top_n} Suspicious Users")
    print(user_summary.to_string(index=False))
    return user_summary


def print_key_findings(predictions_df: pd.DataFrame, feature_importance_df: pd.DataFrame, top_users_df: pd.DataFrame) -> None:
    """Print slide-friendly findings."""
    anomaly_count = int(predictions_df["predicted_anomaly"].sum())
    anomaly_subset = predictions_df[predictions_df["predicted_anomaly"] == 1]
    off_hour_rate = float(anomaly_subset["off_hours_activity"].mean() * 100) if not anomaly_subset.empty else 0.0
    strongest_feature = str(feature_importance_df.iloc[0]["feature"]) if not feature_importance_df.empty else "N/A"
    riskiest_user = str(top_users_df.iloc[0]["user_id"]) if not top_users_df.empty else "N/A"

    print("\n" + "=" * 72)
    print("Key Findings")
    print("=" * 72)
    print(f"* {anomaly_count} anomalies detected in total")
    print(f"* {off_hour_rate:.1f}% of detected anomalies occurred during off-hours")
    print(f"* Strongest indicator by score association: {strongest_feature}")
    print(f"* Highest-risk user in the current run: {riskiest_user}")


def run_detection_pipeline(
    dataset_path: Optional[str] = None,
    input_df: Optional[pd.DataFrame] = None,
    contamination: float = 0.08,
    show_plots: bool = True,
) -> DetectionOutput:
    """Run the full anomaly detection workflow."""
    raw_df = input_df.copy() if input_df is not None else load_dataset(dataset_path)
    feature_df = engineer_features(raw_df)
    preprocessor, _, _ = build_preprocessor(feature_df)
    _, anomaly_scores, predictions = train_isolation_forest(feature_df, preprocessor, contamination=contamination)
    threshold = float(np.percentile(anomaly_scores, (1.0 - contamination) * 100))
    predictions_df = build_predictions_table(feature_df, anomaly_scores, predictions, threshold)
    metrics = evaluate_predictions(predictions_df)
    feature_importance_df = estimate_feature_importance(feature_df, anomaly_scores)
    figures = create_visualizations(predictions_df, threshold)
    top_users_df = summarize_top_suspicious_users(predictions_df)
    print_key_findings(predictions_df, feature_importance_df, top_users_df)

    predictions_df.to_csv(RESULTS_CSV, index=False)
    top_users_df.to_csv(TOP_USERS_CSV, index=False)
    print(f"\nSaved scored results to: {os.path.abspath(RESULTS_CSV)}")
    print(f"Saved top suspicious users to: {os.path.abspath(TOP_USERS_CSV)}")
    print(f"Saved plots to: {os.path.abspath(SCORES_HIST_PATH)}, {os.path.abspath(SCATTER_PATH)}, {os.path.abspath(TREND_PATH)}")

    if show_plots:
        plt.show()
    else:
        plt.close("all")

    metrics["isolation_forest_threshold"] = threshold
    metrics["top_suspicious_users_count"] = float(len(top_users_df))

    return DetectionOutput(
        raw_df=raw_df,
        processed_df=feature_df,
        predictions_df=predictions_df,
        top_users_df=top_users_df,
        feature_importance_df=feature_importance_df,
        metrics=metrics,
        threshold=threshold,
        figures=figures,
    )


def _running_in_streamlit() -> bool:
    """Return True when launched by Streamlit."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


def _build_ui_user_risk_table(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Create compact user-level risk table for dashboard."""
    if predictions_df.empty:
        return pd.DataFrame(columns=["user_id", "anomaly_score", "risk_level", "anomaly_events", "off_hours_activity_count"])
    user_table = (
        predictions_df.groupby("user_id", as_index=False)
        .agg(
            anomaly_score=("anomaly_score", "max"),
            anomaly_events=("predicted_anomaly", "sum"),
            off_hours_activity_count=("off_hours_activity", "sum"),
        )
        .sort_values("anomaly_score", ascending=False)
        .reset_index(drop=True)
    )
    user_table["risk_level"] = np.select(
        [
            user_table["anomaly_score"] >= predictions_df["anomaly_score"].quantile(0.90),
            user_table["anomaly_score"] >= predictions_df["anomaly_score"].quantile(0.75),
        ],
        ["High", "Medium"],
        default="Low",
    )
    return user_table


def _render_dashboard_styles() -> None:
    """Apply dashboard CSS styling."""
    import streamlit as st

    st.markdown(
        """
        <style>
        .main .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1400px; }
        .dashboard-title { font-size: 2.4rem; font-weight: 800; color: #F8FAFC; margin-bottom: 0.2rem; }
        .dashboard-subtitle { font-size: 1.05rem; color: #93C5FD; margin-bottom: 0.8rem; }
        .dashboard-description { background: rgba(30, 41, 59, 0.75); border: 1px solid rgba(148,163,184,0.35); padding: 1rem 1.2rem; border-radius: 14px; color: #E5E7EB; margin-bottom: 1.2rem; }
        .metric-card { background: rgba(15, 23, 42, 0.65); border: 1px solid rgba(148,163,184,0.35); border-radius: 16px; padding: 1rem 1.2rem; box-shadow: 0 4px 14px rgba(0,0,0,0.25); }
        .metric-label { font-size: 0.9rem; color: #CBD5E1; margin-bottom: 0.35rem; }
        .metric-value { font-size: 1.9rem; font-weight: 800; color: #F8FAFC; }
        .insights-panel { background: linear-gradient(135deg, rgba(30,41,59,0.92) 0%, rgba(15,23,42,0.90) 100%); border: 1px solid rgba(96,165,250,0.45); border-radius: 16px; padding: 1rem 1.2rem; color: #E5E7EB; }
        div.stButton > button { width: 100%; border-radius: 12px; font-weight: 700; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_metric_card(title: str, value: str) -> None:
    """Render one KPI card."""
    import streamlit as st

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{title}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _build_insights_text(predictions_df: pd.DataFrame, feature_importance_df: pd.DataFrame) -> str:
    """Generate plain-language insights panel content."""
    anomaly_df = predictions_df[predictions_df["predicted_anomaly"] == 1]
    high_risk_users = predictions_df[predictions_df["risk_level"] == "High"]["user_id"].nunique()
    off_hour_rate = float(anomaly_df["off_hours_activity"].mean() * 100) if not anomaly_df.empty else 0.0
    top_feature = str(feature_importance_df.iloc[0]["feature"]).replace("_", " ") if not feature_importance_df.empty else "behavioral activity volume"
    return (
        "<div class='insights-panel'>"
        "<b>Key Insights</b><br><br>"
        f"* {off_hour_rate:.1f}% of anomalous events occurred during off-hours.<br>"
        f"* {top_feature.title()} appears to be the strongest anomaly indicator in this run.<br>"
        f"* {high_risk_users} users were flagged as high risk and should be reviewed first."
        "</div>"
    )


def render_streamlit_dashboard() -> None:
    """Render the Streamlit dashboard."""
    import streamlit as st

    st.set_page_config(page_title="Insider Threat Detection", layout="wide")
    _render_dashboard_styles()

    st.markdown('<div class="dashboard-title">Insider Threat Detection Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="dashboard-subtitle">Behavior-based anomaly detection for compliance monitoring</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="dashboard-description">
        This dashboard scores user activity records with Isolation Forest and highlights suspicious behavior patterns
        such as off-hours access, unusual file activity, and abnormal communication volume.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.header("Detection Controls")
    use_sample_data = st.sidebar.checkbox("Use sample dataset", value=True)
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"], disabled=use_sample_data)
    contamination = st.sidebar.slider("Expected anomaly ratio", min_value=0.01, max_value=0.30, value=0.08, step=0.01)
    st.sidebar.markdown("---")
    risk_filter = st.sidebar.multiselect("Risk level filter", options=["High", "Medium", "Low"], default=["High", "Medium", "Low"])
    anomaly_only = st.sidebar.checkbox("Show anomalies only", value=True)
    high_risk_only = st.sidebar.checkbox("Show only high-risk users", value=False)
    run_detection = st.sidebar.button("Run Detection", type="primary")

    if use_sample_data:
        st.info("Sample dataset is ready. Click 'Run Detection' in the sidebar to run the demo.")
    elif uploaded_file is not None:
        st.success(f"Uploaded '{uploaded_file.name}' successfully. Click 'Run Detection' in the sidebar.")
    else:
        st.info("Upload a CSV file or enable 'Use sample dataset' from the sidebar, then click 'Run Detection'.")

    if run_detection:
        with st.spinner("Preparing data, training Isolation Forest, and scoring anomalies..."):
            try:
                if use_sample_data:
                    input_df = generate_synthetic_dataset()
                else:
                    if uploaded_file is None:
                        st.error("No valid dataset is available for detection. Upload a valid CSV or use the sample dataset.")
                        return
                    input_df = load_uploaded_csv_optimized(uploaded_file)
                    if input_df.empty:
                        st.error("The uploaded CSV did not produce any usable activity records.")
                        return

                output = run_detection_pipeline(input_df=input_df, contamination=contamination, show_plots=False)
            except Exception as exc:
                st.error(f"Detection failed: {exc}")
                return

        st.success("Detection completed successfully. Review the KPIs, charts, and suspicious users below.")

        filtered_predictions = output.predictions_df[output.predictions_df["risk_level"].isin(risk_filter)]
        if anomaly_only:
            filtered_predictions = filtered_predictions[filtered_predictions["predicted_anomaly"] == 1]

        suspicious_users_df = _build_ui_user_risk_table(filtered_predictions).head(20)
        if high_risk_only:
            suspicious_users_df = suspicious_users_df[suspicious_users_df["risk_level"] == "High"]

        total_records = len(output.predictions_df)
        total_anomalies = int(output.predictions_df["predicted_anomaly"].sum())
        anomaly_pct = (total_anomalies / total_records * 100) if total_records else 0.0

        st.subheader("Dashboard Overview")
        metric_1, metric_2, metric_3, metric_4 = st.columns(4)
        with metric_1:
            _render_metric_card("Total Records", f"{total_records:,}")
        with metric_2:
            _render_metric_card("Total Anomalies Detected", f"{total_anomalies:,}")
        with metric_3:
            _render_metric_card("% of Anomalies", f"{anomaly_pct:.2f}%")
        with metric_4:
            _render_metric_card("Model Used", "Isolation Forest")

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Model Performance")
        st.json(
            {
                key: round(value, 4)
                for key, value in output.metrics.items()
                if key in ["accuracy", "precision", "recall", "f1_score", "detected_anomalies", "mean_anomaly_score", "max_anomaly_score"]
            }
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Anomaly Visualizations")
        chart_col_1, chart_col_2 = st.columns(2)
        with chart_col_1:
            st.pyplot(output.figures["histogram"], use_container_width=True)
        with chart_col_2:
            st.pyplot(output.figures["scatter"], use_container_width=True)
        st.pyplot(output.figures["trend"], use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Top 20 Suspicious Users")
        st.dataframe(
            suspicious_users_df[["user_id", "anomaly_score", "risk_level", "anomaly_events", "off_hours_activity_count"]],
            use_container_width=True,
            hide_index=True,
            height=420,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Insights Panel")
        st.markdown(_build_insights_text(output.predictions_df, output.feature_importance_df), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Download Results")
        st.download_button(
            "Download scored results CSV",
            data=output.predictions_df.to_csv(index=False).encode("utf-8"),
            file_name=RESULTS_CSV,
            mime="text/csv",
        )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Insider Threat Detection using Isolation Forest")
    parser.add_argument("--dataset", type=str, default=None, help="Path to CERT-like CSV dataset.")
    parser.add_argument("--contamination", type=float, default=0.08, help="Expected fraction of anomalies.")
    parser.add_argument("--no-show", action="store_true", help="Save plots without opening matplotlib windows.")
    return parser.parse_args()


if __name__ == "__main__":
    if _running_in_streamlit():
        render_streamlit_dashboard()
    else:
        args = parse_args()
        run_detection_pipeline(dataset_path=args.dataset, contamination=args.contamination, show_plots=not args.no_show)
