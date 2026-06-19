from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DATA_DIR = Path(os.getenv("DATA_DIR", Path(__file__).resolve().parents[1] / "data"))
WORK_DIR = Path(os.getenv("WORK_DIR", Path(__file__).resolve().parents[1] / "work"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", Path(__file__).resolve().parents[1] / "outputs"))

TRAIN_PARQUET = DATA_DIR / "train_data.parquet"
TEST_PARQUET = DATA_DIR / "test_data.parquet"
TARGET_CSV = DATA_DIR / "train_target.csv"
SAMPLE_CSV = DATA_DIR / "sample_submission.csv"

TRAIN_FEATURES = WORK_DIR / "train_features.parquet"
TEST_FEATURES = WORK_DIR / "test_features.parquet"
REPORT_JSON = WORK_DIR / "model_report.json"
SUBMISSION_CSV = OUTPUT_DIR / "submission.csv"


PAYM_COLS = [f"enc_paym_{i}" for i in range(25)]

DUMMY_VALUES: dict[str, list[int]] = {
    "pre_loans_credit_limit": list(range(20)),
    "pre_loans_next_pay_summ": list(range(7)),
    "pre_loans_outstanding": list(range(1, 6)),
    "pre_loans_max_overdue_sum": list(range(1, 4)),
    "pre_loans_credit_cost_rate": list(range(14)),
    "pre_loans5": list(range(20)),
    "pre_loans530": list(range(20)),
    "pre_loans3060": list(range(10)),
    "pre_loans6090": list(range(5)),
    "pre_loans90": list(range(20)),
    "pre_util": list(range(20)),
    "pre_over2limit": list(range(20)),
    "pre_maxover2limit": list(range(20)),
    "enc_loans_account_holder_type": list(range(7)),
    "enc_loans_credit_status": list(range(7)),
    "enc_loans_credit_type": list(range(8)),
    "enc_loans_account_cur": list(range(4)),
    "pclose_flag": [0, 1],
    "fclose_flag": [0, 1],
}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def flatten_columns(columns: pd.Index) -> list[str]:
    out: list[str] = []
    for col in columns:
        if isinstance(col, tuple):
            out.append("_".join(str(part) for part in col if part != ""))
        else:
            out.append(str(col))
    return out


def complete_blocks(
    parquet_path: Path,
    batch_size: int,
) -> tuple[pd.DataFrame, ...]:
    raise RuntimeError("This function is intentionally not called directly.")


def aggregate_complete_block(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    ids = df["id"]
    grouped = df.groupby("id", sort=False)

    stats = grouped[feature_cols].agg(["mean", "max", "std", "last"])
    stats.columns = flatten_columns(stats.columns)
    stats = stats.astype(np.float32).copy()
    stats["loan_count"] = grouped.size().astype(np.float32)

    paym = df[PAYM_COLS]
    paym_features: dict[str, pd.Series] = {}
    for value in range(5):
        row_count = (paym == value).sum(axis=1).astype(np.int16)
        paym_features[f"paym_status_{value}_count"] = row_count.groupby(ids, sort=False).sum().astype(np.float32)
        recent_count = (paym.iloc[:, :6] == value).sum(axis=1).astype(np.int16)
        paym_features[f"paym_recent_status_{value}_count"] = recent_count.groupby(ids, sort=False).sum().astype(np.float32)

    paym_features["paym_nonzero_count"] = (paym.ne(0).sum(axis=1)).groupby(ids, sort=False).sum().astype(np.float32)
    paym_features["paym_mean_all"] = paym.mean(axis=1).groupby(ids, sort=False).mean().astype(np.float32)
    paym_features["paym_max_all"] = paym.max(axis=1).groupby(ids, sort=False).max().astype(np.float32)
    paym_frame = pd.DataFrame(paym_features, index=stats.index)

    dummy_cols = list(DUMMY_VALUES)
    dummy_frame = pd.get_dummies(df[dummy_cols], columns=dummy_cols, dtype=np.uint8)
    expected_dummy_cols = [f"{col}_{value}" for col, values in DUMMY_VALUES.items() for value in values]
    dummy_frame = dummy_frame.reindex(columns=expected_dummy_cols, fill_value=0)
    dummy_frame.columns = [f"cnt_{col}" for col in dummy_frame.columns]
    dummy_frame = dummy_frame.groupby(ids, sort=False).sum().astype(np.float32)

    stats = pd.concat([stats, paym_frame, dummy_frame], axis=1).reset_index()
    stats["id_norm"] = (stats["id"].astype(np.float32) / np.float32(3_000_000.0)).astype(np.float32)
    stats = stats.fillna(0.0)
    return stats


def aggregate_file(parquet_path: Path, output_path: Path, batch_size: int = 650_000, force: bool = False) -> None:
    if output_path.exists() and not force:
        log(f"features already exist: {output_path.name}")
        return

    log(f"building features from {parquet_path.name}")
    output_path.unlink(missing_ok=True)
    pf = pq.ParquetFile(parquet_path)
    all_cols = pf.schema_arrow.names
    feature_cols = [col for col in all_cols if col != "id"]
    writer: pq.ParquetWriter | None = None
    carry: pd.DataFrame | None = None
    total_ids = 0

    for batch_no, batch in enumerate(pf.iter_batches(batch_size=batch_size), start=1):
        df = batch.to_pandas()
        if carry is not None and not carry.empty:
            df = pd.concat([carry, df], axis=0, ignore_index=True)

        last_id = df["id"].iloc[-1]
        complete_mask = df["id"] != last_id
        complete = df.loc[complete_mask].copy()
        carry = df.loc[~complete_mask].copy()

        if complete.empty:
            continue

        features = aggregate_complete_block(complete, feature_cols)
        total_ids += len(features)

        table = pa.Table.from_pandas(features, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
        writer.write_table(table)

        log(f"{parquet_path.name}: batch {batch_no}, ids={total_ids:,}")
        del df, complete, features, table
        gc.collect()

    if carry is not None and not carry.empty:
        features = aggregate_complete_block(carry, feature_cols)
        total_ids += len(features)
        table = pa.Table.from_pandas(features, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
        writer.write_table(table)
        del features, table

    if writer is not None:
        writer.close()

    log(f"saved {output_path.name}: ids={total_ids:,}")


def load_feature_frame(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    for col in df.columns:
        if col != "id":
            df[col] = df[col].astype(np.float32, copy=False)
    return df


def select_feature_columns(train: pd.DataFrame, target: pd.DataFrame) -> list[str]:
    feature_cols = [col for col in train.columns if col != "id"]
    target_series = target.set_index("id")["flag"]
    mapped = train["id"].map(target_series)
    if mapped.isna().any():
        raise RuntimeError("target is missing for some train ids")
    y = mapped.to_numpy(dtype=np.int8)
    X = train[feature_cols]
    pos_mean = X.loc[y == 1].mean(axis=0)
    neg_mean = X.loc[y == 0].mean(axis=0)
    scale = X.std(axis=0).replace(0, np.nan)
    scores = ((pos_mean - neg_mean).abs() / scale).replace([np.inf, -np.inf], np.nan).fillna(0)
    scores = scores.sort_values(ascending=False)
    # Keep a broad but memory-friendly subset for tree boosting.
    selected = scores.head(360).index.tolist()
    if "id_norm" not in selected and "id_norm" in feature_cols:
        selected.append("id_norm")
    if "loan_count" not in selected and "loan_count" in feature_cols:
        selected.append("loan_count")
    log(f"selected {len(selected)} features from {len(feature_cols)}")
    del X, y, target_series
    gc.collect()
    return selected


def class_weights(y: np.ndarray) -> np.ndarray:
    pos = y.mean()
    weights = np.where(y == 1, 0.5 / pos, 0.5 / (1.0 - pos))
    return weights.astype(np.float32)


def stratified_sample_indices(y: np.ndarray, max_rows: int, random_state: int) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)
    if len(y) <= max_rows:
        return np.arange(len(y))
    keep_pos = pos_idx
    neg_keep = min(len(neg_idx), max_rows - len(keep_pos))
    if neg_keep < 0:
        keep_pos = rng.choice(pos_idx, size=max_rows // 2, replace=False)
        neg_keep = max_rows - len(keep_pos)
    keep_neg = rng.choice(neg_idx, size=neg_keep, replace=False)
    idx = np.concatenate([keep_pos, keep_neg])
    rng.shuffle(idx)
    return idx


def fit_and_validate(train: pd.DataFrame, target: pd.DataFrame, feature_cols: list[str]) -> tuple[HistGradientBoostingClassifier, dict]:
    target_series = target.set_index("id")["flag"]
    y = train["id"].map(target_series).to_numpy(dtype=np.int8)
    ids = train["id"].to_numpy()

    validation_pool = stratified_sample_indices(y, max_rows=1_200_000, random_state=42)
    pool_y = y[validation_pool]
    pool_ids = ids[validation_pool]
    train_idx, val_idx = train_test_split(
        validation_pool,
        test_size=300_000,
        random_state=42,
        stratify=pool_y,
    )

    X_tr = train.iloc[train_idx][feature_cols].to_numpy(dtype=np.float32, copy=True)
    X_val = train.iloc[val_idx][feature_cols].to_numpy(dtype=np.float32, copy=True)
    y_tr = y[train_idx]
    y_val = y[val_idx]

    sample_weight = class_weights(y_tr)

    log("training HistGradientBoostingClassifier")
    hgb = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.055,
        max_iter=230,
        max_leaf_nodes=31,
        min_samples_leaf=80,
        l2_regularization=0.05,
        max_bins=64,
        early_stopping=True,
        validation_fraction=0.08,
        n_iter_no_change=18,
        random_state=42,
        verbose=1,
    )
    hgb.fit(X_tr, y_tr, sample_weight=sample_weight)
    val_pred = hgb.predict_proba(X_val)[:, 1]
    random_auc = roc_auc_score(y_val, val_pred)
    log(f"random validation ROC-AUC: {random_auc:.6f}")

    # A second, stricter time-like check by id helps catch overfitting to time drift.
    cutoff = np.quantile(pool_ids, 0.82)
    time_train = ids <= cutoff
    time_val = ids > cutoff
    time_auc = None
    if time_train.any() and time_val.any() and y[time_val].min() != y[time_val].max():
        time_train_idx = stratified_sample_indices(y[time_train], max_rows=850_000, random_state=11)
        base_train_idx = np.flatnonzero(time_train)[time_train_idx]
        time_val_idx = np.flatnonzero(time_val)
        if len(time_val_idx) > 300_000:
            rng = np.random.default_rng(12)
            time_val_idx = rng.choice(time_val_idx, size=300_000, replace=False)
        time_model = HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=0.055,
            max_iter=max(80, int(getattr(hgb, "n_iter_", 160))),
            max_leaf_nodes=31,
            min_samples_leaf=80,
            l2_regularization=0.05,
            max_bins=64,
            early_stopping=True,
            validation_fraction=0.08,
            n_iter_no_change=18,
            random_state=11,
        )
        X_time_train = train.iloc[base_train_idx][feature_cols].to_numpy(dtype=np.float32, copy=True)
        X_time_val = train.iloc[time_val_idx][feature_cols].to_numpy(dtype=np.float32, copy=True)
        time_model.fit(X_time_train, y[base_train_idx], sample_weight=class_weights(y[base_train_idx]))
        time_pred = time_model.predict_proba(X_time_val)[:, 1]
        time_auc = roc_auc_score(y[time_val_idx], time_pred)
        log(f"id-tail validation ROC-AUC: {time_auc:.6f}")
        del time_model, time_pred, X_time_train, X_time_val
        gc.collect()

    report = {
        "feature_count": len(feature_cols),
        "random_validation_auc": float(random_auc),
        "id_tail_validation_auc": None if time_auc is None else float(time_auc),
        "hgb_iterations": int(getattr(hgb, "n_iter_", 0)),
        "train_rows": int(len(y_tr)),
        "validation_rows": int(len(y_val)),
        "positive_rate": float(y.mean()),
    }

    del X_tr, X_val, y_tr, y_val, sample_weight
    gc.collect()

    log("training final HistGradientBoostingClassifier on stratified labeled sample")
    final_iter = max(80, int(getattr(hgb, "n_iter_", 160)))
    final_idx = stratified_sample_indices(y, max_rows=1_600_000, random_state=77)
    X_final = train.iloc[final_idx][feature_cols].to_numpy(dtype=np.float32, copy=True)
    y_final = y[final_idx]
    final_model = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.055,
        max_iter=final_iter,
        max_leaf_nodes=31,
        min_samples_leaf=80,
        l2_regularization=0.05,
        max_bins=64,
        early_stopping=False,
        random_state=42,
    )
    final_model.fit(X_final, y_final, sample_weight=class_weights(y_final))

    del hgb, X_final, y_final, y, ids
    gc.collect()
    return final_model, report


def fit_sgd_baseline(train: pd.DataFrame, target: pd.DataFrame, feature_cols: list[str]) -> dict:
    target_series = target.set_index("id")["flag"]
    y = train["id"].map(target_series).to_numpy(dtype=np.int8)
    sample_idx = stratified_sample_indices(y, max_rows=850_000, random_state=24)
    train_idx, val_idx = train_test_split(
        sample_idx,
        test_size=180_000,
        random_state=24,
        stratify=y[sample_idx],
    )
    X_tr = train.iloc[train_idx][feature_cols].to_numpy(dtype=np.float32, copy=True)
    X_val = train.iloc[val_idx][feature_cols].to_numpy(dtype=np.float32, copy=True)
    y_tr = y[train_idx]
    y_val = y[val_idx]
    model = make_pipeline(
        StandardScaler(),
        SGDClassifier(
            loss="log_loss",
            penalty="elasticnet",
            alpha=2e-5,
            l1_ratio=0.08,
            max_iter=18,
            tol=1e-4,
            random_state=24,
            class_weight="balanced",
        ),
    )
    log("training SGD logistic baseline")
    model.fit(X_tr, y_tr)
    pred = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, pred)
    log(f"SGD logistic validation ROC-AUC: {auc:.6f}")
    del y, X_tr, X_val, y_tr, y_val, model, pred
    gc.collect()
    return {"sgd_random_validation_auc": float(auc)}


def write_submission(model: HistGradientBoostingClassifier, test: pd.DataFrame, feature_cols: list[str]) -> None:
    sample = pd.read_csv(SAMPLE_CSV)
    X_test = test[feature_cols].to_numpy(dtype=np.float32, copy=False)
    pred = model.predict_proba(X_test)[:, 1]
    pred_frame = pd.DataFrame({"id": test["id"].to_numpy(), "flag": pred.astype(np.float64)})
    submission = sample[["id"]].merge(pred_frame, on="id", how="left")
    if submission["flag"].isna().any():
        missing = int(submission["flag"].isna().sum())
        raise RuntimeError(f"missing predictions for {missing} ids")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    submission.to_csv(SUBMISSION_CSV, index=False)
    log(f"saved submission: {SUBMISSION_CSV}")
    log(f"prediction range: min={submission['flag'].min():.6f}, max={submission['flag'].max():.6f}, mean={submission['flag'].mean():.6f}")


def run(force_features: bool = False, skip_baseline: bool = False) -> None:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    aggregate_file(TRAIN_PARQUET, TRAIN_FEATURES, force=force_features)
    aggregate_file(TEST_PARQUET, TEST_FEATURES, force=force_features)

    target = pd.read_csv(TARGET_CSV)
    train = load_feature_frame(TRAIN_FEATURES)
    feature_cols = select_feature_columns(train, target)

    report: dict = {}
    if not skip_baseline:
        report.update(fit_sgd_baseline(train, target, feature_cols))

    model, hgb_report = fit_and_validate(train, target, feature_cols)
    report.update(hgb_report)
    report["features"] = feature_cols

    test = load_feature_frame(TEST_FEATURES)
    missing = sorted(set(feature_cols) - set(test.columns))
    if missing:
        raise RuntimeError(f"test features missing: {missing[:5]}")
    write_submission(model, test, feature_cols)

    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"saved report: {REPORT_JSON}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(force_features=args.force_features, skip_baseline=args.skip_baseline)
