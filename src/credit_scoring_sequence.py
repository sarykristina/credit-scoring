from __future__ import annotations

import gc
import json
import time
from pathlib import Path
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from credit_scoring_compact import class_weights, logit
from credit_scoring_improve import (
    BASIC_TEST,
    BASIC_TRAIN,
    EXTRA_TEST,
    EXTRA_TRAIN,
    OUTPUT_DIR,
    SAMPLE_CSV,
    TARGET_CSV,
    calibrate_to_prior,
    score_feature_file,
)


WORK_DIR = Path(os.getenv("WORK_DIR", Path(__file__).resolve().parents[1] / "work"))
SEQ_TRAIN = WORK_DIR / "train_features_seq_v1.parquet"
SEQ_TEST = WORK_DIR / "test_features_seq_v1.parquet"
SUBMISSION_CSV = OUTPUT_DIR / "submission_sequence.csv"
REPORT_JSON = WORK_DIR / "model_report_sequence.json"


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def score_file_fast(path: Path, target_map: pd.Series, batch_size: int = 70_000) -> pd.Series:
    pf = pq.ParquetFile(path)
    cols = [col for col in pf.schema_arrow.names if col != "id"]
    n_cols = len(cols)
    pos_sum = np.zeros(n_cols, dtype=np.float64)
    neg_sum = np.zeros(n_cols, dtype=np.float64)
    total_sum = np.zeros(n_cols, dtype=np.float64)
    total_sq = np.zeros(n_cols, dtype=np.float64)
    pos_n = neg_n = total_n = 0

    for batch_no, batch in enumerate(pf.iter_batches(batch_size=batch_size, columns=["id", *cols]), start=1):
        chunk = batch.to_pandas()
        y = chunk["id"].map(target_map).to_numpy(dtype=np.int8)
        x = chunk[cols].to_numpy(dtype=np.float32, copy=False)
        pos = y == 1
        neg = ~pos
        if pos.any():
            pos_sum += x[pos].sum(axis=0)
            pos_n += int(pos.sum())
        if neg.any():
            neg_sum += x[neg].sum(axis=0)
            neg_n += int(neg.sum())
        total_sum += x.sum(axis=0)
        total_sq += np.square(x, dtype=np.float64).sum(axis=0)
        total_n += len(chunk)
        if batch_no % 8 == 0:
            log(f"scored {path.name}: batch {batch_no}")
        del chunk, x, y
        gc.collect()

    pos_mean = pos_sum / max(pos_n, 1)
    neg_mean = neg_sum / max(neg_n, 1)
    mean = total_sum / max(total_n, 1)
    var = total_sq / max(total_n, 1) - mean * mean
    std = np.sqrt(np.maximum(var, 1e-12))
    scores = np.abs(pos_mean - neg_mean) / std
    return pd.Series(scores, index=cols).replace([np.inf, -np.inf], 0).fillna(0).sort_values(ascending=False)


def select_features() -> tuple[list[str], list[str], list[str], dict]:
    target = pd.read_csv(TARGET_CSV)
    target_map = target.set_index("id")["flag"]
    log("sequence scoring basic features")
    basic_scores = score_feature_file(BASIC_TRAIN, target_map).sort_values(ascending=False)
    log("sequence scoring extra features")
    extra_scores = score_feature_file(EXTRA_TRAIN, target_map).sort_values(ascending=False)
    log("sequence scoring sequence features")
    seq_scores = score_file_fast(SEQ_TRAIN, target_map)

    basic_cols = basic_scores.head(210).index.tolist()
    extra_cols = extra_scores.head(110).index.tolist()
    seq_cols = seq_scores.head(190).index.tolist()
    for col in ("id_norm", "loan_count"):
        if col in basic_scores.index and col not in basic_cols:
            basic_cols.append(col)

    log(f"selected: basic={len(basic_cols)}, extra={len(extra_cols)}, seq={len(seq_cols)}")
    report = {
        "basic_top": basic_scores.head(40).to_dict(),
        "extra_top": extra_scores.head(40).to_dict(),
        "seq_top": seq_scores.head(60).to_dict(),
    }
    return basic_cols, extra_cols, seq_cols, report


def make_sample_ids(max_train: int = 720_000, val_size: int = 240_000, seed: int = 807) -> tuple[set[int], set[int], pd.Series]:
    target = pd.read_csv(TARGET_CSV)
    y = target["flag"].to_numpy(dtype=np.int8)
    ids = target["id"].to_numpy()
    idx = np.arange(len(target))
    train_pool, val_idx = train_test_split(idx, test_size=val_size, random_state=seed, stratify=y)
    rng = np.random.default_rng(seed)
    pool_y = y[train_pool]
    pos = train_pool[pool_y == 1]
    neg = train_pool[pool_y == 0]
    neg_take = min(len(neg), max_train - len(pos))
    train_idx = np.concatenate([pos, rng.choice(neg, size=neg_take, replace=False)])
    rng.shuffle(train_idx)
    log(f"sample ids: train={len(train_idx):,}, val={len(val_idx):,}")
    return set(map(int, ids[train_idx])), set(map(int, ids[val_idx])), target.set_index("id")["flag"]


def load_rows(path: Path, columns: list[str], ids_needed: set[int], label: str, batch_size: int = 120_000) -> pd.DataFrame:
    pf = pq.ParquetFile(path)
    chunks: list[pd.DataFrame] = []
    for batch_no, batch in enumerate(pf.iter_batches(batch_size=batch_size, columns=["id", *columns]), start=1):
        chunk = batch.to_pandas()
        chunk = chunk.loc[chunk["id"].isin(ids_needed)]
        if not chunk.empty:
            chunks.append(chunk)
        if batch_no % 8 == 0:
            log(f"loaded {label}: batch {batch_no}, rows={sum(len(c) for c in chunks):,}")
    out = pd.concat(chunks, axis=0, ignore_index=True)
    for col in columns:
        out[col] = out[col].astype(np.float32, copy=False)
    log(f"loaded {label}: {len(out):,} rows")
    return out


def load_sample_matrix(
    basic_cols: list[str],
    extra_cols: list[str],
    seq_cols: list[str],
    ids_needed: set[int],
) -> pd.DataFrame:
    basic = load_rows(BASIC_TRAIN, basic_cols, ids_needed, "basic")
    extra = load_rows(EXTRA_TRAIN, extra_cols, ids_needed, "extra").rename(columns={c: f"extra__{c}" for c in extra_cols})
    seq = load_rows(SEQ_TRAIN, seq_cols, ids_needed, "seq").rename(columns={c: f"seq__{c}" for c in seq_cols})
    data = basic.merge(extra, on="id", how="inner").merge(seq, on="id", how="inner")
    del basic, extra, seq
    gc.collect()
    log(f"merged sequence sample matrix: {data.shape}")
    return data


def train_models(data: pd.DataFrame, target_map: pd.Series, train_ids: set[int], val_ids: set[int]):
    y = data["id"].map(target_map).to_numpy(dtype=np.int8)
    features = [c for c in data.columns if c != "id"]
    train_mask = data["id"].isin(train_ids).to_numpy()
    val_mask = data["id"].isin(val_ids).to_numpy()
    x_train = data.loc[train_mask, features].to_numpy(dtype=np.float32, copy=True)
    y_train = y[train_mask]
    x_val = data.loc[val_mask, features].to_numpy(dtype=np.float32, copy=True)
    y_val = y[val_mask]

    configs = [
        (
            "seq_hgb_leaf63",
            {
                "loss": "log_loss",
                "learning_rate": 0.042,
                "max_iter": 300,
                "max_leaf_nodes": 63,
                "min_samples_leaf": 55,
                "l2_regularization": 0.025,
                "max_bins": 128,
                "early_stopping": True,
                "validation_fraction": 0.08,
                "n_iter_no_change": 24,
                "random_state": 807,
                "verbose": 1,
            },
            0.95,
        ),
        (
            "seq_hgb_leaf31",
            {
                "loss": "log_loss",
                "learning_rate": 0.052,
                "max_iter": 240,
                "max_leaf_nodes": 31,
                "min_samples_leaf": 90,
                "l2_regularization": 0.09,
                "max_bins": 128,
                "early_stopping": True,
                "validation_fraction": 0.08,
                "n_iter_no_change": 22,
                "random_state": 808,
                "verbose": 0,
            },
            1.0,
        ),
    ]

    models = []
    val_preds = []
    report = {"model_aucs": {}, "feature_count": len(features)}
    for name, params, weight_multiplier in configs:
        log(f"training {name}")
        model = HistGradientBoostingClassifier(**params)
        model.fit(x_train, y_train, sample_weight=class_weights(y_train, weight_multiplier))
        pred = model.predict_proba(x_val)[:, 1]
        auc = roc_auc_score(y_val, pred)
        log(f"{name} validation ROC-AUC: {auc:.6f}; iterations={getattr(model, 'n_iter_', None)}")
        report["model_aucs"][name] = float(auc)
        report[f"{name}_iterations"] = int(getattr(model, "n_iter_", 0))
        models.append(model)
        val_preds.append(pred)
        gc.collect()

    best_auc = -1.0
    best_weights = [1.0, 0.0]
    for w in np.linspace(0, 1, 41):
        score = w * logit(val_preds[0]) + (1 - w) * logit(val_preds[1])
        auc = roc_auc_score(y_val, score)
        if auc > best_auc:
            best_auc = float(auc)
            best_weights = [float(w), float(1 - w)]
    log(f"sequence ensemble validation ROC-AUC: {best_auc:.6f}; weights={best_weights}")
    report["ensemble_auc"] = best_auc
    report["ensemble_weights"] = best_weights
    del x_train, x_val
    gc.collect()
    return models, best_weights, features, report


def predict_test(
    models,
    weights: list[float],
    basic_cols: list[str],
    extra_cols: list[str],
    seq_cols: list[str],
    features: list[str],
    prior: float,
) -> pd.DataFrame:
    basic_pf = pq.ParquetFile(BASIC_TEST)
    extra_pf = pq.ParquetFile(EXTRA_TEST)
    seq_pf = pq.ParquetFile(SEQ_TEST)
    n_groups = basic_pf.num_row_groups
    parts: list[pd.DataFrame] = []
    for rg in range(n_groups):
        basic = basic_pf.read_row_group(rg, columns=["id", *basic_cols]).to_pandas()
        extra = extra_pf.read_row_group(rg, columns=["id", *extra_cols]).to_pandas().rename(columns={c: f"extra__{c}" for c in extra_cols})
        seq = seq_pf.read_row_group(rg, columns=["id", *seq_cols]).to_pandas().rename(columns={c: f"seq__{c}" for c in seq_cols})
        data = basic.merge(extra, on="id", how="inner").merge(seq, on="id", how="inner")
        for col in features:
            data[col] = data[col].astype(np.float32, copy=False)
        x = data[features].to_numpy(dtype=np.float32, copy=True)
        score = np.zeros(len(data), dtype=np.float64)
        for model, weight in zip(models, weights):
            score += weight * logit(model.predict_proba(x)[:, 1])
        raw = 1.0 / (1.0 + np.exp(-score))
        pred = calibrate_to_prior(raw, prior)
        parts.append(pd.DataFrame({"id": data["id"].to_numpy(), "flag": pred}))
        log(f"predicted sequence test row-group {rg + 1}/{n_groups}")
        del basic, extra, seq, data, x, raw, pred
        gc.collect()
    return pd.concat(parts, axis=0, ignore_index=True)


def main() -> None:
    basic_cols, extra_cols, seq_cols, feature_report = select_features()
    train_ids, val_ids, target_map = make_sample_ids()
    data = load_sample_matrix(basic_cols, extra_cols, seq_cols, train_ids | val_ids)
    models, weights, features, model_report = train_models(data, target_map, train_ids, val_ids)
    prior = float(pd.read_csv(TARGET_CSV)["flag"].mean())
    pred = predict_test(models, weights, basic_cols, extra_cols, seq_cols, features, prior)
    sample = pd.read_csv(SAMPLE_CSV)
    submission = sample[["id"]].merge(pred, on="id", how="left")
    if submission["flag"].isna().any():
        raise RuntimeError(f"missing predictions: {int(submission['flag'].isna().sum())}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    submission.to_csv(SUBMISSION_CSV, index=False, float_format="%.7f")
    report = {
        **feature_report,
        **model_report,
        "basic_features": len(basic_cols),
        "extra_features": len(extra_cols),
        "seq_features": len(seq_cols),
        "submission": str(SUBMISSION_CSV),
        "submission_bytes": SUBMISSION_CSV.stat().st_size,
        "prediction_min": float(submission["flag"].min()),
        "prediction_max": float(submission["flag"].max()),
        "prediction_mean": float(submission["flag"].mean()),
    }
    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"saved sequence submission: {SUBMISSION_CSV}; size={SUBMISSION_CSV.stat().st_size / 1_000_000:.2f} MB")
    log(f"prediction range: min={submission['flag'].min():.6f}, max={submission['flag'].max():.6f}, mean={submission['flag'].mean():.6f}")


if __name__ == "__main__":
    main()
