from __future__ import annotations

import argparse
import gc
import json
import math
import warnings
import time
from pathlib import Path
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


DATA_DIR = Path(os.getenv("DATA_DIR", Path(__file__).resolve().parents[1] / "data"))
WORK_DIR = Path(os.getenv("WORK_DIR", Path(__file__).resolve().parents[1] / "work"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", Path(__file__).resolve().parents[1] / "outputs"))

TRAIN_PARQUET = DATA_DIR / "train_data.parquet"
TEST_PARQUET = DATA_DIR / "test_data.parquet"
TARGET_CSV = DATA_DIR / "train_target.csv"
SAMPLE_CSV = DATA_DIR / "sample_submission.csv"

BASIC_TRAIN = WORK_DIR / "train_features.parquet"
BASIC_TEST = WORK_DIR / "test_features.parquet"
EXTRA_TRAIN = WORK_DIR / "train_features_extra_v2.parquet"
EXTRA_TEST = WORK_DIR / "test_features_extra_v2.parquet"
REPORT_JSON = WORK_DIR / "model_report_improved.json"
SUBMISSION_CSV = OUTPUT_DIR / "submission_improved.csv"

PAYM_COLS = [f"enc_paym_{i}" for i in range(25)]

EXTRA_DUMMY_VALUES: dict[str, list[int]] = {
    "pre_since_opened": list(range(20)),
    "pre_since_confirmed": list(range(18)),
    "pre_pterm": list(range(18)),
    "pre_fterm": list(range(17)),
    "pre_till_pclose": list(range(17)),
    "pre_till_fclose": list(range(16)),
}
EXTRA_DUMMY_VALUES.update({col: list(range(5)) for col in PAYM_COLS})

RECENT_COLS = [
    "pre_since_opened",
    "pre_since_confirmed",
    "pre_pterm",
    "pre_fterm",
    "pre_till_pclose",
    "pre_till_fclose",
    "pre_loans_credit_limit",
    "pre_loans_next_pay_summ",
    "pre_loans_outstanding",
    "pre_loans_max_overdue_sum",
    "pre_loans_credit_cost_rate",
    "pre_loans5",
    "pre_loans530",
    "pre_loans3060",
    "pre_loans6090",
    "pre_loans90",
    "is_zero_loans5",
    "is_zero_loans530",
    "is_zero_loans3060",
    "is_zero_loans6090",
    "is_zero_loans90",
    "pre_util",
    "pre_over2limit",
    "pre_maxover2limit",
    "is_zero_util",
    "is_zero_over2limit",
    "is_zero_maxover2limit",
    "enc_loans_account_holder_type",
    "enc_loans_credit_status",
    "enc_loans_credit_type",
    "enc_loans_account_cur",
    "pclose_flag",
    "fclose_flag",
]


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


def add_dummy_counts(df: pd.DataFrame, ids: pd.Series, loan_count: pd.Series) -> pd.DataFrame:
    dummy_cols = list(EXTRA_DUMMY_VALUES)
    expected = [f"{col}_{value}" for col, values in EXTRA_DUMMY_VALUES.items() for value in values]
    dummies = pd.get_dummies(df[dummy_cols], columns=dummy_cols, dtype=np.uint8)
    dummies = dummies.reindex(columns=expected, fill_value=0)
    dummies.columns = [f"xcnt_{col}" for col in dummies.columns]
    counts = dummies.groupby(ids, sort=False).sum().astype(np.float32)
    # Proportions make the same signal easier for tree splits to use when history length varies.
    props = counts.div(loan_count, axis=0).astype(np.float32)
    props.columns = [col.replace("xcnt_", "xprop_", 1) for col in counts.columns]
    return pd.concat([counts, props], axis=1)


def add_recent_features(df: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for k in (2, 3, 5, 10):
        tail = df.groupby("id", sort=False).tail(k)
        agg = tail.groupby("id", sort=False)[RECENT_COLS].agg(["mean", "max", "min"])
        agg.columns = [f"xrecent{k}_{name}" for name in flatten_columns(agg.columns)]
        frames.append(agg.astype(np.float32))
    return pd.concat(frames, axis=1)


def add_payment_summary(df: pd.DataFrame, ids: pd.Series) -> pd.DataFrame:
    paym = df[PAYM_COLS]
    loan_summary = pd.DataFrame(
        {
            "paym_zero": (paym == 0).sum(axis=1),
            "paym_one": (paym == 1).sum(axis=1),
            "paym_two": (paym == 2).sum(axis=1),
            "paym_three": (paym == 3).sum(axis=1),
            "paym_four": (paym == 4).sum(axis=1),
            "paym_nonzero": paym.ne(0).sum(axis=1),
            "paym_recent_nonzero": paym.iloc[:, :6].ne(0).sum(axis=1),
            "paym_recent_bad12": paym.iloc[:, :6].isin([1, 2]).sum(axis=1),
            "paym_old_bad12": paym.iloc[:, 19:25].isin([1, 2]).sum(axis=1),
            "paym_max": paym.max(axis=1),
            "paym_mean": paym.mean(axis=1),
            "paym_recent_mean": paym.iloc[:, :6].mean(axis=1),
            "paym_old_mean": paym.iloc[:, 19:25].mean(axis=1),
            "paym_trend_recent_minus_old": paym.iloc[:, :6].mean(axis=1) - paym.iloc[:, 19:25].mean(axis=1),
        },
        index=df.index,
    )
    agg = loan_summary.groupby(ids, sort=False).agg(["mean", "max", "std", "sum", "last"])
    agg.columns = [f"x{name}" for name in flatten_columns(agg.columns)]
    return agg.fillna(0.0).astype(np.float32)


def add_cross_counts(df: pd.DataFrame, ids: pd.Series) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    combos = {
        "status_type": (
            df["enc_loans_credit_status"] * 10 + df["enc_loans_credit_type"],
            [s * 10 + t for s in range(7) for t in range(8)],
        ),
        "status_holder": (
            df["enc_loans_credit_status"] * 10 + df["enc_loans_account_holder_type"],
            [s * 10 + h for s in range(7) for h in range(7)],
        ),
        "type_cur": (
            df["enc_loans_credit_type"] * 10 + df["enc_loans_account_cur"],
            [t * 10 + c for t in range(8) for c in range(4)],
        ),
    }
    for name, (series, values) in combos.items():
        dummies = pd.get_dummies(series, dtype=np.uint8)
        dummies = dummies.reindex(columns=values, fill_value=0)
        dummies.columns = [f"xcnt_{name}_{value}" for value in values]
        pieces.append(dummies.groupby(ids, sort=False).sum().astype(np.float32))
    return pd.concat(pieces, axis=1)


def aggregate_extra_block(df: pd.DataFrame) -> pd.DataFrame:
    ids = df["id"]
    loan_count = df.groupby("id", sort=False).size().astype(np.float32)
    pieces = [
        pd.DataFrame({"id": loan_count.index, "xloan_count_check": loan_count.to_numpy(dtype=np.float32)}).set_index("id"),
        add_dummy_counts(df, ids, loan_count),
        add_recent_features(df),
        add_payment_summary(df, ids),
        add_cross_counts(df, ids),
    ]
    features = pd.concat(pieces, axis=1).reset_index().fillna(0.0)
    return features


def aggregate_extra_file(parquet_path: Path, output_path: Path, batch_size: int = 650_000, force: bool = False) -> None:
    if output_path.exists() and not force:
        log(f"extra features already exist: {output_path.name}")
        return

    log(f"building extra features from {parquet_path.name}")
    output_path.unlink(missing_ok=True)
    pf = pq.ParquetFile(parquet_path)
    writer: pq.ParquetWriter | None = None
    carry: pd.DataFrame | None = None
    total_ids = 0

    for batch_no, batch in enumerate(pf.iter_batches(batch_size=batch_size), start=1):
        df = batch.to_pandas()
        if carry is not None and not carry.empty:
            df = pd.concat([carry, df], axis=0, ignore_index=True)

        last_id = df["id"].iloc[-1]
        complete = df.loc[df["id"] != last_id].copy()
        carry = df.loc[df["id"] == last_id].copy()
        if complete.empty:
            continue

        features = aggregate_extra_block(complete)
        total_ids += len(features)
        table = pa.Table.from_pandas(features, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
        writer.write_table(table)
        log(f"{parquet_path.name}: extra batch {batch_no}, ids={total_ids:,}")
        del df, complete, features, table
        gc.collect()

    if carry is not None and not carry.empty:
        features = aggregate_extra_block(carry)
        total_ids += len(features)
        table = pa.Table.from_pandas(features, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
        writer.write_table(table)
        del features, table

    if writer is not None:
        writer.close()
    log(f"saved {output_path.name}: ids={total_ids:,}")


def feature_columns(path: Path) -> list[str]:
    cols = pq.ParquetFile(path).schema_arrow.names
    return [col for col in cols if col != "id"]


def score_feature_file(path: Path, target_map: pd.Series) -> pd.Series:
    pf = pq.ParquetFile(path)
    cols = [col for col in pf.schema_arrow.names if col != "id"]
    pos_sum = pd.Series(0.0, index=cols)
    neg_sum = pd.Series(0.0, index=cols)
    total_sum = pd.Series(0.0, index=cols)
    total_sq = pd.Series(0.0, index=cols)
    pos_n = 0
    neg_n = 0
    total_n = 0
    for batch_no, batch in enumerate(pf.iter_batches(batch_size=120_000), start=1):
        chunk = batch.to_pandas()
        y = chunk["id"].map(target_map).to_numpy(dtype=np.int8)
        x = chunk[cols]
        pos = y == 1
        neg = ~pos
        if pos.any():
            pos_sum += x.loc[pos].sum(axis=0)
            pos_n += int(pos.sum())
        if neg.any():
            neg_sum += x.loc[neg].sum(axis=0)
            neg_n += int(neg.sum())
        total_sum += x.sum(axis=0)
        total_sq += (x.astype(np.float64) ** 2).sum(axis=0)
        total_n += len(chunk)
        if batch_no % 8 == 0:
            log(f"scored {path.name}: row batch {batch_no}")
        del chunk, x, y
        gc.collect()

    pos_mean = pos_sum / max(pos_n, 1)
    neg_mean = neg_sum / max(neg_n, 1)
    mean = total_sum / max(total_n, 1)
    var = (total_sq / max(total_n, 1)) - mean * mean
    std = np.sqrt(np.maximum(var, 1e-12))
    scores = ((pos_mean - neg_mean).abs() / std).replace([np.inf, -np.inf], 0).fillna(0)
    return scores


def select_features(top_n: int) -> tuple[list[str], list[str], pd.Series]:
    target = pd.read_csv(TARGET_CSV)
    target_map = target.set_index("id")["flag"]
    log("scoring basic features")
    basic_scores = score_feature_file(BASIC_TRAIN, target_map)
    log("scoring extra features")
    extra_scores = score_feature_file(EXTRA_TRAIN, target_map)
    scores = pd.concat(
        [
            basic_scores.rename(index=lambda name: f"basic::{name}"),
            extra_scores.rename(index=lambda name: f"extra::{name}"),
        ]
    ).sort_values(ascending=False)
    keep = scores.head(top_n).index.tolist()

    must_keep = ["basic::id_norm", "basic::loan_count"]
    for name in must_keep:
        if name in scores.index and name not in keep:
            keep.append(name)

    basic_keep = [name.split("::", 1)[1] for name in keep if name.startswith("basic::")]
    extra_keep = [name.split("::", 1)[1] for name in keep if name.startswith("extra::")]
    log(f"selected features: basic={len(basic_keep)}, extra={len(extra_keep)}, total={len(basic_keep) + len(extra_keep)}")
    return basic_keep, extra_keep, scores


def read_feature_matrix(basic_path: Path, extra_path: Path, basic_cols: list[str], extra_cols: list[str]) -> pd.DataFrame:
    basic = pd.read_parquet(basic_path, columns=["id", *basic_cols])
    extra = pd.read_parquet(extra_path, columns=["id", *extra_cols])
    if not basic["id"].equals(extra["id"]):
        extra = extra.set_index("id").loc[basic["id"]].reset_index()
    basic_data = basic[basic_cols].astype(np.float32, copy=False)
    extra_data = extra[extra_cols].astype(np.float32, copy=False).rename(columns=lambda col: f"extra__{col}")
    return pd.concat([basic[["id"]].reset_index(drop=True), basic_data.reset_index(drop=True), extra_data.reset_index(drop=True)], axis=1).copy()


def class_weights(y: np.ndarray, positive_multiplier: float = 1.0) -> np.ndarray:
    pos = y.mean()
    weights = np.where(y == 1, 0.5 / pos, 0.5 / (1.0 - pos))
    weights = weights.astype(np.float32)
    weights[y == 1] *= np.float32(positive_multiplier)
    return weights


def stratified_sample_indices(y: np.ndarray, max_rows: int, random_state: int) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    if len(y) <= max_rows:
        return np.arange(len(y))
    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)
    keep_pos = pos_idx
    neg_keep = max_rows - len(keep_pos)
    if neg_keep < 0:
        keep_pos = rng.choice(pos_idx, size=max_rows // 2, replace=False)
        neg_keep = max_rows - len(keep_pos)
    keep_neg = rng.choice(neg_idx, size=neg_keep, replace=False)
    idx = np.concatenate([keep_pos, keep_neg])
    rng.shuffle(idx)
    return idx


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-8, 1 - 1e-8)
    return np.log(p / (1 - p))


def train_hgb_candidate(
    name: str,
    train_df: pd.DataFrame,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    feature_cols: list[str],
    params: dict,
    weight_multiplier: float,
) -> tuple[HistGradientBoostingClassifier, np.ndarray, float]:
    log(f"training candidate {name}")
    x_train = train_df.iloc[train_idx][feature_cols].to_numpy(dtype=np.float32, copy=True)
    x_val = train_df.iloc[val_idx][feature_cols].to_numpy(dtype=np.float32, copy=True)
    y_train = y[train_idx]
    y_val = y[val_idx]
    model = HistGradientBoostingClassifier(**params)
    model.fit(x_train, y_train, sample_weight=class_weights(y_train, weight_multiplier))
    pred = model.predict_proba(x_val)[:, 1]
    auc = roc_auc_score(y_val, pred)
    log(f"{name} validation ROC-AUC: {auc:.6f}; iterations={getattr(model, 'n_iter_', None)}")
    del x_train, x_val, y_train
    gc.collect()
    return model, pred, float(auc)


def train_extra_trees_candidate(
    train_df: pd.DataFrame,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    feature_cols: list[str],
) -> tuple[ExtraTreesClassifier, np.ndarray, float]:
    log("training ExtraTrees diversity candidate")
    rng = np.random.default_rng(123)
    pos_idx = train_idx[y[train_idx] == 1]
    neg_idx = train_idx[y[train_idx] == 0]
    neg_take = min(len(neg_idx), len(pos_idx) * 12)
    fit_idx = np.concatenate([pos_idx, rng.choice(neg_idx, size=neg_take, replace=False)])
    rng.shuffle(fit_idx)
    cols = feature_cols[:420]
    model = ExtraTreesClassifier(
        n_estimators=260,
        max_features=0.55,
        min_samples_leaf=25,
        max_depth=None,
        bootstrap=False,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=123,
    )
    x_train = train_df.iloc[fit_idx][cols].to_numpy(dtype=np.float32, copy=True)
    x_val = train_df.iloc[val_idx][cols].to_numpy(dtype=np.float32, copy=True)
    model.fit(x_train, y[fit_idx])
    pred = model.predict_proba(x_val)[:, 1]
    auc = roc_auc_score(y[val_idx], pred)
    log(f"ExtraTrees validation ROC-AUC: {auc:.6f}")
    del x_train, x_val
    gc.collect()
    return model, pred, float(auc)


def fit_final_hgb(
    name: str,
    train_df: pd.DataFrame,
    y: np.ndarray,
    feature_cols: list[str],
    params: dict,
    validation_model: HistGradientBoostingClassifier,
    weight_multiplier: float,
    max_rows: int,
    seed: int,
) -> HistGradientBoostingClassifier:
    final_params = params.copy()
    final_params["early_stopping"] = False
    final_params["max_iter"] = max(120, int(getattr(validation_model, "n_iter_", params.get("max_iter", 260))))
    final_params.pop("validation_fraction", None)
    final_params.pop("n_iter_no_change", None)
    final_params["random_state"] = seed
    idx = stratified_sample_indices(y, max_rows=max_rows, random_state=seed)
    x_final = train_df.iloc[idx][feature_cols].to_numpy(dtype=np.float32, copy=True)
    model = HistGradientBoostingClassifier(**final_params)
    log(f"training final {name}: rows={len(idx):,}, iter={final_params['max_iter']}")
    model.fit(x_final, y[idx], sample_weight=class_weights(y[idx], weight_multiplier))
    del x_final
    gc.collect()
    return model


def fit_final_extra_trees(
    train_df: pd.DataFrame,
    y: np.ndarray,
    feature_cols: list[str],
) -> ExtraTreesClassifier:
    rng = np.random.default_rng(321)
    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)
    neg_take = min(len(neg_idx), len(pos_idx) * 14)
    idx = np.concatenate([pos_idx, rng.choice(neg_idx, size=neg_take, replace=False)])
    rng.shuffle(idx)
    cols = feature_cols[:420]
    model = ExtraTreesClassifier(
        n_estimators=340,
        max_features=0.55,
        min_samples_leaf=25,
        bootstrap=False,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=321,
    )
    log(f"training final ExtraTrees: rows={len(idx):,}")
    x_final = train_df.iloc[idx][cols].to_numpy(dtype=np.float32, copy=True)
    model.fit(x_final, y[idx])
    del x_final
    gc.collect()
    return model


def calibrate_to_prior(scores: np.ndarray, prior: float) -> np.ndarray:
    scores = np.clip(scores, 1e-8, 1 - 1e-8)
    ratio = prior / (1.0 - prior)
    return (scores * ratio) / (1.0 - scores + scores * ratio)


def run(force_extra: bool = False, top_n: int = 720, skip_extra_trees: bool = False) -> None:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    aggregate_extra_file(TRAIN_PARQUET, EXTRA_TRAIN, force=force_extra)
    aggregate_extra_file(TEST_PARQUET, EXTRA_TEST, force=force_extra)

    basic_cols, extra_cols, scores = select_features(top_n)
    train_df = read_feature_matrix(BASIC_TRAIN, EXTRA_TRAIN, basic_cols, extra_cols)
    target = pd.read_csv(TARGET_CSV)
    target_map = target.set_index("id")["flag"]
    y = train_df["id"].map(target_map).to_numpy(dtype=np.int8)
    feature_cols = [col for col in train_df.columns if col != "id"]

    pool_idx = stratified_sample_indices(y, max_rows=1_300_000, random_state=91)
    train_idx, val_idx = train_test_split(
        pool_idx,
        test_size=320_000,
        random_state=91,
        stratify=y[pool_idx],
    )

    candidates = [
        (
            "hgb_wide_leaf63",
            {
                "loss": "log_loss",
                "learning_rate": 0.042,
                "max_iter": 360,
                "max_leaf_nodes": 63,
                "min_samples_leaf": 55,
                "l2_regularization": 0.015,
                "max_bins": 128,
                "early_stopping": True,
                "validation_fraction": 0.08,
                "n_iter_no_change": 22,
                "random_state": 91,
                "verbose": 1,
            },
            0.95,
            1_550_000,
            191,
        ),
        (
            "hgb_regular_leaf31",
            {
                "loss": "log_loss",
                "learning_rate": 0.05,
                "max_iter": 310,
                "max_leaf_nodes": 31,
                "min_samples_leaf": 90,
                "l2_regularization": 0.08,
                "max_bins": 128,
                "early_stopping": True,
                "validation_fraction": 0.08,
                "n_iter_no_change": 20,
                "random_state": 92,
                "verbose": 0,
            },
            1.0,
            1_500_000,
            192,
        ),
    ]

    validation_predictions: dict[str, np.ndarray] = {}
    validation_aucs: dict[str, float] = {}
    fitted_validation_models: dict[str, HistGradientBoostingClassifier] = {}

    for name, params, weight_multiplier, _, _ in candidates:
        model, pred, auc = train_hgb_candidate(
            name, train_df, y, train_idx, val_idx, feature_cols, params, weight_multiplier
        )
        fitted_validation_models[name] = model
        validation_predictions[name] = pred
        validation_aucs[name] = auc

    extra_trees_model = None
    if not skip_extra_trees:
        extra_trees_model, pred, auc = train_extra_trees_candidate(train_df, y, train_idx, val_idx, feature_cols)
        validation_predictions["extra_trees"] = pred
        validation_aucs["extra_trees"] = auc

    y_val = y[val_idx]
    ensemble_grid: list[tuple[float, dict[str, float]]] = []
    names = list(validation_predictions)
    if len(names) == 2:
        for w in np.linspace(0.0, 1.0, 21):
            weights = {names[0]: float(w), names[1]: float(1 - w)}
            score = sum(weights[name] * logit(validation_predictions[name]) for name in names)
            auc = roc_auc_score(y_val, score)
            ensemble_grid.append((float(auc), weights))
    else:
        for w1 in np.linspace(0.0, 1.0, 11):
            for w2 in np.linspace(0.0, 1.0 - w1, 11):
                w3 = 1.0 - w1 - w2
                weights = {names[0]: float(w1), names[1]: float(w2), names[2]: float(w3)}
                score = sum(weights[name] * logit(validation_predictions[name]) for name in names)
                auc = roc_auc_score(y_val, score)
                ensemble_grid.append((float(auc), weights))
    best_ensemble_auc, best_weights = max(ensemble_grid, key=lambda item: item[0])
    log(f"best validation ensemble ROC-AUC: {best_ensemble_auc:.6f}; weights={best_weights}")

    cutoff = np.quantile(train_df["id"].to_numpy(), 0.82)
    tail_mask = train_df["id"].to_numpy() > cutoff
    tail_auc = None
    # Use the validation models for a comparable tail readout; it is diagnostic, not final model selection.
    if tail_mask.any() and y[tail_mask].min() != y[tail_mask].max():
        tail_idx = np.flatnonzero(tail_mask)
        if len(tail_idx) > 320_000:
            rng = np.random.default_rng(555)
            tail_idx = rng.choice(tail_idx, size=320_000, replace=False)
        x_tail = train_df.iloc[tail_idx][feature_cols].to_numpy(dtype=np.float32, copy=True)
        tail_score = np.zeros(len(tail_idx), dtype=np.float64)
        for name, weight in best_weights.items():
            if name == "extra_trees":
                cols = feature_cols[:420]
                x_tail_et = train_df.iloc[tail_idx][cols].to_numpy(dtype=np.float32, copy=True)
                tail_score += weight * logit(extra_trees_model.predict_proba(x_tail_et)[:, 1])
                del x_tail_et
            else:
                tail_score += weight * logit(fitted_validation_models[name].predict_proba(x_tail)[:, 1])
        tail_auc = roc_auc_score(y[tail_idx], tail_score)
        log(f"diagnostic id-tail ensemble ROC-AUC: {tail_auc:.6f}")
        del x_tail, tail_score
        gc.collect()

    final_models: dict[str, object] = {}
    for name, params, weight_multiplier, max_rows, seed in candidates:
        if best_weights.get(name, 0.0) > 0:
            final_models[name] = fit_final_hgb(
                name,
                train_df,
                y,
                feature_cols,
                params,
                fitted_validation_models[name],
                weight_multiplier,
                max_rows,
                seed,
            )
    if best_weights.get("extra_trees", 0.0) > 0:
        final_models["extra_trees"] = fit_final_extra_trees(train_df, y, feature_cols)

    del train_df
    gc.collect()

    test_df = read_feature_matrix(BASIC_TEST, EXTRA_TEST, basic_cols, extra_cols)
    test_scores = np.zeros(len(test_df), dtype=np.float64)
    for name, weight in best_weights.items():
        if weight == 0:
            continue
        if name == "extra_trees":
            cols = feature_cols[:420]
            pred = final_models[name].predict_proba(test_df[cols].to_numpy(dtype=np.float32, copy=True))[:, 1]
        else:
            pred = final_models[name].predict_proba(test_df[feature_cols].to_numpy(dtype=np.float32, copy=True))[:, 1]
        test_scores += weight * logit(pred)
        log(f"predicted test with {name}")
        del pred
        gc.collect()

    raw_pred = 1.0 / (1.0 + np.exp(-test_scores))
    prior = float(target["flag"].mean())
    pred = calibrate_to_prior(raw_pred, prior)
    sample = pd.read_csv(SAMPLE_CSV)
    prediction_frame = pd.DataFrame({"id": test_df["id"].to_numpy(), "flag": pred})
    submission = sample[["id"]].merge(prediction_frame, on="id", how="left")
    if submission["flag"].isna().any():
        raise RuntimeError(f"missing predictions: {int(submission['flag'].isna().sum())}")
    submission.to_csv(SUBMISSION_CSV, index=False, float_format="%.7f")

    report = {
        "top_n": top_n,
        "feature_count": len(feature_cols),
        "basic_feature_count": len(basic_cols),
        "extra_feature_count": len(extra_cols),
        "validation_aucs": validation_aucs,
        "best_ensemble_auc": best_ensemble_auc,
        "best_weights": best_weights,
        "id_tail_ensemble_auc": tail_auc,
        "submission": str(SUBMISSION_CSV),
        "submission_bytes": SUBMISSION_CSV.stat().st_size,
        "prediction_min": float(submission["flag"].min()),
        "prediction_max": float(submission["flag"].max()),
        "prediction_mean": float(submission["flag"].mean()),
        "top_features": scores.head(80).to_dict(),
    }
    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"saved improved submission: {SUBMISSION_CSV}; size={SUBMISSION_CSV.stat().st_size / 1_000_000:.2f} MB")
    log(f"prediction range: min={submission['flag'].min():.6f}, max={submission['flag'].max():.6f}, mean={submission['flag'].mean():.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-extra", action="store_true")
    parser.add_argument("--top-n", type=int, default=720)
    parser.add_argument("--skip-extra-trees", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(force_extra=args.force_extra, top_n=args.top_n, skip_extra_trees=args.skip_extra_trees)
