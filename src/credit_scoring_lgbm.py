from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from credit_scoring_compact import logit
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
from credit_scoring_sequence import SEQ_TEST, SEQ_TRAIN, score_file_fast


WORK_DIR = Path(os.getenv("WORK_DIR", Path(__file__).resolve().parents[1] / "work"))
DIST_TRAIN = WORK_DIR / "train_features_dist_v1.parquet"
DIST_TEST = WORK_DIR / "test_features_dist_v1.parquet"
PAT_TRAIN = WORK_DIR / "train_features_paypat_v1.parquet"
PAT_TEST = WORK_DIR / "test_features_paypat_v1.parquet"
SUBMISSION_CSV = OUTPUT_DIR / "submission_lgbm.csv"
REPORT_JSON = WORK_DIR / "model_report_lgbm.json"
LAST_REPORT_JSON = WORK_DIR / "model_report_lgbm_last.json"
SCORES_JSON = WORK_DIR / "feature_scores_lgbm.json"
CURRENT_BEST_AUC = 0.0
MODEL_SET = "all"
POS_SCALE_MULT = 0.45


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def read_score_cache() -> tuple[pd.Series, pd.Series, pd.Series, pd.Series | None, pd.Series | None] | None:
    if not SCORES_JSON.exists():
        return None
    data = json.loads(SCORES_JSON.read_text(encoding="utf-8"))
    return (
        pd.Series(data["basic"], dtype=np.float64).sort_values(ascending=False),
        pd.Series(data["extra"], dtype=np.float64).sort_values(ascending=False),
        pd.Series(data["seq"], dtype=np.float64).sort_values(ascending=False),
        pd.Series(data["dist"], dtype=np.float64).sort_values(ascending=False) if "dist" in data else None,
        pd.Series(data["pat"], dtype=np.float64).sort_values(ascending=False) if "pat" in data else None,
    )


def write_score_cache(
    basic: pd.Series,
    extra: pd.Series,
    seq: pd.Series,
    dist: pd.Series | None = None,
    pat: pd.Series | None = None,
) -> None:
    payload = {
        "basic": basic.sort_values(ascending=False).to_dict(),
        "extra": extra.sort_values(ascending=False).to_dict(),
        "seq": seq.sort_values(ascending=False).to_dict(),
    }
    if dist is not None:
        payload["dist"] = dist.sort_values(ascending=False).to_dict()
    if pat is not None:
        payload["pat"] = pat.sort_values(ascending=False).to_dict()
    SCORES_JSON.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def select_features(
    basic_n: int,
    extra_n: int,
    seq_n: int,
    dist_n: int,
    pat_n: int = 0,
    force_rescore: bool = False,
) -> tuple[list[str], list[str], list[str], list[str], list[str], dict]:
    target = pd.read_csv(TARGET_CSV)
    target_map = target.set_index("id")["flag"]
    cached = None if force_rescore else read_score_cache()
    if cached is None:
        log("LGBM scoring basic features")
        basic_scores = score_feature_file(BASIC_TRAIN, target_map).sort_values(ascending=False)
        log("LGBM scoring extra features")
        extra_scores = score_feature_file(EXTRA_TRAIN, target_map).sort_values(ascending=False)
        log("LGBM scoring sequence features")
        seq_scores = score_file_fast(SEQ_TRAIN, target_map)
        dist_scores = None
        pat_scores = None
        if dist_n > 0:
            log("LGBM scoring distribution features")
            dist_scores = score_file_fast(DIST_TRAIN, target_map, batch_size=35_000)
        if pat_n > 0:
            log("LGBM scoring payment-pattern features")
            pat_scores = score_file_fast(PAT_TRAIN, target_map, batch_size=70_000)
        write_score_cache(basic_scores, extra_scores, seq_scores, dist_scores, pat_scores)
    else:
        basic_scores, extra_scores, seq_scores, dist_scores, pat_scores = cached
        log("LGBM loaded cached feature scores")
        if dist_n > 0 and dist_scores is None:
            log("LGBM scoring distribution features")
            dist_scores = score_file_fast(DIST_TRAIN, target_map, batch_size=35_000)
            write_score_cache(basic_scores, extra_scores, seq_scores, dist_scores, pat_scores)
        if pat_n > 0 and pat_scores is None:
            log("LGBM scoring payment-pattern features")
            pat_scores = score_file_fast(PAT_TRAIN, target_map, batch_size=70_000)
            write_score_cache(basic_scores, extra_scores, seq_scores, dist_scores, pat_scores)

    basic_cols = basic_scores.head(basic_n).index.tolist()
    extra_cols = extra_scores.head(extra_n).index.tolist()
    seq_cols = seq_scores.head(seq_n).index.tolist()
    dist_cols = [] if dist_n <= 0 or dist_scores is None else dist_scores.head(dist_n).index.tolist()
    pat_cols = [] if pat_n <= 0 or pat_scores is None else pat_scores.head(pat_n).index.tolist()
    for col in ("id_norm", "loan_count"):
        if col in basic_scores.index and col not in basic_cols:
            basic_cols.append(col)
    report = {
        "basic_top": basic_scores.head(50).to_dict(),
        "extra_top": extra_scores.head(50).to_dict(),
        "seq_top": seq_scores.head(80).to_dict(),
        "dist_top": {} if dist_scores is None else dist_scores.head(80).to_dict(),
        "pat_top": {} if pat_scores is None else pat_scores.head(80).to_dict(),
    }
    log(
        "LGBM selected features: "
        f"basic={len(basic_cols)}, extra={len(extra_cols)}, seq={len(seq_cols)}, "
        f"dist={len(dist_cols)}, pat={len(pat_cols)}"
    )
    return basic_cols, extra_cols, seq_cols, dist_cols, pat_cols, report


def make_sample_ids(
    max_train: int,
    val_size: int,
    seed: int,
) -> tuple[set[int], set[int], pd.Series]:
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
    log(f"LGBM sample ids: train={len(train_idx):,}, val={len(val_idx):,}")
    return set(map(int, ids[train_idx])), set(map(int, ids[val_idx])), target.set_index("id")["flag"]


def load_rows(path: Path, columns: list[str], ids_needed: set[int], label: str, batch_size: int = 160_000) -> pd.DataFrame:
    pf = pq.ParquetFile(path)
    chunks: list[pd.DataFrame] = []
    rows = 0
    for batch_no, batch in enumerate(pf.iter_batches(batch_size=batch_size, columns=["id", *columns]), start=1):
        chunk = batch.to_pandas()
        chunk = chunk.loc[chunk["id"].isin(ids_needed)]
        if not chunk.empty:
            rows += len(chunk)
            chunks.append(chunk)
        if batch_no % 8 == 0:
            log(f"LGBM loaded {label}: batch {batch_no}, rows={rows:,}")
        del chunk
        gc.collect()
    out = pd.concat(chunks, axis=0, ignore_index=True)
    for col in columns:
        out[col] = out[col].astype(np.float32, copy=False)
    log(f"LGBM loaded {label}: {len(out):,} rows")
    return out


def load_matrix(
    basic_cols: list[str],
    extra_cols: list[str],
    seq_cols: list[str],
    dist_cols: list[str],
    pat_cols: list[str],
    ids_needed: set[int],
) -> pd.DataFrame:
    basic = load_rows(BASIC_TRAIN, basic_cols, ids_needed, "basic")
    extra = load_rows(EXTRA_TRAIN, extra_cols, ids_needed, "extra").rename(
        columns={col: f"extra__{col}" for col in extra_cols}
    )
    seq = load_rows(SEQ_TRAIN, seq_cols, ids_needed, "seq").rename(columns={col: f"seq__{col}" for col in seq_cols})
    data = basic.merge(extra, on="id", how="inner").merge(seq, on="id", how="inner")
    if dist_cols:
        dist = load_rows(DIST_TRAIN, dist_cols, ids_needed, "dist").rename(
            columns={col: f"dist__{col}" for col in dist_cols}
        )
        data = data.merge(dist, on="id", how="inner")
        del dist
    if pat_cols:
        pat = load_rows(PAT_TRAIN, pat_cols, ids_needed, "paypat").rename(
            columns={col: f"pat__{col}" for col in pat_cols}
        )
        data = data.merge(pat, on="id", how="inner")
        del pat
    del basic, extra, seq
    gc.collect()
    log(f"LGBM merged train matrix: {data.shape}")
    return data


def lgb_dataset(x: pd.DataFrame, y: np.ndarray) -> lgb.Dataset:
    return lgb.Dataset(x, label=y, free_raw_data=False)


def train_candidates(
    data: pd.DataFrame,
    target_map: pd.Series,
    train_ids: set[int],
    val_ids: set[int],
) -> tuple[list[lgb.Booster], list[float], list[str], dict]:
    features = [col for col in data.columns if col != "id"]
    y = data["id"].map(target_map).to_numpy(dtype=np.int8)
    train_mask = data["id"].isin(train_ids).to_numpy()
    val_mask = data["id"].isin(val_ids).to_numpy()
    x_train = data.loc[train_mask, features].astype(np.float32, copy=False)
    y_train = y[train_mask]
    x_val = data.loc[val_mask, features].astype(np.float32, copy=False)
    y_val = y[val_mask]

    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    ratio = neg / max(pos, 1)
    log(f"LGBM train positives={pos:,}, negatives={neg:,}, ratio={ratio:.2f}")

    all_configs = [
        (
            "lgbm_gbdt_balanced",
            {
                "objective": "binary",
                "metric": "auc",
                "learning_rate": 0.030,
                "num_leaves": 96,
                "min_data_in_leaf": 90,
                "feature_fraction": 0.82,
                "bagging_fraction": 0.84,
                "bagging_freq": 1,
                "lambda_l1": 0.02,
                "lambda_l2": 1.2,
                "max_bin": 127,
                "force_col_wise": True,
                "scale_pos_weight": min(ratio, 24.0),
                "verbosity": -1,
                "seed": 1701,
                "feature_fraction_seed": 1702,
                "bagging_seed": 1703,
                "num_threads": -1,
            },
            3200,
        ),
        (
            "lgbm_gbdt_rankish",
            {
                "objective": "binary",
                "metric": "auc",
                "learning_rate": 0.038,
                "num_leaves": 64,
                "min_data_in_leaf": 140,
                "feature_fraction": 0.76,
                "bagging_fraction": 0.78,
                "bagging_freq": 1,
                "lambda_l1": 0.00,
                "lambda_l2": 2.2,
                "max_bin": 127,
                "force_col_wise": True,
                "scale_pos_weight": max(1.0, min(ratio * POS_SCALE_MULT, 13.0)),
                "verbosity": -1,
                "seed": 2701,
                "feature_fraction_seed": 2702,
                "bagging_seed": 2703,
                "num_threads": -1,
            },
            2600,
        ),
    ]
    if MODEL_SET == "balanced":
        configs = all_configs[:1]
    elif MODEL_SET == "rankish":
        configs = all_configs[1:]
    elif MODEL_SET == "goss":
        configs = [
            (
                "lgbm_goss",
                {
                    "boosting_type": "goss",
                    "objective": "binary",
                    "metric": "auc",
                    "learning_rate": 0.035,
                    "num_leaves": 80,
                    "min_data_in_leaf": 120,
                    "feature_fraction": 0.80,
                    "top_rate": 0.24,
                    "other_rate": 0.12,
                    "lambda_l1": 0.00,
                    "lambda_l2": 2.8,
                    "max_bin": 127,
                    "force_col_wise": True,
                    "scale_pos_weight": max(1.0, min(ratio * 0.55, 13.0)),
                    "verbosity": -1,
                    "seed": 3701,
                    "feature_fraction_seed": 3702,
                    "num_threads": -1,
                },
                3000,
            )
        ]
    elif MODEL_SET == "rankish_ensemble":
        configs = [
            all_configs[1],
            (
                "lgbm_rankish_deeper_seed",
                {
                    "objective": "binary",
                    "metric": "auc",
                    "learning_rate": 0.034,
                    "num_leaves": 80,
                    "min_data_in_leaf": 115,
                    "feature_fraction": 0.80,
                    "bagging_fraction": 0.80,
                    "bagging_freq": 1,
                    "lambda_l1": 0.00,
                    "lambda_l2": 3.0,
                    "max_bin": 127,
                    "force_col_wise": True,
                    "scale_pos_weight": max(1.0, min(ratio * POS_SCALE_MULT, 13.0)),
                    "verbosity": -1,
                    "seed": 8701,
                    "feature_fraction_seed": 8702,
                    "bagging_seed": 8703,
                    "num_threads": -1,
                },
                3000,
            ),
        ]
    else:
        configs = all_configs

    train_set = lgb_dataset(x_train, y_train)
    val_set = lgb_dataset(x_val, y_val)
    validation_preds: list[np.ndarray] = []
    validation_models: list[lgb.Booster] = []
    report: dict = {"model_aucs": {}, "best_iterations": {}, "feature_count": len(features)}

    for name, params, rounds in configs:
        log(f"training {name}")
        model = lgb.train(
            params,
            train_set,
            num_boost_round=rounds,
            valid_sets=[val_set],
            valid_names=["valid"],
            callbacks=[lgb.early_stopping(180, first_metric_only=True), lgb.log_evaluation(25)],
        )
        pred = model.predict(x_val, num_iteration=model.best_iteration)
        auc = roc_auc_score(y_val, pred)
        log(f"{name} validation ROC-AUC: {auc:.6f}; best_iteration={model.best_iteration}")
        validation_models.append(model)
        validation_preds.append(pred)
        report["model_aucs"][name] = float(auc)
        report["best_iterations"][name] = int(model.best_iteration)
        gc.collect()

    if len(validation_preds) == 1:
        best_auc = float(next(iter(report["model_aucs"].values())))
        best_weights = [1.0]
    else:
        best_auc = -1.0
        best_weights = [1.0, 0.0]
        for w in np.linspace(0.0, 1.0, 41):
            score = w * logit(validation_preds[0]) + (1.0 - w) * logit(validation_preds[1])
            auc = roc_auc_score(y_val, score)
            if auc > best_auc:
                best_auc = float(auc)
                best_weights = [float(w), float(1.0 - w)]
    log(f"LGBM validation ensemble ROC-AUC: {best_auc:.6f}; weights={best_weights}")
    report["ensemble_auc"] = best_auc
    report["ensemble_weights"] = best_weights
    if best_auc <= CURRENT_BEST_AUC:
        report["final_skipped"] = True
        log(f"LGBM did not beat current local best {CURRENT_BEST_AUC:.6f}; skipping final fit")
        del x_train, x_val, train_set, val_set
        gc.collect()
        return validation_models, best_weights, features, report

    final_models: list[lgb.Booster] = []
    x_final = data[features].astype(np.float32, copy=False)
    y_final = y
    final_set = lgb_dataset(x_final, y_final)
    for (name, params, _), validation_model in zip(configs, validation_models):
        final_rounds = int(validation_model.best_iteration or 500)
        final_params = params.copy()
        final_params["seed"] += 100
        log(f"training final {name}: rows={len(x_final):,}, rounds={final_rounds}")
        final_model = lgb.train(
            final_params,
            final_set,
            num_boost_round=final_rounds,
            callbacks=[lgb.log_evaluation(100)],
        )
        final_models.append(final_model)
        gc.collect()

    del x_train, x_val, x_final, train_set, val_set, final_set
    gc.collect()
    return final_models, best_weights, features, report


def predict_test(
    models: list[lgb.Booster],
    weights: list[float],
    basic_cols: list[str],
    extra_cols: list[str],
    seq_cols: list[str],
    dist_cols: list[str],
    pat_cols: list[str],
    features: list[str],
    prior: float,
) -> pd.DataFrame:
    basic_pf = pq.ParquetFile(BASIC_TEST)
    extra_pf = pq.ParquetFile(EXTRA_TEST)
    seq_pf = pq.ParquetFile(SEQ_TEST)
    n_groups = basic_pf.num_row_groups
    dist_index = None
    if dist_cols:
        dist_all = pd.read_parquet(DIST_TEST, columns=["id", *dist_cols]).rename(
            columns={col: f"dist__{col}" for col in dist_cols}
        )
        dist_index = dist_all.set_index("id")
        del dist_all
    pat_index = None
    if pat_cols:
        pat_all = pd.read_parquet(PAT_TEST, columns=["id", *pat_cols]).rename(
            columns={col: f"pat__{col}" for col in pat_cols}
        )
        pat_index = pat_all.set_index("id")
        del pat_all
    parts: list[pd.DataFrame] = []

    for rg in range(n_groups):
        basic = basic_pf.read_row_group(rg, columns=["id", *basic_cols]).to_pandas()
        extra = extra_pf.read_row_group(rg, columns=["id", *extra_cols]).to_pandas().rename(
            columns={col: f"extra__{col}" for col in extra_cols}
        )
        seq = seq_pf.read_row_group(rg, columns=["id", *seq_cols]).to_pandas().rename(
            columns={col: f"seq__{col}" for col in seq_cols}
        )
        data = basic.merge(extra, on="id", how="inner").merge(seq, on="id", how="inner")
        if dist_index is not None:
            dist = dist_index.loc[data["id"].to_numpy()].reset_index()
            data = data.merge(dist, on="id", how="inner")
            del dist
        if pat_index is not None:
            pat = pat_index.loc[data["id"].to_numpy()].reset_index()
            data = data.merge(pat, on="id", how="inner")
            del pat
        x = data[features].astype(np.float32, copy=False)
        score = np.zeros(len(data), dtype=np.float64)
        for model, weight in zip(models, weights):
            if weight == 0:
                continue
            pred = model.predict(x, num_iteration=model.best_iteration)
            score += weight * logit(pred)
        raw = 1.0 / (1.0 + np.exp(-score))
        pred = calibrate_to_prior(raw, prior)
        parts.append(pd.DataFrame({"id": data["id"].to_numpy(), "flag": pred}))
        log(f"LGBM predicted test row-group {rg + 1}/{n_groups}")
        del basic, extra, seq, data, x, raw, pred
        gc.collect()
    return pd.concat(parts, axis=0, ignore_index=True)


def validate_submission(path: Path) -> dict:
    sample = pd.read_csv(SAMPLE_CSV)
    submission = pd.read_csv(path)
    if len(submission) != len(sample):
        raise RuntimeError(f"bad row count: {len(submission)} != {len(sample)}")
    if not submission["id"].equals(sample["id"]):
        raise RuntimeError("submission id order differs from sample")
    if submission["flag"].isna().any():
        raise RuntimeError("submission contains NaN")
    if not submission["flag"].between(0, 1).all():
        raise RuntimeError("submission probabilities are outside [0, 1]")
    return {
        "rows": int(len(submission)),
        "bytes": int(path.stat().st_size),
        "min": float(submission["flag"].min()),
        "max": float(submission["flag"].max()),
        "mean": float(submission["flag"].mean()),
    }


def run(args: argparse.Namespace) -> None:
    global MODEL_SET, POS_SCALE_MULT
    MODEL_SET = args.model_set
    POS_SCALE_MULT = args.pos_scale_mult
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    basic_cols, extra_cols, seq_cols, dist_cols, pat_cols, feature_report = select_features(
        args.basic_features,
        args.extra_features,
        args.seq_features,
        args.dist_features,
        args.pat_features,
        force_rescore=args.force_rescore,
    )
    train_ids, val_ids, target_map = make_sample_ids(args.max_train, args.val_size, args.seed)
    data = load_matrix(basic_cols, extra_cols, seq_cols, dist_cols, pat_cols, train_ids | val_ids)
    models, weights, features, model_report = train_candidates(data, target_map, train_ids, val_ids)
    if model_report.get("final_skipped"):
        report = {
            **feature_report,
            **model_report,
            "basic_features": len(basic_cols),
            "extra_features": len(extra_cols),
            "seq_features": len(seq_cols),
            "dist_features": len(dist_cols),
            "pat_features": len(pat_cols),
            "max_train": args.max_train,
            "val_size": args.val_size,
            "seed": args.seed,
            "submission": None,
            "beats_current_best_local": False,
        }
        LAST_REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return
    prior = float(pd.read_csv(TARGET_CSV)["flag"].mean())
    pred = predict_test(models, weights, basic_cols, extra_cols, seq_cols, dist_cols, pat_cols, features, prior)
    sample = pd.read_csv(SAMPLE_CSV)
    submission = sample[["id"]].merge(pred, on="id", how="left")
    if submission["flag"].isna().any():
        raise RuntimeError(f"missing predictions: {int(submission['flag'].isna().sum())}")
    submission.to_csv(SUBMISSION_CSV, index=False, float_format="%.7f")
    validation = validate_submission(SUBMISSION_CSV)

    report = {
        **feature_report,
        **model_report,
        "basic_features": len(basic_cols),
        "extra_features": len(extra_cols),
        "seq_features": len(seq_cols),
        "dist_features": len(dist_cols),
        "pat_features": len(pat_cols),
        "max_train": args.max_train,
        "val_size": args.val_size,
        "seed": args.seed,
        "submission": str(SUBMISSION_CSV),
        "submission_validation": validation,
        "beats_current_best_local": bool(model_report["ensemble_auc"] > CURRENT_BEST_AUC),
    }
    LAST_REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"saved LGBM submission: {SUBMISSION_CSV}; size={validation['bytes'] / 1_000_000:.2f} MB")
    log(f"prediction range: min={validation['min']:.6f}, max={validation['max']:.6f}, mean={validation['mean']:.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--basic-features", type=int, default=260)
    parser.add_argument("--extra-features", type=int, default=300)
    parser.add_argument("--seq-features", type=int, default=240)
    parser.add_argument("--dist-features", type=int, default=0)
    parser.add_argument("--pat-features", type=int, default=0)
    parser.add_argument("--max-train", type=int, default=950_000)
    parser.add_argument("--val-size", type=int, default=260_000)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--force-rescore", action="store_true")
    parser.add_argument(
        "--model-set",
        choices=["all", "balanced", "rankish", "goss", "rankish_ensemble"],
        default="all",
    )
    parser.add_argument("--pos-scale-mult", type=float, default=0.45)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
