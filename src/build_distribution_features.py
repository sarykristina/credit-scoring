from __future__ import annotations

import gc
import time
from pathlib import Path
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


DATA_DIR = Path(os.getenv("DATA_DIR", Path(__file__).resolve().parents[1] / "data"))
WORK_DIR = Path(os.getenv("WORK_DIR", Path(__file__).resolve().parents[1] / "work"))

TRAIN_PARQUET = DATA_DIR / "train_data.parquet"
TEST_PARQUET = DATA_DIR / "test_data.parquet"
TRAIN_DIST = WORK_DIR / "train_features_dist_v1.parquet"
TEST_DIST = WORK_DIR / "test_features_dist_v1.parquet"


DUMMY_VALUES: dict[str, list[int]] = {
    "pre_loans_credit_limit": list(range(20)),
    "pre_loans_next_pay_summ": list(range(7)),
    "pre_loans_outstanding": list(range(1, 6)),
    "pre_loans_total_overdue": list(range(2)),
    "pre_loans_max_overdue_sum": list(range(4)),
    "pre_loans_credit_cost_rate": list(range(14)),
    "pre_loans5": [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 13, 16],
    "pre_loans530": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19],
    "pre_loans3060": list(range(10)),
    "pre_loans6090": list(range(5)),
    "pre_loans90": [2, 3, 8, 10, 13, 14, 19],
    "is_zero_loans5": list(range(2)),
    "is_zero_loans530": list(range(2)),
    "is_zero_loans3060": list(range(2)),
    "is_zero_loans6090": list(range(2)),
    "is_zero_loans90": list(range(2)),
    "pre_util": list(range(20)),
    "pre_over2limit": list(range(20)),
    "pre_maxover2limit": list(range(20)),
    "is_zero_util": list(range(2)),
    "is_zero_over2limit": list(range(2)),
    "is_zero_maxover2limit": list(range(2)),
    "enc_loans_account_holder_type": list(range(7)),
    "enc_loans_credit_status": list(range(7)),
    "enc_loans_credit_type": list(range(8)),
    "enc_loans_account_cur": list(range(4)),
    "pclose_flag": list(range(2)),
    "fclose_flag": list(range(2)),
}

RECENT_DUMMY_VALUES: dict[str, list[int]] = {
    "pre_loans_credit_limit": list(range(20)),
    "pre_loans_credit_cost_rate": list(range(14)),
    "pre_loans530": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19],
    "pre_loans3060": list(range(10)),
    "pre_loans6090": list(range(5)),
    "pre_util": list(range(20)),
    "pre_over2limit": list(range(20)),
    "enc_loans_credit_status": list(range(7)),
    "enc_loans_credit_type": list(range(8)),
    "is_zero_loans3060": list(range(2)),
    "is_zero_loans6090": list(range(2)),
    "is_zero_loans90": list(range(2)),
}

CROSS_SPECS: dict[str, tuple[pd.Series, list[int]]] = {}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def expected_columns(prefix: str, values: dict[str, list[int]]) -> list[str]:
    return [f"{prefix}_{col}_{value}" for col, col_values in values.items() for value in col_values]


def dummy_frame(df: pd.DataFrame, columns: dict[str, list[int]], prefix: str) -> pd.DataFrame:
    expected = [f"{col}_{value}" for col, values in columns.items() for value in values]
    dummies = pd.get_dummies(df[list(columns)], columns=list(columns), dtype=np.uint8)
    dummies = dummies.reindex(columns=expected, fill_value=0)
    dummies.columns = [f"{prefix}_{col}" for col in dummies.columns]
    return dummies


def counts_and_props(dummies: pd.DataFrame, ids: pd.Series, loan_count: pd.Series, count_prefix: str, prop_prefix: str) -> pd.DataFrame:
    counts = dummies.groupby(ids, sort=False).sum().astype(np.float32)
    counts.columns = [col.replace(count_prefix, f"{count_prefix}cnt_", 1) for col in counts.columns]
    props = counts.div(loan_count, axis=0).astype(np.float32)
    props.columns = [col.replace(f"{count_prefix}cnt_", prop_prefix, 1) for col in counts.columns]
    return pd.concat([counts, props], axis=1)


def last_one_hot(df: pd.DataFrame, ids_index: pd.Index) -> pd.DataFrame:
    last = df.groupby("id", sort=False).tail(1).set_index("id")
    pieces: list[pd.DataFrame] = []
    for col, values in DUMMY_VALUES.items():
        codes = last[col].reindex(ids_index)
        for value in values:
            pieces.append((codes == value).astype(np.float32).rename(f"dlast_{col}_{value}"))
    return pd.concat(pieces, axis=1)


def recent_props(df: pd.DataFrame, ids_index: pd.Index) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for k in (3, 7):
        tail = df.groupby("id", sort=False).tail(k)
        loan_count = tail.groupby("id", sort=False).size().astype(np.float32)
        dummies = dummy_frame(tail, RECENT_DUMMY_VALUES, f"dr{k}")
        counts = dummies.groupby(tail["id"], sort=False).sum().astype(np.float32)
        props = counts.div(loan_count, axis=0).reindex(ids_index).fillna(0.0).astype(np.float32)
        props.columns = [col.replace(f"dr{k}_", f"dr{k}prop_", 1) for col in props.columns]
        pieces.append(props)
    return pd.concat(pieces, axis=1)


def cross_counts(df: pd.DataFrame, ids: pd.Series, loan_count: pd.Series) -> pd.DataFrame:
    specs = {
        "util_status": (
            df["pre_util"] * 10 + df["enc_loans_credit_status"],
            [u * 10 + s for u in range(20) for s in range(7)],
        ),
        "over_status": (
            df["pre_over2limit"] * 10 + df["enc_loans_credit_status"],
            [u * 10 + s for u in range(20) for s in range(7)],
        ),
        "limit_type": (
            df["pre_loans_credit_limit"] * 10 + df["enc_loans_credit_type"],
            [u * 10 + t for u in range(20) for t in range(8)],
        ),
        "cost_type": (
            df["pre_loans_credit_cost_rate"] * 10 + df["enc_loans_credit_type"],
            [u * 10 + t for u in range(14) for t in range(8)],
        ),
    }
    pieces: list[pd.DataFrame] = []
    for name, (series, values) in specs.items():
        dummies = pd.get_dummies(series, dtype=np.uint8).reindex(columns=values, fill_value=0)
        dummies.columns = [f"dcross_{name}_{value}" for value in values]
        counts = dummies.groupby(ids, sort=False).sum().astype(np.float32)
        props = counts.div(loan_count, axis=0).astype(np.float32)
        props.columns = [col.replace("dcross_", "dcrossprop_", 1) for col in counts.columns]
        pieces.extend([counts, props])
    return pd.concat(pieces, axis=1)


def aggregate_block(df: pd.DataFrame) -> pd.DataFrame:
    ids = df["id"]
    ids_index = pd.Index(ids.drop_duplicates(), name="id")
    loan_count = df.groupby("id", sort=False).size().astype(np.float32)
    all_dummies = dummy_frame(df, DUMMY_VALUES, "d")
    pieces = [
        counts_and_props(all_dummies, ids, loan_count, "d_", "dprop_"),
        last_one_hot(df, ids_index),
        recent_props(df, ids_index),
        cross_counts(df, ids, loan_count),
    ]
    features = pd.concat(pieces, axis=1).reindex(ids_index).fillna(0.0)
    features.insert(0, "id", ids_index.to_numpy(dtype=np.int64))
    return features.reset_index(drop=True)


def build_file(input_path: Path, output_path: Path, batch_size: int = 650_000, force: bool = False) -> None:
    if output_path.exists() and not force:
        log(f"distribution features already exist: {output_path.name}")
        return

    columns = ["id", *DUMMY_VALUES.keys()]
    output_path.unlink(missing_ok=True)
    pf = pq.ParquetFile(input_path)
    writer: pq.ParquetWriter | None = None
    carry: pd.DataFrame | None = None
    total_ids = 0

    log(f"building distribution features from {input_path.name}")
    for batch_no, batch in enumerate(pf.iter_batches(batch_size=batch_size, columns=columns), start=1):
        df = batch.to_pandas()
        if carry is not None and not carry.empty:
            df = pd.concat([carry, df], axis=0, ignore_index=True)

        last_id = df["id"].iloc[-1]
        complete = df.loc[df["id"] != last_id].copy()
        carry = df.loc[df["id"] == last_id].copy()
        if complete.empty:
            continue

        features = aggregate_block(complete)
        total_ids += len(features)
        table = pa.Table.from_pandas(features, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
        writer.write_table(table)
        log(f"{input_path.name}: dist batch {batch_no}, ids={total_ids:,}, cols={features.shape[1]}")
        del df, complete, features, table
        gc.collect()

    if carry is not None and not carry.empty:
        features = aggregate_block(carry)
        total_ids += len(features)
        table = pa.Table.from_pandas(features, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
        writer.write_table(table)
        del features, table

    if writer is not None:
        writer.close()
    log(f"saved {output_path.name}: ids={total_ids:,}")


if __name__ == "__main__":
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    build_file(TRAIN_PARQUET, TRAIN_DIST)
    build_file(TEST_PARQUET, TEST_DIST)
