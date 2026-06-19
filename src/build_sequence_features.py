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
TRAIN_SEQ = WORK_DIR / "train_features_seq_v1.parquet"
TEST_SEQ = WORK_DIR / "test_features_seq_v1.parquet"

N_LAST = 14
SOURCE_COLS = [
    "rn",
    "pre_since_opened",
    "pre_since_confirmed",
    "pre_pterm",
    "pre_fterm",
    "pre_till_pclose",
    "pre_till_fclose",
    "pre_loans_credit_limit",
    "pre_loans_next_pay_summ",
    "pre_loans_outstanding",
    "pre_loans_total_overdue",
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
    *[f"enc_paym_{i}" for i in range(25)],
    "enc_loans_account_holder_type",
    "enc_loans_credit_status",
    "enc_loans_credit_type",
    "enc_loans_account_cur",
    "pclose_flag",
    "fclose_flag",
]


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def sequence_block(df: pd.DataFrame) -> pd.DataFrame:
    df = df[["id", *SOURCE_COLS]]
    tail = df.groupby("id", sort=False).tail(N_LAST).copy()
    tail["_pos"] = tail.groupby("id", sort=False).cumcount(ascending=False).astype(np.uint8)
    ids = pd.Index(df["id"].drop_duplicates(), name="id")
    pieces: list[pd.DataFrame] = []

    for pos in range(N_LAST):
        part = tail.loc[tail["_pos"] == pos, ["id", *SOURCE_COLS]].set_index("id")
        part = part.reindex(ids)
        part.columns = [f"seq{pos}_{col}" for col in SOURCE_COLS]
        pieces.append(part)

    out = pd.concat(pieces, axis=1)
    out = out.fillna(255).clip(0, 255).astype(np.uint8)
    out.insert(0, "id", ids.to_numpy(dtype=np.int64))

    # Compact recency deltas that are not just raw sequence values.
    if N_LAST >= 2:
        out["seq_delta0_1_credit_limit"] = (
            out["seq0_pre_loans_credit_limit"].astype(np.int16)
            - out["seq1_pre_loans_credit_limit"].astype(np.int16)
            + 128
        ).clip(0, 255).astype(np.uint8)
        out["seq_delta0_1_util"] = (
            out["seq0_pre_util"].astype(np.int16) - out["seq1_pre_util"].astype(np.int16) + 128
        ).clip(0, 255).astype(np.uint8)
        out["seq_delta0_1_status"] = (
            out["seq0_enc_loans_credit_status"].astype(np.int16)
            - out["seq1_enc_loans_credit_status"].astype(np.int16)
            + 128
        ).clip(0, 255).astype(np.uint8)
    return out.reset_index(drop=True)


def build_file(input_path: Path, output_path: Path, batch_size: int = 650_000, force: bool = False) -> None:
    if output_path.exists() and not force:
        log(f"sequence features already exist: {output_path.name}")
        return

    output_path.unlink(missing_ok=True)
    pf = pq.ParquetFile(input_path)
    writer: pq.ParquetWriter | None = None
    carry: pd.DataFrame | None = None
    total_ids = 0
    columns = ["id", *SOURCE_COLS]

    log(f"building sequence features from {input_path.name}")
    for batch_no, batch in enumerate(pf.iter_batches(batch_size=batch_size, columns=columns), start=1):
        df = batch.to_pandas()
        if carry is not None and not carry.empty:
            df = pd.concat([carry, df], axis=0, ignore_index=True)

        last_id = df["id"].iloc[-1]
        complete = df.loc[df["id"] != last_id].copy()
        carry = df.loc[df["id"] == last_id].copy()
        if complete.empty:
            continue

        features = sequence_block(complete)
        total_ids += len(features)
        table = pa.Table.from_pandas(features, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
        writer.write_table(table)
        log(f"{input_path.name}: seq batch {batch_no}, ids={total_ids:,}")
        del df, complete, features, table
        gc.collect()

    if carry is not None and not carry.empty:
        features = sequence_block(carry)
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
    build_file(TRAIN_PARQUET, TRAIN_SEQ)
    build_file(TEST_PARQUET, TEST_SEQ)
