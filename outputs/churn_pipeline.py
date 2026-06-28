#!/usr/bin/env python3
"""
Large-scale fintech churn prediction pipeline.

Raw processing is Spark-only. The final account-level feature table is small
enough for LightGBM/Optuna on most competition machines; if it is still too
large, lower --train-sample-frac or use a bigger driver.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import lightgbm as lgb
import matplotlib
import numpy as np
import optuna
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from pyspark import StorageLevel
from pyspark.ml.feature import Imputer, StringIndexer
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from sklearn.metrics import roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")


@dataclass
class Config:
    data_dir: str = "/Users/shanzidemon/Documents/CompNSU"
    work_dir: str = "./work/churn_pipeline"
    output_dir: str = "./outputs"
    cutoff_date: str = "2024-04-01"
    churn_horizon_days: int = 30
    valid_frac: float = 0.20
    train_sample_frac: float = 1.0
    optuna_trials: int = 50
    random_seed: int = 42
    spark_partitions: int = 800
    driver_memory: str = "8g"
    executor_memory: str = "8g"
    spark_max_result_size: str = "4g"
    reuse_feature_store: bool = False
    account_col: str = "ACCOUNT_ID"
    src_account_col: str = "SRC_ACCOUNT"
    dst_account_col: str = "DST_ACCOUNT"
    txn_date_col: str = "TRX_DATETIME"
    txn_amount_col: str = "TRX_AMT"
    txn_type_col: str = "TRX_TYPE"
    balance_date_col: str = "DATE"
    balance_amount_col: str = "AVAILABLE_BALANCE"
    label_col: str = "CHURN"


def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    for field, value in asdict(Config()).items():
        arg = "--" + field.replace("_", "-")
        if isinstance(value, bool):
            parser.add_argument(arg, action="store_true")
        else:
            parser.add_argument(arg, type=type(value), default=value)
    return Config(**vars(parser.parse_args()))


def build_spark(cfg: Config) -> SparkSession:
    return (
        SparkSession.builder.appName("fictipay-churn-prediction")
        .config("spark.sql.shuffle.partitions", str(cfg.spark_partitions))
        .config("spark.default.parallelism", str(cfg.spark_partitions))
        .config("spark.driver.memory", cfg.driver_memory)
        .config("spark.executor.memory", cfg.executor_memory)
        .config("spark.driver.maxResultSize", cfg.spark_max_result_size)
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .getOrCreate()
    )


def existing_paths(paths: Iterable[Path]) -> list[str]:
    return [str(p) for p in paths if p.exists()]


def discover_inputs(data_dir: str) -> dict[str, list[str] | str | None]:
    root = Path(data_dir)
    parquet_dirs = [p for p in root.rglob("*") if p.is_dir()]
    parquet_files = [p for p in root.rglob("*.parquet") if p.is_file()]
    csv_files = [p for p in root.rglob("*.csv") if p.is_file()]

    def dirs_named(*names: str) -> list[str]:
        lowered = {n.lower() for n in names}
        return [str(p) for p in parquet_dirs if p.name.lower() in lowered]

    def files_matching(*tokens: str) -> list[str]:
        lowered = [t.lower() for t in tokens]
        return [str(p) for p in parquet_files if any(t in p.name.lower() or t in str(p.parent).lower() for t in lowered)]

    def first_csv(*tokens: str) -> str | None:
        lowered = [t.lower() for t in tokens]
        return next((str(p) for p in csv_files if all(t in p.name.lower() for t in lowered)), None)

    def first_parquet(*tokens: str) -> str | None:
        lowered = [t.lower() for t in tokens]
        return next((str(p) for p in parquet_files if all(t in p.name.lower() for t in lowered)), None)

    transaction_paths = existing_paths(
        [
            root / "transactions",
            root / "transaction",
            root / "trx",
            root / "public" / "transactions",
        ]
    ) or dirs_named("transactions", "transaction", "trx") or files_matching("trx", "transaction")

    balance_paths = existing_paths(
        [
            root / "dayend_balance",
            root / "DayEndBalance",
            root / "balance",
            root / "public" / "dayend_balance",
        ]
    ) or dirs_named("dayend_balance", "dayendbalance", "balance") or files_matching("balance")

    return {
        "transactions": transaction_paths,
        "balances": balance_paths,
        "kyc": next(
            (
                str(p)
                for p in [
                    root / "kyc.parquet",
                    root / "KYC.parquet",
                    root / "public" / "kyc.parquet",
                ]
                if p.exists()
            ),
            first_parquet("kyc"),
        ),
        "train_labels": next(
            (
                str(p)
                for p in [
                    root / "train_labels.csv",
                    root / "public" / "train_labels.csv",
                    root / "labels.csv",
                ]
                if p.exists()
            ),
            first_csv("train", "label") or first_csv("labels"),
        ),
        "test_ids": next(
            (
                str(p)
                for p in [
                    root / "test.csv",
                    root / "public" / "test.csv",
                    root / "sample_submission.csv",
                    root / "public" / "sample_submission.csv",
                ]
                if p.exists()
            ),
            first_csv("test") or first_csv("sample", "submission"),
        ),
    }


def read_any(spark: SparkSession, paths: list[str] | str, fmt_hint: str | None = None) -> DataFrame:
    if isinstance(paths, str):
        path_list = [paths]
    else:
        path_list = paths
    if not path_list:
        raise FileNotFoundError("No input paths found.")
    suffix = Path(path_list[0]).suffix.lower()
    if fmt_hint == "csv" or suffix == ".csv":
        return spark.read.option("header", True).option("inferSchema", True).csv(path_list)
    if fmt_hint == "parquet" or suffix == ".parquet" or Path(path_list[0]).is_dir():
        return spark.read.parquet(*path_list)
    return spark.read.option("header", True).option("inferSchema", True).csv(path_list)


def parquet_files(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            files.extend(sorted(p for p in path.rglob("*.parquet") if p.is_file()))
        elif path.suffix.lower() == ".parquet":
            files.append(path)
    return files


def cast_table_timestamps_to_us(table: pa.Table) -> pa.Table:
    arrays = []
    fields = []
    changed = False
    for field, column in zip(table.schema, table.itercolumns()):
        if pa.types.is_timestamp(field.type) and field.type.unit == "ns":
            target_type = pa.timestamp("us", tz=field.type.tz)
            arrays.append(column.cast(target_type, safe=False))
            fields.append(pa.field(field.name, target_type, nullable=field.nullable, metadata=field.metadata))
            changed = True
        else:
            arrays.append(column)
            fields.append(field)
    if not changed:
        return table
    return pa.Table.from_arrays(arrays, schema=pa.schema(fields, metadata=table.schema.metadata))


def parquet_has_timestamp_nanos(path: Path) -> bool:
    schema = pq.ParquetFile(path).schema_arrow
    return any(pa.types.is_timestamp(field.type) and field.type.unit == "ns" for field in schema)


def make_spark_compatible_parquet(paths: list[str], output_dir: str, batch_size: int = 250_000) -> list[str]:
    source_files = parquet_files(paths)
    if not source_files:
        return paths
    if not any(parquet_has_timestamp_nanos(path) for path in source_files):
        return paths

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"Converting {len(source_files)} Parquet file(s) with TIMESTAMP(NANOS) to Spark-compatible TIMESTAMP_MICROS...")

    converted: list[str] = []
    for idx, source in enumerate(source_files):
        target = out_root / f"part-{idx:05d}-{source.stem}.parquet"
        converted.append(str(target))
        if target.exists():
            continue

        pf = pq.ParquetFile(source)
        writer = None
        try:
            for batch in pf.iter_batches(batch_size=batch_size):
                table = cast_table_timestamps_to_us(pa.Table.from_batches([batch]))
                if writer is None:
                    writer = pq.ParquetWriter(target, table.schema, compression="snappy")
                writer.write_table(table)
        finally:
            if writer is not None:
                writer.close()
    return converted


def normalize_inputs(cfg: Config, spark: SparkSession) -> tuple[DataFrame, DataFrame, DataFrame, DataFrame, DataFrame]:
    paths = discover_inputs(cfg.data_dir)
    if not paths["transactions"] or not paths["balances"] or not paths["kyc"]:
        raise FileNotFoundError(
            f"Could not discover transactions, balances, and KYC under {cfg.data_dir}. "
            f"Discovered: {json.dumps(paths, indent=2)}"
        )

    compat_root = Path(cfg.work_dir) / "spark_compatible_parquet"
    trx_paths = make_spark_compatible_parquet(list(paths["transactions"]), str(compat_root / "transactions"))
    bal_paths = make_spark_compatible_parquet(list(paths["balances"]), str(compat_root / "balances"))
    kyc_paths = make_spark_compatible_parquet([str(paths["kyc"])], str(compat_root / "kyc"))

    trx = read_any(spark, trx_paths)
    bal = read_any(spark, bal_paths)
    kyc = read_any(spark, kyc_paths)
    labels = read_any(spark, paths["train_labels"], "csv") if paths["train_labels"] else None
    test_ids = read_any(spark, paths["test_ids"], "csv") if paths["test_ids"] else None

    if labels is None:
        raise FileNotFoundError("train_labels.csv was not found.")
    if test_ids is None:
        raise FileNotFoundError("test.csv or sample_submission.csv was not found.")

    trx = (
        trx.withColumnRenamed(cfg.src_account_col, "ACCOUNT_ID")
        .withColumnRenamed(cfg.dst_account_col, "DST_ACCOUNT_ID")
        .withColumnRenamed(cfg.txn_date_col, "TXN_TS")
        .withColumnRenamed(cfg.txn_amount_col, "TXN_AMOUNT")
        .withColumnRenamed(cfg.txn_type_col, "TXN_TYPE")
        .withColumn("TXN_TS", F.to_timestamp("TXN_TS"))
        .withColumn("TXN_DATE", F.to_date("TXN_TS"))
        .withColumn("TXN_AMOUNT", F.col("TXN_AMOUNT").cast("double"))
    )
    bal = (
        bal.withColumnRenamed(cfg.account_col, "ACCOUNT_ID")
        .withColumnRenamed(cfg.balance_date_col, "BAL_DATE")
        .withColumnRenamed(cfg.balance_amount_col, "BALANCE")
        .withColumn("BAL_DATE", F.to_date("BAL_DATE"))
        .withColumn("BALANCE", F.col("BALANCE").cast("double"))
    )
    kyc = kyc.withColumnRenamed(cfg.account_col, "ACCOUNT_ID")
    labels = labels.withColumnRenamed(cfg.account_col, "ACCOUNT_ID").withColumnRenamed(cfg.label_col, "CHURN")
    test_ids = test_ids.withColumnRenamed(cfg.account_col, "ACCOUNT_ID").select("ACCOUNT_ID").distinct()

    cutoff = F.to_date(F.lit(cfg.cutoff_date))
    trx = trx.filter(F.col("TXN_DATE") < cutoff)
    bal = bal.filter(F.col("BAL_DATE") < cutoff)

    return trx, bal, kyc, labels, test_ids


def maybe_customer_ids(kyc: DataFrame) -> DataFrame:
    if "ACCOUNT_TYPE" in kyc.columns:
        return kyc.filter(F.lower(F.col("ACCOUNT_TYPE")) == "customer").select("ACCOUNT_ID").distinct()
    return kyc.select("ACCOUNT_ID").distinct()


def add_window_flags(df: DataFrame, date_col: str, cutoff_date: str) -> DataFrame:
    cutoff = F.to_date(F.lit(cutoff_date))
    days_ago = F.datediff(cutoff, F.col(date_col))
    return (
        df.withColumn("days_before_cutoff", days_ago)
        .withColumn("in_7d", F.col("days_before_cutoff").between(1, 7).cast("int"))
        .withColumn("in_30d", F.col("days_before_cutoff").between(1, 30).cast("int"))
        .withColumn("in_90d", F.col("days_before_cutoff").between(1, 90).cast("int"))
    )


def build_transaction_features(trx: DataFrame, cfg: Config) -> DataFrame:
    t = add_window_flags(trx, "TXN_DATE", cfg.cutoff_date).persist(StorageLevel.DISK_ONLY)
    type_lower = F.lower(F.coalesce(F.col("TXN_TYPE").cast("string"), F.lit("")))
    is_p2p = type_lower.rlike("p2p|send|transfer")
    is_bill = type_lower.rlike("bill")
    is_merchant = type_lower.rlike("merchant|pay")
    is_cashout = type_lower.rlike("cash.?out|cashout")
    days = F.col("days_before_cutoff")

    base = t.groupBy("ACCOUNT_ID").agg(
        F.count("*").alias("txn_count_90d"),
        F.sum(F.when(days == 1, 1).otherwise(0)).alias("txn_count_1d"),
        F.sum(F.when(days.between(1, 2), 1).otherwise(0)).alias("txn_count_2d"),
        F.sum(F.when(days.between(1, 3), 1).otherwise(0)).alias("txn_count_3d"),
        F.sum(F.when(days.between(1, 5), 1).otherwise(0)).alias("txn_count_5d"),
        F.sum(F.when(days.between(1, 14), 1).otherwise(0)).alias("txn_count_14d"),
        F.sum(F.when(days.between(1, 21), 1).otherwise(0)).alias("txn_count_21d"),
        F.sum(F.when(days.between(1, 45), 1).otherwise(0)).alias("txn_count_45d"),
        F.sum(F.when(days.between(1, 60), 1).otherwise(0)).alias("txn_count_60d"),
        F.sum(F.when(days.between(31, 60), 1).otherwise(0)).alias("txn_count_31_60d"),
        F.sum(F.when(days.between(61, 90), 1).otherwise(0)).alias("txn_count_61_90d"),
        F.sum(F.when(F.col("in_30d") == 1, 1).otherwise(0)).alias("txn_count_30d"),
        F.sum(F.when(F.col("in_7d") == 1, 1).otherwise(0)).alias("txn_count_7d"),
        F.sum("TXN_AMOUNT").alias("txn_sum_90d"),
        F.sum(F.when(days == 1, F.col("TXN_AMOUNT")).otherwise(0.0)).alias("txn_sum_1d"),
        F.sum(F.when(days.between(1, 2), F.col("TXN_AMOUNT")).otherwise(0.0)).alias("txn_sum_2d"),
        F.sum(F.when(days.between(1, 3), F.col("TXN_AMOUNT")).otherwise(0.0)).alias("txn_sum_3d"),
        F.sum(F.when(days.between(1, 5), F.col("TXN_AMOUNT")).otherwise(0.0)).alias("txn_sum_5d"),
        F.sum(F.when(days.between(1, 14), F.col("TXN_AMOUNT")).otherwise(0.0)).alias("txn_sum_14d"),
        F.sum(F.when(days.between(1, 21), F.col("TXN_AMOUNT")).otherwise(0.0)).alias("txn_sum_21d"),
        F.sum(F.when(days.between(1, 45), F.col("TXN_AMOUNT")).otherwise(0.0)).alias("txn_sum_45d"),
        F.sum(F.when(days.between(1, 60), F.col("TXN_AMOUNT")).otherwise(0.0)).alias("txn_sum_60d"),
        F.sum(F.when(days.between(31, 60), F.col("TXN_AMOUNT")).otherwise(0.0)).alias("txn_sum_31_60d"),
        F.sum(F.when(days.between(61, 90), F.col("TXN_AMOUNT")).otherwise(0.0)).alias("txn_sum_61_90d"),
        F.sum(F.when(F.col("in_30d") == 1, F.col("TXN_AMOUNT")).otherwise(0.0)).alias("txn_sum_30d"),
        F.sum(F.when(F.col("in_7d") == 1, F.col("TXN_AMOUNT")).otherwise(0.0)).alias("txn_sum_7d"),
        F.avg("TXN_AMOUNT").alias("avg_txn_amount"),
        F.stddev("TXN_AMOUNT").alias("std_txn_amount"),
        F.max("TXN_AMOUNT").alias("max_txn_amount"),
        F.min("TXN_AMOUNT").alias("min_txn_amount"),
        F.expr("percentile_approx(TXN_AMOUNT, 0.5)").alias("median_txn_amount"),
        F.countDistinct("DST_ACCOUNT_ID").alias("unique_dst_accounts"),
        F.countDistinct("TXN_TYPE").alias("unique_txn_types"),
        F.countDistinct(F.when(days.between(1, 7), F.col("DST_ACCOUNT_ID"))).alias("unique_dst_accounts_7d"),
        F.countDistinct(F.when(F.col("in_30d") == 1, F.col("DST_ACCOUNT_ID"))).alias("unique_dst_accounts_30d"),
        F.countDistinct(F.when(F.col("in_30d") == 1, F.col("TXN_DATE"))).alias("active_days_30d"),
        F.countDistinct(F.when(days.between(1, 7), F.col("TXN_DATE"))).alias("active_days_7d"),
        F.countDistinct(F.when(days.between(1, 14), F.col("TXN_DATE"))).alias("active_days_14d"),
        F.countDistinct(F.when(days.between(1, 21), F.col("TXN_DATE"))).alias("active_days_21d"),
        F.countDistinct(F.col("TXN_DATE")).alias("active_days_90d"),
        F.max("TXN_DATE").alias("last_txn_date"),
        F.min("TXN_DATE").alias("first_txn_date"),
        F.sum(F.when(is_p2p, 1).otherwise(0)).alias("p2p_count_90d"),
        F.sum(F.when(is_bill, 1).otherwise(0)).alias("bill_count_90d"),
        F.sum(F.when(is_merchant, 1).otherwise(0)).alias("merchant_count_90d"),
        F.sum(F.when(is_cashout, 1).otherwise(0)).alias("cashout_count_90d"),
        F.sum(F.when(is_p2p & (F.col("in_30d") == 1), 1).otherwise(0)).alias("p2p_count_30d"),
        F.sum(F.when(is_bill & (F.col("in_30d") == 1), 1).otherwise(0)).alias("bill_count_30d"),
        F.sum(F.when(is_merchant & (F.col("in_30d") == 1), 1).otherwise(0)).alias("merchant_count_30d"),
        F.sum(F.when(is_cashout & (F.col("in_30d") == 1), 1).otherwise(0)).alias("cashout_count_30d"),
        F.max(F.when(is_p2p, F.col("TXN_DATE"))).alias("last_p2p_date"),
        F.max(F.when(is_bill, F.col("TXN_DATE"))).alias("last_bill_date"),
        F.max(F.when(is_merchant, F.col("TXN_DATE"))).alias("last_merchant_date"),
    )

    w = Window.partitionBy("ACCOUNT_ID").orderBy("TXN_DATE")
    gaps = (
        t.select("ACCOUNT_ID", "TXN_DATE")
        .distinct()
        .withColumn("prev_txn_date", F.lag("TXN_DATE").over(w))
        .withColumn("txn_gap_days", F.datediff("TXN_DATE", "prev_txn_date"))
        .groupBy("ACCOUNT_ID")
        .agg(
            F.max("txn_gap_days").alias("max_txn_gap_days"),
            F.avg("txn_gap_days").alias("avg_txn_gap_days"),
        )
    )

    cutoff = F.to_date(F.lit(cfg.cutoff_date))
    features = (
        base.join(gaps, "ACCOUNT_ID", "left")
        .withColumn("days_since_last_transaction", F.datediff(cutoff, "last_txn_date"))
        .withColumn("days_since_first_transaction", F.datediff(cutoff, "first_txn_date"))
        .withColumn("days_since_last_p2p", F.datediff(cutoff, "last_p2p_date"))
        .withColumn("days_since_last_bill", F.datediff(cutoff, "last_bill_date"))
        .withColumn("days_since_last_merchant", F.datediff(cutoff, "last_merchant_date"))
        .withColumn("p2p_ratio", F.col("p2p_count_90d") / F.greatest(F.col("txn_count_90d"), F.lit(1)))
        .withColumn("bill_ratio", F.col("bill_count_90d") / F.greatest(F.col("txn_count_90d"), F.lit(1)))
        .withColumn("merchant_ratio", F.col("merchant_count_90d") / F.greatest(F.col("txn_count_90d"), F.lit(1)))
        .withColumn("cashout_ratio", F.col("cashout_count_90d") / F.greatest(F.col("txn_count_90d"), F.lit(1)))
        .withColumn("p2p_ratio_30d", F.col("p2p_count_30d") / F.greatest(F.col("txn_count_30d"), F.lit(1)))
        .withColumn("bill_ratio_30d", F.col("bill_count_30d") / F.greatest(F.col("txn_count_30d"), F.lit(1)))
        .withColumn("merchant_ratio_30d", F.col("merchant_count_30d") / F.greatest(F.col("txn_count_30d"), F.lit(1)))
        .withColumn("cashout_ratio_30d", F.col("cashout_count_30d") / F.greatest(F.col("txn_count_30d"), F.lit(1)))
        .withColumn("txn_count_1d_to_7d", F.col("txn_count_1d") / F.greatest(F.col("txn_count_7d"), F.lit(1)))
        .withColumn("txn_count_3d_to_14d", F.col("txn_count_3d") / F.greatest(F.col("txn_count_14d"), F.lit(1)))
        .withColumn("txn_count_7d_to_14d", F.col("txn_count_7d") / F.greatest(F.col("txn_count_14d"), F.lit(1)))
        .withColumn("txn_count_14d_to_30d", F.col("txn_count_14d") / F.greatest(F.col("txn_count_30d"), F.lit(1)))
        .withColumn("txn_count_21d_to_45d", F.col("txn_count_21d") / F.greatest(F.col("txn_count_45d"), F.lit(1)))
        .withColumn("txn_count_7d_to_30d", F.col("txn_count_7d") / F.greatest(F.col("txn_count_30d"), F.lit(1)))
        .withColumn("txn_count_30d_to_90d", F.col("txn_count_30d") / F.greatest(F.col("txn_count_90d"), F.lit(1)))
        .withColumn("txn_sum_1d_to_7d", F.col("txn_sum_1d") / F.greatest(F.col("txn_sum_7d"), F.lit(1.0)))
        .withColumn("txn_sum_3d_to_14d", F.col("txn_sum_3d") / F.greatest(F.col("txn_sum_14d"), F.lit(1.0)))
        .withColumn("txn_sum_7d_to_14d", F.col("txn_sum_7d") / F.greatest(F.col("txn_sum_14d"), F.lit(1.0)))
        .withColumn("txn_sum_14d_to_30d", F.col("txn_sum_14d") / F.greatest(F.col("txn_sum_30d"), F.lit(1.0)))
        .withColumn("txn_sum_7d_to_30d", F.col("txn_sum_7d") / F.greatest(F.col("txn_sum_30d"), F.lit(1.0)))
        .withColumn("txn_sum_30d_to_90d", F.col("txn_sum_30d") / F.greatest(F.col("txn_sum_90d"), F.lit(1.0)))
        .withColumn("txn_count_recent_vs_prev30", F.col("txn_count_30d") / F.greatest(F.col("txn_count_31_60d"), F.lit(1)))
        .withColumn("txn_count_prev30_vs_old30", F.col("txn_count_31_60d") / F.greatest(F.col("txn_count_61_90d"), F.lit(1)))
        .withColumn("txn_sum_recent_vs_prev30", F.col("txn_sum_30d") / F.greatest(F.col("txn_sum_31_60d"), F.lit(1.0)))
        .withColumn("txn_sum_prev30_vs_old30", F.col("txn_sum_31_60d") / F.greatest(F.col("txn_sum_61_90d"), F.lit(1.0)))
        .withColumn("active_day_ratio_7d", F.col("active_days_7d") / F.lit(7.0))
        .withColumn("active_day_ratio_14d", F.col("active_days_14d") / F.lit(14.0))
        .withColumn("active_day_ratio_21d", F.col("active_days_21d") / F.lit(21.0))
        .withColumn("active_day_ratio_30d", F.col("active_days_30d") / F.lit(30.0))
        .withColumn("active_day_ratio_90d", F.col("active_days_90d") / F.lit(90.0))
        .withColumn("active_day_7d_to_30d", F.col("active_days_7d") / F.greatest(F.col("active_days_30d"), F.lit(1)))
        .withColumn("active_day_14d_to_30d", F.col("active_days_14d") / F.greatest(F.col("active_days_30d"), F.lit(1)))
        .withColumn("dst_diversity_7d_to_30d", F.col("unique_dst_accounts_7d") / F.greatest(F.col("unique_dst_accounts_30d"), F.lit(1)))
        .withColumn("dst_per_txn_30d", F.col("unique_dst_accounts_30d") / F.greatest(F.col("txn_count_30d"), F.lit(1)))
        .withColumn("txn_per_active_day_30d", F.col("txn_count_30d") / F.greatest(F.col("active_days_30d"), F.lit(1)))
        .withColumn("txn_per_active_day_90d", F.col("txn_count_90d") / F.greatest(F.col("active_days_90d"), F.lit(1)))
        .withColumn("recency_x_count_30d", F.col("days_since_last_transaction") * F.log1p(F.col("txn_count_30d")))
        .withColumn("recency_x_sum_30d", F.col("days_since_last_transaction") * F.log1p(F.col("txn_sum_30d")))
        .withColumn("activity_frequency_score", F.col("txn_count_90d") / F.greatest(F.col("days_since_first_transaction"), F.lit(1)))
        .drop("last_txn_date", "first_txn_date", "last_p2p_date", "last_bill_date", "last_merchant_date")
    )
    t.unpersist()
    return features


def build_balance_features(bal: DataFrame, cfg: Config) -> DataFrame:
    b = add_window_flags(bal, "BAL_DATE", cfg.cutoff_date).persist(StorageLevel.DISK_ONLY)
    w_desc = Window.partitionBy("ACCOUNT_ID").orderBy(F.col("BAL_DATE").desc())
    w_asc = Window.partitionBy("ACCOUNT_ID").orderBy(F.col("BAL_DATE").asc())

    agg = b.groupBy("ACCOUNT_ID").agg(
        F.avg(F.when(F.col("days_before_cutoff") == 1, F.col("BALANCE"))).alias("avg_balance_1d"),
        F.avg(F.when(F.col("days_before_cutoff").between(1, 3), F.col("BALANCE"))).alias("avg_balance_3d"),
        F.avg(F.when(F.col("days_before_cutoff").between(1, 7), F.col("BALANCE"))).alias("avg_balance_7d"),
        F.avg(F.when(F.col("days_before_cutoff").between(1, 14), F.col("BALANCE"))).alias("avg_balance_14d"),
        F.min(F.when(F.col("days_before_cutoff").between(1, 7), F.col("BALANCE"))).alias("balance_min_7d"),
        F.max(F.when(F.col("days_before_cutoff").between(1, 7), F.col("BALANCE"))).alias("balance_max_7d"),
        F.stddev(F.when(F.col("days_before_cutoff").between(1, 7), F.col("BALANCE"))).alias("balance_std_7d"),
        F.sum(F.when((F.col("days_before_cutoff").between(1, 7)) & (F.col("BALANCE") <= 0), 1).otherwise(0)).alias("zero_balance_days_7d"),
        F.sum(F.when(F.col("days_before_cutoff").between(1, 7), 1).otherwise(0)).alias("balance_obs_7d"),
        F.avg(F.when(F.col("in_30d") == 1, F.col("BALANCE"))).alias("avg_balance_30d"),
        F.avg(F.when(F.col("days_before_cutoff").between(31, 60), F.col("BALANCE"))).alias("avg_balance_31_60d"),
        F.avg(F.when(F.col("days_before_cutoff").between(61, 90), F.col("BALANCE"))).alias("avg_balance_61_90d"),
        F.stddev(F.when(F.col("in_30d") == 1, F.col("BALANCE"))).alias("balance_std_30d"),
        F.min(F.when(F.col("in_30d") == 1, F.col("BALANCE"))).alias("balance_min_30d"),
        F.max(F.when(F.col("in_30d") == 1, F.col("BALANCE"))).alias("balance_max_30d"),
        F.avg(F.when(F.col("in_90d") == 1, F.col("BALANCE"))).alias("avg_balance_90d"),
        F.stddev(F.when(F.col("in_90d") == 1, F.col("BALANCE"))).alias("balance_std_90d"),
        F.sum(F.when((F.col("in_30d") == 1) & (F.col("BALANCE") <= 0), 1).otherwise(0)).alias("zero_balance_days_30d"),
        F.sum(F.when(F.col("in_30d") == 1, 1).otherwise(0)).alias("balance_obs_30d"),
        F.expr("percentile_approx(CASE WHEN in_30d = 1 THEN BALANCE END, 0.5)").alias("median_balance_30d"),
    )
    last_bal = (
        b.withColumn("rn", F.row_number().over(w_desc))
        .filter(F.col("rn") == 1)
        .select("ACCOUNT_ID", F.col("BALANCE").alias("last_balance"))
    )
    first_bal = (
        b.withColumn("rn", F.row_number().over(w_asc))
        .filter(F.col("rn") == 1)
        .select("ACCOUNT_ID", F.col("BALANCE").alias("first_balance_90d"))
    )

    features = (
        agg.join(last_bal, "ACCOUNT_ID", "left")
        .join(first_bal, "ACCOUNT_ID", "left")
        .withColumn("zero_balance_ratio_7d", F.col("zero_balance_days_7d") / F.greatest(F.col("balance_obs_7d"), F.lit(1)))
        .withColumn("zero_balance_ratio", F.col("zero_balance_days_30d") / F.greatest(F.col("balance_obs_30d"), F.lit(1)))
        .withColumn("balance_cv_7d", F.col("balance_std_7d") / F.greatest(F.abs(F.col("avg_balance_7d")), F.lit(1.0)))
        .withColumn("balance_cv_30d", F.col("balance_std_30d") / F.greatest(F.abs(F.col("avg_balance_30d")), F.lit(1.0)))
        .withColumn("balance_1d_to_7d", F.col("avg_balance_1d") / F.greatest(F.abs(F.col("avg_balance_7d")), F.lit(1.0)))
        .withColumn("balance_3d_to_14d", F.col("avg_balance_3d") / F.greatest(F.abs(F.col("avg_balance_14d")), F.lit(1.0)))
        .withColumn("balance_7d_to_30d", F.col("avg_balance_7d") / F.greatest(F.abs(F.col("avg_balance_30d")), F.lit(1.0)))
        .withColumn("balance_14d_to_30d", F.col("avg_balance_14d") / F.greatest(F.abs(F.col("avg_balance_30d")), F.lit(1.0)))
        .withColumn("balance_drop_30d", F.col("avg_balance_90d") - F.col("avg_balance_30d"))
        .withColumn("balance_recent_vs_prev30", F.col("avg_balance_30d") / F.greatest(F.abs(F.col("avg_balance_31_60d")), F.lit(1.0)))
        .withColumn("balance_prev30_vs_old30", F.col("avg_balance_31_60d") / F.greatest(F.abs(F.col("avg_balance_61_90d")), F.lit(1.0)))
        .withColumn("balance_last_vs_first", F.col("last_balance") / F.greatest(F.abs(F.col("first_balance_90d")), F.lit(1.0)))
    )
    b.unpersist()
    return features


def prefix_feature_columns(df: DataFrame, prefix: str) -> DataFrame:
    return df.select(
        "ACCOUNT_ID",
        *[F.col(c).alias(f"{prefix}_{c}") for c in df.columns if c != "ACCOUNT_ID"],
    )


def build_incoming_transactions(trx: DataFrame) -> DataFrame:
    return trx.select(
        F.col("DST_ACCOUNT_ID").alias("ACCOUNT_ID"),
        F.col("ACCOUNT_ID").alias("DST_ACCOUNT_ID"),
        "TXN_TS",
        "TXN_DATE",
        "TXN_AMOUNT",
        "TXN_TYPE",
    ).filter(F.col("ACCOUNT_ID").isNotNull())


def build_kyc_features(kyc: DataFrame, cfg: Config) -> DataFrame:
    cols = ["ACCOUNT_ID"]
    cutoff = F.to_date(F.lit(cfg.cutoff_date))
    df = kyc
    if "ACCOUNT_OPEN_DATE" in df.columns:
        df = df.withColumn("ACCOUNT_OPEN_DATE", F.to_date("ACCOUNT_OPEN_DATE"))
        df = df.withColumn("account_age_days", F.datediff(cutoff, "ACCOUNT_OPEN_DATE"))
        cols.append("account_age_days")

    for col in ["GENDER", "REGION", "ACCOUNT_TYPE"]:
        if col in df.columns:
            indexer = StringIndexer(inputCol=col, outputCol=f"{col.lower()}_idx", handleInvalid="keep")
            df = indexer.fit(df).transform(df)
            cols.append(f"{col.lower()}_idx")
    return df.select(*cols).dropDuplicates(["ACCOUNT_ID"])


def assemble_features(
    trx_f: DataFrame,
    bal_f: DataFrame,
    kyc_f: DataFrame,
    population: DataFrame,
) -> DataFrame:
    df = population.select("ACCOUNT_ID").distinct()
    df = df.join(trx_f, "ACCOUNT_ID", "left").join(bal_f, "ACCOUNT_ID", "left").join(kyc_f, "ACCOUNT_ID", "left")

    def first_existing(*cols: str):
        existing = [F.col(c) for c in cols if c in df.columns]
        return F.coalesce(*existing, F.lit(0.0)) if existing else F.lit(0.0)

    out_sum_30d = first_existing("out_txn_sum_30d", "txn_sum_30d")
    in_sum_30d = first_existing("in_txn_sum_30d")
    out_sum_90d = first_existing("out_txn_sum_90d", "txn_sum_90d")
    in_sum_90d = first_existing("in_txn_sum_90d")
    out_count_30d = first_existing("out_txn_count_30d", "txn_count_30d")
    in_count_30d = first_existing("in_txn_count_30d")
    active_days = F.greatest(
        first_existing("out_active_days_90d"),
        first_existing("in_active_days_90d"),
        F.lit(1.0),
    )
    df = df.withColumn("out_txn_to_balance_ratio", out_sum_30d / F.greatest(F.abs(F.col("avg_balance_30d")), F.lit(1.0)))
    df = df.withColumn("in_txn_to_balance_ratio", in_sum_30d / F.greatest(F.abs(F.col("avg_balance_30d")), F.lit(1.0)))
    df = df.withColumn("net_flow_30d", in_sum_30d - out_sum_30d)
    df = df.withColumn("net_flow_90d", in_sum_90d - out_sum_90d)
    df = df.withColumn("in_out_amount_ratio_30d", in_sum_30d / F.greatest(out_sum_30d, F.lit(1.0)))
    df = df.withColumn("in_out_count_ratio_30d", in_count_30d / F.greatest(out_count_30d, F.lit(1.0)))
    df = df.withColumn("amount_per_active_day_90d", (out_sum_90d + in_sum_90d) / active_days)

    numeric_cols = [c for c, t in df.dtypes if c != "ACCOUNT_ID" and t in {"double", "float", "int", "bigint", "smallint"}]
    if numeric_cols:
        df = df.fillna(9999, subset=[c for c in numeric_cols if "days_since" in c or "gap_days" in c])
        df = df.fillna(0, subset=[c for c in numeric_cols if "days_since" not in c and "gap_days" not in c])
        imputed_cols = [f"{c}__imputed" for c in numeric_cols]
        imputer = Imputer(inputCols=numeric_cols, outputCols=imputed_cols).setStrategy("median")
        df = imputer.fit(df).transform(df)
        for original, imputed in zip(numeric_cols, imputed_cols):
            df = df.drop(original).withColumnRenamed(imputed, original)
    return df.fillna(0)


def write_feature_store(df: DataFrame, path: str) -> None:
    (
        df.repartition("ACCOUNT_ID")
        .write.mode("overwrite")
        .option("compression", "snappy")
        .parquet(path)
    )


def precision_recall_at_k(y_true: np.ndarray, y_prob: np.ndarray, k_frac: float = 0.10) -> tuple[float, float]:
    n = max(1, int(math.ceil(len(y_true) * k_frac)))
    order = np.argsort(-y_prob)[:n]
    positives = y_true.sum()
    precision = float(y_true[order].mean()) if n else 0.0
    recall = float(y_true[order].sum() / positives) if positives else 0.0
    return precision, recall


def choose_time_validation(pdf: pd.DataFrame, cfg: Config) -> tuple[pd.Series, pd.Series]:
    date_candidates = ["LABEL_DATE", "SNAPSHOT_DATE", "AS_OF_DATE", "ACCOUNT_OPEN_DATE"]
    available = [c for c in date_candidates if c in pdf.columns]
    if not available:
        raise ValueError(
            "No time column found for a leakage-safe validation split. Add LABEL_DATE/SNAPSHOT_DATE/AS_OF_DATE "
            "to labels or keep ACCOUNT_OPEN_DATE in KYC. Refusing to do a random split."
        )
    split_col = available[0]
    dates = pd.to_datetime(pdf[split_col], errors="coerce")
    threshold = dates.quantile(1.0 - cfg.valid_frac)
    valid_mask = dates >= threshold
    train_mask = ~valid_mask
    print(f"Time validation split uses {split_col}; threshold={threshold.date() if pd.notna(threshold) else threshold}")
    return train_mask, valid_mask


def train_lightgbm(train_pdf: pd.DataFrame, cfg: Config) -> tuple[lgb.Booster, list[str], dict[str, float]]:
    train_mask, valid_mask = choose_time_validation(train_pdf, cfg)
    if cfg.train_sample_frac < 1.0:
        keep = train_pdf.loc[train_mask].sample(frac=cfg.train_sample_frac, random_state=cfg.random_seed).index
        train_mask = train_pdf.index.isin(keep)

    drop_cols = {"ACCOUNT_ID", "CHURN", "LABEL_DATE", "SNAPSHOT_DATE", "AS_OF_DATE", "ACCOUNT_OPEN_DATE"}
    features = [c for c in train_pdf.columns if c not in drop_cols]
    X_train = train_pdf.loc[train_mask, features]
    y_train = train_pdf.loc[train_mask, "CHURN"].astype(int).values
    X_valid = train_pdf.loc[valid_mask, features]
    y_valid = train_pdf.loc[valid_mask, "CHURN"].astype(int).values

    pos = max(1, int(y_train.sum()))
    neg = max(1, int(len(y_train) - y_train.sum()))
    scale_pos_weight = neg / pos

    dtrain = lgb.Dataset(X_train, y_train, free_raw_data=False)
    dvalid = lgb.Dataset(X_valid, y_valid, reference=dtrain, free_raw_data=False)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.08, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 14),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 500),
            "subsample": trial.suggest_float("subsample", 0.60, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.60, 1.0),
            "scale_pos_weight": scale_pos_weight,
            "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 5.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 10.0),
            "verbosity": -1,
            "seed": cfg.random_seed,
            "feature_pre_filter": False,
        }
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=5000,
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(0)],
        )
        preds = model.predict(X_valid, num_iteration=model.best_iteration)
        return roc_auc_score(y_valid, preds)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=cfg.random_seed))
    study.optimize(objective, n_trials=cfg.optuna_trials, show_progress_bar=True)

    best_params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "scale_pos_weight": scale_pos_weight,
        "verbosity": -1,
        "seed": cfg.random_seed,
        "feature_pre_filter": False,
        **study.best_params,
    }
    model = lgb.train(
        best_params,
        dtrain,
        num_boost_round=8000,
        valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(250), lgb.log_evaluation(100)],
    )
    valid_pred = model.predict(X_valid, num_iteration=model.best_iteration)
    p10, r10 = precision_recall_at_k(y_valid, valid_pred, 0.10)
    metrics = {
        "auc_roc": float(roc_auc_score(y_valid, valid_pred)),
        "precision_at_10pct": p10,
        "recall_at_10pct": r10,
        "best_iteration": int(model.best_iteration or 0),
        "scale_pos_weight": float(scale_pos_weight),
        "optuna_best_auc": float(study.best_value),
    }

    full_y = train_pdf["CHURN"].astype(int).values
    full_pos = max(1, int(full_y.sum()))
    full_neg = max(1, int(len(full_y) - full_y.sum()))
    final_params = dict(best_params)
    final_params["scale_pos_weight"] = full_neg / full_pos
    final_rounds = int(model.best_iteration or metrics["best_iteration"] or 300)
    final_model = lgb.train(
        final_params,
        lgb.Dataset(train_pdf[features], full_y),
        num_boost_round=final_rounds,
        callbacks=[lgb.log_evaluation(100)],
    )
    metrics["final_train_rows"] = int(len(train_pdf))
    metrics["final_num_boost_round"] = final_rounds
    return final_model, features, metrics


def save_shap(model: lgb.Booster, train_pdf: pd.DataFrame, features: list[str], output_dir: str, seed: int) -> None:
    import shap

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    sample = train_pdf[features].sample(n=min(20000, len(train_pdf)), random_state=seed)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, sample, show=False, max_display=30)
    plt.tight_layout()
    plt.savefig(out / "shap_summary.png", dpi=180, bbox_inches="tight")
    plt.close()

    importance = (
        pd.DataFrame({"feature": features, "gain": model.feature_importance(importance_type="gain")})
        .sort_values("gain", ascending=False)
    )
    importance.to_csv(out / "feature_importance.csv", index=False)


def main() -> None:
    cfg = parse_args()
    Path(cfg.work_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    spark = build_spark(cfg)
    spark.sparkContext.setLogLevel("WARN")

    print("Loading inputs...")
    trx, bal, kyc, labels, test_ids = normalize_inputs(cfg, spark)
    customers = maybe_customer_ids(kyc)
    population = labels.select("ACCOUNT_ID").unionByName(test_ids.select("ACCOUNT_ID")).distinct().join(customers, "ACCOUNT_ID", "inner")
    feature_path = str(Path(cfg.work_dir) / "account_features")

    if cfg.reuse_feature_store and Path(feature_path).exists():
        print(f"Reusing existing feature store from {feature_path}")
        features = spark.read.parquet(feature_path).persist(StorageLevel.DISK_ONLY)
    else:
        print("Building transaction features...")
        outgoing_trx = trx.join(population, "ACCOUNT_ID", "inner")
        incoming_trx = build_incoming_transactions(trx).join(population, "ACCOUNT_ID", "inner")
        trx_f = prefix_feature_columns(build_transaction_features(outgoing_trx, cfg), "out")
        in_trx_f = prefix_feature_columns(build_transaction_features(incoming_trx, cfg), "in")
        print("Building balance features...")
        bal_f = build_balance_features(bal.join(population, "ACCOUNT_ID", "inner"), cfg)
        print("Building KYC features...")
        kyc_f = build_kyc_features(kyc.join(population, "ACCOUNT_ID", "inner"), cfg)

        print("Assembling account-level feature store...")
        features = assemble_features(
            trx_f.join(in_trx_f, "ACCOUNT_ID", "outer"),
            bal_f,
            kyc_f,
            population,
        ).persist(StorageLevel.DISK_ONLY)
        write_feature_store(features, feature_path)
        print(f"Feature store written to {feature_path}")

    labels_aug = labels
    for c in ["LABEL_DATE", "SNAPSHOT_DATE", "AS_OF_DATE"]:
        if c in labels.columns:
            labels_aug = labels_aug.withColumn(c, F.to_date(c))
    if "ACCOUNT_OPEN_DATE" in kyc.columns and "ACCOUNT_OPEN_DATE" not in labels_aug.columns:
        labels_aug = labels_aug.join(kyc.select("ACCOUNT_ID", F.to_date("ACCOUNT_OPEN_DATE").alias("ACCOUNT_OPEN_DATE")), "ACCOUNT_ID", "left")

    train_df = features.join(labels_aug, "ACCOUNT_ID", "inner")
    test_df = features.join(test_ids, "ACCOUNT_ID", "inner")

    print("Collecting account-level train/test matrices for LightGBM...")
    train_pdf = train_df.toPandas()
    test_pdf = test_df.toPandas()
    train_pdf["CHURN"] = train_pdf["CHURN"].astype(int)

    print("Training LightGBM with Optuna...")
    model, feature_cols, metrics = train_lightgbm(train_pdf, cfg)
    model_path = str(Path(cfg.output_dir) / "lightgbm_churn_model.txt")
    model.save_model(model_path)

    print("Saving explainability artifacts...")
    save_shap(model, train_pdf, feature_cols, cfg.output_dir, cfg.random_seed)

    print("Scoring test accounts...")
    test_pred = model.predict(test_pdf[feature_cols], num_iteration=model.best_iteration)
    predictions = pd.DataFrame({"ACCOUNT_ID": test_pdf["ACCOUNT_ID"], "CHURN_PROB": test_pred})
    predictions.to_csv(Path(cfg.output_dir) / "predictions.csv", index=False)

    with open(Path(cfg.output_dir) / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(Path(cfg.output_dir) / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"Saved predictions to {Path(cfg.output_dir) / 'predictions.csv'}")
    spark.stop()


if __name__ == "__main__":
    main()
