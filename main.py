"""Example usage of the local feature store."""

import os
import pandas as pd
from datetime import datetime

from src import Entity, Feature, FeatureStore



def main():
    # Clean up any leftover Parquet file from a previous run.
    if os.path.exists("offline.parquet"):
        os.remove("offline.parquet")

    # Sample transaction data
    data = {
        "attempt_id": [1001, 1002, 1003, 1004, 1005, 1006],
        "user_id": [1, 1, 2, 1, 2, 1],
        "store_id": [10, 10, 20, 20, 20, 10],
        "card": ["XXX", "XXX", "", "C", "B", "A"],
        "price": [100, 150, 200, 120, 180, 90],
        "ts": [
            datetime(2024, 1, 1),
            datetime(2024, 1, 2),
            datetime(2024, 1, 2),
            datetime(2024, 1, 3),
            datetime(2024, 1, 3),
            datetime(2024, 1, 4),
        ],
    }
    df = pd.DataFrame(data)

    # ---------------------------------------------------------------
    # Define entities
    # ---------------------------------------------------------------
    user  = Entity("user",  "user_id")
    store = Entity("store", "store_id")

    # ---------------------------------------------------------------
    # Define features  (some with time windows)
    # ---------------------------------------------------------------
    user_avg_price   = Feature("avg_price",          user,  "mean",    on="price")
    user_total_spent = Feature("total_spent",         user,  "sum",     on="price")
    user_tx_count    = Feature("tx_count",            user,  "count")
    user_cards_2d    = Feature("distinct_cards_2d",   user,  "nunique", on="card", window="2d")
    user_tx_store10  = Feature("tx_count_store10",    user,  "count",
                               where=lambda g: g["store_id"] == 10)

    store_avg_price  = Feature("avg_price",           store, "mean",    on="price")
    store_total_rev  = Feature("total_revenue",        store, "sum",     on="price")

    # ---------------------------------------------------------------
    # Example 1: run() — batch processing, one shot
    # ---------------------------------------------------------------
    print("=" * 60)
    print("EXAMPLE 1 — run(df, id_col=...)")
    print("=" * 60)

    fs = FeatureStore(timestamp_col="ts")
    for f in [user_avg_price, user_total_spent, user_tx_count, user_cards_2d,
              user_tx_store10, store_avg_price, store_total_rev]:
        fs.register(f)

    results = fs.run(df, id_col="attempt_id")
    for attempt_id, entity_features in results.items():
        print(f"\n  attempt_id={attempt_id}")
        for entity_name, feats in entity_features.items():
            print(f"    {entity_name}: {feats}")

    print("\n--- Online features after run() ---")
    print(f"  User 1:   {fs.get_online_features('user', user_id=1)}")
    print(f"  User 2:   {fs.get_online_features('user', user_id=2)}")
    print(f"  Store 10: {fs.get_online_features('store', store_id=10)}")

    print("\n--- Offline: avg_price history for User 1 (read from Parquet) ---")
    for rec in fs.get_offline_features("user", feature_name="avg_price", user_id=1):
        print(f"  {rec.timestamp.date()}: avg_price = {rec.value:.2f}")

    # ---------------------------------------------------------------
    # Example 2: step() — process one row at a time (streaming)
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("EXAMPLE 2 — step(row)  — streaming / real-time")
    print("=" * 60)

    if os.path.exists("offline.parquet"):
        os.remove("offline.parquet")

    fs2 = FeatureStore(timestamp_col="ts")
    for f in [user_avg_price, user_tx_count, user_cards_2d]:
        fs2.register(f)

    df_sorted = df.sort_values("ts").reset_index(drop=True)
    for _, row in df_sorted.iterrows():
        features = fs2.step(row)
        print(f"  attempt {int(row['attempt_id'])} "
              f"(user {int(row['user_id'])}, ts={row['ts'].date()}) → {features}")

    # ---------------------------------------------------------------
    # Example 3: run() without id_col — uses DataFrame index
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("EXAMPLE 3 — run(df)  — no id_col, keyed by index")
    print("=" * 60)

    if os.path.exists("offline.parquet"):
        os.remove("offline.parquet")

    fs3 = FeatureStore(timestamp_col="ts")
    for f in [user_avg_price, user_tx_count]:
        fs3.register(f)

    results3 = fs3.run(df)
    for idx, entity_features in results3.items():
        print(f"  row {idx}: {entity_features}")


if __name__ == "__main__":
    main()
