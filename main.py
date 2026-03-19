"""Example usage of the feature engine — e-commerce clickstream."""

import logging
import shutil

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src import Entity, Feature, FeatureEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)

_EVENT_TYPES = ["view", "add_to_cart", "purchase"]
_EVENT_WEIGHTS = [0.70, 0.20, 0.10]
_CATEGORIES = ["electronics", "clothing", "books", "sports", "home"]
_DEVICES = ["mobile", "desktop", "tablet"]


def make_synthetic_df(n: int = 1_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = datetime(2024, 1, 1)
    seconds = rng.integers(0, 90 * 24 * 3600, size=n)
    category = rng.choice(_CATEGORIES, size=n)
    user_id = rng.integers(1, 51, size=n)
    return pd.DataFrame({
        "event_id":          np.arange(1, n + 1),
        "user_id":           user_id,
        "product_id":        rng.integers(1, 101, size=n),
        "category":          category,
        "event_type":        rng.choice(_EVENT_TYPES, size=n, p=_EVENT_WEIGHTS),
        "price_usd":         rng.uniform(5, 500, size=n).round(2),
        "device_type":       rng.choice(_DEVICES, size=n),
        "_user_x_category":  [f"{u}|{c}" for u, c in zip(user_id, category)],
        "ts":                [base + timedelta(seconds=int(s)) for s in seconds],
    })


def main():
    shutil.rmtree("offline_store", ignore_errors=True)

    df = make_synthetic_df(n=1_000_000)

    # ---------------------------------------------------------------
    # Define entities
    # ---------------------------------------------------------------
    user             = Entity("user",             key="user_id")
    product          = Entity("product",          key="product_id")
    user_in_category = Entity("user_in_category", key="_user_x_category")

    # ---------------------------------------------------------------
    # Define features
    # ---------------------------------------------------------------
    _is_purchase    = lambda g: g["event_type"] == "purchase"
    _is_add_to_cart = lambda g: g["event_type"] == "add_to_cart"

    user_features = [
        Feature("num_events",            user, "count"),
        Feature("num_purchases",         user, "count",   where=_is_purchase),
        Feature("num_add_to_cart",       user, "count",   where=_is_add_to_cart),
        Feature("total_spent",           user, "sum",     on="price_usd",  where=_is_purchase),
        Feature("mean_price_viewed",     user, "mean",    on="price_usd"),
        Feature("distinct_products_7d",  user, "nunique", on="product_id", window="7d"),
        Feature("distinct_categories",   user, "nunique", on="category"),
        Feature("typical_device",        user, "mode",    on="device_type"),
        Feature("num_events_5m",         user, "count",                    window="5m"),
        Feature("num_events_1h",         user, "count",                    window="1h"),
        Feature("num_events_30d",        user, "count",                    window="30d"),
        Feature("num_purchases_1h",      user, "count",   where=_is_purchase, window="1h"),
        Feature("num_purchases_30d",     user, "count",   where=_is_purchase, window="30d"),
        Feature("mean_price_1h",         user, "mean",    on="price_usd",  window="1h"),
        Feature("mean_price_30d",        user, "mean",    on="price_usd",  window="30d"),
    ]

    product_features = [
        Feature("num_views",     product, "count"),
        Feature("num_purchases", product, "count", where=_is_purchase),
        Feature("mean_price",    product, "mean",  on="price_usd"),
        Feature("std_price",     product, "std",   on="price_usd"),
    ]

    user_in_category_features = [
        Feature("num_events",    user_in_category, "count"),
        Feature("num_purchases", user_in_category, "count", where=_is_purchase),
        Feature("total_spent",   user_in_category, "sum",   on="price_usd", where=_is_purchase),
    ]

    all_features = user_features + product_features + user_in_category_features

    # ---------------------------------------------------------------
    # Example 1: compute() — batch mode
    # ---------------------------------------------------------------
    print("=" * 60)
    print("EXAMPLE 1 — compute(df) — batch mode")
    print("=" * 60)

    fe = FeatureEngine(timestamp_col="ts")
    for f in all_features:
        fe.register(f)

    report = fe.compute(df, verbose=True, log_every=100_000)
    print("\nReport:", report)

    print("\nFirst 5 rows with feature columns:")
    display_cols = ["event_id", "user_id", "product_id", "event_type", "ts"] + report["feature_columns"]
    print(df[display_cols].head().to_string(index=False))

    print("\n--- Online features after compute() ---")
    print(f"  User 1:    {fe.get_online_features('user', user_id=1)}")
    print(f"  Product 1: {fe.get_online_features('product', product_id=1)}")

    # ---------------------------------------------------------------
    # Example 2: restore_online_from_offline()
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("EXAMPLE 2 — restore_online_from_offline()")
    print("=" * 60)

    fe2 = FeatureEngine(timestamp_col="ts", offline_path="offline_store")
    for f in all_features:
        fe2.register(f)

    n_loaded = fe2.restore_online_from_offline()
    print(f"\nLoaded {n_loaded} entries from offline backup.")
    print(f"  User 1 (restored): {fe2.get_online_features('user', user_id=1)}")

    # ---------------------------------------------------------------
    # Example 3: update() — streaming mode
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("EXAMPLE 3 — update(row) — streaming mode (first 5 events)")
    print("=" * 60)

    shutil.rmtree("offline_store", ignore_errors=True)

    fe3 = FeatureEngine(timestamp_col="ts")
    for f in user_features[:5]:
        fe3.register(f)

    df_sorted = df.sort_values("ts").reset_index(drop=True)
    for _, row in df_sorted.head(5).iterrows():
        features = fe3.update(row)
        print(
            f"  event {int(row['event_id'])} "
            f"(user {int(row['user_id'])}, {row['event_type']}, "
            f"ts={row['ts'].date()}) → {features}"
        )


if __name__ == "__main__":
    main()
