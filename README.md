# feature_engine

A lightweight **feature engine** for event-driven feature computation. Define features once, feed events one at a time with `update()` or compute features across a full DataFrame with `compute()`. Supports time-windowed aggregations, row-level filters, online serving (in-memory) and offline history (Parquet).

## Installation

```bash
pip install pandas pyarrow
```

## Core concepts

| Concept | What it is |
|---|---|
| `Entity` | The thing you compute features for (e.g. a user, a product). Identified by a single key column. |
| `Feature` | A named aggregation over an entity's historical rows, optionally within a time window and/or a row filter. |
| `FeatureEngine` | Registers features, drives computation, and serves results. Maintains an online store (latest values) and an offline store (full history). |

The engine maintains two backends:
- **Online** — in-memory dict of the latest feature values per entity key. Fast lookups for real-time scoring.
- **Offline** — append-only Parquet directory with the full history of every feature computation. Never grows unbounded in RAM.

## Quick start

```python
from src import Entity, Feature, FeatureEngine

# 1. Define entities
user    = Entity("user",    key="user_id")
product = Entity("product", key="product_id")

# 2. Define features
num_events       = Feature("num_events",       user,    "count")
total_spent      = Feature("total_spent",      user,    "sum",     on="price_usd",
                            where=lambda g: g["event_type"] == "purchase")
distinct_prod_7d = Feature("distinct_prod_7d", user,    "nunique", on="product_id", window="7d")
num_views_1h     = Feature("num_views_1h",     user,    "count",   window="1h")
product_views    = Feature("num_views",        product, "count")

# 3. Create engine and register features
fe = FeatureEngine(timestamp_col="ts")
fe.register(num_events)
fe.register(total_spent)
fe.register(distinct_prod_7d)
fe.register(num_views_1h)
fe.register(product_views)

# 4a. Streaming mode — one event at a time
features = fe.update(row)

# 4b. Batch mode — full DataFrame, enriched in place
report = fe.compute(df)
# df["user:num_events"], df["user:total_spent"], df["product:num_views"] are now populated
```

## Execution modes

### `update(row)` — streaming

Process one event at a time. Each call appends the row to an internal buffer and recomputes features over that buffer. Use this when data arrives incrementally (e.g. from a message queue or a real-time stream).

```python
row = {
    "user_id": 7, "product_id": 42, "category": "electronics",
    "event_type": "purchase", "price_usd": 199.99, "ts": datetime(...)
}

features = fe.update(row)
# {
#   "user":    {"num_events": 3, "total_spent": 349.97, "num_views_1h": 1},
#   "product": {"num_views": 5}
# }
```

Useful for scoring a new event in real time — call `update()`, inspect the returned features, make a decision.

> **Note:** The buffer grows indefinitely. A future improvement is to periodically dump older rows to an optimised data file and reload only the rows within the widest registered window.

### `compute(df)` — batch

Process a full DataFrame at once. The engine works directly on the supplied DataFrame — no internal buffer is used and no copy is made. Feature values are written onto `df` in place as new columns named `"{entity}:{feature}"`. Offline records are written to Parquet in batches so IO is not a per-row bottleneck.

```python
report = fe.compute(df)
# df now has new columns: "user:num_events", "user:total_spent", "product:num_views", ...
# report → {
#   "rows_processed": 1000000,
#   "features_computed": 5,
#   "records_written": 5000000,
#   "elapsed_seconds": 12.4,
#   "feature_columns": ["product:num_views", "user:num_events", ...]
# }
```

Use this for offline experiments, historical replay, and dataset enrichment.

## Querying the stores

```python
# Latest values (online store)
fe.get_online_features("user", user_id=7)
# {"num_events": 42, "total_spent": 1320.50, "distinct_prod_7d": 8, ...}

# Full history (offline store, read from Parquet)
records = fe.get_offline_features("user", feature_name="total_spent", user_id=7)
for rec in records:
    print(rec.timestamp, rec.value)

# Point-in-time query
records = fe.get_offline_features("user", feature_name="total_spent",
                                  as_of=datetime(2024, 2, 1), user_id=7)
```

### Restoring the online store from disk

After a restart, repopulate the in-memory online store from the Parquet backup without recomputing anything:

```python
fe = FeatureEngine(timestamp_col="ts", offline_path="offline_store")
# ... register features ...
n = fe.restore_online_from_offline()
print(f"Loaded {n} entries")  # ready for real-time lookups immediately
```

## Feature definition

```python
Feature(name, entity, aggregation, on=None, window=None, where=None)
```

| Parameter | Type | Description |
|---|---|---|
| `name` | `str` | Feature name, e.g. `"total_spent"` |
| `entity` | `Entity` | The entity this feature belongs to |
| `aggregation` | `str \| Callable` | Built-in name (see table below) or a custom `(DataFrame) -> value` function |
| `on` | `str \| None` | Column the aggregation reads. Required for all built-ins except `"count"`. For custom callables, used for upfront column validation. |
| `window` | `str \| None` | Optional time window: `"5m"`, `"1h"`, `"30d"`, etc. Supported units: `s`, `m`, `h`, `d`. |
| `where` | `Callable \| None` | Optional row filter applied after windowing. Receives the entity DataFrame, must return a boolean Series. |

### Built-in aggregations

| Name | Equivalent |
|---|---|
| `"mean"` | `g[col].mean()` |
| `"sum"` | `g[col].sum()` |
| `"count"` | `len(g)` |
| `"nunique"` | `g[col].nunique()` |
| `"min"` | `g[col].min()` |
| `"max"` | `g[col].max()` |
| `"std"` | `g[col].std()` |
| `"first"` | `g[col].iloc[0]` |
| `"last"` | `g[col].iloc[-1]` |
| `"mode"` | most frequent value in `g[col]` |

For anything else, pass a callable:

```python
Feature("p95_price", user, lambda g: g["price_usd"].quantile(0.95), on="price_usd")
```

### Time windows

When `window` is set, only rows within `[timestamp - window, timestamp]` are passed to the aggregation:

```python
# Distinct products viewed by this user in the last 7 days
distinct_prod_7d = Feature("distinct_prod_7d", user, "nunique", on="product_id", window="7d")

# Purchase count in the last hour (burst detection)
purchases_1h = Feature("purchases_1h", user, "count",
                        where=lambda g: g["event_type"] == "purchase", window="1h")
```

The window string is validated at construction time — an invalid format raises `ValueError` immediately.

### Row filters

When `where` is set, it is applied after windowing, before the aggregation:

```python
# Total spent on purchases only
total_spent = Feature("total_spent", user, "sum", on="price_usd",
                       where=lambda g: g["event_type"] == "purchase")

# High-value purchases in the last 30 days
big_purchases_30d = Feature("big_purchases_30d", user, "count",
                             window="30d", where=lambda g: g["price_usd"] > 200)
```

### Composite entities

Use synthetic key columns to scope features to a combination of entities:

```python
# User behaviour within a specific product category
user_in_category = Entity("user_in_category", key="_user_x_category")

# Build the composite key before calling compute()
df["_user_x_category"] = df["user_id"].astype(str) + "|" + df["category"]

spend_in_category = Feature("total_spent", user_in_category, "sum",
                             on="price_usd", where=lambda g: g["event_type"] == "purchase")
```

## Crash recovery (batch mode)

For long-running `compute()` calls, pass `checkpoint_dir` to save progress every N rows. If the process dies, re-run with the same path to resume from the last checkpoint:

```python
report = fe.compute(
    df,
    checkpoint_dir="run_checkpoint",
    checkpoint_every=500_000,
)
```

The checkpoint is deleted automatically on successful completion. The fast (vectorised) path always reruns from scratch; only the slow (row-by-row) path is checkpointed.

## Validation

The engine validates inputs before doing any work:

- **`update(row)`** — raises `ValueError` if the timestamp column or any declared feature column is missing from the row dict.
- **`compute(df)`** — raises `ValueError` if the timestamp column is missing from the DataFrame columns.
- **`Feature(...)`** — raises `ValueError` at construction if the aggregation name is unknown, if a required `on` column is missing, or if the `window` string is not a valid format.
- A warning is logged (but execution continues) when an entity key column is missing from a row or DataFrame.

## Project structure

```
src/
  entity.py         # Entity definition
  feature.py        # Feature definition, built-in aggregation registry, window parsing
  engine.py         # FeatureEngine: computation, streaming, batch, and serving
  storage.py        # OnlineStorage (in-memory) + OfflineStorage (Parquet, append-only)
main.py             # Usage examples (e-commerce clickstream)
```

## Configuration

```python
FeatureEngine(
    timestamp_col="ts",          # column used as event timestamp (default: "ts")
    offline_path="offline_store" # directory for the Parquet history (default: "offline_store")
)
```

In streaming mode, records are buffered in memory (up to 500 at a time) and flushed automatically; call `fe.flush()` to force-write any pending records. In batch mode, records are flushed in 100 K-row batches during `compute()`.
