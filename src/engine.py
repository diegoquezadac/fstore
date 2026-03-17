"""FeatureEngine - register features, feed events, get feature values."""

import json
import os
import shutil
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Any

from src.entity import Entity
from src.feature import Feature, parse_window
from src.storage import OnlineStorage, OfflineStorage, FeatureRecord

logger = logging.getLogger(__name__)

# Aggregations that can be computed for all rows at once via pandas vectorised ops.
# Requires: built-in aggregation name, no time window.
# `where` filters are handled via NaN masking so they don't block this path.
_VECTORIZABLE_AGGS = frozenset({"mean", "sum", "count", "min", "max", "std", "nunique"})


class FeatureEngine:
    """Computes and serves features from event data.

    Define features once, then feed events either one at a time (``update``)
    or as a full DataFrame (``compute``).  The engine maintains two stores:

    - **Online** — in-memory dict of the latest value per entity key.
      Use ``get_online_features`` for low-latency, real-time lookups.
    - **Offline** — append-only Parquet directory with the full history of
      every feature computation.  Never grows unbounded in RAM.

    Example — streaming::

        fe = FeatureEngine(timestamp_col="ts")
        fe.register(avg_price)

        for _, row in df.iterrows():
            features = fe.update(row)
            # {"user": {"avg_price": 120.5}}

    Example — batch::

        fe = FeatureEngine(timestamp_col="ts")
        fe.register(avg_price)

        report = fe.compute(df)
        # df now has a new column "user:avg_price"
        # report → {"rows_processed": 6, "features_computed": 1, ...}

    Args:
        timestamp_col: Name of the column used as the event timestamp.
        offline_path: Directory used by the offline store (Parquet, append-only).
    """

    def __init__(self, timestamp_col: str = "ts", offline_path: str = "offline_store"):
        self.timestamp_col = timestamp_col
        self._features: dict[str, Feature] = {}   # "{entity}:{feature}" → Feature
        self._entities: dict[str, Entity] = {}    # entity_name → Entity
        self._buffer: list[dict] = []             # raw rows in arrival order
        self.online = OnlineStorage()
        self.offline = OfflineStorage(path=offline_path)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, feature: Feature):
        """Register a feature definition."""
        self._features[f"{feature.entity.name}:{feature.name}"] = feature
        self._entities[feature.entity.name] = feature.entity

    # ------------------------------------------------------------------
    # Streaming API
    # ------------------------------------------------------------------

    def update(self, row: dict | Any) -> dict[str, dict[str, Any]]:
        """Process one event and return updated feature values.

        The row is appended to an internal buffer and every registered feature
        whose entity key is present in *row* is recomputed over that buffer
        (subject to any time window).

        Args:
            row: A dict or pandas Series representing one event.

        Returns:
            ``{entity_name: {feature_name: value}}`` for every entity in row.

        Note:
            The buffer grows indefinitely. A future improvement is to
            periodically flush older rows to disk and retain only the rows
            within the widest registered time window.
        """
        if hasattr(row, "to_dict"):
            row = row.to_dict()

        if self.timestamp_col not in row:
            raise ValueError(
                f"timestamp column '{self.timestamp_col}' not found in row keys: {list(row.keys())}"
            )
        self._validate_row(row)

        self._buffer.append(row)
        timestamp = row.get(self.timestamp_col)

        df = pd.DataFrame(self._buffer)

        updated: dict[str, dict[str, Any]] = {}

        for feature in self._features.values():
            entity = feature.entity

            if entity.key not in row:
                logger.warning("feature '%s' skipped: key '%s' not in row", feature.name, entity.key)
                continue

            entity_key = row[entity.key]
            entity_df = df[df[entity.key] == entity_key].copy()
            entity_df = self._apply_window(entity_df, feature, timestamp)
            if feature.where is not None:
                entity_df = entity_df[feature.where(entity_df)]
            value = feature.aggregation(entity_df)

            self.online.upsert(entity.name, entity_key, feature.name, value)
            self.offline.append(
                FeatureRecord(
                    entity_name=entity.name,
                    entity_key=entity_key,
                    feature_name=feature.name,
                    value=value,
                    timestamp=timestamp,
                )
            )

            updated.setdefault(entity.name, {})[feature.name] = value

        return updated

    # ------------------------------------------------------------------
    # Batch API
    # ------------------------------------------------------------------

    def compute(
        self,
        df: pd.DataFrame,
        verbose: bool = False,
        log_every: int = 100,
        checkpoint_dir: str | None = None,
        checkpoint_every: int = 100_000,
    ) -> dict[str, Any]:
        """Compute features for a full DataFrame, adding results as new columns.

        Sorts *df* in place by timestamp and writes feature values directly
        onto it as new columns named ``"{entity}:{feature}"``.  Does not use
        the internal buffer.  Offline records are flushed in 100 K-row batches.

        Features split into two execution paths:

        - **Fast** — vectorised ``groupby().expanding()`` ops, one call per
          feature.  Used when the aggregation is a built-in and there is no
          time window.
        - **Slow** — row-by-row with ``searchsorted``.  Used for windowed
          aggregations and custom callables.  Supports crash recovery via
          ``checkpoint_dir``.

        Args:
            df: Input DataFrame. Must contain ``timestamp_col``. Modified in
                place — feature columns are added directly onto it.
            verbose: Emit detailed per-phase timing logs (INFO level).
                     Enable with ``logging.basicConfig(level=logging.INFO)``.
            log_every: Progress log interval in rows (default 100).
            checkpoint_dir: Directory for crash-recovery checkpoints saved
                during the slow path. Pass the same path on a re-run to
                resume automatically. Deleted on success.
            checkpoint_every: Rows between checkpoint writes (default 100 000).

        Returns:
            A summary report dict with keys: ``rows_processed``,
            ``features_computed``, ``fast_features``, ``slow_features``,
            ``records_written``, ``elapsed_seconds``, ``feature_columns``.
        """
        if self.timestamp_col not in df.columns:
            raise ValueError(
                f"timestamp column '{self.timestamp_col}' not found in DataFrame columns: {list(df.columns)}"
            )

        start = time.perf_counter()
        df.sort_values(self.timestamp_col, inplace=True, ignore_index=True)
        n_rows = len(df)
        input_cols = list(df.columns)

        feat_items = list(self._features.items())
        col_names  = {fk: f"{feat.entity.name}:{feat.name}" for fk, feat in feat_items}

        # Checkpoint paths (resolved once; None when checkpointing is disabled).
        _ckpt_meta   = os.path.join(checkpoint_dir, "meta.json")           if checkpoint_dir else None
        _ckpt_values = os.path.join(checkpoint_dir, "slow_values.parquet") if checkpoint_dir else None
        resume_slow_row = 0

        # Split features into two paths:
        #   fast — vectorised pandas ops over the whole df at once (no Python loop)
        #   slow — row-by-row with searchsorted (needed for time-windowed or custom aggs)
        fast_feats = [
            (fk, feat) for fk, feat in feat_items
            if feat._agg_name in _VECTORIZABLE_AGGS and feat.window is None
        ]
        slow_feats = [
            (fk, feat) for fk, feat in feat_items
            if not (feat._agg_name in _VECTORIZABLE_AGGS and feat.window is None)
        ]

        if verbose:
            logger.info(
                "[compute] %d fast features (vectorised), %d slow features (row-by-row)",
                len(fast_feats), len(slow_feats),
            )

        # feature_values: col_name -> numpy array (fast) or list (slow)
        feature_values: dict[str, Any] = {}

        # Load slow-path values saved by a prior run that was interrupted.
        if _ckpt_meta and os.path.exists(_ckpt_meta):
            with open(_ckpt_meta) as _f:
                _meta = json.load(_f)
            if _meta.get("n_rows") == n_rows:
                resume_slow_row = _meta.get("slow_row", 0)
                if resume_slow_row > 0 and _ckpt_values and os.path.exists(_ckpt_values):
                    _ckpt_df = pd.read_parquet(_ckpt_values)
                    for _col in _ckpt_df.columns:
                        # Restore computed rows; pad the rest with None.
                        feature_values[_col] = (
                            _ckpt_df[_col].tolist()
                            + [None] * (n_rows - resume_slow_row)
                        )
                logger.info(
                    "[compute] Resuming from checkpoint — slow row %d/%d",
                    resume_slow_row, n_rows,
                )
            else:
                logger.warning(
                    "[compute] Checkpoint n_rows mismatch (%d vs %d) — starting fresh",
                    _meta.get("n_rows"), n_rows,
                )
                resume_slow_row = 0

        _BATCH_SIZE = 100_000
        records: list[FeatureRecord] = []
        total_records_written = 0

        _t: dict[str, float] = {
            "vectorized": 0.0,   # fast path: _compute_vectorized()
            "records":    0.0,   # fast path: online upsert + FeatureRecord creation
            "slice":      0.0,   # slow path: searchsorted + DataFrame build
            "where":      0.0,   # slow path: where predicate
            "agg":        0.0,   # slow path: aggregation
            "online":     0.0,   # slow path: online store upsert
            "offline_io": 0.0,   # Parquet writes (both paths)
        }

        def _flush(recs: list) -> None:
            if verbose:
                _t0 = time.perf_counter()
            self.offline.batch_write(recs)
            if verbose:
                _t["offline_io"] += time.perf_counter() - _t0

        # ----------------------------------------------------------------
        # Fast path — one vectorised call per feature, no row loop
        # ----------------------------------------------------------------
        for fk, feat in fast_feats:
            col_name = col_names[fk]
            ek = feat.entity.key

            if ek not in df.columns:
                logger.warning("feature '%s' skipped: key '%s' not in DataFrame", feat.name, ek)
                continue

            if verbose:
                _t0 = time.perf_counter()
            values = self._compute_vectorized(df, feat)
            if verbose:
                _t["vectorized"] += time.perf_counter() - _t0

            feature_values[col_name] = values

            if verbose:
                _t0 = time.perf_counter()

            # Online store: last value per entity key (df is timestamp-sorted).
            val_series = pd.Series(values, index=df.index)
            for key_val, last_val in val_series.groupby(df[ek]).last().items():
                if hasattr(key_val, "item"):
                    key_val = key_val.item()
                self.online.upsert(feat.entity.name, key_val, feat.name, last_val)

            # FeatureRecords — one per row, no computation.
            new_recs = [
                FeatureRecord(
                    entity_name=feat.entity.name,
                    entity_key=ek_v.item() if hasattr(ek_v, "item") else ek_v,
                    feature_name=feat.name,
                    value=val,
                    timestamp=ts,
                )
                for ek_v, ts, val in zip(df[ek], df[self.timestamp_col], values)
            ]
            records.extend(new_recs)

            if verbose:
                _t["records"] += time.perf_counter() - _t0

            if len(records) >= _BATCH_SIZE:
                _flush(records)
                total_records_written += len(records)
                records.clear()

        if verbose and fast_feats:
            elapsed = time.perf_counter() - start
            logger.info(
                "[compute] fast path done in %.3fs (%.0f rows/s equivalent) | "
                "vectorized=%.3fs records=%.3fs offline_io=%.3fs",
                elapsed, n_rows / elapsed,
                _t["vectorized"], _t["records"], _t["offline_io"],
            )

        # ----------------------------------------------------------------
        # Slow path — row-by-row with searchsorted, only for slow_feats
        # ----------------------------------------------------------------
        if slow_feats:
            # Build per-entity groups only for entity key cols used by slow features.
            slow_key_cols = {
                feat.entity.key for _, feat in slow_feats
                if feat.entity.key in df.columns
            }
            if verbose:
                _t0 = time.perf_counter()
            groups: dict[str, dict[Any, tuple[pd.DatetimeIndex, dict[str, np.ndarray]]]] = {}
            for key_col in slow_key_cols:
                groups[key_col] = {}
                for key_val, grp in df.groupby(key_col, sort=False):
                    if hasattr(key_val, "item"):
                        key_val = key_val.item()
                    groups[key_col][key_val] = (
                        pd.DatetimeIndex(grp[self.timestamp_col]),
                        {col: grp[col].to_numpy() for col in input_cols},
                    )
            if verbose:
                logger.info("[compute] slow-path groupby done in %.3fs", time.perf_counter() - _t0)

            windows = {
                fk: parse_window(feat.window) if feat.window else None
                for fk, feat in slow_feats
            }
            for fk, feat in slow_feats:
                if col_names[fk] not in feature_values:  # may be pre-loaded from checkpoint
                    feature_values[col_names[fk]] = [None] * n_rows

            for i, row in df.iloc[resume_slow_row:].iterrows():
                timestamp = row[self.timestamp_col]

                for feat_key, feature in slow_feats:
                    entity   = feature.entity
                    col_name = col_names[feat_key]

                    if entity.key not in groups:
                        logger.warning("feature '%s' skipped: key '%s' not in DataFrame", feature.name, entity.key)
                        continue

                    entity_key = row[entity.key]
                    if hasattr(entity_key, "item"):
                        entity_key = entity_key.item()

                    ts_idx, arrays = groups[entity.key][entity_key]

                    if verbose:
                        _t0 = time.perf_counter()
                    pos       = ts_idx.searchsorted(timestamp, side="right")
                    window_td = windows[feat_key]
                    start_pos = (
                        ts_idx.searchsorted(timestamp - window_td, side="left")
                        if window_td is not None else 0
                    )
                    s = slice(start_pos, pos)
                    if feature.where is not None:
                        entity_df = pd.DataFrame({col: arrays[col][s] for col in input_cols})
                    elif feature.columns:
                        entity_df = pd.DataFrame({col: arrays[col][s] for col in feature.columns})
                    else:
                        entity_df = pd.DataFrame(index=range(pos - start_pos))
                    if verbose:
                        _t["slice"] += time.perf_counter() - _t0

                    if feature.where is not None:
                        if verbose:
                            _t0 = time.perf_counter()
                        entity_df = entity_df[feature.where(entity_df)]
                        if verbose:
                            _t["where"] += time.perf_counter() - _t0

                    if verbose:
                        _t0 = time.perf_counter()
                    value = feature.aggregation(entity_df)
                    if verbose:
                        _t["agg"] += time.perf_counter() - _t0

                    if verbose:
                        _t0 = time.perf_counter()
                    self.online.upsert(entity.name, entity_key, feature.name, value)
                    if verbose:
                        _t["online"] += time.perf_counter() - _t0

                    feature_values[col_name][i] = value
                    records.append(FeatureRecord(
                        entity_name=entity.name,
                        entity_key=entity_key,
                        feature_name=feature.name,
                        value=value,
                        timestamp=timestamp,
                    ))

                if len(records) >= _BATCH_SIZE:
                    _flush(records)
                    total_records_written += len(records)
                    records.clear()

                if _ckpt_meta and (i + 1) % checkpoint_every == 0:
                    os.makedirs(checkpoint_dir, exist_ok=True)  # type: ignore[arg-type]
                    _slow_cols = {
                        col_names[fk]: feature_values[col_names[fk]][: i + 1]
                        for fk, _ in slow_feats
                    }
                    pd.DataFrame(_slow_cols).to_parquet(_ckpt_values, index=False)
                    with open(_ckpt_meta, "w") as _f:
                        json.dump({"n_rows": n_rows, "slow_row": i + 1}, _f)
                    if verbose:
                        logger.info("[compute] Checkpoint saved at row %d/%d", i + 1, n_rows)

                if verbose and (i + 1) % log_every == 0:
                    elapsed = time.perf_counter() - start
                    logger.info(
                        "[compute] slow %d/%d rows (%.0f rows/s) | "
                        "slice=%.3fs where=%.3fs agg=%.3fs online=%.3fs offline_io=%.3fs",
                        i + 1, n_rows, (i + 1) / elapsed,
                        _t["slice"], _t["where"], _t["agg"], _t["online"], _t["offline_io"],
                    )

        # Assign all feature columns to df at once.
        for col, values in feature_values.items():
            df[col] = values

        if records:
            _flush(records)
            total_records_written += len(records)

        if checkpoint_dir and os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)
            if verbose:
                logger.info("[compute] Checkpoint removed — run completed successfully")

        elapsed = time.perf_counter() - start

        if verbose:
            total_instrumented = sum(_t.values()) or 1.0
            logger.info("[compute] DONE — %d rows in %.3fs (%.0f rows/s)", n_rows, elapsed, n_rows / elapsed)
            logger.info("[compute] Timing breakdown:")
            for phase, t in _t.items():
                logger.info("  %-14s  %7.3fs  (%5.1f%%)", phase, t, 100 * t / total_instrumented)

        feature_cols = sorted(col_names.values())
        return {
            "rows_processed":    n_rows,
            "features_computed": len(self._features),
            "fast_features":     len(fast_feats),
            "slow_features":     len(slow_feats),
            "records_written":   total_records_written,
            "elapsed_seconds":   round(elapsed, 4),
            "feature_columns":   feature_cols,
        }

    # ------------------------------------------------------------------
    # Vectorised helper
    # ------------------------------------------------------------------

    def _compute_vectorized(self, df: pd.DataFrame, feature: Feature) -> np.ndarray:
        """Compute feature values for every row at once using pandas vectorised ops."""
        ek  = feature.entity.key
        agg = feature._agg_name
        on  = feature.on

        if feature.where is not None:
            mask = feature.where(df)
            if agg == "count":
                return mask.astype(int).groupby(df[ek], dropna=False).cumsum().to_numpy()
            else:
                data = df[on].where(mask, other=np.nan)
                result = getattr(data.groupby(df[ek], dropna=False).expanding(), agg)()
                return result.droplevel(0).sort_index().to_numpy()

        if agg == "nunique":
            is_first = (~df.duplicated(subset=[ek, on], keep="first")).astype(int)
            return is_first.groupby(df[ek], dropna=False).cumsum().to_numpy()

        if agg == "count":
            return (df.groupby(ek, dropna=False).cumcount() + 1).to_numpy()

        result = getattr(df.groupby(ek, dropna=False)[on].expanding(), agg)()
        return result.droplevel(0).sort_index().to_numpy()

    # ------------------------------------------------------------------
    # Serving
    # ------------------------------------------------------------------

    def get_online_features(self, entity_name: str, **entity_keys) -> dict[str, Any]:
        """Return the latest feature values for an entity key."""
        entity = self._entities.get(entity_name)
        if not entity:
            return {}
        return self.online.get(entity_name, entity_keys.get(entity.key))

    def get_offline_features(
        self,
        entity_name: str,
        feature_name: str | None = None,
        as_of: datetime | None = None,
        **entity_keys,
    ) -> list[FeatureRecord]:
        """Return historical feature records for an entity key."""
        entity = self._entities.get(entity_name)
        if not entity:
            return []

        entity_key = entity_keys.get(entity.key)

        if feature_name:
            return self.offline.get_historical(entity_name, entity_key, feature_name, as_of)

        results: list[FeatureRecord] = []
        for f in self._features.values():
            if f.entity.name == entity_name:
                results.extend(
                    self.offline.get_historical(entity_name, entity_key, f.name, as_of)
                )
        return sorted(results, key=lambda r: r.timestamp)

    def flush(self):
        """Flush any buffered offline records to disk immediately."""
        self.offline.flush()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_window(
        self, entity_df: pd.DataFrame, feature: Feature, timestamp: Any
    ) -> pd.DataFrame:
        if feature.window is None or timestamp is None:
            return entity_df
        cutoff = timestamp - parse_window(feature.window)
        return entity_df[entity_df[self.timestamp_col] >= cutoff]

    def _validate_row(self, row: dict):
        """Raise if any feature's declared columns are missing from a row dict."""
        for feature in self._features.values():
            missing = [c for c in feature.columns if c not in row]
            if missing:
                raise ValueError(
                    f"feature '{feature.name}' requires columns {feature.columns}, "
                    f"but {missing} are missing from the row"
                )
