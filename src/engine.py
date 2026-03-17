"""Engine - core computation and storage logic.

Handles all the complex machinery: incremental row processing, time-windowed
aggregations, batch execution, and both online (latest) and offline
(historical) feature serving.
"""

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


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class Engine:
    """Core computation engine for the feature store.

    Maintains an in-memory row buffer and computes features incrementally.
    The online store (latest values) is kept in memory; the offline store
    (full history) is persisted to a Parquet file so it never grows
    unbounded in RAM.

    The storage layer is injected so it can be swapped for remote or
    file-backed implementations without touching this class.

    Args:
        timestamp_col: Name of the column used as the event timestamp.
        offline_path: Path to the Parquet file used by the offline store.
    """

    def __init__(self, timestamp_col: str = "ts", offline_path: str = "offline.parquet"):
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
        """Register a feature (and its entity)."""
        self._features[f"{feature.entity.name}:{feature.name}"] = feature
        self._entities[feature.entity.name] = feature.entity

    # ------------------------------------------------------------------
    # Incremental (streaming) API
    # ------------------------------------------------------------------

    def step(self, row: dict | Any) -> dict[str, dict[str, Any]]:
        """Process a single row and update the feature stores.

        The row is appended to the internal buffer.  For every registered
        feature whose entity key columns are all present in *row*, the
        feature is recomputed over all buffered rows for that entity (subject
        to any time window).

        Args:
            row: A dict or pandas Series representing one event.

        Returns:
            {entity_name: {feature_name: value}} for every entity found in row.
        """
        if hasattr(row, "to_dict"):
            row = row.to_dict()

        if self.timestamp_col not in row:
            raise ValueError(
                f"timestamp column '{self.timestamp_col}' not found in row keys: {list(row.keys())}"
            )
        self._validate_row(row)

        # NOTE: The buffer grows indefinitely. A future improvement would be to
        # periodically dump older rows to an optimised data file (e.g. Parquet
        # or Arrow IPC) and reload only the rows that fall within the widest
        # registered time window, rather than keeping every row in memory.
        self._buffer.append(row)
        timestamp = row.get(self.timestamp_col)

        # Build a DataFrame from the buffer once per step call.
        df = pd.DataFrame(self._buffer)

        updated: dict[str, dict[str, Any]] = {}

        for feature in self._features.values():
            entity = feature.entity

            if entity.key not in row:
                print(f"[warning] feature '{feature.name}' skipped: key '{entity.key}' not in row")
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

    def run(
        self,
        df: pd.DataFrame,
        verbose: bool = False,
        log_every: int = 100,
        checkpoint_dir: str | None = None,
        checkpoint_every: int = 100_000,
    ) -> dict[str, Any]:
        """Process a full DataFrame in timestamp order, adding computed features
        as new columns directly on *df*.

        Unlike ``step()``, this method does **not** use the internal buffer — it
        works entirely on the supplied DataFrame, which must contain all rows
        upfront.  The DataFrame is sorted in place by timestamp and feature
        values are written onto it as new columns named
        ``"{entity}__{feature}"`` — no copy is made.

        Offline records are flushed to Parquet in batches of 100 K rows so IO
        is never a per-row bottleneck.

        Args:
            df: Input DataFrame. Must contain ``timestamp_col``. Modified in
                place — feature values are added as new columns.
            verbose: When True, emit detailed timing logs via the ``engine``
                     logger (INFO level). Use ``logging.basicConfig(level=logging.INFO)``
                     to see them. Defaults to False.
            log_every: How often (in rows) to emit a progress log when
                       verbose=True. Defaults to 100.
            checkpoint_dir: Directory to store checkpoint files.  When set,
                the slow path saves its progress every ``checkpoint_every``
                rows so a failed run can resume without restarting from row 0.
                If a valid checkpoint is found at startup the run resumes
                automatically; on successful completion the directory is
                removed.  The fast (vectorised) path always reruns from
                scratch — it typically takes only seconds.
            checkpoint_every: How often (in rows) to write a checkpoint
                during the slow path. Defaults to 100 000.

        Returns:
            A summary report with execution statistics.
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
        col_names  = {fk: f"{feat.entity.name}__{feat.name}" for fk, feat in feat_items}

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
                "[run] %d fast features (vectorised), %d slow features (row-by-row)",
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
                    "[run] Resuming from checkpoint — slow row %d/%d",
                    resume_slow_row, n_rows,
                )
            else:
                logger.warning(
                    "[run] Checkpoint n_rows mismatch (%d vs %d) — starting fresh",
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
                "[run] fast path done in %.3fs (%.0f rows/s equivalent) | "
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
                logger.info("[run] slow-path groupby done in %.3fs", time.perf_counter() - _t0)

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
                    pos        = ts_idx.searchsorted(timestamp, side="right")
                    window_td  = windows[feat_key]
                    start_pos  = (
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
                        logger.info("[run] Checkpoint saved at row %d/%d", i + 1, n_rows)

                if verbose and (i + 1) % log_every == 0:
                    elapsed = time.perf_counter() - start
                    logger.info(
                        "[run] slow %d/%d rows (%.0f rows/s) | "
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
                logger.info("[run] Checkpoint removed — run completed successfully")

        elapsed = time.perf_counter() - start

        if verbose:
            total_instrumented = sum(_t.values()) or 1.0
            logger.info("[run] DONE — %d rows in %.3fs (%.0f rows/s)", n_rows, elapsed, n_rows / elapsed)
            logger.info("[run] Timing breakdown:")
            for phase, t in _t.items():
                logger.info("  %-14s  %7.3fs  (%5.1f%%)", phase, t, 100 * t / total_instrumented)

        feature_cols = sorted(col_names.values())
        return {
            "rows_processed":   n_rows,
            "features_computed": len(self._features),
            "fast_features":    len(fast_feats),
            "slow_features":    len(slow_feats),
            "records_written":  total_records_written,
            "elapsed_seconds":  round(elapsed, 4),
            "feature_columns":  feature_cols,
        }

    # ------------------------------------------------------------------
    # Vectorised helper
    # ------------------------------------------------------------------

    def _compute_vectorized(self, df: pd.DataFrame, feature: Feature) -> np.ndarray:
        """Compute feature values for every row at once using pandas vectorised ops.

        Returns a numpy array aligned with df's integer index (0 … n-1).

        Conditions for calling this:
          - feature._agg_name is in _VECTORIZABLE_AGGS
          - feature.window is None
          - feature.where may or may not be set (handled via NaN masking)
        """
        ek  = feature.entity.key
        agg = feature._agg_name
        on  = feature.on

        if feature.where is not None:
            # Apply the predicate to the full df — it's a row-wise boolean comparison
            # so it produces the same mask regardless of whether it sees one entity or all.
            mask = feature.where(df)
            if agg == "count":
                # Cumulative count of matching rows per entity key.
                return mask.astype(int).groupby(df[ek]).cumsum().to_numpy()
            else:
                # Zero out non-matching rows as NaN; expanding agg skips NaN.
                data = df[on].where(mask, other=np.nan)
                result = getattr(data.groupby(df[ek]).expanding(), agg)()
                return result.droplevel(0).sort_index().to_numpy()

        if agg == "nunique":
            # For each row, count distinct (entity_key, value) pairs seen so far.
            # Mark first occurrence of each pair → cumsum within entity = expanding nunique.
            is_first = (~df.duplicated(subset=[ek, on], keep="first")).astype(int)
            return is_first.groupby(df[ek]).cumsum().to_numpy()

        if agg == "count":
            return (df.groupby(ek).cumcount() + 1).to_numpy()

        # mean, sum, min, max, std — expanding window
        result = getattr(df.groupby(ek)[on].expanding(), agg)()
        return result.droplevel(0).sort_index().to_numpy()

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def get_online_features(
        self, entity_name: str, **entity_keys
    ) -> dict[str, Any]:
        """Return the latest feature values for an entity."""
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
        """Return historical feature records for an entity."""
        entity = self._entities.get(entity_name)
        if not entity:
            return []

        entity_key = entity_keys.get(entity.key)

        if feature_name:
            return self.offline.get_historical(
                entity_name, entity_key, feature_name, as_of
            )

        all_features = [
            f for f in self._features.values() if f.entity.name == entity_name
        ]
        results: list[FeatureRecord] = []
        for f in all_features:
            results.extend(
                self.offline.get_historical(entity_name, entity_key, f.name, as_of)
            )
        return sorted(results, key=lambda r: r.timestamp)

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
