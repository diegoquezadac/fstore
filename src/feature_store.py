"""FeatureStore - public interface for registering and querying features.

All heavy lifting lives in Engine.  This file is intentionally thin so it
stays easy to read and extend.
"""

from datetime import datetime
from typing import Any

import pandas as pd

from src.engine import Engine
from src.feature import Feature
from src.storage import FeatureRecord


class FeatureStore:
    """Public interface for the local feature store.

    Online store: in-memory (always fast).
    Offline store: Parquet file (never grows unbounded in RAM).

    Two execution modes:

    **Streaming** — ``step(row)``: process one event at a time. Each call
    appends the row to an internal buffer and recomputes features over that
    buffer. Use this when data arrives incrementally (e.g. a message queue).

    **Experimental** — ``run(df)``: process a full DataFrame at once. No buffer
    is used — the engine works directly on the supplied DataFrame, adds feature
    columns onto it in place, and flushes all offline records in a single batch
    write. Use this for offline experiments and historical replay.

    Example — streaming::

        fs = FeatureStore(timestamp_col="ts")
        fs.register(avg_price)

        for _, row in df.iterrows():
            features = fs.step(row)
            # {"user": {"avg_price": 120.5}}

    Example — experimental::

        fs = FeatureStore(timestamp_col="ts")
        fs.register(avg_price)

        report = fs.run(df)
        # df now has a new column "user__avg_price"
        # report → {"rows_processed": 6, "features_computed": 1, ...}
    """

    def __init__(
        self,
        timestamp_col: str = "ts",
        offline_path: str = "offline_store",
    ):
        self._engine = Engine(timestamp_col=timestamp_col, offline_path=offline_path)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def register(self, feature: Feature):
        """Register a feature definition."""
        self._engine.register(feature)

    # ------------------------------------------------------------------
    # Incremental (streaming) API
    # ------------------------------------------------------------------

    def step(self, row: dict | Any) -> dict[str, dict[str, Any]]:
        """Process one row and return updated features for entities in that row.

        Args:
            row: A dict or pandas Series representing one event.

        Returns:
            {entity_name: {feature_name: value}} for every entity found in row.
        """
        return self._engine.step(row)

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
        """Process a full DataFrame in timestamp order (experimental mode).

        Modifies *df* in place by adding feature values as new columns named
        ``"{entity}__{feature}"``. Offline records are flushed in 100 K-row
        batches. Does not use the internal buffer.

        Args:
            df: Input DataFrame. Must contain ``timestamp_col``. Modified in
                place — feature columns are added directly onto it.
            verbose: Emit detailed per-phase timing logs (INFO level).
                     Enable with ``logging.basicConfig(level=logging.INFO)``.
            log_every: Progress log interval in rows (default 100).
            checkpoint_dir: Directory for crash-recovery checkpoints saved
                during the slow (row-by-row) path. Pass the same path on a
                re-run to resume automatically. Deleted on success.
            checkpoint_every: Rows between checkpoint writes (default 100 000).

        Returns:
            A summary report dict: rows_processed, features_computed,
            records_written, elapsed_seconds, feature_columns.
        """
        return self._engine.run(
            df,
            verbose=verbose,
            log_every=log_every,
            checkpoint_dir=checkpoint_dir,
            checkpoint_every=checkpoint_every,
        )

    # ------------------------------------------------------------------
    # Serving
    # ------------------------------------------------------------------

    def get_online_features(
        self, entity_name: str, **entity_keys
    ) -> dict[str, Any]:
        """Return the latest feature values for an entity key."""
        return self._engine.get_online_features(entity_name, **entity_keys)

    def get_offline_features(
        self,
        entity_name: str,
        feature_name: str | None = None,
        as_of: datetime | None = None,
        **entity_keys,
    ) -> list[FeatureRecord]:
        """Return historical feature records for an entity key."""
        return self._engine.get_offline_features(
            entity_name, feature_name, as_of, **entity_keys
        )

    def get_feature_vector(self, entity_name: str, **entity_keys) -> dict[str, Any]:
        """Alias for get_online_features."""
        return self.get_online_features(entity_name, **entity_keys)

    def flush(self):
        """Flush any buffered offline records to disk immediately."""
        self._engine.offline.flush()
