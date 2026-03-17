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

    Example — row-by-row (streaming) usage::

        user = Entity("user", "user_id")
        avg_price = Feature("avg_price", user, lambda g: g["price"].mean())

        fs = FeatureStore(timestamp_col="ts")
        fs.register(avg_price)

        for _, row in df.iterrows():
            features = fs.step(row)
            # {"user": {"avg_price": 120.5}}

    Example — batch usage with a fixed dataset::

        results = fs.run(df, id_col="attempt_id")
        # {attempt_id: {"user": {"avg_price": ...}, ...}}

    Example — batch with no id column (uses DataFrame index)::

        results = fs.run(df)
    """

    def __init__(
        self,
        timestamp_col: str = "ts",
        offline_path: str = "offline.parquet",
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
        id_col: str | None = None,
    ) -> dict[Any, dict[str, dict[str, Any]]]:
        """Process a full DataFrame and return a per-row feature snapshot.

        Rows are processed in ascending timestamp order.  Each feature is
        computed once per unique entity key per timestamp (not once per row),
        so this is faster than calling ``step`` in a loop.

        Args:
            df: Input DataFrame.
            id_col: Column that identifies each row in the output dict.
                    When omitted the DataFrame index is used.

        Returns:
            {id_value: {entity_name: {feature_name: value}}}
        """
        return self._engine.run(df, id_col)

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
