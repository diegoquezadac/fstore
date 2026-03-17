"""Storage backends for online and offline feature serving."""

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd


# Entity keys are plain scalars (int or str), serialised to JSON strings for
# uniform Parquet storage.

def _key_to_str(key: Any) -> str:
    # numpy scalars (int64, float64 …) are not JSON-serialisable by default.
    if hasattr(key, "item"):
        key = key.item()
    return json.dumps(key)


def _str_to_key(s: str) -> Any:
    return json.loads(s)


# ---------------------------------------------------------------------------
# FeatureRecord
# ---------------------------------------------------------------------------

@dataclass
class FeatureRecord:
    """A single feature value with metadata."""
    entity_name: str
    entity_key: Any
    feature_name: str
    value: Any
    timestamp: datetime


# ---------------------------------------------------------------------------
# OnlineStorage  (in-memory — serves the latest value per entity key)
# ---------------------------------------------------------------------------

class OnlineStorage:
    """In-memory storage for latest feature values.

    Structure: {entity_name: {entity_key: {feature_name: value}}}
    """

    def __init__(self):
        self._data: dict[str, dict[Any, dict[str, Any]]] = {}

    def upsert(self, entity_name: str, entity_key: Any, feature_name: str, value: Any):
        self._data.setdefault(entity_name, {}).setdefault(entity_key, {})[feature_name] = value

    def get(self, entity_name: str, entity_key: Any) -> dict[str, Any]:
        return self._data.get(entity_name, {}).get(entity_key, {})

    def get_all(self, entity_name: str) -> dict[Any, dict[str, Any]]:
        return self._data.get(entity_name, {})


# ---------------------------------------------------------------------------
# OfflineStorage  (Parquet-backed — never grows unbounded in memory)
# ---------------------------------------------------------------------------

class OfflineStorage:
    """Parquet-backed storage for the full feature history.

    Records are accumulated in a small in-memory buffer and flushed to disk
    automatically when the buffer is full, or on-demand via ``flush()``.
    Reads always flush first so the Parquet file is always up-to-date.

    The file is created lazily on the first flush.  Pass ``path`` to control
    where it lives — useful for keeping test runs or experiments isolated.

    Designed so that swapping the backend (e.g. to S3, Delta Lake, Lance …)
    only requires changing this class.
    """

    _FLUSH_THRESHOLD = 500  # records to buffer before an automatic flush

    def __init__(self, path: str = "offline.parquet"):
        self.path = path
        self._pending: list[dict] = []

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def append(self, record: FeatureRecord):
        """Buffer a record; flush to disk when the buffer is full."""
        self._pending.append({
            "entity_name": record.entity_name,
            "entity_key": _key_to_str(record.entity_key),
            "feature_name": record.feature_name,
            "value": record.value,
            "timestamp": record.timestamp,
        })
        if len(self._pending) >= self._FLUSH_THRESHOLD:
            self.flush()

    def flush(self):
        """Write all pending records to the Parquet file."""
        if not self._pending:
            return
        new_df = pd.DataFrame(self._pending)
        if os.path.exists(self.path):
            existing = pd.read_parquet(self.path)
            combined = pd.concat([existing, new_df], ignore_index=True)
        else:
            combined = new_df
        combined.to_parquet(self.path, index=False)
        self._pending.clear()

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def get_historical(
        self,
        entity_name: str,
        entity_key: Any,
        feature_name: str,
        as_of: datetime | None = None,
    ) -> list[FeatureRecord]:
        """Return historical records, optionally capped at as_of."""
        self.flush()
        if not os.path.exists(self.path):
            return []

        key_str = _key_to_str(entity_key)
        df = pd.read_parquet(
            self.path,
            filters=[
                ("entity_name", "=", entity_name),
                ("entity_key", "=", key_str),
                ("feature_name", "=", feature_name),
            ],
        )
        if as_of is not None:
            df = df[df["timestamp"] <= as_of]

        return [
            FeatureRecord(
                entity_name=row.entity_name,
                entity_key=_str_to_key(row.entity_key),
                feature_name=row.feature_name,
                value=row.value,
                timestamp=row.timestamp,
            )
            for row in df.sort_values("timestamp").itertuples()
        ]

    def get_all_records(self) -> list[FeatureRecord]:
        """Return every record ever written."""
        self.flush()
        if not os.path.exists(self.path):
            return []
        df = pd.read_parquet(self.path)
        return [
            FeatureRecord(
                entity_name=row.entity_name,
                entity_key=_str_to_key(row.entity_key),
                feature_name=row.feature_name,
                value=row.value,
                timestamp=row.timestamp,
            )
            for row in df.itertuples()
        ]
