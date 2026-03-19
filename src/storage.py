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
    # pandas NA, float NaN, None → JSON null
    try:
        if pd.isna(key):
            return "null"
    except (TypeError, ValueError):
        pass
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

    Uses an **append-only directory layout**: every flush writes a new
    ``part_NNNNNNNN.parquet`` file inside ``path/``.  No existing data is
    ever re-read on write — each flush is O(batch_size), not O(total_rows).

    ``pd.read_parquet`` transparently combines all part files when reading,
    and pyarrow's predicate pushdown still applies per-file.

    The directory is created lazily on the first flush.  Pass ``path`` to
    control where it lives — useful for isolating test runs.

    Designed so that swapping the backend (e.g. to S3, Delta Lake, Lance …)
    only requires changing this class.
    """

    _FLUSH_THRESHOLD = 500  # records to buffer before an automatic flush

    def __init__(self, path: str = "offline_store"):
        self.path = path
        self._pending: list[dict] = []
        self._part_count: int = 0

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _next_part_path(self) -> str:
        p = os.path.join(self.path, f"part_{self._part_count:08d}.parquet")
        self._part_count += 1
        return p

    @staticmethod
    def _records_to_df(rows: list[dict]) -> "pd.DataFrame":
        return pd.DataFrame(rows)

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
        """Write buffered records as a new part file — O(pending), no re-read."""
        if not self._pending:
            return
        os.makedirs(self.path, exist_ok=True)
        self._records_to_df(self._pending).to_parquet(self._next_part_path(), index=False)
        self._pending.clear()

    def batch_write(self, records: list["FeatureRecord"]):
        """Write a batch of records as a single new part file — O(batch_size), no re-read.

        Intended for experimental/batch mode where all records are known upfront.
        """
        if not records:
            return
        os.makedirs(self.path, exist_ok=True)
        rows = [{
            "entity_name": r.entity_name,
            "entity_key": _key_to_str(r.entity_key),
            "feature_name": r.feature_name,
            "value": r.value,
            "timestamp": r.timestamp,
        } for r in records]
        self._records_to_df(rows).to_parquet(self._next_part_path(), index=False)

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

    def get_latest_values(self) -> pd.DataFrame:
        """Return the latest value per (entity_name, entity_key, feature_name).

        Returns a DataFrame with columns: entity_name, entity_key, feature_name, value.
        entity_key values are returned as raw Python objects (JSON-decoded).
        """
        self.flush()
        if not os.path.exists(self.path):
            return pd.DataFrame(columns=["entity_name", "entity_key", "feature_name", "value"])
        parts = sorted(
            os.path.join(self.path, f)
            for f in os.listdir(self.path)
            if f.endswith(".parquet")
        )
        if not parts:
            return pd.DataFrame(columns=["entity_name", "entity_key", "feature_name", "value"])
        # Read each part file individually so pandas handles int/float promotion
        # on concat, avoiding Arrow's strict cross-file schema unification.
        df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        df = df.sort_values("timestamp").groupby(
            ["entity_name", "entity_key", "feature_name"], sort=False
        ).last().reset_index()
        df["entity_key"] = df["entity_key"].map(_str_to_key)
        return df[["entity_name", "entity_key", "feature_name", "value"]]
