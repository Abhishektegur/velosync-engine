"""
VeloSync — Log-Structured Vector Synchronization Engine for Offline-First AI Applications.

VeloSync provides edge devices (mobile phones, IoT nodes, embedded gateways) with:

  1. A durable local vector store on SQLite, with embeddings stored as packed
     binary BLOBs alongside a logical Write-Ahead Log (WAL) table that records
     every mutation (INSERT / UPDATE / DELETE) under a monotonically increasing
     Log Sequence Number (LSN).
  2. Pure-Python cosine similarity search over the local store (no numpy, no
     external dependencies of any kind).
  3. A synchronization engine that computes the delta of unsynchronized WAL
     entries, compacts them into a minimal sync payload, and reconciles
     concurrent remote edits using "Semantic Version Vector" conflict
     resolution: vector clocks establish causality; truly concurrent edits are
     resolved by semantic weight (confidence), falling back to
     last-write-wins on wall-clock timestamps.

Design notes
------------
* The SQLite native journal is also switched to WAL mode (PRAGMA
  journal_mode=WAL) for crash safety and concurrent readers. This is distinct
  from VeloSync's *logical* WAL table (``wal_log``), which exists at the
  application layer to drive replication, not durability.
* Embeddings are encoded as little-endian IEEE-754 float64 sequences via
  ``struct``. Dimension is stored redundantly in its own column so the schema
  is self-describing and corrupted BLOBs are detectable.
* The WAL stores a full after-image of each record (state-based log) rather
  than a diff. This makes log compaction trivial — the latest entry per
  vector_id is sufficient — at the cost of slightly larger log rows.

Standard library only: sqlite3, struct, math, json, uuid, time, heapq,
hashlib, logging, dataclasses, enum, typing.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import logging
import math
import sqlite3
import struct
import time
import uuid
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "VeloSyncError",
    "DimensionMismatchError",
    "VectorNotFoundError",
    "CorruptEmbeddingError",
    "OpType",
    "VectorRecord",
    "WalEntry",
    "SearchResult",
    "ConflictResolution",
    "VeloSyncStore",
    "SyncEngine",
    "MockCloudVectorDB",
]

logger = logging.getLogger("velosync")

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class VeloSyncError(Exception):
    """Base class for all VeloSync errors."""


class DimensionMismatchError(VeloSyncError):
    """Raised when a vector's dimension does not match the store's dimension."""


class VectorNotFoundError(VeloSyncError):
    """Raised when an operation references a vector_id that does not exist."""


class CorruptEmbeddingError(VeloSyncError):
    """Raised when a stored BLOB cannot be decoded into the declared dimension."""


# ---------------------------------------------------------------------------
# Core data model
# ---------------------------------------------------------------------------


class OpType(str, Enum):
    """Mutation types recorded in the logical WAL."""

    INSERT = "INSERT"
    UPDATE = "UPDATE"
    DELETE = "DELETE"


@dataclass(frozen=True)
class VectorRecord:
    """A single versioned vector entry.

    Attributes:
        vector_id: Globally unique identifier (UUID4 string by default).
        embedding: The dense embedding. Empty for tombstones is permitted.
        metadata: Arbitrary JSON-serializable metadata.
        semantic_weight: Confidence / importance score in [0, +inf). Used as
            the primary tiebreaker for concurrent conflicting edits.
        version_vector: Vector clock mapping device_id -> logical counter.
            Encodes causal history across devices.
        updated_at: Wall-clock UNIX timestamp of the last mutation (LWW
            fallback only — never trusted for causality).
        is_deleted: Tombstone flag. Deleted records are retained so deletions
            replicate correctly and can win/lose conflicts.
    """

    vector_id: str
    embedding: Tuple[float, ...]
    metadata: Dict[str, Any] = field(default_factory=dict)
    semantic_weight: float = 1.0
    version_vector: Dict[str, int] = field(default_factory=dict)
    updated_at: float = 0.0
    is_deleted: bool = False

    def to_wire(self) -> Dict[str, Any]:
        """Serialize to a JSON-safe dict for WAL payloads and sync transport."""
        return {
            "vector_id": self.vector_id,
            "embedding": list(self.embedding),
            "metadata": self.metadata,
            "semantic_weight": self.semantic_weight,
            "version_vector": self.version_vector,
            "updated_at": self.updated_at,
            "is_deleted": self.is_deleted,
        }

    @staticmethod
    def from_wire(data: Mapping[str, Any]) -> "VectorRecord":
        """Deserialize from a wire dict, validating required fields."""
        try:
            return VectorRecord(
                vector_id=str(data["vector_id"]),
                embedding=tuple(float(x) for x in data["embedding"]),
                metadata=dict(data.get("metadata") or {}),
                semantic_weight=float(data.get("semantic_weight", 1.0)),
                version_vector={
                    str(k): int(v)
                    for k, v in (data.get("version_vector") or {}).items()
                },
                updated_at=float(data.get("updated_at", 0.0)),
                is_deleted=bool(data.get("is_deleted", False)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise VeloSyncError(f"Malformed wire record: {exc}") from exc


@dataclass(frozen=True)
class WalEntry:
    """One row of the logical write-ahead log."""

    lsn: int
    op: OpType
    vector_id: str
    record: VectorRecord
    created_at: float


@dataclass(frozen=True)
class SearchResult:
    """One nearest-neighbor hit."""

    vector_id: str
    similarity: float
    metadata: Dict[str, Any]
    semantic_weight: float


class ConflictResolution(str, Enum):
    """Outcome labels emitted by the conflict resolver (for audit logging)."""

    LOCAL_DOMINATES = "local_dominates"          # causal: local saw remote's edit
    REMOTE_DOMINATES = "remote_dominates"        # causal: remote saw local's edit
    IDENTICAL_HISTORY = "identical_history"      # same version vector; no-op
    CONCURRENT_SEMANTIC_LOCAL = "concurrent_semantic_local"
    CONCURRENT_SEMANTIC_REMOTE = "concurrent_semantic_remote"
    CONCURRENT_LWW_LOCAL = "concurrent_lww_local"
    CONCURRENT_LWW_REMOTE = "concurrent_lww_remote"
    NO_LOCAL_COPY = "no_local_copy"              # remote record is new to us


# ---------------------------------------------------------------------------
# Binary embedding codec (pure stdlib)
# ---------------------------------------------------------------------------

_FLOAT64_SIZE = 8


def encode_embedding(embedding: Sequence[float]) -> bytes:
    """Pack an embedding into a little-endian float64 BLOB.

    Args:
        embedding: Sequence of floats (may be empty for tombstones).

    Returns:
        Packed bytes of length ``8 * len(embedding)``.

    Raises:
        VeloSyncError: If any element is not coercible to float.
    """
    try:
        return struct.pack(f"<{len(embedding)}d", *(float(x) for x in embedding))
    except (struct.error, TypeError, ValueError) as exc:
        raise VeloSyncError(f"Cannot encode embedding: {exc}") from exc


def decode_embedding(blob: bytes, dimension: int) -> Tuple[float, ...]:
    """Unpack a BLOB into a float tuple, verifying the declared dimension.

    Args:
        blob: Raw bytes from the ``embedding`` column.
        dimension: Expected vector dimension from the ``dimension`` column.

    Returns:
        Tuple of floats of length ``dimension``.

    Raises:
        CorruptEmbeddingError: If the BLOB length disagrees with ``dimension``.
    """
    expected = dimension * _FLOAT64_SIZE
    if len(blob) != expected:
        raise CorruptEmbeddingError(
            f"BLOB is {len(blob)} bytes but dimension={dimension} "
            f"requires {expected} bytes"
        )
    return struct.unpack(f"<{dimension}d", blob)


# ---------------------------------------------------------------------------
# Pure-Python vector math
# ---------------------------------------------------------------------------


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute cosine similarity between two equal-length vectors.

    Single-pass accumulation of dot product and both squared norms.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine similarity in [-1.0, 1.0]. Returns 0.0 if either vector has
        zero magnitude (similarity is undefined; 0.0 is the conventional
        neutral value).

    Raises:
        DimensionMismatchError: If the vectors have different lengths.
    """
    if len(a) != len(b):
        raise DimensionMismatchError(
            f"Cannot compare vectors of dimension {len(a)} and {len(b)}"
        )
    dot = 0.0
    norm_a_sq = 0.0
    norm_b_sq = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a_sq += x * x
        norm_b_sq += y * y
    if norm_a_sq <= 0.0 or norm_b_sq <= 0.0:
        return 0.0
    sim = dot / (math.sqrt(norm_a_sq) * math.sqrt(norm_b_sq))
    # Clamp accumulated floating-point drift back into the valid range.
    return max(-1.0, min(1.0, sim))


# ---------------------------------------------------------------------------
# Version vector (vector clock) algebra
# ---------------------------------------------------------------------------


def compare_version_vectors(
    a: Mapping[str, int], b: Mapping[str, int]
) -> str:
    """Determine the causal relationship between two version vectors.

    Args:
        a: First version vector (device_id -> counter).
        b: Second version vector.

    Returns:
        One of:
          * ``"equal"``      — identical histories.
          * ``"a_after_b"``  — a causally dominates b (a saw all of b's edits).
          * ``"b_after_a"``  — b causally dominates a.
          * ``"concurrent"`` — neither dominates; edits happened in parallel.
    """
    a_ge_b = all(a.get(k, 0) >= v for k, v in b.items())
    b_ge_a = all(b.get(k, 0) >= v for k, v in a.items())
    if a_ge_b and b_ge_a:
        return "equal"
    if a_ge_b:
        return "a_after_b"
    if b_ge_a:
        return "b_after_a"
    return "concurrent"


def merge_version_vectors(
    a: Mapping[str, int], b: Mapping[str, int]
) -> Dict[str, int]:
    """Component-wise maximum of two version vectors (join in the lattice)."""
    merged = dict(a)
    for device, counter in b.items():
        if counter > merged.get(device, 0):
            merged[device] = counter
    return merged


# ---------------------------------------------------------------------------
# SQLite storage layer
# ---------------------------------------------------------------------------

_SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS vectors (
    vector_id       TEXT    PRIMARY KEY,
    embedding       BLOB    NOT NULL,
    dimension       INTEGER NOT NULL,
    metadata        TEXT    NOT NULL DEFAULT '{}',
    semantic_weight REAL    NOT NULL DEFAULT 1.0,
    version_vector  TEXT    NOT NULL DEFAULT '{}',
    updated_at      REAL    NOT NULL,
    is_deleted      INTEGER NOT NULL DEFAULT 0
                            CHECK (is_deleted IN (0, 1))
);

-- Logical write-ahead log. Every mutation appends one row inside the same
-- transaction that mutates `vectors`, so the log and the table can never
-- diverge. `lsn` is the replication cursor.
CREATE TABLE IF NOT EXISTS wal_log (
    lsn        INTEGER PRIMARY KEY AUTOINCREMENT,
    op         TEXT    NOT NULL CHECK (op IN ('INSERT', 'UPDATE', 'DELETE')),
    vector_id  TEXT    NOT NULL,
    payload    TEXT    NOT NULL,   -- full after-image of the record (JSON)
    created_at REAL    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_wal_vector_id ON wal_log (vector_id);
CREATE INDEX IF NOT EXISTS idx_vectors_live
    ON vectors (vector_id) WHERE is_deleted = 0;

-- Singleton row tracking the replication frontier.
CREATE TABLE IF NOT EXISTS sync_state (
    id              INTEGER PRIMARY KEY CHECK (id = 1),
    last_synced_lsn INTEGER NOT NULL DEFAULT 0,
    last_sync_at    REAL
);

INSERT OR IGNORE INTO sync_state (id, last_synced_lsn, last_sync_at)
VALUES (1, 0, NULL);
"""


class VeloSyncStore:
    """Durable local vector store with a logical WAL for replication.

    All mutations are transactional: the row in ``vectors`` and its WAL entry
    commit atomically or not at all.

    Args:
        db_path: SQLite file path, or ``":memory:"`` for ephemeral stores.
        dimension: Fixed embedding dimension enforced on every write/query.
        device_id: Stable identifier of this edge device; advances this
            device's component in every record's version vector.

    Raises:
        VeloSyncError: If the database cannot be opened or initialized.
        ValueError: If ``dimension`` is not a positive integer.
    """

    def __init__(self, db_path: str, dimension: int, device_id: str) -> None:
        if dimension <= 0:
            raise ValueError(f"dimension must be positive, got {dimension}")
        if not device_id:
            raise ValueError("device_id must be a non-empty string")
        self.db_path = db_path
        self.dimension = dimension
        self.device_id = device_id
        try:
            self._conn = sqlite3.connect(db_path)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA synchronous=NORMAL;")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()
        except sqlite3.Error as exc:
            raise VeloSyncError(f"Failed to initialize store at {db_path!r}: {exc}") from exc
        logger.info("Store ready: path=%s dim=%d device=%s", db_path, dimension, device_id)

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()

    def __enter__(self) -> "VeloSyncStore":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    # -- internal helpers --------------------------------------------------

    def _validate_dimension(self, embedding: Sequence[float]) -> None:
        if len(embedding) != self.dimension:
            raise DimensionMismatchError(
                f"Store dimension is {self.dimension}, got vector of "
                f"dimension {len(embedding)}"
            )

    def _row_to_record(self, row: sqlite3.Row) -> VectorRecord:
        return VectorRecord(
            vector_id=row["vector_id"],
            embedding=decode_embedding(row["embedding"], row["dimension"]),
            metadata=json.loads(row["metadata"]),
            semantic_weight=row["semantic_weight"],
            version_vector=json.loads(row["version_vector"]),
            updated_at=row["updated_at"],
            is_deleted=bool(row["is_deleted"]),
        )

    def _write_record(
        self, record: VectorRecord, op: OpType, log_wal: bool
    ) -> Optional[int]:
        """Atomically upsert a record row and (optionally) append a WAL entry.

        ``log_wal=False`` is used when applying *remote* changes during sync,
        so that replicated edits are not echoed back to the server on the
        next push.

        Returns:
            The new LSN if a WAL entry was written, else ``None``.
        """
        now = time.time()
        blob = encode_embedding(record.embedding)
        try:
            with self._conn:  # implicit BEGIN ... COMMIT/ROLLBACK
                self._conn.execute(
                    """
                    INSERT INTO vectors
                        (vector_id, embedding, dimension, metadata,
                         semantic_weight, version_vector, updated_at, is_deleted)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (vector_id) DO UPDATE SET
                        embedding       = excluded.embedding,
                        dimension       = excluded.dimension,
                        metadata        = excluded.metadata,
                        semantic_weight = excluded.semantic_weight,
                        version_vector  = excluded.version_vector,
                        updated_at      = excluded.updated_at,
                        is_deleted      = excluded.is_deleted
                    """,
                    (
                        record.vector_id,
                        blob,
                        len(record.embedding),
                        json.dumps(record.metadata, separators=(",", ":")),
                        record.semantic_weight,
                        json.dumps(record.version_vector, separators=(",", ":")),
                        record.updated_at,
                        int(record.is_deleted),
                    ),
                )
                if not log_wal:
                    return None
                cursor = self._conn.execute(
                    """
                    INSERT INTO wal_log (op, vector_id, payload, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        op.value,
                        record.vector_id,
                        json.dumps(record.to_wire(), separators=(",", ":")),
                        now,
                    ),
                )
                lsn = cursor.lastrowid
        except sqlite3.Error as exc:
            raise VeloSyncError(f"Write failed for {record.vector_id}: {exc}") from exc
        logger.debug("WAL append: lsn=%s op=%s id=%s", lsn, op.value, record.vector_id)
        return lsn

    def _bump_clock(self, base: Mapping[str, int]) -> Dict[str, int]:
        """Advance this device's component of a version vector."""
        clock = dict(base)
        clock[self.device_id] = clock.get(self.device_id, 0) + 1
        return clock

    # -- public mutation API -------------------------------------------------

    def insert(
        self,
        embedding: Sequence[float],
        metadata: Optional[Dict[str, Any]] = None,
        semantic_weight: float = 1.0,
        vector_id: Optional[str] = None,
    ) -> VectorRecord:
        """Insert a new vector.

        Args:
            embedding: Dense vector of the store's fixed dimension.
            metadata: Optional JSON-serializable metadata.
            semantic_weight: Confidence score; must be non-negative.
            vector_id: Optional explicit id; a UUID4 is generated otherwise.

        Returns:
            The committed :class:`VectorRecord`.

        Raises:
            DimensionMismatchError: On wrong dimension.
            VeloSyncError: If the id already exists or the write fails.
        """
        self._validate_dimension(embedding)
        if semantic_weight < 0:
            raise ValueError("semantic_weight must be non-negative")
        vid = vector_id or str(uuid.uuid4())
        if self.get(vid, include_deleted=True) is not None:
            raise VeloSyncError(f"vector_id {vid!r} already exists; use update()")
        record = VectorRecord(
            vector_id=vid,
            embedding=tuple(float(x) for x in embedding),
            metadata=dict(metadata or {}),
            semantic_weight=float(semantic_weight),
            version_vector=self._bump_clock({}),
            updated_at=time.time(),
            is_deleted=False,
        )
        self._write_record(record, OpType.INSERT, log_wal=True)
        return record

    def update(
        self,
        vector_id: str,
        embedding: Optional[Sequence[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        semantic_weight: Optional[float] = None,
    ) -> VectorRecord:
        """Update an existing vector's embedding, metadata, and/or weight.

        Only the supplied fields change; the version vector is advanced and a
        WAL UPDATE entry is appended.

        Raises:
            VectorNotFoundError: If the id does not exist or is a tombstone.
        """
        current = self.get(vector_id)
        if current is None:
            raise VectorNotFoundError(f"No live vector with id {vector_id!r}")
        if embedding is not None:
            self._validate_dimension(embedding)
        if semantic_weight is not None and semantic_weight < 0:
            raise ValueError("semantic_weight must be non-negative")
        updated = replace(
            current,
            embedding=(
                tuple(float(x) for x in embedding)
                if embedding is not None
                else current.embedding
            ),
            metadata=dict(metadata) if metadata is not None else current.metadata,
            semantic_weight=(
                float(semantic_weight)
                if semantic_weight is not None
                else current.semantic_weight
            ),
            version_vector=self._bump_clock(current.version_vector),
            updated_at=time.time(),
        )
        self._write_record(updated, OpType.UPDATE, log_wal=True)
        return updated

    def delete(self, vector_id: str) -> VectorRecord:
        """Tombstone a vector (soft delete) so the deletion replicates.

        Raises:
            VectorNotFoundError: If the id does not exist or is already deleted.
        """
        current = self.get(vector_id)
        if current is None:
            raise VectorNotFoundError(f"No live vector with id {vector_id!r}")
        tombstone = replace(
            current,
            version_vector=self._bump_clock(current.version_vector),
            updated_at=time.time(),
            is_deleted=True,
        )
        self._write_record(tombstone, OpType.DELETE, log_wal=True)
        return tombstone

    def apply_remote(self, record: VectorRecord) -> None:
        """Write a remotely-authored record without generating a WAL entry.

        Used by the sync engine when a remote record wins conflict resolution;
        bypassing the WAL prevents the change from being pushed back to the
        server on the next sync cycle (echo suppression).
        """
        if record.embedding and len(record.embedding) != self.dimension:
            raise DimensionMismatchError(
                f"Remote record {record.vector_id!r} has dimension "
                f"{len(record.embedding)}, store expects {self.dimension}"
            )
        self._write_record(record, OpType.UPDATE, log_wal=False)

    # -- public read API -----------------------------------------------------

    def get(
        self, vector_id: str, include_deleted: bool = False
    ) -> Optional[VectorRecord]:
        """Fetch one record by id, or ``None`` if absent (or tombstoned)."""
        try:
            row = self._conn.execute(
                "SELECT * FROM vectors WHERE vector_id = ?", (vector_id,)
            ).fetchone()
        except sqlite3.Error as exc:
            raise VeloSyncError(f"Read failed for {vector_id!r}: {exc}") from exc
        if row is None:
            return None
        record = self._row_to_record(row)
        if record.is_deleted and not include_deleted:
            return None
        return record

    def iter_live_records(self) -> Iterator[VectorRecord]:
        """Yield all non-deleted records (streaming cursor, low memory)."""
        try:
            cursor = self._conn.execute(
                "SELECT * FROM vectors WHERE is_deleted = 0"
            )
            for row in cursor:
                yield self._row_to_record(row)
        except sqlite3.Error as exc:
            raise VeloSyncError(f"Scan failed: {exc}") from exc

    def count(self, include_deleted: bool = False) -> int:
        """Number of records in the store."""
        sql = (
            "SELECT COUNT(*) FROM vectors"
            if include_deleted
            else "SELECT COUNT(*) FROM vectors WHERE is_deleted = 0"
        )
        return int(self._conn.execute(sql).fetchone()[0])

    def search(
        self,
        query: Sequence[float],
        top_k: int = 5,
        min_similarity: float = -1.0,
    ) -> List[SearchResult]:
        """Brute-force cosine k-NN over all live vectors.

        On-device corpora are typically small (10^3–10^5 vectors), where an
        exact linear scan in pure Python is both simplest and fully accurate.
        A bounded min-heap keeps memory at O(top_k).

        Args:
            query: Query embedding of the store's dimension.
            top_k: Maximum number of results, must be >= 1.
            min_similarity: Discard hits below this threshold.

        Returns:
            Results sorted by similarity, descending.
        """
        self._validate_dimension(query)
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        q = tuple(float(x) for x in query)
        # Min-heap of (similarity, tie_breaker, result); heap[0] is the worst hit.
        heap: List[Tuple[float, str, SearchResult]] = []
        for record in self.iter_live_records():
            sim = cosine_similarity(q, record.embedding)
            if sim < min_similarity:
                continue
            result = SearchResult(
                vector_id=record.vector_id,
                similarity=sim,
                metadata=record.metadata,
                semantic_weight=record.semantic_weight,
            )
            item = (sim, record.vector_id, result)
            if len(heap) < top_k:
                heapq.heappush(heap, item)
            elif sim > heap[0][0]:
                heapq.heapreplace(heap, item)
        return [item[2] for item in sorted(heap, key=lambda t: t[0], reverse=True)]

    # -- WAL / sync-state accessors -------------------------------------------

    def last_synced_lsn(self) -> int:
        """The highest LSN known to be durably acknowledged by the server."""
        row = self._conn.execute(
            "SELECT last_synced_lsn FROM sync_state WHERE id = 1"
        ).fetchone()
        return int(row["last_synced_lsn"])

    def head_lsn(self) -> int:
        """The highest LSN ever written locally (0 if the log is empty)."""
        row = self._conn.execute("SELECT MAX(lsn) AS m FROM wal_log").fetchone()
        return int(row["m"] or 0)

    def wal_entries_after(self, lsn: int) -> List[WalEntry]:
        """All WAL entries with LSN strictly greater than ``lsn``, in order."""
        try:
            rows = self._conn.execute(
                "SELECT * FROM wal_log WHERE lsn > ? ORDER BY lsn ASC", (lsn,)
            ).fetchall()
        except sqlite3.Error as exc:
            raise VeloSyncError(f"WAL read failed: {exc}") from exc
        entries: List[WalEntry] = []
        for row in rows:
            entries.append(
                WalEntry(
                    lsn=row["lsn"],
                    op=OpType(row["op"]),
                    vector_id=row["vector_id"],
                    record=VectorRecord.from_wire(json.loads(row["payload"])),
                    created_at=row["created_at"],
                )
            )
        return entries

    def mark_synced(self, lsn: int) -> None:
        """Advance the replication frontier after a server acknowledgment.

        Idempotent and monotonic: the frontier never moves backwards.
        """
        try:
            with self._conn:
                self._conn.execute(
                    """
                    UPDATE sync_state
                    SET last_synced_lsn = MAX(last_synced_lsn, ?),
                        last_sync_at = ?
                    WHERE id = 1
                    """,
                    (lsn, time.time()),
                )
        except sqlite3.Error as exc:
            raise VeloSyncError(f"Failed to mark LSN {lsn} synced: {exc}") from exc

    def truncate_wal(self) -> int:
        """Garbage-collect WAL entries at or below the synced frontier.

        Returns:
            Number of log rows removed.
        """
        frontier = self.last_synced_lsn()
        try:
            with self._conn:
                cursor = self._conn.execute(
                    "DELETE FROM wal_log WHERE lsn <= ?", (frontier,)
                )
            return cursor.rowcount
        except sqlite3.Error as exc:
            raise VeloSyncError(f"WAL truncation failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Synchronization engine
# ---------------------------------------------------------------------------


class SyncEngine:
    """Computes push payloads and reconciles pull responses for one device.

    Protocol (state-based delta replication):

      PUSH:  client sends every WAL after-image past ``last_synced_lsn``,
             log-compacted to one record per vector_id (the latest one wins
             locally, so only it matters remotely).
      PULL:  server replies with records other devices changed; each is merged
             through :meth:`resolve_conflict`.
      ACK:   server acknowledges the payload's ``head_lsn``; the client
             advances its replication frontier and may truncate the WAL.
    """

    PROTOCOL_VERSION = 1

    def __init__(self, store: VeloSyncStore) -> None:
        self.store = store

    # -- push side -----------------------------------------------------------

    def pending_changes(self) -> List[WalEntry]:
        """WAL entries not yet acknowledged by the server, in LSN order."""
        return self.store.wal_entries_after(self.store.last_synced_lsn())

    def compact(self, entries: Sequence[WalEntry]) -> List[WalEntry]:
        """Log compaction: keep only the final entry per vector_id.

        Because each WAL row stores a full after-image, intermediate states
        are redundant for replication — e.g. INSERT then DELETE of the same
        vector compacts to a single DELETE tombstone.
        """
        latest: Dict[str, WalEntry] = {}
        for entry in entries:  # ascending LSN, so later entries overwrite
            latest[entry.vector_id] = entry
        return sorted(latest.values(), key=lambda e: e.lsn)

    def build_sync_payload(self) -> Dict[str, Any]:
        """Assemble the JSON-ready push payload for the server.

        Returns:
            Dict with device identity, the LSN window covered, compacted
            change list, and a SHA-256 integrity checksum over the changes.
        """
        pending = self.pending_changes()
        compacted = self.compact(pending)
        changes = [
            {
                "lsn": e.lsn,
                "op": e.op.value,
                "record": e.record.to_wire(),
            }
            for e in compacted
        ]
        body = json.dumps(changes, separators=(",", ":"), sort_keys=True)
        base_lsn = self.store.last_synced_lsn()
        return {
            "protocol_version": self.PROTOCOL_VERSION,
            "device_id": self.store.device_id,
            "base_lsn": base_lsn,
            # The WAL may have been truncated past sync; the head never
            # regresses below the acknowledged frontier.
            "head_lsn": max(self.store.head_lsn(), base_lsn),
            "change_count": len(changes),
            "raw_wal_entries": len(pending),
            "changes": changes,
            "checksum_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
            "generated_at": time.time(),
        }

    # -- conflict resolution ---------------------------------------------------

    @staticmethod
    def resolve_conflict(
        local: Optional[VectorRecord], remote: VectorRecord
    ) -> Tuple[VectorRecord, ConflictResolution]:
        """Semantic Version Vector conflict resolution.

        Decision procedure:
          1. No local copy            -> take remote.
          2. Causality via version vectors:
               local ⊒ remote         -> keep local (it already saw remote).
               remote ⊒ local         -> take remote.
               equal                  -> no-op (keep local).
          3. Concurrent edits         -> higher ``semantic_weight`` wins.
          4. Equal weights            -> last-write-wins on ``updated_at``
                                         (ties break toward remote, so all
                                         replicas converge identically).

        The winner's version vector is merged with the loser's so the
        resolved record causally dominates both histories (no conflict
        re-detection on subsequent syncs).

        Returns:
            ``(winning_record, resolution_label)``.
        """
        if local is None:
            return remote, ConflictResolution.NO_LOCAL_COPY

        relation = compare_version_vectors(local.version_vector, remote.version_vector)
        if relation == "equal":
            return local, ConflictResolution.IDENTICAL_HISTORY
        if relation == "a_after_b":
            return local, ConflictResolution.LOCAL_DOMINATES
        if relation == "b_after_a":
            return remote, ConflictResolution.REMOTE_DOMINATES

        # Concurrent: semantic weight, then LWW.
        merged_clock = merge_version_vectors(
            local.version_vector, remote.version_vector
        )
        if local.semantic_weight > remote.semantic_weight:
            winner, label = local, ConflictResolution.CONCURRENT_SEMANTIC_LOCAL
        elif remote.semantic_weight > local.semantic_weight:
            winner, label = remote, ConflictResolution.CONCURRENT_SEMANTIC_REMOTE
        elif local.updated_at > remote.updated_at:
            winner, label = local, ConflictResolution.CONCURRENT_LWW_LOCAL
        else:
            winner, label = remote, ConflictResolution.CONCURRENT_LWW_REMOTE
        return replace(winner, version_vector=merged_clock), label

    # -- pull side --------------------------------------------------------------

    def apply_remote_changes(
        self, remote_records: Sequence[Mapping[str, Any]]
    ) -> List[Tuple[str, ConflictResolution]]:
        """Merge a batch of remote records into the local store.

        Each record is resolved against the local copy. Remote wins are
        written via :meth:`VeloSyncStore.apply_remote` (no WAL echo); local
        wins with a merged clock are re-written through the WAL so the merged
        causal history propagates back to the server.

        Returns:
            ``[(vector_id, resolution_label), ...]`` audit trail.
        """
        outcomes: List[Tuple[str, ConflictResolution]] = []
        for wire in remote_records:
            remote = VectorRecord.from_wire(wire)
            local = self.store.get(remote.vector_id, include_deleted=True)
            winner, label = self.resolve_conflict(local, remote)
            if label in (
                ConflictResolution.IDENTICAL_HISTORY,
                ConflictResolution.LOCAL_DOMINATES,
            ):
                pass  # local state already correct; nothing to write
            elif label in (
                ConflictResolution.CONCURRENT_SEMANTIC_LOCAL,
                ConflictResolution.CONCURRENT_LWW_LOCAL,
            ):
                # Local content won but its clock was merged — persist via the
                # WAL so the dominating history is pushed on the next cycle.
                self.store._write_record(winner, OpType.UPDATE, log_wal=True)
            else:
                self.store.apply_remote(winner)
            logger.info(
                "Conflict resolution: id=%s outcome=%s", remote.vector_id, label.value
            )
            outcomes.append((remote.vector_id, label))
        return outcomes

    # -- acknowledgment ----------------------------------------------------------

    def acknowledge(self, acked_head_lsn: int, truncate: bool = True) -> None:
        """Record the server's durable acknowledgment of a pushed payload."""
        self.store.mark_synced(acked_head_lsn)
        if truncate:
            removed = self.store.truncate_wal()
            logger.info("WAL truncated: %d entries GC'd", removed)


# ---------------------------------------------------------------------------
# Mock cloud endpoint (demo only)
# ---------------------------------------------------------------------------


class MockCloudVectorDB:
    """In-memory stand-in for the central cloud vector database.

    Accepts push payloads, verifies their checksum, stores records, and can
    fabricate a concurrent remote edit to exercise conflict resolution.
    """

    def __init__(self) -> None:
        self._records: Dict[str, Dict[str, Any]] = {}

    def receive_push(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        """Validate and ingest a client push; return an ACK envelope."""
        body = json.dumps(payload["changes"], separators=(",", ":"), sort_keys=True)
        digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
        if digest != payload["checksum_sha256"]:
            return {"status": "rejected", "reason": "checksum_mismatch"}
        for change in payload["changes"]:
            self._records[change["record"]["vector_id"]] = dict(change["record"])
        return {
            "status": "ok",
            "acked_head_lsn": payload["head_lsn"],
            "ingested": len(payload["changes"]),
        }

    def fabricate_concurrent_edit(
        self,
        vector_id: str,
        editing_device: str,
        new_metadata: Dict[str, Any],
        semantic_weight: float,
    ) -> Dict[str, Any]:
        """Simulate another device having edited ``vector_id`` concurrently.

        The fabricated record's version vector advances only the *other*
        device's counter from the last state the cloud saw — which is exactly
        what produces a concurrent (incomparable) clock pair.
        """
        if vector_id not in self._records:
            raise KeyError(f"Cloud has no record {vector_id!r}")
        base = dict(self._records[vector_id])
        clock = dict(base.get("version_vector") or {})
        clock[editing_device] = clock.get(editing_device, 0) + 1
        base.update(
            metadata=new_metadata,
            semantic_weight=semantic_weight,
            version_vector=clock,
            updated_at=time.time(),
        )
        self._records[vector_id] = base
        return base


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def _pretty(payload: Mapping[str, Any], max_embedding_preview: int = 3) -> str:
    """Render a sync payload as JSON with embeddings truncated for readability."""

    def shrink(obj: Any) -> Any:
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                if k == "embedding" and isinstance(v, list):
                    head = [round(x, 4) for x in v[:max_embedding_preview]]
                    out[k] = head + [f"... ({len(v)} dims)"]
                else:
                    out[k] = shrink(v)
            return out
        if isinstance(obj, list):
            return [shrink(x) for x in obj]
        return obj

    return json.dumps(shrink(dict(payload)), indent=2)


def main() -> None:
    """End-to-end simulation of the VeloSync offline-first lifecycle."""
    import random

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    rng = random.Random(42)
    dim = 8

    print("=" * 72)
    print("VeloSync demo — offline-first vector store + log-structured sync")
    print("=" * 72)

    # 1. Initialize local SQLite storage (in-memory for the demo; pass a file
    #    path for a persistent on-device store).
    store = VeloSyncStore(db_path=":memory:", dimension=dim, device_id="edge-phone-01")
    engine = SyncEngine(store)
    cloud = MockCloudVectorDB()

    # 2. Create / index 10 mock vectors.
    labels = [
        "coffee shop review", "battery health log", "voice memo: groceries",
        "photo caption: sunset", "support ticket draft", "sensor anomaly note",
        "meeting summary", "recipe: pad thai", "navigation landmark",
        "device error trace",
    ]
    records: List[VectorRecord] = []
    for label in labels:
        embedding = [rng.uniform(-1.0, 1.0) for _ in range(dim)]
        rec = store.insert(
            embedding=embedding,
            metadata={"label": label, "source": "on-device-encoder"},
            semantic_weight=round(rng.uniform(0.5, 1.0), 3),
        )
        records.append(rec)
    print(f"\n[1] Indexed {store.count()} vectors locally "
          f"(head LSN = {store.head_lsn()})")

    # 3. Local nearest-neighbor query — pure-Python cosine similarity.
    target = records[3]
    query = [x + rng.uniform(-0.05, 0.05) for x in target.embedding]  # noisy probe
    hits = store.search(query, top_k=3)
    print("\n[2] Local k-NN query (noisy probe of "
          f"'{target.metadata['label']}'):")
    for rank, hit in enumerate(hits, start=1):
        print(f"    #{rank}  sim={hit.similarity:+.4f}  weight={hit.semantic_weight:.3f}"
              f"  label={hit.metadata['label']!r}")
    assert hits[0].vector_id == target.vector_id, "sanity: probe should match target"

    # 4. First sync while "online": push everything, server ACKs.
    payload = engine.build_sync_payload()
    ack = cloud.receive_push(payload)
    engine.acknowledge(ack["acked_head_lsn"])
    print(f"\n[3] Initial sync: pushed {ack['ingested']} records, "
          f"server ACKed LSN {ack['acked_head_lsn']}; "
          f"pending now = {len(engine.pending_changes())}")

    # 5. Device goes OFFLINE — local edits accumulate in the WAL.
    print("\n[4] -- connectivity lost: performing offline edits --")
    store.update(
        records[0].vector_id,
        metadata={"label": "coffee shop review", "edited": "offline", "stars": 5},
        semantic_weight=0.95,
    )
    store.update(
        records[1].vector_id,
        embedding=[x * 0.9 for x in records[1].embedding],
        semantic_weight=0.40,  # device is *less* confident after re-encoding
    )
    store.delete(records[2].vector_id)
    new_rec = store.insert(
        embedding=[rng.uniform(-1.0, 1.0) for _ in range(dim)],
        metadata={"label": "offline note: wifi password", "source": "on-device-encoder"},
        semantic_weight=0.88,
    )
    # Churn that log compaction should collapse: update the same record twice.
    store.update(new_rec.vector_id, metadata={"label": "offline note: wifi password",
                                              "revised": True})
    print(f"    2 updates, 1 delete, 1 insert (+1 redundant update) -> "
          f"{len(engine.pending_changes())} raw WAL entries pending")

    # Meanwhile, ANOTHER device edits two of the same vectors in the cloud.
    remote_a = cloud.fabricate_concurrent_edit(
        records[0].vector_id, editing_device="edge-tablet-07",
        new_metadata={"label": "coffee shop review", "edited": "tablet", "stars": 2},
        semantic_weight=0.60,   # lower confidence than our 0.95 -> local should win
    )
    remote_b = cloud.fabricate_concurrent_edit(
        records[1].vector_id, editing_device="edge-tablet-07",
        new_metadata={"label": "battery health log", "edited": "tablet"},
        semantic_weight=0.85,   # higher confidence than our 0.40 -> remote should win
    )

    # 6. Connectivity restored — build the push payload.
    payload = engine.build_sync_payload()
    print("\n[5] -- connectivity restored: sync payload ready to POST --")
    print(_pretty(payload))
    print(f"    note: {payload['raw_wal_entries']} WAL entries compacted to "
          f"{payload['change_count']} changes")

    # 7. Push, then pull the server's concurrent records and resolve conflicts.
    ack = cloud.receive_push(payload)
    pull = [
        dict(remote_a),                          # concurrent, lower remote weight
        dict(remote_b),                          # concurrent, higher remote weight
    ]
    outcomes = engine.apply_remote_changes(pull)
    print("\n[6] Conflict resolution outcomes:")
    for vid, label in outcomes:
        local_now = store.get(vid, include_deleted=True)
        assert local_now is not None
        print(f"    {vid[:8]}…  {label.value:<28} "
              f"final_weight={local_now.semantic_weight:.2f} "
              f"final_meta={local_now.metadata.get('edited', 'local')!r} "
              f"clock={local_now.version_vector}")

    # 8. Acknowledge and garbage-collect the WAL.
    engine.acknowledge(ack["acked_head_lsn"])
    print(f"\n[7] Final state: {store.count()} live vectors, "
          f"{store.count(include_deleted=True) - store.count()} tombstone(s), "
          f"synced LSN = {store.last_synced_lsn()}, "
          f"pending = {len(engine.pending_changes())} "
          f"(local conflict-wins remain queued for the next push cycle)")

    store.close()
    print("\nDemo complete.")


if __name__ == "__main__":
    main()
