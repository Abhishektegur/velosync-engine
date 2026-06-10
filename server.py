"""
VeloSync Server — central master store and synchronization authority.

A FastAPI service that fronts a SQLite master database (``server_store.db``)
and implements the VeloSync replication protocol:

  POST /sync
      Receives a client's compacted WAL payload. The server:
        1. Verifies the SHA-256 integrity checksum over the change list.
        2. Rejects replays (payloads whose ``head_lsn`` is not beyond the
           device's acknowledged frontier) idempotently.
        3. Resolves every incoming record against the master copy using
           Semantic Version Vector logic (causality first, then semantic
           weight, then last-write-wins).
        4. Persists every winner; appends accepted changes to a server-side
           change feed (``change_log``) keyed by a global ``server_seq``.
        5. Returns "pull" records the client must apply: server-side conflict
           winners, merged-clock records (so clocks converge immediately),
           and changes originated by *other* devices since this device's last
           cursor — with echo suppression so a device never receives its own
           pushes back.

  POST /ack
      The client confirms it durably applied the pull set. Only then does the
      server advance the device's cursors (client WAL LSN + server_seq), which
      makes the protocol crash-safe: a client that dies mid-pull simply
      re-pulls on its next sync.

  GET /health, GET /stats
      Liveness and observability.

Dependencies: fastapi, uvicorn — plus the zero-dependency ``velosync`` module
for the shared data model and conflict-resolution algebra (single source of
truth between client and server).

Run:
    uvicorn server:app --host 127.0.0.1 --port 8000
    # or simply:  python3 server.py
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from velosync import (
    ConflictResolution,
    OpType,
    SyncEngine,
    VectorRecord,
    VeloSyncError,
    decode_embedding,
    encode_embedding,
)

logger = logging.getLogger("velosync.server")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

DB_PATH = "server_store.db"
PROTOCOL_VERSION = 1


# ---------------------------------------------------------------------------
# Wire models (request/response validation)
# ---------------------------------------------------------------------------


class SyncPayload(BaseModel):
    """Push payload produced by ``client.HTTPSyncEngine.build_sync_payload``.

    ``changes`` is deliberately typed as raw dicts (not nested models) so the
    server can recompute the checksum over byte-identical JSON before any
    coercion happens.
    """

    protocol_version: int
    device_id: str = Field(min_length=1)
    base_lsn: int = Field(ge=0)
    head_lsn: int = Field(ge=0)
    change_count: int = Field(ge=0)
    raw_wal_entries: int = Field(ge=0)
    changes: List[Dict[str, Any]]
    checksum_sha256: str = Field(min_length=64, max_length=64)
    generated_at: float


class AckPayload(BaseModel):
    """Client confirmation that a sync round was durably applied."""

    device_id: str = Field(min_length=1)
    acked_head_lsn: int = Field(ge=0)
    acked_server_seq: int = Field(ge=0)


# ---------------------------------------------------------------------------
# Master store
# ---------------------------------------------------------------------------

_SERVER_SCHEMA = """
CREATE TABLE IF NOT EXISTS vectors (
    vector_id       TEXT    PRIMARY KEY,
    embedding       BLOB    NOT NULL,
    dimension       INTEGER NOT NULL,
    metadata        TEXT    NOT NULL DEFAULT '{}',
    semantic_weight REAL    NOT NULL DEFAULT 1.0,
    version_vector  TEXT    NOT NULL DEFAULT '{}',
    updated_at      REAL    NOT NULL,
    is_deleted      INTEGER NOT NULL DEFAULT 0 CHECK (is_deleted IN (0, 1))
);

-- Global, totally ordered feed of accepted changes. Drives cross-device
-- fan-out: a device pulls every entry past its cursor that it did not author.
CREATE TABLE IF NOT EXISTS change_log (
    server_seq    INTEGER PRIMARY KEY AUTOINCREMENT,
    vector_id     TEXT    NOT NULL,
    origin_device TEXT    NOT NULL,
    created_at    REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_change_log_origin ON change_log (origin_device);

-- Per-device replication cursors. Advanced only on explicit /ack, which makes
-- pulls re-deliverable (at-least-once) if a client crashes mid-apply.
CREATE TABLE IF NOT EXISTS device_cursors (
    device_id            TEXT PRIMARY KEY,
    last_acked_client_lsn INTEGER NOT NULL DEFAULT 0,
    last_acked_server_seq INTEGER NOT NULL DEFAULT 0,
    updated_at           REAL    NOT NULL
);
"""


class MasterStore:
    """Thread-safe SQLite master store behind the FastAPI handlers.

    FastAPI may execute sync endpoints on a thread pool, so a single
    connection (``check_same_thread=False``) is serialized behind a lock;
    every public method is one atomic critical section / transaction.
    """

    def __init__(self, db_path: str) -> None:
        self._lock = threading.Lock()
        try:
            self._conn = sqlite3.connect(db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA synchronous=NORMAL;")
            self._conn.executescript(_SERVER_SCHEMA)
            self._conn.commit()
        except sqlite3.Error as exc:
            raise RuntimeError(f"Cannot open master store {db_path!r}: {exc}") from exc
        logger.info("Master store ready at %s", db_path)

    # -- low-level row mapping ------------------------------------------------

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> VectorRecord:
        return VectorRecord(
            vector_id=row["vector_id"],
            embedding=decode_embedding(row["embedding"], row["dimension"]),
            metadata=json.loads(row["metadata"]),
            semantic_weight=row["semantic_weight"],
            version_vector=json.loads(row["version_vector"]),
            updated_at=row["updated_at"],
            is_deleted=bool(row["is_deleted"]),
        )

    def _get_unlocked(self, vector_id: str) -> Optional[VectorRecord]:
        row = self._conn.execute(
            "SELECT * FROM vectors WHERE vector_id = ?", (vector_id,)
        ).fetchone()
        return self._row_to_record(row) if row else None

    def _upsert_unlocked(self, record: VectorRecord) -> None:
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
                encode_embedding(record.embedding),
                len(record.embedding),
                json.dumps(record.metadata, separators=(",", ":")),
                record.semantic_weight,
                json.dumps(record.version_vector, separators=(",", ":")),
                record.updated_at,
                int(record.is_deleted),
            ),
        )

    def _append_change_unlocked(self, vector_id: str, origin_device: str) -> int:
        cursor = self._conn.execute(
            "INSERT INTO change_log (vector_id, origin_device, created_at) "
            "VALUES (?, ?, ?)",
            (vector_id, origin_device, time.time()),
        )
        assert cursor.lastrowid is not None
        return cursor.lastrowid

    def _cursors_unlocked(self, device_id: str) -> Tuple[int, int]:
        row = self._conn.execute(
            "SELECT last_acked_client_lsn, last_acked_server_seq "
            "FROM device_cursors WHERE device_id = ?",
            (device_id,),
        ).fetchone()
        if row is None:
            return 0, 0
        return int(row["last_acked_client_lsn"]), int(row["last_acked_server_seq"])

    # -- protocol operations ----------------------------------------------------

    def process_sync(self, payload: SyncPayload) -> Dict[str, Any]:
        """Apply a push payload and compute the pull set for the device.

        Runs entirely inside one transaction so a crash mid-payload leaves
        the master store untouched (all-or-nothing ingestion).
        """
        device = payload.device_id
        with self._lock, self._conn:
            last_client_lsn, last_server_seq = self._cursors_unlocked(device)

            # Replay / idempotency guard: a payload the device already had
            # acknowledged carries nothing new. Still serve the pull set so a
            # client that lost the previous response can recover.
            is_replay = payload.head_lsn <= last_client_lsn and payload.change_count > 0

            conflicts: List[Dict[str, str]] = []
            pull: Dict[str, VectorRecord] = {}

            if not is_replay:
                for change in payload.changes:
                    try:
                        incoming = VectorRecord.from_wire(change["record"])
                        op = OpType(change["op"])  # validated against enum
                    except (KeyError, ValueError, VeloSyncError) as exc:
                        raise HTTPException(
                            status_code=422, detail=f"Malformed change entry: {exc}"
                        ) from exc

                    master = self._get_unlocked(incoming.vector_id)
                    # resolve_conflict(local=master copy, remote=incoming)
                    winner, label = SyncEngine.resolve_conflict(master, incoming)
                    conflicts.append(
                        {"vector_id": incoming.vector_id, "resolution": label.value}
                    )

                    if label in (
                        ConflictResolution.NO_LOCAL_COPY,
                        ConflictResolution.REMOTE_DOMINATES,
                    ):
                        # Clean fast-forward: client is strictly ahead.
                        self._upsert_unlocked(winner)
                        self._append_change_unlocked(winner.vector_id, device)
                    elif label == ConflictResolution.IDENTICAL_HISTORY:
                        pass  # byte-for-byte same causal state; nothing to do
                    elif label == ConflictResolution.LOCAL_DOMINATES:
                        # Master already saw this edit and more: client must
                        # catch up. Nothing stored; master copy goes to pull.
                        pull[winner.vector_id] = winner
                    else:
                        # Concurrent edits. The winner carries a merged clock
                        # that dominates both histories. Persist it, feed it,
                        # and send it back so the client's clock converges in
                        # this same round (even when the client's *content*
                        # won, its local clock lacks the merge).
                        origin = (
                            device
                            if label
                            in (
                                ConflictResolution.CONCURRENT_SEMANTIC_REMOTE,
                                ConflictResolution.CONCURRENT_LWW_REMOTE,
                            )
                            else "server-merge"
                        )
                        self._upsert_unlocked(winner)
                        self._append_change_unlocked(winner.vector_id, origin)
                        pull[winner.vector_id] = winner
                        logger.info(
                            "Conflict on %s: %s (weights local=%.2f incoming=%.2f)",
                            incoming.vector_id[:8],
                            label.value,
                            master.semantic_weight if master else float("nan"),
                            incoming.semantic_weight,
                        )

            # Cross-device fan-out: every feed entry past the device's cursor
            # that this device did not author (echo suppression). Conflict
            # pulls computed above take precedence on collision.
            rows = self._conn.execute(
                """
                SELECT DISTINCT vector_id FROM change_log
                WHERE server_seq > ? AND origin_device != ?
                """,
                (last_server_seq, device),
            ).fetchall()
            for row in rows:
                vid = row["vector_id"]
                if vid not in pull:
                    rec = self._get_unlocked(vid)
                    if rec is not None:
                        pull[vid] = rec

            head_seq_row = self._conn.execute(
                "SELECT COALESCE(MAX(server_seq), 0) AS m FROM change_log"
            ).fetchone()
            server_seq = int(head_seq_row["m"])

        return {
            "status": "ok",
            "protocol_version": PROTOCOL_VERSION,
            "replay_detected": is_replay,
            "acked_head_lsn": payload.head_lsn,
            "server_seq": server_seq,
            "applied": payload.change_count if not is_replay else 0,
            "conflicts": conflicts,
            "pull": [rec.to_wire() for rec in pull.values()],
        }

    def process_ack(self, ack: AckPayload) -> Dict[str, Any]:
        """Advance a device's cursors (monotonic, idempotent)."""
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO device_cursors
                    (device_id, last_acked_client_lsn, last_acked_server_seq, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT (device_id) DO UPDATE SET
                    last_acked_client_lsn =
                        MAX(device_cursors.last_acked_client_lsn,
                            excluded.last_acked_client_lsn),
                    last_acked_server_seq =
                        MAX(device_cursors.last_acked_server_seq,
                            excluded.last_acked_server_seq),
                    updated_at = excluded.updated_at
                """,
                (ack.device_id, ack.acked_head_lsn, ack.acked_server_seq, time.time()),
            )
            lsn, seq = self._cursors_unlocked(ack.device_id)
        logger.info(
            "ACK from %s: client_lsn=%d server_seq=%d", ack.device_id, lsn, seq
        )
        return {
            "status": "ok",
            "device_id": ack.device_id,
            "last_acked_client_lsn": lsn,
            "last_acked_server_seq": seq,
        }

    def stats(self) -> Dict[str, Any]:
        """Lightweight observability snapshot."""
        with self._lock:
            live = self._conn.execute(
                "SELECT COUNT(*) FROM vectors WHERE is_deleted = 0"
            ).fetchone()[0]
            tombstones = self._conn.execute(
                "SELECT COUNT(*) FROM vectors WHERE is_deleted = 1"
            ).fetchone()[0]
            feed = self._conn.execute(
                "SELECT COALESCE(MAX(server_seq), 0) FROM change_log"
            ).fetchone()[0]
            devices = [
                dict(row)
                for row in self._conn.execute(
                    "SELECT * FROM device_cursors ORDER BY device_id"
                ).fetchall()
            ]
        return {
            "live_vectors": live,
            "tombstones": tombstones,
            "server_seq": feed,
            "devices": devices,
        }


# ---------------------------------------------------------------------------
# Checksum verification
# ---------------------------------------------------------------------------


def verify_checksum(payload: SyncPayload) -> None:
    """Recompute the SHA-256 over the change list and compare to the claim.

    Must serialize exactly as the client does (compact separators, sorted
    keys) so the digest is byte-identical. Raises HTTP 400 on mismatch.
    """
    body = json.dumps(payload.changes, separators=(",", ":"), sort_keys=True)
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    if digest != payload.checksum_sha256:
        logger.warning(
            "Checksum mismatch from %s: claimed=%s computed=%s",
            payload.device_id,
            payload.checksum_sha256[:12],
            digest[:12],
        )
        raise HTTPException(
            status_code=400,
            detail={
                "error": "checksum_mismatch",
                "claimed": payload.checksum_sha256,
                "computed": digest,
            },
        )


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="VeloSync Server",
    description="Master store and conflict-resolution authority for VeloSync edge devices.",
    version="1.0.0",
)
store = MasterStore(DB_PATH)


@app.get("/health")
def health() -> Dict[str, Any]:
    """Liveness probe used by clients before attempting a sync round."""
    return {"status": "ok", "protocol_version": PROTOCOL_VERSION, "time": time.time()}


@app.get("/stats")
def stats() -> Dict[str, Any]:
    """Master-store observability: counts, feed head, device cursors."""
    return store.stats()


@app.post("/sync")
def sync(payload: SyncPayload) -> Dict[str, Any]:
    """Ingest a client push and return the pull set + LSN to acknowledge."""
    if payload.protocol_version != PROTOCOL_VERSION:
        raise HTTPException(
            status_code=409,
            detail=f"Unsupported protocol version {payload.protocol_version}; "
            f"server speaks {PROTOCOL_VERSION}",
        )
    if payload.change_count != len(payload.changes):
        raise HTTPException(
            status_code=422,
            detail=f"change_count={payload.change_count} but "
            f"{len(payload.changes)} changes present",
        )
    verify_checksum(payload)
    try:
        return store.process_sync(payload)
    except HTTPException:
        raise
    except (VeloSyncError, sqlite3.Error) as exc:
        logger.exception("Sync processing failed for %s", payload.device_id)
        raise HTTPException(status_code=500, detail=f"sync_failed: {exc}") from exc


@app.post("/ack")
def ack(payload: AckPayload) -> Dict[str, Any]:
    """Durably advance the device's replication cursors."""
    try:
        return store.process_ack(payload)
    except sqlite3.Error as exc:
        logger.exception("Ack processing failed for %s", payload.device_id)
        raise HTTPException(status_code=500, detail=f"ack_failed: {exc}") from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
