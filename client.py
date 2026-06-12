"""
VeloSync Client — zero-dependency edge device with HTTP synchronization.

Extends the local-only ``velosync`` engine with a network transport built
entirely on the standard library (``urllib.request`` — no ``requests``, no
``httpx``). One call to :meth:`HTTPSyncEngine.sync` performs a complete
replication round against a running VeloSync server:

    1. PUSH   — compile + compact pending WAL entries past the synced LSN,
                checksum them, POST to ``/sync``.
    2. PULL   — apply the server's pull set locally through Semantic Version
                Vector conflict resolution, with WAL echo suppression so
                replicated records are never pushed back.
    3. ACK    — POST ``/ack`` confirming the round was durably applied; only
                then advance the local frontier and truncate the WAL.

If any step fails (connection refused, timeout, HTTP error, malformed
response), the local frontier does NOT advance: the WAL is retained and the
next ``sync()`` retries the exact same window. Combined with the server's
idempotent replay detection, this yields at-least-once delivery with
exactly-once *effect*.

Standard library only. Requires ``velosync.py`` alongside this file.

Run the demo (server must be up first):
    python3 server.py            # terminal 1
    python3 client.py            # terminal 2
"""

from __future__ import annotations

import json
import logging
import os
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

from velosync import (
    ConflictResolution,
    SyncEngine,
    VectorRecord,
    VeloSyncError,
    VeloSyncStore,
)

logger = logging.getLogger("velosync.client")

DEFAULT_SERVER_URL = "http://127.0.0.1:8000"


# ---------------------------------------------------------------------------
# Transport-layer exceptions
# ---------------------------------------------------------------------------


class SyncTransportError(VeloSyncError):
    """Network-level failure: connection refused, DNS, timeout, bad gateway.

    Retriable — local state is untouched and the WAL window is preserved.
    """


class SyncRejectedError(VeloSyncError):
    """The server understood the request and rejected it (4xx).

    Not blindly retriable: indicates checksum mismatch, protocol skew, or a
    malformed payload that will fail again unchanged.
    """

    def __init__(self, status: int, detail: Any) -> None:
        super().__init__(f"Server rejected request (HTTP {status}): {detail}")
        self.status = status
        self.detail = detail


@dataclass(frozen=True)
class SyncReport:
    """Outcome of one full sync round, suitable for logs and telemetry."""

    pushed_changes: int
    raw_wal_entries: int
    pulled_records: int
    conflicts: Tuple[Tuple[str, str], ...]   # (vector_id, resolution_label)
    acked_head_lsn: int
    acked_server_seq: int
    wal_entries_truncated: int
    replay_detected: bool
    round_trip_seconds: float


# ---------------------------------------------------------------------------
# HTTP sync engine
# ---------------------------------------------------------------------------


class HTTPSyncEngine(SyncEngine):
    """SyncEngine subclass that replicates over HTTP via ``urllib``.

    Args:
        store: The local :class:`VeloSyncStore`.
        server_url: Base URL of the VeloSync server (no trailing path).
        timeout: Per-request socket timeout in seconds.
        max_retries: Transport-level retry attempts per request (exponential
            backoff). Rejections (4xx) are never retried.
    """

    def __init__(
        self,
        store: VeloSyncStore,
        server_url: str = DEFAULT_SERVER_URL,
        timeout: float = 10.0,
        max_retries: int = 3,
    ) -> None:
        super().__init__(store)
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if max_retries < 1:
            raise ValueError("max_retries must be >= 1")
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries

    # -- low-level HTTP ------------------------------------------------------

    def _request_json(
        self,
        path: str,
        body: Optional[Mapping[str, Any]] = None,
        method: str = "GET",
    ) -> Dict[str, Any]:
        """Issue one HTTP request and decode the JSON response.

        Retries transport failures with exponential backoff (0.5s, 1s, 2s…).
        4xx responses raise :class:`SyncRejectedError` immediately; 5xx are
        treated as transient and retried.

        Raises:
            SyncRejectedError: On any 4xx response.
            SyncTransportError: When all retries are exhausted or the
                response body is not valid JSON.
        """
        url = f"{self.server_url}{path}"
        data = (
            json.dumps(body, separators=(",", ":")).encode("utf-8")
            if body is not None
            else None
        )
        last_error: Optional[BaseException] = None

        for attempt in range(1, self.max_retries + 1):
            request = urllib.request.Request(
                url,
                data=data,
                method=method,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "User-Agent": f"velosync-client/{self.PROTOCOL_VERSION}",
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                    raw = resp.read().decode("utf-8")
                try:
                    return json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise SyncTransportError(
                        f"Server returned non-JSON body from {path}: {exc}"
                    ) from exc
            except urllib.error.HTTPError as exc:
                detail_raw = exc.read().decode("utf-8", errors="replace")
                try:
                    detail: Any = json.loads(detail_raw).get("detail", detail_raw)
                except (json.JSONDecodeError, AttributeError):
                    detail = detail_raw
                if 400 <= exc.code < 500:
                    raise SyncRejectedError(exc.code, detail) from exc
                last_error = SyncTransportError(
                    f"HTTP {exc.code} from {path}: {detail}"
                )
            except urllib.error.URLError as exc:
                last_error = SyncTransportError(
                    f"Cannot reach {url}: {exc.reason}"
                )
            except (socket.timeout, TimeoutError) as exc:
                last_error = SyncTransportError(
                    f"Request to {url} timed out after {self.timeout}s"
                )

            if attempt < self.max_retries:
                backoff = 0.5 * (2 ** (attempt - 1))
                logger.warning(
                    "Transport failure (attempt %d/%d): %s — retrying in %.1fs",
                    attempt, self.max_retries, last_error, backoff,
                )
                time.sleep(backoff)

        assert last_error is not None
        raise last_error

    # -- protocol steps --------------------------------------------------------

    def server_healthy(self) -> bool:
        """Probe ``/health``; never raises (returns False when unreachable)."""
        try:
            return self._request_json("/health").get("status") == "ok"
        except (SyncTransportError, SyncRejectedError):
            return False

    def sync(self) -> SyncReport:
        """Execute one full PUSH → PULL → ACK replication round.

        Returns:
            A :class:`SyncReport` describing what moved in each direction.

        Raises:
            SyncTransportError: Network failure — safe to retry later; the
                local WAL window is preserved.
            SyncRejectedError: Protocol-level rejection — inspect ``.detail``.
            VeloSyncError: Malformed pull records or local write failure.
        """
        started = time.monotonic()

        # ---- PUSH ----------------------------------------------------------
        payload = self.build_sync_payload()
        logger.info(
            "PUSH: %d compacted change(s) from %d WAL entries "
            "(window LSN %d → %d)",
            payload["change_count"],
            payload["raw_wal_entries"],
            payload["base_lsn"],
            payload["head_lsn"],
        )
        response = self._request_json("/sync", body=payload, method="POST")
        if response.get("status") != "ok":
            raise SyncRejectedError(200, f"unexpected sync response: {response}")

        # ---- PULL ----------------------------------------------------------
        pull_records: List[Mapping[str, Any]] = response.get("pull", [])
        outcomes = self.apply_remote_changes(pull_records)  # echo-suppressed
        if pull_records:
            logger.info("PULL: applied %d remote record(s)", len(pull_records))

        # ---- ACK -----------------------------------------------------------
        acked_lsn = int(response["acked_head_lsn"])
        acked_seq = int(response["server_seq"])
        self._request_json(
            "/ack",
            body={
                "device_id": self.store.device_id,
                "acked_head_lsn": acked_lsn,
                "acked_server_seq": acked_seq,
            },
            method="POST",
        )

        # Only after the server has durably recorded the ACK do we advance
        # the local frontier and garbage-collect the WAL.
        self.store.mark_synced(acked_lsn)
        truncated = self.store.truncate_wal()

        report = SyncReport(
            pushed_changes=payload["change_count"],
            raw_wal_entries=payload["raw_wal_entries"],
            pulled_records=len(pull_records),
            conflicts=tuple(
                (vid, label.value) for vid, label in outcomes
                if label != ConflictResolution.NO_LOCAL_COPY
            ),
            acked_head_lsn=acked_lsn,
            acked_server_seq=acked_seq,
            wal_entries_truncated=truncated,
            replay_detected=bool(response.get("replay_detected", False)),
            round_trip_seconds=time.monotonic() - started,
        )
        logger.info(
            "Sync complete: pushed=%d pulled=%d truncated=%d rtt=%.0fms",
            report.pushed_changes,
            report.pulled_records,
            report.wal_entries_truncated,
            report.round_trip_seconds * 1000,
        )
        return report


# ---------------------------------------------------------------------------
# Demo: two edge devices converging through the server
# ---------------------------------------------------------------------------

_DIM = 8


def _fresh_store(path: str, device_id: str) -> VeloSyncStore:
    """Open a clean on-disk store for a repeatable demo run."""
    for suffix in ("", "-wal", "-shm"):
        try:
            os.remove(path + suffix)
        except FileNotFoundError:
            pass
    return VeloSyncStore(db_path=path, dimension=_DIM, device_id=device_id)


def _banner(text: str) -> None:
    print(f"\n{'-' * 72}\n{text}\n{'-' * 72}")


def main() -> None:
    """Simulate two devices: offline edits, conflicting writes, convergence."""
    import random

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    rng = random.Random(7)

    # --- preflight: is the server up? ------------------------------------
    probe = HTTPSyncEngine(
        VeloSyncStore(":memory:", _DIM, "probe"), DEFAULT_SERVER_URL, timeout=3.0,
        max_retries=1,
    )
    if not probe.server_healthy():
        print(
            "ERROR: VeloSync server is not reachable at "
            f"{DEFAULT_SERVER_URL}.\n"
            "Start it first in another terminal:\n"
            "    python3 server.py\n"
            "  or\n"
            "    uvicorn server:app --host 127.0.0.1 --port 8000"
        )
        raise SystemExit(1)
    probe.store.close()

    print("=" * 72)
    print("VeloSync distributed demo — two edge devices, one master server")
    print("=" * 72)

    # Use randomized device IDs so that repeated runs start with clean cursors on the server
    run_suffix = str(int(time.time()))[-6:]
    phone = _fresh_store("client_phone.db", f"edge-phone-{run_suffix}")
    tablet = _fresh_store("client_tablet.db", f"edge-tablet-{run_suffix}")
    phone_sync = HTTPSyncEngine(phone)
    tablet_sync = HTTPSyncEngine(tablet)

    # [1] Phone indexes 5 vectors locally and pushes them.
    _banner("[1] Phone indexes 5 vectors offline, then syncs (initial push)")
    labels = [
        "coffee shop review", "battery health log", "voice memo: groceries",
        "photo caption: sunset", "meeting summary",
    ]
    phone_records = [
        phone.insert(
            embedding=[rng.uniform(-1.0, 1.0) for _ in range(_DIM)],
            metadata={"label": label, "origin": "phone"},
            semantic_weight=round(rng.uniform(0.5, 1.0), 3),
        )
        for label in labels
    ]
    report = phone_sync.sync()
    print(f"    phone: pushed={report.pushed_changes} pulled={report.pulled_records} "
          f"acked_lsn={report.acked_head_lsn}")

    # [2] Tablet starts empty and receives everything via pull fan-out.
    _banner("[2] Tablet (empty) syncs and receives the phone's records")
    report = tablet_sync.sync()
    print(f"    tablet: pushed={report.pushed_changes} pulled={report.pulled_records} "
          f"-> {tablet.count()} live vectors locally")
    probe_vec = list(phone_records[0].embedding)
    hit = tablet.search(probe_vec, top_k=1)[0]
    print(f"    tablet local k-NN sanity check: sim={hit.similarity:+.4f} "
          f"label={hit.metadata['label']!r}")

    # [3] Both devices go offline and edit the SAME vector concurrently.
    contested = phone_records[0].vector_id
    _banner("[3] Offline: both devices edit the same vector concurrently")
    phone.update(
        contested,
        metadata={"label": "coffee shop review", "stars": 5, "edited_by": "phone"},
        semantic_weight=0.95,   # phone is highly confident
    )
    tablet.update(
        contested,
        metadata={"label": "coffee shop review", "stars": 2, "edited_by": "tablet"},
        semantic_weight=0.60,   # tablet is less confident
    )
    # Plus non-conflicting offline work on each side.
    phone.delete(phone_records[2].vector_id)
    new_on_tablet = tablet.insert(
        embedding=[rng.uniform(-1.0, 1.0) for _ in range(_DIM)],
        metadata={"label": "tablet sketch note", "origin": "tablet"},
        semantic_weight=0.80,
    )
    print(f"    contested vector: {contested[:8]}…  "
          f"(phone weight 0.95 vs tablet weight 0.60)")
    print(f"    phone pending WAL entries:  {len(phone_sync.pending_changes())}")
    print(f"    tablet pending WAL entries: {len(tablet_sync.pending_changes())}")

    # [4] Tablet reconnects first: its edit lands on the master unopposed.
    _banner("[4] Tablet reconnects and syncs first")
    report = tablet_sync.sync()
    print(f"    tablet: pushed={report.pushed_changes} pulled={report.pulled_records} "
          f"conflicts={list(report.conflicts)}")

    # [5] Phone reconnects: the server detects concurrency on the contested
    #     vector and resolves by semantic weight (phone's 0.95 beats 0.60).
    _banner("[5] Phone reconnects — server resolves the concurrent edit")
    report = phone_sync.sync()
    print(f"    phone: pushed={report.pushed_changes} pulled={report.pulled_records}")
    for vid, label in report.conflicts:
        print(f"      resolution {vid[:8]}… -> {label}")
    winner_on_phone = phone.get(contested)
    assert winner_on_phone is not None
    print(f"    phone's copy now: stars={winner_on_phone.metadata.get('stars')} "
          f"edited_by={winner_on_phone.metadata.get('edited_by')!r} "
          f"weight={winner_on_phone.semantic_weight:.2f} "
          f"clock={winner_on_phone.version_vector}")

    # [6] Tablet syncs again and converges on the phone's winning edit, and
    #     the phone has already received the tablet's offline insert.
    _banner("[6] Tablet syncs again — full convergence")
    report = tablet_sync.sync()
    print(f"    tablet: pushed={report.pushed_changes} pulled={report.pulled_records}")
    a = phone.get(contested)
    b = tablet.get(contested)
    assert a is not None and b is not None
    converged = (
        a.metadata == b.metadata
        and a.semantic_weight == b.semantic_weight
        and a.version_vector == b.version_vector
    )
    print(f"    contested vector identical on both devices: {converged}")
    print(f"      metadata = {a.metadata}")
    print(f"      clock    = {a.version_vector}")
    sketch_on_phone = phone.get(new_on_tablet.vector_id)
    print(f"    tablet's offline insert visible on phone: "
          f"{sketch_on_phone is not None and sketch_on_phone.metadata['label']!r}")
    print(f"    live counts — phone: {phone.count()}, tablet: {tablet.count()} "
          f"(tombstoned delete replicated)")

    # [7] Idle sync: nothing pending on either side.
    _banner("[7] Steady state — idle sync rounds move nothing")
    r1, r2 = phone_sync.sync(), tablet_sync.sync()
    print(f"    phone:  pushed={r1.pushed_changes} pulled={r1.pulled_records}")
    print(f"    tablet: pushed={r2.pushed_changes} pulled={r2.pulled_records}")

    phone.close()
    tablet.close()
    print("\nDistributed demo complete — both replicas converged.")


if __name__ == "__main__":
    main()
