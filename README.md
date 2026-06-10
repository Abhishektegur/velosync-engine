# VeloSync ⚡

VeloSync is a **zero-dependency, offline-first vector database synchronization engine** written in Python. It is designed for edge environments (mobile phones, IoT nodes, embedded devices) that need to perform fast local vector searches on SQLite and synchronize mutations bidirectionally with a central cloud database over HTTP.

---

## System Architecture

```
         +----------------------------------+          +----------------------------------+
         |     Device A (edge-phone-01)     |          |    Device B (edge-tablet-07)     |
         +----------------------------------+          +----------------------------------+
          | SQLite DB |   | Logical WAL log |           | SQLite DB |   | Logical WAL log |
          +-----------+   +-----------------+           +-----------+   +-----------------+
                      \       /                                     \       /
                    (HTTPSyncEngine)                                (HTTPSyncEngine)
                        \   /                                           \   /
                         v v                                             v v
                   POST /sync (Replicate)                          POST /sync (Replicate)
                   POST /ack  (Commit)                             POST /ack  (Commit)
                         \   /                                           \   /
                          \ /                                             / /
                           v                                             v
                 +-------------------------------------------------------------+
                 |                     FastAPI HTTP Server                     |
                 +-------------------------------------------------------------+
                 |                Master Store (server_store.db)               |
                 +-------------------------------------------------------------+
```

### Key Components

1. **Local Edge Storage (`client.py`)**: 
   * A SQLite file database on the device (`client_phone.db`, `client_tablet.db`).
   * Embeddings are packed as raw binary `float64` BLOBs using the `struct` module (low memory, high speed).
   * A local Write-Ahead Log (WAL) table logs every insert, update, or delete with an incrementing LSN.
   * Nearest neighbor search uses pure-Python cosine similarity and a bounded min-heap (`heapq`) to maintain $O(k)$ memory.
2. **FastAPI Sync Server (`server.py`)**:
   * A lightweight HTTP web service hosting a master database (`server_store.db`).
   * Implements transaction-safe concurrency control using Python thread locks.
   * Reconciles edits, computes change logs, and maintains device cursors.
3. **HTTP Replication Protocol**:
   * **`POST /sync`**: Ingests the client's compacted WAL payload. The server validates a SHA-256 payload checksum, runs conflict resolution, and returns a "pull" set containing changes from other devices and resolved conflict states.
   * **`POST /ack`**: Once the client successfully applies the pull set locally, it sends an ACK. Only then does the server advance the replication cursors, making the sync process **crash-safe**.

---

## 2026 Core Architectural Design Details

### 1. Vector Clock Causality + Semantic Resolution
Standard systems use wall-clock timestamps for conflict resolution, which breaks due to clock drift. VeloSync uses **Version Vectors** to track history. 
If two devices edit the same vector concurrently (in parallel):
1. The server compares version vectors to detect the conflict.
2. The edit with the higher **Semantic Weight** (confidence score) wins.
3. The server **merges their clocks** (taking the component-wise max). The client that won gets the merged clock sent back so the resolved state causally dominates both devices' futures, preventing conflict loops.

### 2. Log Compaction & Echo Suppression
*   **Log Compaction**: If an offline client updates a vector multiple times, the client compacts the log, sending only the final state to the server to minimize bandwidth.
*   **Echo Suppression**: The server tracks a global `server_seq` sequence feed. When a client pulls changes, the server filters out entries originally authored by that device (`device_id`) so the client never downloads its own uploads.

---

## Setup & Running the Simulation

### 1. Install Dependencies
Install FastAPI and Uvicorn on the server host:
```bash
pip install fastapi uvicorn
```

### 2. Start the Server
Start the FastAPI server (it will initialize `server_store.db` and listen on port 8000):
```bash
python server.py
```

### 3. Run the Distributed Sync Simulation
In another terminal, run the client simulation (it will initialize two local databases, simulate offline edits, and synchronize them over HTTP):
```bash
python client.py
```

---

## Verification & Output Log

Running the simulation output:
1. **Initial Push**: Phone creates and indexes 5 vectors offline and synchronizes them to the Server.
2. **Pull Fan-out**: Tablet starts empty, syncs, and downloads the 5 vectors from the Server.
3. **Offline Conflicts**:
   - Both devices go offline.
   - Both devices modify the same contested vector. Phone sets `stars = 5` with confidence `0.95`. Tablet sets `stars = 2` with confidence `0.60`.
   - Phone deletes a vector. Tablet inserts a new vector.
4. **Replication & Convergence**:
   - Tablet reconnects first and pushes its edit.
   - Phone reconnects. The server detects the concurrent conflict on the contested vector.
   - Because Phone's weight (`0.95`) beats Tablet's weight (`0.60`), Phone's edit wins.
   - Tablet syncs again and downloads the winning Phone edit, achieving **full convergence**. Both devices have identical tables and clocks, and the deleted vector tombstone has replicated to both.
