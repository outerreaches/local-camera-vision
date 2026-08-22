from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from camera_vision.models import DetectionEvent, IdentityResult, NotificationState


class EventStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    camera_name TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    snapshot_path TEXT NOT NULL,
                    score REAL,
                    labels TEXT NOT NULL,
                    boxes TEXT NOT NULL,
                    identity TEXT NOT NULL,
                    notification TEXT NOT NULL,
                    raw TEXT NOT NULL
                )
                """
            )

    def add_event(self, event: DetectionEvent) -> DetectionEvent:
        data = event.model_dump(mode="json")
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO events (created_at, camera_name, prompt, snapshot_path, score, labels, boxes, identity, notification, raw)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    data["created_at"],
                    data["camera_name"],
                    data["prompt"],
                    data["snapshot_path"],
                    data["score"],
                    json.dumps(data["labels"]),
                    json.dumps(data["boxes"]),
                    json.dumps(data["identity"]),
                    json.dumps(data["notification"]),
                    json.dumps(data["raw"]),
                ),
            )
            event_id = int(cursor.lastrowid)
        return event.model_copy(update={"id": event_id})

    def list_events(self, limit: int = 50) -> list[DetectionEvent]:
        limit = max(1, min(limit, 500))
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [self._row_to_event(row) for row in rows]

    def get_event(self, event_id: int) -> DetectionEvent | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_event(row)

    def latest_event(self) -> DetectionEvent | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM events ORDER BY id DESC LIMIT 1").fetchone()
        if row is None:
            return None
        return self._row_to_event(row)

    def _row_to_event(self, row: sqlite3.Row) -> DetectionEvent:
        payload: dict[str, Any] = {
            "id": row["id"],
            "created_at": row["created_at"],
            "camera_name": row["camera_name"],
            "prompt": row["prompt"],
            "snapshot_path": row["snapshot_path"],
            "score": row["score"],
            "labels": json.loads(row["labels"]),
            "boxes": json.loads(row["boxes"]),
            "identity": IdentityResult.model_validate(json.loads(row["identity"])),
            "notification": NotificationState.model_validate(json.loads(row["notification"])),
            "raw": json.loads(row["raw"]),
        }
        return DetectionEvent.model_validate(payload)
