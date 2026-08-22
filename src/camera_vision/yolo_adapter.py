from __future__ import annotations

import atexit
import json
import logging
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class YoloDetection(BaseModel):
    bbox: list[float]  # [x1, y1, x2, y2]
    confidence: float
    class_name: str = "person"


class YoloResult(BaseModel):
    person_detected: bool
    detections: list[YoloDetection] = Field(default_factory=list)
    elapsed_ms: float = 0.0
    error: str | None = None


class YoloAdapter:
    """YOLO person detector via a persistent subprocess.

    Spawns scripts/yolo_detect_server.py as a long-lived subprocess using
    a configured Python environment.
    The worker loads YOLOv11n once and accepts JSON-line requests on stdin.

    Fallback: if the worker can't start, the
    adapter degrades gracefully — detect() returns person_detected=False
    instead of crashing.
    """

    _DETECT_SERVER = "yolo_detect_server.py"
    _STARTUP_TIMEOUT = 60   # model load + warm-up
    _INFER_TIMEOUT = 30     # per-frame inference
    _IDLE_RESTART = 300     # restart worker if idle this long

    def __init__(
        self,
        backend: Literal["mock", "real", "auto"] = "mock",
        model_path: str | Path = "yolo11n.pt",
        confidence: float = 0.5,
        python_path: str | Path = ".venv/bin/python3",
    ):
        self.backend = backend
        self.model_path = Path(model_path)
        configured_python = Path(python_path)
        if python_path == ".venv/bin/python3" and os.name == "nt":
            configured_python = Path(".venv/Scripts/python.exe")
        self._venv_python = configured_python
        self._scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
        self._detect_server = self._scripts_dir / self._DETECT_SERVER
        self.confidence = confidence

        # Worker state
        self._worker: subprocess.Popen | None = None
        self._worker_lock = threading.Lock()
        self._last_used = 0.0
        self._startup_error: str | None = None
        atexit.register(self._stop_worker)

    # ── public interface ──────────────────────────────────────────────

    @property
    def available(self) -> bool:
        return self._venv_python.exists() and self._detect_server.exists()

    @property
    def worker_alive(self) -> bool:
        with self._worker_lock:
            return self._worker is not None and self._worker.poll() is None

    def detect(self, image_path: str | Path) -> YoloResult:
        """Run person detection on a single frame. Blocks until result."""
        if self.backend == "mock":
            return YoloResult(person_detected=False)

        if not self.available:
            return YoloResult(
                person_detected=False,
                error="YOLO worker not available (virtual environment or script missing)",
            )

        worker = self._ensure_worker()
        if worker is None:
            return YoloResult(
                person_detected=False,
                error=self._startup_error or "YOLO worker failed to start",
            )

        self._last_used = time.time()

        request = json.dumps({
            "image_path": str(image_path),
            "confidence": self.confidence,
        }) + "\n"

        try:
            with self._worker_lock:
                if self._worker is None or self._worker.poll() is not None:
                    return YoloResult(person_detected=False, error="worker died")
                self._worker.stdin.write(request)
                self._worker.stdin.flush()
                raw = self._worker.stdout.readline()

            if not raw:
                return YoloResult(person_detected=False, error="worker returned empty response")

            data = json.loads(raw)
            if data.get("error"):
                return YoloResult(person_detected=False, error=data["error"])

            detections = [
                YoloDetection(
                    bbox=d["bbox"],
                    confidence=d["confidence"],
                    class_name=d.get("class", "person"),
                )
                for d in data.get("detections", [])
            ]
            return YoloResult(
                person_detected=data.get("person_detected", False),
                detections=detections,
                elapsed_ms=data.get("elapsed_ms", 0.0),
            )
        except Exception as exc:
            logger.error("YOLO detect failed: %s", exc)
            return YoloResult(person_detected=False, error=str(exc))

    # ── worker lifecycle ──────────────────────────────────────────────

    def _ensure_worker(self) -> subprocess.Popen | None:
        """Return running worker, starting one if needed."""
        with self._worker_lock:
            if self._worker is not None and self._worker.poll() is None:
                return self._worker

            # Clear stale error
            self._startup_error = None

            try:
                self._worker = subprocess.Popen(
                    [str(self._venv_python), str(self._detect_server), "--model", str(self.model_path)],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=str(self._scripts_dir.parent),
                )
            except Exception as exc:
                self._startup_error = f"Failed to spawn YOLO worker: {exc}"
                logger.error(self._startup_error)
                return None

        # Wait for readiness signal
        start = time.time()
        while time.time() - start < self._STARTUP_TIMEOUT:
            if self._worker.poll() is not None:
                stderr = self._worker.stderr.read() if self._worker.stderr else ""
                self._startup_error = f"Worker exited early (code {self._worker.returncode}): {stderr[:300]}"
                logger.error(self._startup_error)
                self._worker = None
                return None

            try:
                raw = self._worker.stdout.readline()
                if not raw:
                    continue
                data = json.loads(raw.strip())
                if data.get("status") == "ready":
                    logger.info(
                        "YOLO worker ready: %s (load %s ms, device=%s)",
                        data.get("model"), data.get("load_ms"), data.get("device"),
                    )
                    return self._worker
                elif data.get("status") == "loading":
                    logger.debug("YOLO worker loading...")
            except (json.JSONDecodeError, Exception):
                continue

        self._startup_error = f"Worker startup timed out after {self._STARTUP_TIMEOUT}s"
        logger.error(self._startup_error)
        self._stop_worker()
        return None

    def _stop_worker(self) -> None:
        with self._worker_lock:
            if self._worker is not None:
                try:
                    self._worker.stdin.close()
                except Exception:
                    pass
                try:
                    self._worker.terminate()
                    self._worker.wait(timeout=5)
                except Exception:
                    try:
                        self._worker.kill()
                    except Exception:
                        pass
                self._worker = None
