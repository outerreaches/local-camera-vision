from __future__ import annotations

import asyncio
import logging
import os
import secrets
from datetime import datetime, timedelta, timezone
from importlib import resources
from pathlib import Path
from uuid import uuid4

import httpx
import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from camera_vision.capture import capture_frame
from camera_vision.events import EventStore
from camera_vision.models import DetectionEvent, NotificationState, VisionSettings
from camera_vision.nvr_poller import CameraDashboardPoller

from camera_vision.settings import DEFAULT_SETTINGS_PATH, load_settings, save_settings
from camera_vision.vlm_client import VlmClient
from camera_vision.yolo_adapter import YoloAdapter

# Load local env overrides (RTSP creds, tokens) — must happen before any config reads
try:
    from dotenv import load_dotenv
    _env_file = Path(__file__).resolve().parents[2] / ".env.local"
    if _env_file.exists():
        load_dotenv(_env_file)
except ImportError:
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_WEB_DIR = Path(__file__).resolve().parent / "web"
DEV_WEB_DIR = PROJECT_ROOT / "web"
WEB_DIR = SOURCE_WEB_DIR if SOURCE_WEB_DIR.exists() else DEV_WEB_DIR


class TestPromptRequest(BaseModel):
    prompt: str | None = None
    snapshot_path: str | None = None


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").lower() in {"1", "true", "yes", "on"}


def _web_dir() -> Path | None:
    if SOURCE_WEB_DIR.exists():
        return SOURCE_WEB_DIR
    if DEV_WEB_DIR.exists():
        return DEV_WEB_DIR
    try:
        candidate = resources.files("camera_vision").joinpath("web")
        if candidate.is_dir():
            return Path(str(candidate))
    except Exception:
        return None
    return None


def _merge_editable_settings(current: VisionSettings, requested: VisionSettings) -> VisionSettings:
    """Merge HTTP-editable settings while preserving runtime filesystem paths."""

    merged = requested.model_copy(deep=True)
    merged.service.database_path = current.service.database_path
    merged.service.snapshot_dir = current.service.snapshot_dir
    merged.service.data_dir = current.service.data_dir
    merged.service.tmp_dir = current.service.tmp_dir

    merged.face_recognition.known_faces_dir = current.face_recognition.known_faces_dir
    return merged


def _validated_snapshot_path(snapshot_path: str, snapshot_dir: Path) -> Path:
    base = snapshot_dir.resolve()
    target = Path(snapshot_path).expanduser().resolve()
    if target != base and base not in target.parents:
        raise HTTPException(status_code=400, detail="snapshot_path must be under configured snapshot_dir")
    if not target.is_file():
        raise HTTPException(status_code=400, detail="snapshot_path does not exist")
    return target


def create_app(
    settings_path: str | Path = DEFAULT_SETTINGS_PATH,
    db_path: str | Path | None = None,
    snapshot_dir: str | Path | None = None,
    *,
    api_token: str | None = None,
    allow_unsafe_no_auth: bool | None = None,
    initialize_runtime: bool = True,
) -> FastAPI:
    app = FastAPI(title="Local Camera Vision", version="0.1.0")
    app.state.settings_path = Path(settings_path)
    settings = load_settings(settings_path)
    app.state.settings = settings
    app.state.db_path_override = Path(db_path) if db_path is not None else None
    app.state.snapshot_dir_override = Path(snapshot_dir) if snapshot_dir is not None else None
    app.state.db_path = app.state.db_path_override or Path(settings.service.database_path)
    app.state.snapshot_dir = app.state.snapshot_dir_override or Path(settings.service.snapshot_dir)
    app.state.api_token = api_token if api_token is not None else os.environ.get("CAMERA_VISION_API_TOKEN")
    app.state.allow_unsafe_no_auth = _env_flag("CAMERA_VISION_ALLOW_UNSAFE_NO_AUTH") if allow_unsafe_no_auth is None else allow_unsafe_no_auth
    app.state.capture_frame = capture_frame

    app.state.runtime_initialized = False

    _pipeline_log = logging.getLogger("camera_vision.pipeline")

    app.state.nvr_poller = CameraDashboardPoller(
        dashboard_url=settings.camera_dashboard.url,
        channel=settings.camera_dashboard.channel,
        trigger_types=settings.camera_dashboard.trigger_types,
    )
    app.state.vlm_client = VlmClient(
        server_url=settings.vlm.server_url,
        model=settings.vlm.model,
        max_tokens=settings.vlm.max_tokens,
        timeout_seconds=settings.vlm.timeout_seconds,
        classification_prompt=settings.vlm.classification_prompt,
    )
    app.state.yolo_adapter = YoloAdapter(
        backend=settings.yolo.backend,
        model_path=settings.yolo.model_path,
        python_path=settings.yolo.python_path,
        confidence=settings.yolo.confidence,
    )
    app.state._pipeline_task = None
    app.state._yolo_task = None
    app.state._retention_task = None
    app.state._last_trigger_ts = 0.0
    app.state._pipeline_enabled = settings.pipeline.enabled
    app.state._pipeline_cooldown = settings.pipeline.cooldown_seconds

    # Server-side tracking of last LLM request times (watches id_task deltas)
    app.state._server_tracker = {
        "vlm": {"last_id_task": 0, "last_request_ts": None},
    }
    app.state._http_client = httpx.Client(timeout=5.0)

    def ensure_runtime() -> None:
        if app.state.runtime_initialized:
            return
        app.state.snapshot_dir.mkdir(parents=True, exist_ok=True)
        app.state.event_store = EventStore(app.state.db_path)
        app.state.runtime_initialized = True

    def recent_vlm_context() -> tuple[str, int] | None:
        """Return the previous event's context_description and age when recent enough."""
        threshold_seconds = getattr(app.state.settings.pipeline, "recent_context_seconds", 60)
        if threshold_seconds <= 0:
            return None
        ensure_runtime()
        event = app.state.event_store.latest_event()
        if event is None:
            return None
        created_at = event.created_at
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        age = datetime.now(timezone.utc) - created_at
        if age > timedelta(seconds=threshold_seconds):
            return None
        vlm = event.raw.get("vlm") if isinstance(event.raw, dict) else None
        if not isinstance(vlm, dict) or vlm.get("error"):
            return None
        context = vlm.get("context_description")
        if not isinstance(context, str) or not context.strip():
            return None
        return context.strip(), max(0, round(age.total_seconds()))

    if initialize_runtime:
        ensure_runtime()

    async def _pipeline_loop() -> None:
        interval = settings.pipeline.cooldown_seconds if settings.pipeline.cooldown_seconds > 0 else 60
        cooldown = settings.pipeline.cooldown_seconds
        _pipeline_log.info("Pipeline started (interval=%ds, cooldown=%ds)", interval, cooldown)

        while True:
            try:
                trigger = await app.state.nvr_poller.poll_once()
                if trigger is not None:
                    now = asyncio.get_running_loop().time()
                    if cooldown > 0 and app.state._last_trigger_ts > 0 and (now - app.state._last_trigger_ts) < cooldown:
                        _pipeline_log.debug("Trigger suppressed by cooldown")
                    else:
                        app.state._last_trigger_ts = now
                        _pipeline_log.info("Trigger: %s on channel %d", trigger.detection_type, trigger.channel)
                        try:
                            ensure_runtime()
                            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                            uid = uuid4().hex[:8]
                            snapshot = app.state.snapshot_dir / f"pipeline-{ts}-{uid}.jpg"
                            app.state.capture_frame(
                                snapshot,
                                channel=settings.camera.channel,
                                timeout_seconds=settings.camera.capture_timeout_seconds,
                                camera_config=settings.camera,
                            )
                            _pipeline_log.info("Captured frame: %s", snapshot)

                            # Collect frame burst: initial + 2 follow-up frames
                            burst_paths: list[Path] = [snapshot]
                            for i in range(2):
                                await asyncio.sleep(0.5)
                                ts_burst = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                                burst_snap = app.state.snapshot_dir / f"burst-{ts_burst}-{uuid4().hex[:8]}.jpg"
                                app.state.capture_frame(
                                    burst_snap,
                                    channel=settings.camera.channel,
                                    timeout_seconds=settings.camera.capture_timeout_seconds,
                                    camera_config=settings.camera,
                                )
                                burst_paths.append(burst_snap)
                            _pipeline_log.info("Frame burst captured: %d frames", len(burst_paths))

                            prior_context = recent_vlm_context()
                            previous_context = prior_context[0] if prior_context else None
                            previous_context_age_seconds = prior_context[1] if prior_context else None
                            result = await app.state.vlm_client.classify(
                                burst_paths,
                                previous_context=previous_context,
                                previous_context_age_seconds=previous_context_age_seconds,
                            )
                            _pipeline_log.info("VLM: person=%s activity=%s suspicious=%s",
                                              result.person_present, result.activity, result.suspicious)
                            event = DetectionEvent(
                                camera_name=settings.camera.name,
                                prompt=settings.vlm.classification_prompt,
                                snapshot_path=str(snapshot),
                                score=0.0,
                                labels=[result.activity or "unknown"],
                                boxes=[],
                                notification=NotificationState(
                                    status="pending" if result.suspicious else "skipped",
                                    reason="pipeline auto-classification",
                                ),
                                raw={
                                    "vlm": result.to_dict(),
                                    "previous_context": previous_context,
                                    "previous_context_age_seconds": previous_context_age_seconds,
                                    "trigger": {"type": trigger.detection_type, "channel": trigger.channel},
                                    "frame_burst": [str(p) for p in burst_paths],
                                },
                            )
                            saved = app.state.event_store.add_event(event)
                            _pipeline_log.info("Event saved: id=%s", saved.id)
                        except Exception:
                            _pipeline_log.exception("Pipeline processing failed for trigger")
            except Exception:
                _pipeline_log.exception("Pipeline loop iteration failed")
            await asyncio.sleep(interval)

    async def _yolo_poll_loop() -> None:
        """Poll RTSP frames, run YOLO person detection, trigger VLM pipeline on detection."""
        yolo_enabled = settings.yolo.enabled
        if not yolo_enabled:
            _pipeline_log.info("YOLO poller disabled")
            return

        interval = max(1, settings.yolo.poll_interval_seconds)
        cooldown = settings.pipeline.cooldown_seconds
        _pipeline_log.info("YOLO poller started (interval=%ds, confidence=%.2f)", interval, settings.yolo.confidence)

        while True:
            try:
                ensure_runtime()
                ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                uid = uuid4().hex[:8]
                snapshot = app.state.snapshot_dir / f"yolo-poll-{ts}-{uid}.jpg"
                app.state.capture_frame(
                    snapshot,
                    channel=settings.camera.channel,
                    timeout_seconds=settings.camera.capture_timeout_seconds,
                    camera_config=settings.camera,
                )

                yolo_result = app.state.yolo_adapter.detect(snapshot)
                _pipeline_log.debug(
                    "YOLO: person=%s detections=%d elapsed=%.1fms",
                    yolo_result.person_detected,
                    len(yolo_result.detections),
                    yolo_result.elapsed_ms,
                )

                if yolo_result.person_detected:
                    now = asyncio.get_running_loop().time()
                    if cooldown > 0 and app.state._last_trigger_ts > 0 and (now - app.state._last_trigger_ts) < cooldown:
                        _pipeline_log.debug("YOLO trigger suppressed by cooldown")
                        # Person detected but suppressed — clean up snapshot
                        try:
                            snapshot.unlink(missing_ok=True)
                        except Exception:
                            pass
                    else:
                        app.state._last_trigger_ts = now
                        _pipeline_log.info("YOLO detected person! Capturing frame burst + VLM classification...")
                        try:
                            # Collect frame burst: YOLO snapshot + 2 follow-up frames
                            burst_paths: list[Path] = [snapshot]
                            for i in range(2):
                                await asyncio.sleep(0.5)
                                ts_burst = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                                burst_snap = app.state.snapshot_dir / f"yolo-burst-{ts_burst}-{uuid4().hex[:8]}.jpg"
                                app.state.capture_frame(
                                    burst_snap,
                                    channel=settings.camera.channel,
                                    timeout_seconds=settings.camera.capture_timeout_seconds,
                                    camera_config=settings.camera,
                                )
                                burst_paths.append(burst_snap)
                            _pipeline_log.info("Frame burst captured: %d frames", len(burst_paths))

                            prior_context = recent_vlm_context()
                            previous_context = prior_context[0] if prior_context else None
                            previous_context_age_seconds = prior_context[1] if prior_context else None
                            result = await app.state.vlm_client.classify(
                                burst_paths,
                                previous_context=previous_context,
                                previous_context_age_seconds=previous_context_age_seconds,
                            )
                            _pipeline_log.info(
                                "YOLO→VLM: person=%s activity=%s suspicious=%s",
                                result.person_present, result.activity, result.suspicious,
                            )
                            event = DetectionEvent(
                                camera_name=settings.camera.name,
                                prompt=settings.vlm.classification_prompt,
                                snapshot_path=str(snapshot),
                                score=max((d.confidence for d in yolo_result.detections), default=0.0),
                                labels=[result.activity or "unknown"],
                                boxes=[
                                    {"box": d.bbox, "label": "person", "score": d.confidence}
                                    for d in yolo_result.detections
                                ],
                                notification=NotificationState(
                                    status="pending" if result.suspicious else "skipped",
                                    reason="yolo person detection → vlm classification",
                                ),
                                raw={
                                    "vlm": result.to_dict(),
                                    "previous_context": previous_context,
                                    "previous_context_age_seconds": previous_context_age_seconds,
                                    "yolo": yolo_result.model_dump(mode="json"),
                                    "trigger": {"type": "yolo", "source": "person_detector"},
                                    "frame_burst": [str(p) for p in burst_paths],
                                },
                            )
                            saved = app.state.event_store.add_event(event)
                            _pipeline_log.info("YOLO event saved: id=%s", saved.id)
                        except Exception:
                            _pipeline_log.exception("YOLO→VLM pipeline failed")
                else:
                    # No person detected — clean up snapshot immediately
                    try:
                        snapshot.unlink(missing_ok=True)
                    except Exception:
                        pass
            except Exception:
                _pipeline_log.exception("YOLO poll iteration failed")
            await asyncio.sleep(interval)

    async def _retention_cleanup_loop() -> None:
        """Periodically delete snapshots older than retention_hours."""
        while True:
            try:
                retention_hours = app.state.settings.pipeline.retention_hours
                if retention_hours > 0:
                    import time as _time
                    cutoff = _time.time() - (retention_hours * 3600)
                    snapshot_dir = app.state.snapshot_dir
                    deleted = 0
                    freed_bytes = 0
                    for entry in os.scandir(snapshot_dir):
                        if entry.is_file():
                            try:
                                st = entry.stat()
                                if st.st_mtime < cutoff:
                                    freed_bytes += st.st_size
                                    os.remove(entry.path)
                                    deleted += 1
                            except OSError:
                                pass
                    if deleted:
                        _pipeline_log.info(
                            "Retention cleanup: deleted %d old snapshots (freed %.1f MB)",
                            deleted, freed_bytes / (1024 * 1024),
                        )
            except Exception:
                _pipeline_log.exception("Retention cleanup failed")
            await asyncio.sleep(3600)  # Run every hour

    async def require_mutation_auth(
        request: Request,
        authorization: str | None = Header(default=None),
        x_api_token: str | None = Header(default=None, alias="X-API-Token"),
    ) -> None:
        token = request.app.state.api_token
        if request.app.state.allow_unsafe_no_auth and not token:
            return
        if not token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="CAMERA_VISION_API_TOKEN is required for mutating endpoints",
                headers={"WWW-Authenticate": "Bearer"},
            )
        supplied = x_api_token
        if authorization and authorization.lower().startswith("bearer "):
            supplied = authorization.split(" ", 1)[1]
        if not supplied or not secrets.compare_digest(supplied, token):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="invalid or missing API token",
                headers={"WWW-Authenticate": "Bearer"},
            )

    web_dir = _web_dir()
    if web_dir is not None:
        app.mount("/web", StaticFiles(directory=web_dir), name="web")

    @app.get("/")
    def root() -> Response:
        if web_dir is None:
            return HTMLResponse("<h1>Camera Vision</h1><p>Dashboard assets are not installed.</p>")
        return FileResponse(web_dir / "index.html")

    @app.get("/healthz")
    def healthz() -> dict[str, object]:
        return {
            "status": "ok",
            "service": "camera-vision",

            "pipeline": {
                "enabled": getattr(app.state, '_pipeline_enabled', False),
                "cooldown_seconds": getattr(app.state, '_pipeline_cooldown', 0),
                "recent_context_seconds": app.state.settings.pipeline.recent_context_seconds,
                "last_trigger_ts": getattr(app.state, '_last_trigger_ts', 0.0),
            },
            "yolo": {
                "enabled": app.state.settings.yolo.enabled,
                "backend": app.state.settings.yolo.backend,
                "available": app.state.yolo_adapter.available,
                "worker_alive": app.state.yolo_adapter.worker_alive,
            },
        }

    @app.get("/api/servers/status")
    def servers_status() -> dict[str, object]:
        """Report the configured VLM server status."""
        tracker = app.state._server_tracker
        client = app.state._http_client
        result: dict[str, object] = {}

        servers = {
            "vlm": str(app.state.settings.vlm.server_url).rstrip("/"),
        }

        for name, base_url in servers.items():
            entry: dict[str, object] = {"status": "error", "id_task": None, "last_request_ts": None}
            try:
                resp = client.get(f"{base_url}/slots")
                if resp.status_code == 200:
                    slots = resp.json()
                    # Find max id_task across all slots (multi-slot servers)
                    id_task = 0
                    for s in (slots or []):
                        t = s.get("id_task")
                        if t is not None and t > id_task:
                            id_task = t
                    trk = tracker[name]
                    if id_task != trk["last_id_task"]:
                        trk["last_id_task"] = id_task
                        trk["last_request_ts"] = datetime.now(timezone.utc)
                    entry["status"] = "ok"
                    entry["id_task"] = id_task
                    ts = trk["last_request_ts"]
                    entry["last_request_ts"] = ts.isoformat() if ts else None
                else:
                    entry["error"] = f"HTTP {resp.status_code}"
            except Exception as exc:
                entry["error"] = str(exc)

            result[name] = entry

        return result

    @app.get("/api/settings", response_model=VisionSettings)
    def get_settings() -> VisionSettings:
        return app.state.settings

    @app.put("/api/settings", response_model=VisionSettings, dependencies=[Depends(require_mutation_auth)])
    def put_settings(settings: VisionSettings) -> VisionSettings:
        merged = _merge_editable_settings(app.state.settings, settings)
        app.state.settings = merged
        app.state.db_path = app.state.db_path_override or Path(merged.service.database_path)
        app.state.snapshot_dir = app.state.snapshot_dir_override or Path(merged.service.snapshot_dir)
        app.state._pipeline_enabled = merged.pipeline.enabled
        app.state._pipeline_cooldown = merged.pipeline.cooldown_seconds
        app.state.settings.camera_dashboard = merged.camera_dashboard
        app.state.settings.yolo = merged.yolo
        app.state.vlm_client = VlmClient(
            server_url=merged.vlm.server_url,
            model=merged.vlm.model,
            max_tokens=merged.vlm.max_tokens,
            timeout_seconds=merged.vlm.timeout_seconds,
            classification_prompt=merged.vlm.classification_prompt,
        )

        app.state.runtime_initialized = False
        ensure_runtime()
        save_settings(merged, app.state.settings_path)
        return merged

    @app.get("/api/events", response_model=list[DetectionEvent])
    def list_events(limit: int = 50) -> list[DetectionEvent]:
        ensure_runtime()
        return app.state.event_store.list_events(limit=limit)

    @app.get("/api/events/{event_id}", response_model=DetectionEvent)
    def get_event(event_id: int) -> DetectionEvent:
        ensure_runtime()
        event = app.state.event_store.get_event(event_id)
        if event is None:
            raise HTTPException(status_code=404, detail="event not found")
        return event

    @app.get("/api/snapshots/{path:path}")
    def get_snapshot(path: str) -> FileResponse:
        base = app.state.snapshot_dir.resolve()
        target = (base / path).resolve()
        if not target.is_file() or (base not in target.parents and target != base):
            raise HTTPException(status_code=404, detail="snapshot not found")
        return FileResponse(target)

    @app.post("/api/capture-now", dependencies=[Depends(require_mutation_auth)])
    def capture_now() -> dict[str, str]:
        ensure_runtime()
        settings = app.state.settings
        output = app.state.snapshot_dir / f"capture-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}.jpg"
        path = app.state.capture_frame(
            output,
            channel=settings.camera.channel,
            timeout_seconds=settings.camera.capture_timeout_seconds,
            camera_config=settings.camera,
        )
        return {"snapshot_path": str(path)}

    @app.post("/api/test-prompt", dependencies=[Depends(require_mutation_auth)])
    async def test_prompt(request: TestPromptRequest) -> dict[str, object]:
        ensure_runtime()
        settings = app.state.settings
        prompt = request.prompt or next((p.text for p in settings.prompts if p.enabled), "person in the scene")
        if request.snapshot_path:
            image_path = _validated_snapshot_path(request.snapshot_path, app.state.snapshot_dir)
        else:
            image_path = app.state.snapshot_dir / f"test-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}.jpg"
            app.state.capture_frame(
                image_path,
                channel=settings.camera.channel,
                timeout_seconds=settings.camera.capture_timeout_seconds,
                camera_config=settings.camera,
            )
        result = await app.state.vlm_client.classify(image_path)
        event = DetectionEvent(
            camera_name=settings.camera.name,
            prompt=prompt,
            snapshot_path=str(image_path),
            score=0.0,
            labels=[result.activity or "unknown"],
            boxes=[],
            notification=NotificationState(status="skipped", reason="manual test endpoint"),
            raw={"vlm": result.to_dict()},
        )
        saved = app.state.event_store.add_event(event)
        return {"vlm": result.to_dict(), "event": saved.model_dump(mode="json")}

    @app.get("/api/pipeline/status")
    async def pipeline_status() -> dict[str, object]:
        now = asyncio.get_running_loop().time()
        last = app.state._last_trigger_ts
        return {
            "enabled": app.state._pipeline_enabled,
            "cooldown_seconds": app.state._pipeline_cooldown,
            "recent_context_seconds": app.state.settings.pipeline.recent_context_seconds,
            "last_trigger_ts": last,
            "seconds_since_last_trigger": None if last == 0.0 else round(now - last, 1),
        }

    @app.post("/api/pipeline/classify", dependencies=[Depends(require_mutation_auth)])
    async def pipeline_classify(request: TestPromptRequest) -> dict[str, object]:
        ensure_runtime()
        settings = app.state.settings
        if request.snapshot_path:
            image_path = _validated_snapshot_path(request.snapshot_path, app.state.snapshot_dir)
        else:
            image_path = app.state.snapshot_dir / f"manual-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}.jpg"
            app.state.capture_frame(
                image_path,
                channel=settings.camera.channel,
                timeout_seconds=settings.camera.capture_timeout_seconds,
                camera_config=settings.camera,
            )
        prior_context = recent_vlm_context()
        previous_context = prior_context[0] if prior_context else None
        previous_context_age_seconds = prior_context[1] if prior_context else None
        result = await app.state.vlm_client.classify(
            image_path,
            previous_context=previous_context,
            previous_context_age_seconds=previous_context_age_seconds,
        )
        event = DetectionEvent(
            camera_name=settings.camera.name,
            prompt=settings.vlm.classification_prompt,
            snapshot_path=str(image_path),
            score=0.0,
            labels=[result.activity or "unknown"],
            boxes=[],
            notification=NotificationState(status="skipped", reason="manual classification endpoint"),
            raw={
                "vlm": result.to_dict(),
                "previous_context": previous_context,
                "previous_context_age_seconds": previous_context_age_seconds,
            },
        )
        saved = app.state.event_store.add_event(event)
        return {"vlm": result.to_dict(), "event": saved.model_dump(mode="json")}

    @app.on_event("startup")
    async def _startup() -> None:
        if app.state._pipeline_enabled:
            if settings.camera_dashboard.enabled:
                app.state._pipeline_task = asyncio.create_task(_pipeline_loop())
            else:
                _pipeline_log.info("NVR poller disabled")
            if settings.yolo.enabled:
                app.state._yolo_task = asyncio.create_task(_yolo_poll_loop())
            else:
                _pipeline_log.info("YOLO poller disabled")
            app.state._retention_task = asyncio.create_task(_retention_cleanup_loop())

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        for task in (app.state._pipeline_task, app.state._yolo_task, app.state._retention_task):
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        if app.state.nvr_poller is not None:
            await app.state.nvr_poller.close()

    return app


app = create_app(initialize_runtime=False)


def main() -> None:
    settings = load_settings(DEFAULT_SETTINGS_PATH)
    uvicorn.run("camera_vision.app:app", host=settings.service.host, port=settings.service.port, reload=False)
