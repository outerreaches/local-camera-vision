from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ServiceConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8091
    data_dir: str = "data"
    database_path: str = "data/events.sqlite3"
    snapshot_dir: str = "data/snapshots"
    tmp_dir: str = "data/tmp"


class CameraConfig(BaseModel):
    name: str = "Camera"
    channel: int = 0
    host_env: str = "RTSP_HOST"
    rtsp_url_template_env: str = "RTSP_URL_TEMPLATE"
    user_env: str = "RTSP_USER"
    password_env: str = "RTSP_PASSWORD"
    port_env: str = "RTSP_PORT"
    default_rtsp_port: int = 554
    stream_path_template: str = "h264Preview_{stream_number:02d}_main"
    capture_timeout_seconds: int = 15


class PromptConfig(BaseModel):
    name: str = "person_in_scene"
    text: str = "person in the scene"
    enabled: bool = True
    min_score: float = 0.5


class ZoneConfig(BaseModel):
    name: str = "full_frame"
    enabled: bool = True
    polygon: list[list[int]] = Field(default_factory=list)


class NotificationConfig(BaseModel):
    enabled: bool = False
    cooldown_seconds: int = 300
    dry_run: bool = True


class FaceRecognitionConfig(BaseModel):
    enabled: bool = False
    known_faces_dir: str = "data/faces"
    threshold: float = 0.6


class CameraDashboardConfig(BaseModel):
    enabled: bool = False
    url: str = "http://127.0.0.1:8090"
    poll_interval_seconds: int = 5
    channel: int = 0
    trigger_types: list[str] = Field(default_factory=lambda: ["people"])


class VlmConfig(BaseModel):
    enabled: bool = False
    backend: Literal["gemma4-e2b", "mock"] = "mock"
    server_url: str = "http://127.0.0.1:8084"
    model: str = "gemma-4-e2b"
    max_tokens: int = 1600
    timeout_seconds: int = 60
    classification_prompt: str = (
        "You are analyzing frames from a fixed local camera. "
        "Classify visible people, animals, objects, movement, and relevant scene context. "
        "Classify the scene and reply as strict JSON with these keys:\n"
        "  person_present (boolean): true if any person is visible.\n"
        "  direction (string or null): 'approaching' if moving toward the camera, "
        "'leaving' if moving away, 'standing' if stationary, null if no person.\n"
        "  carrying (string or null): any items the person is holding or carrying.\n"
        "  activity (string): what the person appears to be doing.\n"
        "  suspicious (boolean): true if the behavior seems suspicious (loitering, "
        "checking handles, hiding face, etc).\n"
        "  urgency (string or null): 'low', 'medium', or 'high'.\n"
        "  summary (string): one-sentence description of the scene for display.\n"
        "  context_description (string): two to three sentences describing the situation across "
        "these frames, including visible people/animals, movement, location, and relevant continuity. "
        "This will be used as context for the next detection.\n"
        "Return ONLY the JSON object, no markdown fences or extra text."
    )


class PipelineConfig(BaseModel):
    enabled: bool = False
    cooldown_seconds: int = 60
    recent_context_seconds: int = 60
    retention_hours: int = 24


class YoloDetectorConfig(BaseModel):
    enabled: bool = False
    backend: Literal["mock", "real", "auto"] = "mock"
    confidence: float = 0.5
    poll_interval_seconds: int = 3
    model_path: str = "yolo11n.pt"
    python_path: str = ".venv/bin/python3"


class VisionSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    service: ServiceConfig = Field(default_factory=ServiceConfig)
    camera: CameraConfig = Field(default_factory=CameraConfig)
    prompts: list[PromptConfig] = Field(default_factory=lambda: [PromptConfig()])
    zones: list[ZoneConfig] = Field(default_factory=lambda: [ZoneConfig()])

    notifications: NotificationConfig = Field(default_factory=NotificationConfig)
    face_recognition: FaceRecognitionConfig = Field(default_factory=FaceRecognitionConfig)
    camera_dashboard: CameraDashboardConfig = Field(default_factory=CameraDashboardConfig)
    vlm: VlmConfig = Field(default_factory=VlmConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    yolo: YoloDetectorConfig = Field(default_factory=YoloDetectorConfig)


class IdentityResult(BaseModel):
    name: str = "unknown"
    confidence: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class NotificationState(BaseModel):
    status: Literal["pending", "sent", "skipped", "failed"] = "skipped"
    reason: str | None = None
    sent_at: datetime | None = None


class DetectionEvent(BaseModel):
    id: int | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    camera_name: str
    prompt: str
    snapshot_path: str
    score: float | None = None
    labels: list[str] = Field(default_factory=list)
    boxes: list[dict[str, Any]] = Field(default_factory=list)
    identity: IdentityResult = Field(default_factory=IdentityResult)
    notification: NotificationState = Field(default_factory=NotificationState)
    raw: dict[str, Any] = Field(default_factory=dict)
