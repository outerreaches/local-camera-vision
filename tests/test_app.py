from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from camera_vision.app import create_app
from camera_vision.models import DetectionEvent, VisionSettings

from camera_vision.settings import load_settings, save_settings
from camera_vision.vlm_client import VlmClassificationResult


def test_pipeline_does_not_start_nvr_poller_when_disabled(tmp_path, monkeypatch):
    settings_path = tmp_path / "config.yaml"
    settings = VisionSettings()
    settings.pipeline.enabled = True
    settings.camera_dashboard.enabled = False
    settings.yolo.enabled = False
    save_settings(settings, settings_path)
    app = create_app(
        settings_path=settings_path,
        db_path=tmp_path / "events.sqlite3",
        snapshot_dir=tmp_path / "snapshots",
    )

    class FailingPoller:
        async def poll_once(self):
            raise AssertionError("NVR poller must not run when disabled")

        async def close(self):
            pass

    app.state.nvr_poller = FailingPoller()
    client = TestClient(app)

    with client:
        assert app.state._pipeline_task is None
        assert app.state._yolo_task is None


def test_fastapi_vertical_slice_endpoints(tmp_path, monkeypatch):
    settings_path = tmp_path / "config.yaml"
    db_path = tmp_path / "events.sqlite3"
    snapshot_dir = tmp_path / "snapshots"
    app = create_app(settings_path=settings_path, db_path=db_path, snapshot_dir=snapshot_dir, allow_unsafe_no_auth=True)
    client = TestClient(app)

    def fake_capture(output_path: Path, channel: int = 0, timeout_seconds: int = 15, **_kwargs):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"jpg")
        return output_path

    app.state.capture_frame = fake_capture

    class FakeVlmClient:
        async def classify(self, image_paths, **_kwargs):
            return VlmClassificationResult(
                person_present=True,
                activity="standing",
                suspicious=False,
                summary="A person is visible.",
                context_description="A person is visible in the camera frame.",
            )

    app.state.vlm_client = FakeVlmClient()

    health = client.get("/healthz").json()
    assert health["status"] == "ok"

    assert "pipeline" in health
    assert isinstance(health["pipeline"]["enabled"], bool)
    settings = client.get("/api/settings").json()
    assert settings["service"]["port"] == 8091

    settings["camera"]["name"] = "Updated"
    put = client.put("/api/settings", json=settings)
    assert put.status_code == 200
    assert VisionSettings.model_validate(put.json()).camera.name == "Updated"
    app.state.vlm_client = FakeVlmClient()

    capture = client.post("/api/capture-now")
    assert capture.status_code == 200
    assert Path(capture.json()["snapshot_path"]).exists()

    # pipeline status endpoint
    status = client.get("/api/pipeline/status").json()
    assert "enabled" in status
    assert "cooldown_seconds" in status
    assert "seconds_since_last_trigger" in status

    test_prompt = client.post("/api/test-prompt", json={"prompt": "person at door"})
    assert test_prompt.status_code == 200
    body = test_prompt.json()
    assert "vlm" in body
    assert body["event"]["prompt"] == "person at door"

    events = client.get("/api/events?limit=50").json()
    assert len(events) == 1
    event_id = events[0]["id"]
    assert client.get(f"/api/events/{event_id}").json()["id"] == event_id
    assert client.get("/api/events/99999").status_code == 404

    assert client.get("/").status_code == 200
    assert client.get("/web/app.js").status_code == 200


def test_put_settings_preserves_face_recognition_known_faces_dir(tmp_path):
    settings_path = tmp_path / "config.yaml"
    original_known_faces_dir = str(tmp_path / "known-faces")
    initial_settings = VisionSettings()
    initial_settings.face_recognition.known_faces_dir = original_known_faces_dir
    save_settings(initial_settings, settings_path)
    app = create_app(settings_path=settings_path, allow_unsafe_no_auth=True, initialize_runtime=False)
    client = TestClient(app)

    requested = client.get("/api/settings").json()
    requested["face_recognition"]["known_faces_dir"] = str(tmp_path / "attacker-controlled")
    requested["face_recognition"]["threshold"] = 0.75

    response = client.put("/api/settings", json=requested)

    assert response.status_code == 200
    body = response.json()
    assert body["face_recognition"]["known_faces_dir"] == original_known_faces_dir
    assert body["face_recognition"]["threshold"] == 0.75
    persisted = load_settings(settings_path)
    assert persisted.face_recognition.known_faces_dir == original_known_faces_dir
    assert persisted.face_recognition.threshold == 0.75


def test_pipeline_classify_uses_recent_vlm_context(tmp_path):
    settings_path = tmp_path / "config.yaml"
    db_path = tmp_path / "events.sqlite3"
    snapshot_dir = tmp_path / "snapshots"
    initial_settings = VisionSettings()
    initial_settings.pipeline.recent_context_seconds = 60
    save_settings(initial_settings, settings_path)
    app = create_app(settings_path=settings_path, db_path=db_path, snapshot_dir=snapshot_dir, allow_unsafe_no_auth=True)
    client = TestClient(app)

    def fake_capture(output_path: Path, channel: int = 0, timeout_seconds: int = 15, **_kwargs):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"jpg")
        return output_path

    class FakeVlmClient:
        def __init__(self):
            self.previous_context = None
            self.previous_context_age_seconds = None

        async def classify(self, image_paths, previous_context=None, previous_context_age_seconds=None):
            self.previous_context = previous_context
            self.previous_context_age_seconds = previous_context_age_seconds
            return VlmClassificationResult(
                person_present=True,
                activity="standing",
                suspicious=False,
                summary="A person is standing near the porch.",
                context_description="A person is standing near the porch and remains calm.",
            )

    fake_vlm = FakeVlmClient()
    app.state.capture_frame = fake_capture
    app.state.vlm_client = fake_vlm
    app.state.event_store.add_event(DetectionEvent(
        camera_name="Doorbell",
        prompt="prior",
        snapshot_path="prior.jpg",
        created_at=datetime.now(timezone.utc) - timedelta(seconds=23),
        raw={
            "vlm": {
                "context_description": "A delivery driver previously walked away from the porch.",
                "error": None,
            }
        },
    ))

    response = client.post("/api/pipeline/classify", json={})

    assert response.status_code == 200
    assert fake_vlm.previous_context == "A delivery driver previously walked away from the porch."
    assert 23 <= fake_vlm.previous_context_age_seconds <= 24
    event = response.json()["event"]
    assert event["raw"]["previous_context"] == "A delivery driver previously walked away from the porch."
    assert 23 <= event["raw"]["previous_context_age_seconds"] <= 24
    assert event["raw"]["vlm"]["context_description"] == "A person is standing near the porch and remains calm."
