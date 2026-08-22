from pathlib import Path

from fastapi.testclient import TestClient

from camera_vision.app import create_app
from camera_vision.models import VisionSettings



def _client(tmp_path, *, token: str | None = "secret-token", allow_unsafe_no_auth: bool = False):
    app = create_app(
        settings_path=tmp_path / "config.yaml",
        db_path=tmp_path / "events.sqlite3",
        snapshot_dir=tmp_path / "snapshots",
        api_token=token,
        allow_unsafe_no_auth=allow_unsafe_no_auth,
    )

    def fake_capture(output_path: Path, channel: int = 0, timeout_seconds: int = 15, **_kwargs):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"jpg")
        return output_path

    app.state.capture_frame = fake_capture

    class FakeVlmClient:
        async def classify(self, image_paths, **_kwargs):
            from camera_vision.vlm_client import VlmClassificationResult
            return VlmClassificationResult(person_present=True, activity="standing", summary="A person is visible.")

    app.state.vlm_client = FakeVlmClient()
    return TestClient(app), app


def test_mutating_endpoints_require_auth_token(tmp_path):
    client, _app = _client(tmp_path)
    settings = client.get("/api/settings").json()

    assert client.put("/api/settings", json=settings).status_code == 401
    assert client.post("/api/capture-now").status_code == 401
    assert client.post("/api/test-prompt", json={"prompt": "person"}).status_code == 401


def test_mutating_endpoints_accept_bearer_or_x_api_token(tmp_path):
    client, app = _client(tmp_path)
    settings = client.get("/api/settings").json()
    settings["camera"]["name"] = "Token OK"

    assert client.put("/api/settings", json=settings, headers={"Authorization": "Bearer secret-token"}).status_code == 200
    class FakeVlmClient:
        async def classify(self, image_paths, **_kwargs):
            from camera_vision.vlm_client import VlmClassificationResult
            return VlmClassificationResult(person_present=True, activity="standing", summary="A person is visible.")

    app.state.vlm_client = FakeVlmClient()
    assert client.post("/api/capture-now", headers={"X-API-Token": "secret-token"}).status_code == 200
    assert client.post(
        "/api/test-prompt",
        json={"prompt": "person"},
        headers={"Authorization": "Bearer secret-token"},
    ).status_code == 200


def test_explicit_unsafe_test_app_can_mutate_without_token(tmp_path):
    client, _app = _client(tmp_path, token=None, allow_unsafe_no_auth=True)
    assert client.post("/api/capture-now").status_code == 200


def test_settings_api_does_not_allow_runtime_path_changes(tmp_path):
    client, app = _client(tmp_path, token=None, allow_unsafe_no_auth=True)
    original = VisionSettings.model_validate(client.get("/api/settings").json())
    payload = original.model_dump(mode="json")
    payload["service"]["database_path"] = str(tmp_path / "evil.sqlite3")
    payload["service"]["snapshot_dir"] = str(tmp_path / "evil-snapshots")

    payload["camera"]["name"] = "Allowed Change"

    response = client.put("/api/settings", json=payload)

    assert response.status_code == 200
    updated = VisionSettings.model_validate(response.json())
    assert updated.camera.name == "Allowed Change"
    assert updated.service.database_path == original.service.database_path
    assert updated.service.snapshot_dir == original.service.snapshot_dir

    assert app.state.db_path == tmp_path / "events.sqlite3"
    assert app.state.snapshot_dir == tmp_path / "snapshots"


def test_test_prompt_rejects_snapshot_path_outside_snapshot_dir(tmp_path):
    client, _app = _client(tmp_path, token=None, allow_unsafe_no_auth=True)
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"jpg")

    response = client.post("/api/test-prompt", json={"prompt": "person", "snapshot_path": str(outside)})

    assert response.status_code == 400
    assert "snapshot_dir" in response.json()["detail"]


def test_test_prompt_rejects_missing_snapshot_path_inside_snapshot_dir(tmp_path):
    client, app = _client(tmp_path, token=None, allow_unsafe_no_auth=True)
    missing = app.state.snapshot_dir / "missing.jpg"

    response = client.post("/api/test-prompt", json={"prompt": "person", "snapshot_path": str(missing)})

    assert response.status_code == 400
    assert "does not exist" in response.json()["detail"]
