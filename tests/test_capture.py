import subprocess
import traceback

import pytest

from camera_vision.capture import build_rtsp_url, capture_frame
from camera_vision.models import CameraConfig


def test_build_rtsp_url_from_template(monkeypatch):
    monkeypatch.setenv("RTSP_URL_TEMPLATE", "rtsp://user:secret@cam:554/{channel}")

    assert build_rtsp_url(channel=0) == "rtsp://user:secret@cam:554/0"


def test_build_rtsp_url_from_parts_uses_reolink_channel_zero_default(monkeypatch):
    monkeypatch.delenv("RTSP_URL_TEMPLATE", raising=False)
    monkeypatch.setenv("RTSP_USER", "admin")
    monkeypatch.setenv("RTSP_PASSWORD", "pw")
    monkeypatch.setenv("RTSP_HOST", "camera.example.invalid")
    monkeypatch.setenv("RTSP_PORT", "8554")

    assert build_rtsp_url(channel=0) == "rtsp://admin:pw@camera.example.invalid:8554/h264Preview_01_main"


def test_build_rtsp_url_percent_encodes_slash_in_credentials():
    env = {
        "RTSP_USER": "admin/user",
        "RTSP_PASSWORD": "pw/with/slashes",
        "RTSP_HOST": "camera.example.invalid",
    }

    assert build_rtsp_url(channel=0, env=env) == "rtsp://admin%2Fuser:pw%2Fwith%2Fslashes@camera.example.invalid:554/h264Preview_01_main"


def test_build_rtsp_url_uses_camera_config_env_names_and_stream_template():
    config = CameraConfig(
        host_env="CAM_HOST",
        user_env="CAM_USER",
        password_env="CAM_PASSWORD",
        port_env="CAM_PORT",
        rtsp_url_template_env="CAM_TEMPLATE",
        stream_path_template="stream-{channel}-{stream_number:02d}",
        default_rtsp_port=8554,
    )
    env = {"CAM_HOST": "camera.example.invalid", "CAM_USER": "admin", "CAM_PASSWORD": "pw"}

    assert build_rtsp_url(channel=1, env=env, config=config) == "rtsp://admin:pw@camera.example.invalid:8554/stream-1-02"


def test_build_rtsp_url_requires_credentials_without_template(monkeypatch):
    for key in ("RTSP_URL_TEMPLATE", "RTSP_USER", "RTSP_PASSWORD", "RTSP_HOST", "RTSP_PORT"):
        monkeypatch.delenv(key, raising=False)

    with pytest.raises(ValueError, match="RTSP_HOST"):
        build_rtsp_url(channel=0)


def test_capture_frame_invokes_ffmpeg_without_exposing_credentials(tmp_path, monkeypatch):
    calls = []

    def fake_run(cmd, check, capture_output, text, timeout):
        calls.append(cmd)
        (tmp_path / "frame.jpg").write_bytes(b"jpg")

        class Result:
            stderr = ""

        return Result()

    monkeypatch.setenv("RTSP_URL_TEMPLATE", "rtsp://user:secret@cam:554/{channel}")
    monkeypatch.setattr("camera_vision.capture.subprocess.run", fake_run)

    result = capture_frame(tmp_path / "frame.jpg", channel=0, timeout_seconds=3)

    assert result == tmp_path / "frame.jpg"
    assert calls[0][:2] == ["ffmpeg", "-hide_banner"]
    assert "rtsp://user:secret@cam:554/0" in calls[0]


def test_capture_frame_redacts_rtsp_credentials_from_ffmpeg_stderr(tmp_path, monkeypatch):
    raw_url = "rtsp://user:secret@cam:554/0"

    def fake_run(cmd, check, capture_output, text, timeout):
        raise subprocess.CalledProcessError(
            returncode=1,
            cmd=cmd,
            stderr=f"Connection failed while opening {raw_url}",
        )

    monkeypatch.setenv("RTSP_URL_TEMPLATE", "rtsp://user:secret@cam:554/{channel}")
    monkeypatch.setattr("camera_vision.capture.subprocess.run", fake_run)

    with pytest.raises(RuntimeError) as exc_info:
        capture_frame(tmp_path / "frame.jpg", channel=0, timeout_seconds=3)

    message = str(exc_info.value)
    assert "secret" not in message
    assert raw_url not in message
    assert "rtsp://***:***@cam:554/0" in message


@pytest.mark.parametrize(
    "raised_exception",
    [
        subprocess.CalledProcessError(
            returncode=1,
            cmd=["ffmpeg", "-i", "rtsp://user:secret@cam:554/0", "frame.jpg"],
            stderr="Connection failed while opening rtsp://user:secret@cam:554/0",
        ),
        subprocess.TimeoutExpired(
            cmd=["ffmpeg", "-i", "rtsp://user:secret@cam:554/0", "frame.jpg"],
            timeout=3,
        ),
    ],
)
def test_capture_frame_redacts_rtsp_credentials_from_full_traceback(tmp_path, monkeypatch, raised_exception):
    raw_url = "rtsp://user:secret@cam:554/0"

    def fake_run(cmd, check, capture_output, text, timeout):
        raise raised_exception

    monkeypatch.setenv("RTSP_URL_TEMPLATE", "rtsp://user:secret@cam:554/{channel}")
    monkeypatch.setattr("camera_vision.capture.subprocess.run", fake_run)

    with pytest.raises(RuntimeError) as exc_info:
        capture_frame(tmp_path / "frame.jpg", channel=0, timeout_seconds=3)

    formatted = "".join(traceback.format_exception(exc_info.type, exc_info.value, exc_info.tb))
    assert "secret" not in formatted
    assert raw_url not in formatted
    assert "rtsp://***:***@cam:554/0" in formatted
