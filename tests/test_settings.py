from pathlib import Path

from camera_vision.models import CameraConfig, VisionSettings
from camera_vision.settings import load_settings, save_settings


def test_load_default_config_has_camera_channel_zero():
    settings = load_settings(Path("config/vision-config.yaml"))

    assert settings.service.port == 8091
    assert settings.camera.name == "Camera"
    assert settings.camera.channel == 0



def test_settings_round_trip_yaml(tmp_path):
    settings = VisionSettings(camera=CameraConfig(name="Test", host_env="CAM_HOST", channel=0))
    path = tmp_path / "vision.yaml"

    save_settings(settings, path)
    loaded = load_settings(path)

    assert loaded == settings
