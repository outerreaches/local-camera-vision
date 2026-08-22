from __future__ import annotations

import argparse
import os
import re
import subprocess
from pathlib import Path
from urllib.parse import quote

from camera_vision.models import CameraConfig


def build_rtsp_url(channel: int = 0, env: dict[str, str] | None = None, config: CameraConfig | None = None) -> str:
    source = env if env is not None else os.environ
    camera = config or CameraConfig()
    template = source.get(camera.rtsp_url_template_env)
    if template:
        return template.format(channel=channel)

    host = source.get(camera.host_env)
    user = source.get(camera.user_env)
    password = source.get(camera.password_env)
    port = source.get(camera.port_env, str(camera.default_rtsp_port))
    missing = [
        name
        for name, value in (
            (camera.host_env, host),
            (camera.user_env, user),
            (camera.password_env, password),
        )
        if not value
    ]
    if missing:
        raise ValueError(
            f"Missing RTSP configuration: {', '.join(missing)} or {camera.rtsp_url_template_env}"
        )

    stream_number = channel + 1
    stream_path = camera.stream_path_template.format(channel=channel, stream_number=stream_number)
    return f"rtsp://{quote(user or '', safe='')}:{quote(password or '', safe='')}@{host}:{port}/{stream_path}"


_RTSP_CREDENTIALS_RE = re.compile(r"\brtsp://[^@\s]+@", re.IGNORECASE)


def redact_url(url: str) -> str:
    return redact_credentials(url)


def redact_credentials(value: object) -> str:
    """Redact RTSP userinfo from any command, stderr, or stdout text."""

    text = "" if value is None else str(value)
    return _RTSP_CREDENTIALS_RE.sub(lambda match: f"{match.group(0).split('://', 1)[0]}://***:***@", text)


def _format_process_error(exc: subprocess.CalledProcessError) -> str:
    details = []
    if exc.stderr:
        details.append(f"stderr: {redact_credentials(exc.stderr).strip()}")
    if exc.stdout:
        details.append(f"stdout: {redact_credentials(exc.stdout).strip()}")
    if not details:
        details.append(redact_credentials(exc))
    details.append(f"cmd: {redact_credentials(exc.cmd)}")
    return "; ".join(details)


def capture_frame(
    output_path: str | Path,
    channel: int = 0,
    timeout_seconds: int = 15,
    camera_config: CameraConfig | None = None,
) -> Path:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rtsp_url = build_rtsp_url(channel=channel, config=camera_config)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-rtsp_transport",
        "tcp",
        "-y",
        "-i",
        rtsp_url,
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(target),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=timeout_seconds)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg failed for {redact_url(rtsp_url)}: {_format_process_error(exc)}") from None
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"ffmpeg timed out for {redact_url(rtsp_url)}; cmd: {redact_credentials(exc.cmd)}") from None
    if not target.exists():
        raise RuntimeError(f"ffmpeg completed but did not create {target}")
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Capture a single RTSP camera frame with ffmpeg")
    parser.add_argument("output", type=Path, help="output image path")
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=15)
    args = parser.parse_args(argv)
    path = capture_frame(args.output, channel=args.channel, timeout_seconds=args.timeout)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
