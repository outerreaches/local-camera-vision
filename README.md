# Local Camera Vision

A privacy-conscious local camera pipeline: capture frames, detect objects with YOLO, optionally classify a burst of frames with a local vision-language model, and store structured events in a small dashboard.

## What is this?

This is a self-hosted reference application for event-driven camera vision. It is designed around local inference rather than sending camera footage to a hosted service.

The project has two optional trigger paths:

```text
YOLO trigger:
  RTSP frame -> YOLO detection -> frame burst -> VLM classification -> event

NVR trigger:
  NVR event transition -> frame capture -> frame burst -> VLM classification -> event
```

The NVR poller is disabled by default. YOLO-only operation does not require an NVR integration.

## What does it do?

- captures single frames from an RTSP-compatible camera;
- runs a configurable person detector through a persistent YOLO worker;
- captures a short multi-frame burst after a detection;
- sends the burst to an OpenAI-compatible local VLM endpoint;
- parses structured JSON and common fallback key/value responses;
- stores events and recent context in SQLite;
- serves a lightweight web dashboard and JSON API;
- applies cooldowns and snapshot retention;
- protects mutating endpoints with an API token;
- provides a generic trigger-source seam plus an optional vendor-reference poller.

The default configuration is safe and inactive: localhost binding, mock backends, and no background pipeline.

## What do you need?

### Required

- Linux, macOS, or Windows with Python 3.11+;
- an RTSP-compatible camera for real capture;
- FFmpeg available on `PATH` for real capture;
- a YOLO runtime/model for real detection;
- a local OpenAI-compatible VLM server for classification.

### Tested model

The reference implementation was tested with **Gemma 4 E2B** served by a local llama.cpp-compatible endpoint. Gemma was selected because it offered a useful speed/quality balance for camera-scene classification and could process a burst of multiple frames, giving the classifier temporal context instead of only one still image.

Other vision-language models may work through the same API, but they were not tested by this project.

Recorded reference measurement:

```text
Model: Gemma 4 E2B Q8_0 with projector
Hardware: Radeon 8060S-class local inference system
Single-frame warm wall time: ~7.5 s
Three-frame classification: ~10.3 s
Generation rate: ~54 tok/s for the three-frame path
```

These are measurements from one local system, not performance guarantees.

## How do you set it up?

### 1. Install the package

```bash
git clone https://github.com/outerreaches/local-camera-vision.git
cd local-camera-vision
python -m venv .venv
. .venv/bin/activate              # Windows: .venv\\Scripts\\activate
python -m pip install -e '.[dev]'
```

Install FFmpeg. For the included YOLO worker, install the optional runtime:

```bash
python -m pip install -e '.[yolo]'
```

The repository does not bundle model weights.

### 2. Start in safe mock mode

```bash
local-camera-vision
```

Open `http://127.0.0.1:8091/`. The default config does not start background capture or inference.

Run tests with:

```bash
pytest
```

### 3. Configure a real local camera

Create an ignored local environment file or export variables in your shell:

```bash
export RTSP_HOST='camera.example.invalid'
export RTSP_USER='camera-user'
export RTSP_PASSWORD='replace-with-a-local-secret'
export CAMERA_VISION_API_TOKEN='replace-with-a-long-random-token'
```

Do not put real credentials in YAML, source code, URLs committed to Git, logs, issues, or screenshots.

Edit `config/vision-config.yaml`:

```yaml
pipeline:
  enabled: true

yolo:
  enabled: true
  backend: real
  model_path: yolo11n.pt

vlm:
  enabled: true
  backend: gemma4-e2b
  server_url: http://127.0.0.1:8084

camera_dashboard:
  enabled: false
```

YOLO will poll frames and trigger the burst/VLM path. The NVR poller remains off.

### 4. Add an NVR trigger later (optional)

Set `camera_dashboard.enabled: true` only after configuring an appropriate trigger adapter. The included poller is a reference for a dashboard-backed AI-state API; it is not universal NVR support.

For another NVR, point your coding agent at:

```text
src/camera_vision/nvr_poller.py
tests/test_nvr_poller.py
```

Implement a small adapter that returns normalized `TriggerEvent` values, maps the vendor API/webhook/MQTT/ONVIF events, filters channels and event types, deduplicates false-to-true transitions, and treats NVR outages as non-fatal. Add synthetic tests for active, inactive, duplicate, malformed, unauthorized, and unavailable-NVR cases.

## API and security

Read-only health, settings, events, and snapshot endpoints are available for local use. Mutating endpoints require `CAMERA_VISION_API_TOKEN` using either `Authorization: Bearer [TOKEN]` or `X-API-Token: [TOKEN]`.

The unsafe no-auth bypass is for tests only. Do not expose this application or an NVR directly to the public Internet. Use localhost/LAN controls and TLS/reverse-proxy authentication when remote access is needed.

## Privacy

Camera frames can contain people, property details, timestamps, and metadata. Runtime snapshots, SQLite databases, and logs are ignored and must never be committed. Choose a retention period appropriate for your use and delete footage you no longer need.

## Project structure

```text
src/camera_vision/   application, capture, VLM, YOLO, events, NVR seam
scripts/              persistent YOLO worker
config/               safe example configuration
tests/                offline unit and security tests
web/                  dashboard assets
docs/                 architecture and integration notes
```

## License and third-party models

Original application code is licensed under the MIT License. Third-party runtimes, model implementations, model weights, and camera/NVR software remain under their own licenses and terms. No model weights are included.

The project is intentionally model-agnostic at the API boundary, but Gemma 4 E2B is the only VLM tested in the reference deployment.

## Status

This is an early reference implementation. Review authentication, network exposure, retention, and model output before using it for security-sensitive decisions.

## License

MIT. See [LICENSE](LICENSE).

## Attribution

Reolink is a trademark of its respective owner. This project is not affiliated with or endorsed by Reolink.

YOLO and Gemma are referenced third-party models; consult their upstream documentation and licenses before use.
