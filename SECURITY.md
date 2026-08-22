# Security

Local Camera Vision handles camera frames and may connect to cameras, NVRs, and local model servers.

- Keep camera credentials and API tokens in ignored environment files or a secret manager.
- Do not put credentials in YAML, source, RTSP URLs committed to Git, logs, issues, or screenshots.
- Bind the service to `127.0.0.1` by default. If LAN access is needed, use firewall rules and an authenticated TLS reverse proxy.
- Set `CAMERA_VISION_API_TOKEN` for mutating endpoints.
- The unsafe no-auth test bypass is for local tests only and must not be used on a network-exposed service.
- Treat model output as untrusted input. Do not make security-critical decisions without independent verification.
- Report suspected vulnerabilities privately through the repository's security contact or issue tracker; do not publish credentials or private footage in an issue.

No camera footage, model weights, logs, databases, or credentials are included in the repository.

## Dependencies

Review the upstream security advisories and licenses for Python dependencies, FFmpeg, YOLO, the selected VLM, and any NVR software used in a deployment.

## Data handling

Use the shortest practical snapshot retention period. Remove runtime data before sharing diagnostics.

## Trademark notice

Reolink, Gemma, YOLO, and other names are referenced for compatibility and attribution only. This project is not affiliated with those vendors unless explicitly stated.
