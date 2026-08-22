# Contributing

Thank you for contributing to Local Camera Vision.

## Before opening a pull request

- Run `pytest`.
- Run `python -m compileall -q src scripts tests`.
- Do not include camera frames, video, databases, logs, credentials, tokens, private URLs, or local machine paths.
- Add or update tests for behavior changes.
- Keep integrations optional and preserve the YOLO-only path.
- Document third-party model and runtime requirements separately from the original application code.

## NVR integrations

Place vendor-specific logic behind a small trigger adapter. Include synthetic payload tests and confirm that NVR outages do not disable local YOLO processing.

## Commit messages

Use concise, descriptive commit messages. Avoid including private deployment details in commit messages or test fixtures.

## Pull requests

Describe what changed, how it was tested, and any new configuration or dependency requirements. Redact sensitive data from examples and logs.

## License

By contributing, you agree that your contribution is provided under the MIT License in this repository.
