# Privacy

Local Camera Vision is intended for local processing of camera frames. The application can still expose sensitive data if it is configured carelessly.

## Data handled

Depending on configuration, the application may process:

- camera frames and short frame bursts;
- object detections and bounding boxes;
- VLM descriptions and context summaries;
- timestamps and local configuration;
- SQLite event records.

## Local-first behavior

The reference configuration uses a local RTSP source and a local OpenAI-compatible VLM endpoint. No hosted inference service is required by the application. Verify the behavior and retention policy of any model server or NVR that you add.

## Retention

Snapshots and event records are runtime data. They are ignored by Git and should not be uploaded as part of bug reports or releases. Configure `pipeline.retention_hours` for your needs and remove data that is no longer necessary.

## Sharing diagnostics

Before sharing logs or screenshots, remove:

- camera images and video;
- addresses, hostnames, usernames, and local paths;
- RTSP URLs and credentials;
- event timestamps if they reveal occupancy patterns;
- VLM output describing people or property.

## Responsibility

Follow applicable laws and obtain appropriate notice or consent when monitoring areas where other people may be recorded. Do not use model output as the sole basis for safety, legal, employment, or other high-impact decisions.

No real camera data is included in this repository.
