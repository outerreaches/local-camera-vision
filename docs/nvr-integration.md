# NVR trigger integration

The pipeline can run with YOLO alone. The NVR trigger source is optional and disabled by default.

## Normalized contract

An adapter should translate vendor-specific events into a small normalized event:

```python
@dataclass
class TriggerEvent:
    channel: int
    detection_type: str
    active: bool
    ts: float
```

The current reference poller is in `src/camera_vision/nvr_poller.py` and demonstrates a dashboard-backed AI-state API. It is intentionally not presented as universal NVR support or as a direct vendor SDK.

## Adapting another NVR

1. Identify the NVR's local event API, webhook, MQTT topic, or ONVIF event interface.
2. Keep credentials in an ignored environment file or secret manager.
3. Map vendor payloads into `TriggerEvent`.
4. Emit only false-to-true transitions, or deduplicate equivalent events.
5. Restrict events to configured channels and detection types.
6. Add synthetic tests for active, inactive, duplicate, malformed, unauthorized, and unavailable-NVR responses.
7. Return `None` for non-triggering states and handle NVR errors without taking down the YOLO path.
8. Enable the NVR source only after it is configured and tested.

A coding agent can use the reference implementation and `tests/test_nvr_poller.py` as a starting point, but should inspect the target NVR's current documentation and API behavior rather than assuming the reference payload shape.

## Configuration

Leave this disabled for YOLO-only operation:

```yaml
camera_dashboard:
  enabled: false
```

For an installed adapter, configure its endpoint and enable it:

```yaml
camera_dashboard:
  enabled: true
  url: http://127.0.0.1:8090
  poll_interval_seconds: 5
  channel: 0
  trigger_types:
    - people
```

Do not expose the NVR or camera service to the public Internet. Use network controls and authentication appropriate for the deployment.

## Reolink note

The included implementation follows a generic dashboard-backed AI-state response shape inspired by the project's local Reolink deployment. It is an example adapter, not a complete direct Reolink client. Reolink is a trademark of its respective owner; this project is not affiliated with or endorsed by Reolink.
