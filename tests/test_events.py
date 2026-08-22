from camera_vision.events import EventStore
from camera_vision.models import DetectionEvent, IdentityResult, NotificationState


def test_event_store_add_list_get(tmp_path):
    store = EventStore(tmp_path / "events.sqlite3")
    event = DetectionEvent(
        camera_name="Camera",
        prompt="person in scene",
        snapshot_path="snapshots/test.jpg",
        score=0.88,
        labels=["person"],
        identity=IdentityResult(name="unknown", confidence=0.0),
        notification=NotificationState(status="pending"),
    )

    saved = store.add_event(event)
    listed = store.list_events(limit=10)
    fetched = store.get_event(saved.id)
    latest = store.latest_event()

    assert saved.id is not None
    assert len(listed) == 1
    assert listed[0].prompt == "person in scene"
    assert fetched == saved
    assert latest == saved
    assert store.get_event(9999) is None
