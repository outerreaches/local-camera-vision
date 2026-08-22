from camera_vision.nvr_poller import CameraDashboardPoller


class TestCameraDashboardPoller:
    def test_parse_ai_state_detects_person(self):
        poller = CameraDashboardPoller(dashboard_url="http://fake:8090", channel=0)
        raw = [{"value": {"channel": 0, "people": {"alarm_state": 1}, "vehicle": {"alarm_state": 0}}}]
        result = poller._extract_trigger(raw)
        assert result is not None
        assert result.channel == 0
        assert result.detection_type == "people"
        assert result.active is True

    def test_parse_ai_state_no_person(self):
        poller = CameraDashboardPoller(dashboard_url="http://fake:8090", channel=0)
        raw = [{"value": {"channel": 0, "people": {"alarm_state": 0}}}]
        result = poller._extract_trigger(raw)
        assert result is None

    def test_parse_ai_state_ignores_other_channels(self):
        poller = CameraDashboardPoller(dashboard_url="http://fake:8090", channel=0)
        raw = [{"value": {"channel": 1, "people": {"alarm_state": 1}}}]
        result = poller._extract_trigger(raw)
        assert result is None

    def test_suppresses_duplicate_transitions(self):
        poller = CameraDashboardPoller(dashboard_url="http://fake:8090", channel=0)
        raw = [{"value": {"channel": 0, "people": {"alarm_state": 1}}}]
        first = poller._extract_trigger(raw)
        poller._update_last_state(raw)
        second = poller._extract_trigger(raw)
        assert first is not None
        assert second is None  # same state, no new transition
