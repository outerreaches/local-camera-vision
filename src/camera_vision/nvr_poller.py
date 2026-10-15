from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)


@dataclass
class TriggerEvent:
    channel: int
    detection_type: str  # "people", "vehicle", "pet", "face"
    active: bool
    ts: float


class CameraDashboardPoller:
    def __init__(
        self,
        dashboard_url: str = "http://127.0.0.1:8090",
        channel: int = 0,
        trigger_types: list[str] | None = None,
    ):
        self.dashboard_url = dashboard_url.rstrip("/")
        self.channel = channel
        self.trigger_types = trigger_types or ["people"]
        self._last_state: dict[int, dict[str, bool]] = {}
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(10.0))
        return self._client

    async def fetch_ai_state(self) -> list[dict[str, Any]]:
        client = await self._get_client()
        resp = await client.get(f"{self.dashboard_url}/api/ai-state")
        resp.raise_for_status()
        data = resp.json()
        return data.get("ai", []) if isinstance(data, dict) else []

    def _extract_trigger(self, raw_ai: list[dict[str, Any]]) -> TriggerEvent | None:
        """Extract a trigger event from raw AI state if a relevant alarm is active."""
        for item in raw_ai:
            value = item.get("value", {})
            channel = value.get("channel")
            if channel != self.channel:
                continue
            for det_type in self.trigger_types:
                det = value.get(det_type, {})
                if isinstance(det, dict) and det.get("alarm_state") == 1:
                    prev = self._last_state.get(channel, {}).get(det_type, False)
                    if not prev:  # trigger only on false→true transition
                        return TriggerEvent(
                            channel=channel,
                            detection_type=det_type,
                            active=True,
                            ts=time.monotonic(),
                        )
        return None

    def _update_last_state(self, raw_ai: list[dict[str, Any]]) -> None:
        for item in raw_ai:
            value = item.get("value", {})
            ch = value.get("channel")
            if ch is None:
                continue
            if ch not in self._last_state:
                self._last_state[ch] = {}
            for det_type in self.trigger_types:
                det = value.get(det_type, {})
                self._last_state[ch][det_type] = (
                    isinstance(det, dict) and det.get("alarm_state") == 1
                )

    async def poll_once(self) -> TriggerEvent | None:
        try:
            raw = await self.fetch_ai_state()
            trigger = self._extract_trigger(raw)
            self._update_last_state(raw)
            return trigger
        except Exception:
            logger.exception("Failed to poll NVR trigger state")
            return None

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
