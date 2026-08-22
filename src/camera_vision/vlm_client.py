from __future__ import annotations

import base64
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)


@dataclass
class VlmClassificationResult:
    person_present: bool | None = None
    direction: str | None = None
    carrying: str | None = None
    activity: str | None = None
    suspicious: bool | None = None
    urgency: str | None = None
    summary: str = ""
    error: str | None = None
    raw_response: str = ""
    elapsed_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    context_description: str = ""
    response_source: str = "content"
    finish_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "person_present": self.person_present,
            "direction": self.direction,
            "carrying": self.carrying,
            "activity": self.activity,
            "suspicious": self.suspicious,
            "urgency": self.urgency,
            "summary": self.summary,
            "context_description": self.context_description,
            "error": self.error,
            "raw_response": self.raw_response,
            "elapsed_ms": self.elapsed_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "response_source": self.response_source,
            "finish_reason": self.finish_reason,
        }


class VlmClient:
    DEFAULT_PROMPT = (
        "You are analyzing frames from a fixed local camera. "
        "Classify visible people, animals, objects, movement, and relevant scene context. "
        "Classify the scene and reply as strict JSON with these keys:\n"
        "  person_present (boolean): true if any person is visible.\n"
        "  direction (string or null): 'approaching' if moving toward the camera, "
        "'leaving' if moving away, 'standing' if stationary, null if no person.\n"
        "  carrying (string or null): any items the person is holding or carrying.\n"
        "  activity (string): what the person appears to be doing.\n"
        "  suspicious (boolean): true if the behavior seems suspicious (loitering, "
        "checking handles, hiding face, etc).\n"
        "  urgency (string or null): 'low', 'medium', or 'high'.\n"
        "  summary (string): one-sentence description of the scene for display.\n"
        "  context_description (string): two to three sentences describing the situation across "
        "these frames, including visible people/animals, movement, location, and relevant continuity. "
        "This will be used as context for the next detection.\n"
        "Return ONLY the JSON object, no markdown fences or extra text."
    )

    def __init__(
        self,
        server_url: str = "http://127.0.0.1:8084",
        model: str = "gemma-4-e2b",
        max_tokens: int = 1600,
        timeout_seconds: int = 60,
        classification_prompt: str | None = None,
    ):
        self.server_url = server_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout_seconds
        self.classification_prompt = classification_prompt or self.DEFAULT_PROMPT

    def _build_request(
        self,
        image_paths: str | Path | list[str | Path],
        prompt: str,
    ) -> dict[str, Any]:
        if not isinstance(image_paths, list):
            image_paths = [image_paths]

        content: list[dict[str, Any]] = []
        for img_path in image_paths:
            img_data = Path(img_path).read_bytes()
            b64 = base64.b64encode(img_data).decode()
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })

        content.append({"type": "text", "text": prompt})

        return {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": content,
                }
            ],
            "max_tokens": self.max_tokens,
            "temperature": 0.0,
        }

    def _build_prompt(
        self,
        previous_context: str | None = None,
        previous_context_age_seconds: int | None = None,
    ) -> str:
        """Build a per-request prompt, optionally including one recent prior detection."""
        if not previous_context:
            return self.classification_prompt

        cleaned = previous_context.strip()
        if not cleaned:
            return self.classification_prompt

        age_phrase = ""
        if previous_context_age_seconds is not None:
            seconds = max(0, int(previous_context_age_seconds))
            unit = "second" if seconds == 1 else "seconds"
            age_phrase = f" from {seconds} {unit} before the current frames"

        return (
            f"Recent prior detection context{age_phrase}:\n"
            f'"{cleaned}"\n\n'
            "Use this only as background context for continuity. The current frames are authoritative. "
            "If the prior context conflicts with the current frames, trust the current frames.\n\n"
            f"{self.classification_prompt}"
        )

    def _parse_bool_or_none(self, value: Any) -> bool | None:
        if isinstance(value, bool) or value is None:
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "yes", "1"}:
                return True
            if normalized in {"false", "no", "0", "none", "null", ""}:
                return False if normalized not in {"none", "null", ""} else None
        return None

    def _coerce_key_value_field(self, key: str, value: str) -> Any:
        value = value.strip()
        if value.lower() in {"null", "none", "n/a", ""}:
            return None
        if key in {"person_present", "suspicious"}:
            return self._parse_bool_or_none(value)
        return value

    def _parse_key_value_response(self, raw: str) -> dict[str, Any] | None:
        """Parse model fallback output like `key: value` or `key:value,key:value`.

        Gemma occasionally follows the requested schema but emits plain
        key/value text rather than strict JSON. Preserve that useful
        classification instead of marking the event as unknown.
        """
        expected = {
            "person_present",
            "direction",
            "carrying",
            "activity",
            "suspicious",
            "urgency",
            "summary",
            "context_description",
        }
        candidates: list[dict[str, Any]] = []

        # First handle pure newline-delimited key/value blocks.
        if "\n" in raw:
            line_data: dict[str, Any] = {}
            for line in raw.splitlines():
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                key = key.strip().lower()
                if key in expected:
                    line_data[key] = self._coerce_key_value_field(key, value)

            if line_data:
                candidates.append(line_data)

        # Fallback for one-line and mixed comma/newline-delimited outputs:
        # person_present:true,direction:standing,summary:...\ncontext_description:...
        # Split only at delimiters that introduce a known key so descriptive
        # strings can safely contain ordinary commas.
        import re

        data: dict[str, Any] = {}
        key_pattern = "|".join(re.escape(key) for key in sorted(expected, key=len, reverse=True))
        matches = list(re.finditer(rf"(?:^|[,\n]|\.)\s*({key_pattern})\s*:", raw, flags=re.IGNORECASE))
        for idx, match in enumerate(matches):
            key = match.group(1).strip().lower()
            value_start = match.end()
            value_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(raw)
            if idx + 1 < len(matches) and raw[value_end:value_end + 1] == ".":
                value_end += 1
            value = raw[value_start:value_end].strip()
            if value.endswith(","):
                value = value[:-1].strip()
            data[key] = self._coerce_key_value_field(key, value)
        if data:
            candidates.append(data)

        if not candidates:
            return None
        return max(candidates, key=lambda item: len(item))

    def _incomplete_classification_error(self, data: dict[str, Any]) -> str | None:
        """Detect visibly truncated successful-looking classifications."""
        if self._parse_bool_or_none(data.get("person_present")) is not True:
            return None

        required = ["activity", "suspicious", "summary", "context_description"]
        missing = [key for key in required if data.get(key) is None or data.get(key) == ""]
        if not missing:
            return None
        return f"Incomplete VLM classification: missing {', '.join(missing)}"

    def _parse_classification(self, response: dict[str, Any]) -> VlmClassificationResult:
        try:
            choice = response["choices"][0]
            msg = choice["message"]
            finish_reason = choice.get("finish_reason")
            content = msg.get("content", "")
            raw = content.strip()
            response_source = "content"

            # Gemma thinking model: content may be empty if reasoning consumed
            # all tokens. Fall back to reasoning_content when available.
            if not raw:
                raw = msg.get("reasoning_content", "").strip()
                response_source = "reasoning_content" if raw else "content"

            # Strip markdown code fences
            if raw.startswith("```"):
                lines = raw.split("\n")
                raw = "\n".join(lines[1:]) if len(lines) > 1 else raw
            if raw.endswith("```"):
                raw = raw[:-3].strip()

            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                data = self._parse_key_value_response(raw)
                if data is None:
                    raise

            incomplete_error = self._incomplete_classification_error(data)
            if incomplete_error:
                return VlmClassificationResult(
                    error=incomplete_error,
                    raw_response=raw,
                    response_source=response_source,
                    finish_reason=finish_reason,
                )

            return VlmClassificationResult(
                person_present=self._parse_bool_or_none(data.get("person_present")),
                direction=data.get("direction"),
                carrying=data.get("carrying"),
                activity=data.get("activity"),
                suspicious=self._parse_bool_or_none(data.get("suspicious")),
                urgency=data.get("urgency"),
                summary=data.get("summary", ""),
                context_description=data.get("context_description", ""),
                raw_response=raw,
                response_source=response_source,
                finish_reason=finish_reason,
            )
        except (json.JSONDecodeError, KeyError, IndexError) as exc:
            choice = response.get("choices", [{}])[0]
            msg = choice.get("message", {}) if isinstance(choice, dict) else {}
            content = msg.get("content", "")
            reasoning_content = msg.get("reasoning_content", "")
            raw_response = content or reasoning_content
            return VlmClassificationResult(
                error=f"Failed to parse VLM response: {exc}",
                raw_response=raw_response,
                response_source="content" if content else "reasoning_content",
                finish_reason=choice.get("finish_reason") if isinstance(choice, dict) else None,
            )

    async def classify(
        self,
        image_paths: str | Path | list[str | Path],
        previous_context: str | None = None,
        previous_context_age_seconds: int | None = None,
    ) -> VlmClassificationResult:
        prompt = self._build_prompt(previous_context, previous_context_age_seconds)
        payload = self._build_request(image_paths, prompt)
        t0 = time.perf_counter()
        async with httpx.AsyncClient(timeout=httpx.Timeout(self.timeout)) as client:
            resp = await client.post(
                f"{self.server_url}/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()
        elapsed = (time.perf_counter() - t0) * 1000
        result = self._parse_classification(data)
        result.elapsed_ms = elapsed
        result.prompt_tokens = data.get("usage", {}).get("prompt_tokens", 0)
        result.completion_tokens = data.get("usage", {}).get("completion_tokens", 0)
        return result
