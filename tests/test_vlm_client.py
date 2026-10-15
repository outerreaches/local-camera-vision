from camera_vision.vlm_client import VlmClient


class TestVlmClient:
    def test_parse_valid_json_classification(self):
        client = VlmClient(server_url="http://fake:8084", model="gemma-4-e2b")
        response = {
            "choices": [{
                "message": {
                    "content": '{"person_present": true, "carrying": "package", "activity": "delivering", "suspicious": false, "urgency": "normal", "summary": "A person left a package.", "context_description": "A delivery driver approached the porch and left a package near the door. They appeared to be leaving by the final frame."}'
                }
            }]
        }
        result = client._parse_classification(response)
        assert result.person_present is True
        assert result.carrying == "package"
        assert result.activity == "delivering"
        assert result.suspicious is False
        assert result.urgency == "normal"
        assert result.context_description.startswith("A delivery driver")

    def test_parse_json_with_markdown_fences(self):
        client = VlmClient(server_url="http://fake:8084", model="gemma-4-e2b")
        response = {
            "choices": [{
                "message": {
                    "content": '```json\n{"person_present": false, "carrying": null, "activity": "none", "suspicious": false, "urgency": "normal", "summary": "Empty porch."}\n```'
                }
            }]
        }
        result = client._parse_classification(response)
        assert result.person_present is False
        assert result.activity == "none"

    def test_parse_key_value_classification_fallback(self):
        client = VlmClient(server_url="http://fake:8084", model="gemma-4-e2b")
        response = {
            "choices": [{
                "message": {
                    "content": "person_present: true\ndirection: standing\ncarrying: small object\nactivity: standing\nsuspicious: false\nurgency: low\nsummary: A person and a dog are standing on the walkway near the front door.\ncontext_description: A person and dog remain near the front walkway. Across the frames, they appear mostly stationary and non-suspicious."
                }
            }]
        }
        result = client._parse_classification(response)
        assert result.error is None
        assert result.person_present is True
        assert result.direction == "standing"
        assert result.carrying == "small object"
        assert result.activity == "standing"
        assert result.suspicious is False
        assert result.urgency == "low"
        assert result.summary == "A person and a dog are standing on the walkway near the front door."
        assert result.context_description.startswith("A person and dog remain")

    def test_parse_comma_delimited_key_value_classification_fallback(self):
        client = VlmClient(server_url="http://fake:8084", model="gemma-4-e2b")
        response = {
            "choices": [{
                "message": {
                    "content": "person_present:true,direction:standing,carrying:null,activity:standing on the porch,suspicious:false,urgency:null,summary:A person is standing on the front porch looking out at the yard.,context_description:A person is standing on the porch, looking out across the yard. Across the frames, they remain stationary and calm."
                }
            }]
        }
        result = client._parse_classification(response)
        assert result.error is None
        assert result.person_present is True
        assert result.direction == "standing"
        assert result.carrying is None
        assert result.activity == "standing on the porch"
        assert result.suspicious is False
        assert result.urgency is None
        assert result.summary == "A person is standing on the front porch looking out at the yard."
        assert result.context_description == "A person is standing on the porch, looking out across the yard. Across the frames, they remain stationary and calm."

    def test_parse_key_value_response_when_context_key_follows_sentence_without_delimiter(self):
        client = VlmClient(server_url="http://fake:8084", model="gemma-4-e2b")
        response = {
            "choices": [{
                "message": {
                    "content": "person_present:true,direction:leaving,carrying:null,activity:walking,suspicious:false,urgency:low,summary:A person is walking down a residential driveway away from the camera.context_description:The camera is positioned near a front door, looking out onto a residential area. One person is visible walking away."
                }
            }]
        }

        result = client._parse_classification(response)

        assert result.error is None
        assert result.person_present is True
        assert result.direction == "leaving"
        assert result.activity == "walking"
        assert result.summary == "A person is walking down a residential driveway away from the camera."
        assert result.context_description.startswith("The camera is positioned near a front door")

    def test_parse_mixed_comma_and_newline_key_value_response(self):
        client = VlmClient(server_url="http://fake:8084", model="gemma-4-e2b")
        response = {
            "choices": [{
                "message": {
                    "content": "person_present:true,direction:standing,carrying:phone and a green object,activity:looking at a phone,suspicious:false,urgency:low,summary:A person is standing at the front door looking at their phone.\ncontext_description:The camera captures a person standing stationary at the front door of a brick porch, looking down at a phone. The scene is set during the daytime."
                }
            }]
        }

        result = client._parse_classification(response)

        assert result.error is None
        assert result.person_present is True
        assert result.direction == "standing"
        assert result.carrying == "phone and a green object"
        assert result.activity == "looking at a phone"
        assert result.suspicious is False
        assert result.urgency == "low"
        assert result.summary == "A person is standing at the front door looking at their phone."
        assert result.context_description.startswith("The camera captures a person")

    def test_list_valued_text_fields_become_strings(self):
        # Gemma returned "activity": ["looking through a camera"] on 2026-09-29; the list reached
        # DetectionEvent.labels and failed validation, losing the event
        client = VlmClient(server_url="http://fake:8084", model="gemma-4-e2b")
        response = {
            "choices": [{
                "message": {
                    "content": '{"person_present": true, "direction": ["standing"], "carrying": [], "activity": ["looking through a camera"], "suspicious": false, "urgency": 2, "summary": ["A person at the door.", "They look at the camera."], "context_description": "A person stands at the door."}'
                }
            }]
        }
        result = client._parse_classification(response)
        assert result.error is None
        assert result.activity == "looking through a camera"
        assert result.direction == "standing"
        assert result.carrying is None
        assert result.urgency == "2"
        assert result.summary == "A person at the door. They look at the camera."
        for value in (result.direction, result.activity, result.urgency, result.summary, result.context_description):
            assert isinstance(value, str)

    def test_empty_list_activity_counts_as_missing(self):
        client = VlmClient(server_url="http://fake:8084", model="gemma-4-e2b")
        response = {
            "choices": [{
                "message": {
                    "content": '{"person_present": true, "activity": [], "suspicious": false, "summary": "A person.", "context_description": "A person stands still."}'
                }
            }]
        }
        result = client._parse_classification(response)
        assert result.error is not None
        assert "activity" in result.error

    def test_non_object_json_returns_error(self):
        client = VlmClient(server_url="http://fake:8084", model="gemma-4-e2b")
        response = {"choices": [{"message": {"content": '["person", "standing"]'}}]}
        result = client._parse_classification(response)
        assert result.error is not None
        assert result.activity is None

    def test_incomplete_key_value_response_returns_parse_error(self):
        client = VlmClient(server_url="http://fake:8084", model="gemma-4-e2b")
        response = {
            "choices": [{
                "message": {
                    "content": "person_present:true,direction:approaching,carrying:null,"
                }
            }]
        }

        result = client._parse_classification(response)

        assert result.person_present is None
        assert result.error is not None
        assert "Incomplete VLM classification" in result.error

    def test_empty_content_uses_reasoning_content_for_raw_response(self):
        client = VlmClient(server_url="http://fake:8084", model="gemma-4-e2b")
        response = {
            "choices": [{
                "finish_reason": "length",
                "message": {
                    "content": "",
                    "reasoning_content": "person_present:true,direction:standing,carrying:null,activity:standing,suspicious:false,urgency:low,summary:A person is standing on the porch.,context_description:A person remains stationary on the porch."
                }
            }]
        }

        result = client._parse_classification(response)

        assert result.error is None
        assert result.person_present is True
        assert result.raw_response.startswith("person_present:true")
        assert result.response_source == "reasoning_content"
        assert result.finish_reason == "length"

    def test_build_prompt_includes_previous_context(self):
        client = VlmClient(server_url="http://fake:8084", classification_prompt="Classify current frames.")
        prompt = client._build_prompt("A delivery driver was previously leaving the porch.", 23)
        assert "Recent prior detection context from 23 seconds before the current frames" in prompt
        assert "A delivery driver was previously leaving the porch." in prompt
        assert "current frames are authoritative" in prompt
        assert prompt.endswith("Classify current frames.")

    def test_build_prompt_without_age_preserves_context_header(self):
        client = VlmClient(server_url="http://fake:8084", classification_prompt="Classify current frames.")
        prompt = client._build_prompt("A delivery driver was previously leaving the porch.")
        assert "Recent prior detection context:" in prompt
        assert "before the current frames" not in prompt

    def test_parse_truncated_json_returns_error(self):
        client = VlmClient(server_url="http://fake:8084", model="gemma-4-e2b")
        response = {
            "choices": [{
                "message": {
                    "content": '{"person_present": true, "carrying": "package"'
                }
            }]
        }
        result = client._parse_classification(response)
        assert result.person_present is None
        assert result.error is not None

    def test_build_request_includes_prompt_and_image(self):
        from pathlib import Path
        tmp = Path("/tmp/test-vlm-img.jpg")
        tmp.write_bytes(b"fake-jpeg-data")
        try:
            client = VlmClient(server_url="http://fake:8084", classification_prompt="Test prompt")
            payload = client._build_request(tmp, "Test prompt")
            assert payload["model"] == "gemma-4-e2b"
            assert payload["max_tokens"] == 1600
            assert payload["temperature"] == 0.0
            messages = payload["messages"]
            assert len(messages) == 1
            content = messages[0]["content"]
            assert content[0]["type"] == "image_url"
            assert content[1]["type"] == "text"
            assert content[1]["text"] == "Test prompt"
        finally:
            tmp.unlink(missing_ok=True)

    def test_build_request_multiple_images(self):
        from pathlib import Path
        paths = []
        for i in range(3):
            p = Path(f"/tmp/test-vlm-burst-{i}.jpg")
            p.write_bytes(b"fake-jpeg-data-" + str(i).encode())
            paths.append(p)
        try:
            client = VlmClient(server_url="http://fake:8084", classification_prompt="Analyze sequence")
            payload = client._build_request(paths, "Analyze sequence")
            content = payload["messages"][0]["content"]
            # 3 images + 1 text
            assert len(content) == 4
            for i in range(3):
                assert content[i]["type"] == "image_url"
                assert "data:image/jpeg;base64," in content[i]["image_url"]["url"]
            assert content[3]["type"] == "text"
            assert content[3]["text"] == "Analyze sequence"
        finally:
            for p in paths:
                p.unlink(missing_ok=True)
