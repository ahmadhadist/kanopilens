import asyncio
import base64
import json
from datetime import UTC, datetime

import httpx
import numpy as np
import pytest
from meta_sam_parser._mask_codec import _encode_raster

from app.schemas import HealthSummary, Scan
from app.services.arrival import MetaVisualComparer
from app.services.care import MetaCareGuideWriter
from app.services.health import MetaHealthAnalyzer
from app.services.meta_client import MetaApiError, MetaModelClient


def responses_body(text: str) -> dict:
    return {"output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}]}


def client_with(handler) -> MetaModelClient:
    client = MetaModelClient("https://api.meta.ai/v1", "test-key", "sam-3.1", "muse-spark-1.3", backoff_s=0)
    client._http = httpx.AsyncClient(
        base_url="https://api.meta.ai/v1",
        headers={"Authorization": "Bearer test-key"},
        transport=httpx.MockTransport(handler),
    )
    return client


def test_segment_places_box_local_mask_in_frame():
    # 4x2 box-local raster: left half on.
    raster = [1, 1, 0, 0, 1, 1, 0, 0]
    payload = _encode_raster(raster, 4, 2)
    line = f"<0f>7<|box;x1=10;y1=20;x2=49;y2=39;w=100;h=80;c=0.9|><|mask;x=0;y=0;c=0.9;data=2,4,{payload}|>"
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=responses_body(line))

    [mask] = asyncio.run(client_with(handler).segment(b"jpeg", "leaf", (80, 100)))

    assert seen["auth"] == "Bearer test-key"
    assert seen["body"]["model"] == "sam-3.1"
    content = seen["body"]["input"][0]["content"]
    assert content[0] == {"type": "input_text", "text": "leaf"}
    assert content[1]["image_url"].startswith("data:image/jpeg;base64,")

    assert mask.shape == (80, 100)
    ys, xs = np.nonzero(mask)
    assert (xs.min(), xs.max(), ys.min(), ys.max()) == (10, 29, 20, 39)  # left half of the 40x20 box


def test_segment_with_no_matches_returns_empty():
    client = client_with(lambda r: httpx.Response(200, json=responses_body("")))
    assert asyncio.run(client.segment(b"jpeg", "leaf", (80, 100))) == []


def test_health_analyzer_uses_json_schema_and_scores():
    answer = {
        "issues": [
            {"name": "thrips damage", "category": "pest", "severity": "high", "confidence": 1.0,
             "leaf_index": 5, "recommendation": "Isolate and treat with spinosad."},
        ],
        "notes": "One leaf affected.",
    }
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=responses_body(json.dumps(answer)))

    summary = asyncio.run(MetaHealthAnalyzer(client_with(handler)).analyze(b"jpeg", 2, "Monstera"))

    assert seen["body"]["model"] == "muse-spark-1.3"
    assert seen["body"]["text"]["format"]["type"] == "json_schema"
    assert summary.score == 70
    assert summary.issues[0].leaf_index is None  # 5 is out of range for 2 leaves


def test_retries_rate_limit_then_succeeds():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429, text="slow down")
        return httpx.Response(200, json=responses_body('{"ok": true}'))

    result = asyncio.run(client_with(handler).generate_json("hi", "x", {"type": "object"}))
    assert result == {"ok": True}
    assert len(calls) == 2


def test_client_errors_are_not_retried():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(400, text="bad prompt")

    with pytest.raises(MetaApiError) as err:
        asyncio.run(client_with(handler).generate_json("hi", "x", {"type": "object"}))
    assert err.value.status == 400
    assert len(calls) == 1


def test_timeouts_become_504_after_retries():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timeout", request=request)

    with pytest.raises(MetaApiError) as err:
        asyncio.run(client_with(handler).segment(b"jpeg", "leaf", (10, 10)))
    assert err.value.status == 504


def test_care_guide_writer_parses_schema_output():
    answer = {k: f"{k} text" for k in ["summary", "light", "watering", "humidity", "temperature",
                                         "substrate", "feeding", "after_arrival"]}
    answer["watch_for"] = ["yellowing"]
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=responses_body(json.dumps(answer)))

    scan = Scan(
        id="0" * 32, created_at=datetime.now(UTC), species_hint="Monstera albo", image_width=10, image_height=10,
        leaves=[], variegation=None, health=HealthSummary(provider="mock"), segmentation_provider="local",
    )
    guide = asyncio.run(MetaCareGuideWriter(client_with(handler)).write(scan, "ja", "JP"))
    assert guide.watch_for == ["yellowing"]
    prompt = seen["body"]["input"][0]["content"][0]["text"]
    assert "Japanese" in prompt and "Japan" in prompt and "Monstera albo" in prompt
    assert len(seen["body"]["input"][0]["content"]) == 1  # text only, no image


def test_visual_comparer_sends_before_then_after():
    answer = {"same_plant": "likely", "damage": [], "summary": "Same plant, no damage."}
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=responses_body(json.dumps(answer)))

    result = asyncio.run(MetaVisualComparer(client_with(handler)).compare(b"before", b"after", "Monstera"))
    content = seen["body"]["input"][0]["content"]
    assert [c["type"] for c in content] == ["input_text", "input_image", "input_image"]
    assert content[1]["image_url"].endswith(base64.b64encode(b"before").decode())
    assert content[2]["image_url"].endswith(base64.b64encode(b"after").decode())
    assert seen["body"]["text"]["format"]["name"] == "arrival_comparison"
    assert result.same_plant == "likely"
