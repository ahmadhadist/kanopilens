import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.main import app


def plant_photo() -> bytes:
    bgr = np.full((400, 600, 3), (60, 90, 140), np.uint8)  # brown background (BGR)
    cv2.ellipse(bgr, (160, 200), (100, 80), 0, 0, 360, (50, 130, 40), -1)
    cv2.ellipse(bgr, (420, 200), (110, 90), 0, 0, 360, (50, 130, 40), -1)
    cv2.circle(bgr, (420, 200), 30, (230, 240, 240), -1)  # white patch inside leaf
    ok, buf = cv2.imencode(".jpg", bgr)
    assert ok
    return buf.tobytes()


def upload(client: TestClient, path: str = "/v1/scans", **headers):
    return client.post(
        path,
        files={"image": ("plant.jpg", plant_photo(), "image/jpeg")},
        data={"species_hint": "Monstera 'Thai Constellation'"},
        headers=headers,
    )


def test_scan_roundtrip():
    client = TestClient(app)
    res = upload(client)
    assert res.status_code == 200, res.text
    scan = res.json()
    assert len(scan["leaves"]) == 2
    assert max(leaf["variegation_pct"] for leaf in scan["leaves"]) > 5
    assert scan["health"]["provider"] == "mock"

    assert client.get(f"/v1/scans/{scan['id']}").json()["id"] == scan["id"]
    overlay = client.get(f"/v1/scans/{scan['id']}/overlay.jpg")
    assert overlay.headers["content-type"] == "image/jpeg"
    card = client.get(f"/v1/scans/{scan['id']}/card")
    assert "Thai Constellation" in card.text


def test_rejects_non_image():
    client = TestClient(app)
    res = client.post("/v1/scans", files={"image": ("a.txt", b"hello", "text/plain")})
    assert res.status_code == 415


def test_api_key_required_when_configured(monkeypatch):
    monkeypatch.setenv("API_KEYS", "secret-1,secret-2")
    client = TestClient(app)
    assert upload(client).status_code == 401
    assert upload(client, **{"X-API-Key": "wrong"}).status_code == 401
    assert upload(client, **{"X-API-Key": "secret-2"}).status_code == 200
    # The rate-limited web demo stays open without a key.
    assert upload(client, "/demo/scans").status_code == 200


def test_demo_rate_limit(monkeypatch):
    monkeypatch.setenv("DEMO_SCANS_PER_HOUR", "2")
    client = TestClient(app)
    assert upload(client, "/demo/scans").status_code == 200
    assert upload(client, "/demo/scans").status_code == 200
    assert upload(client, "/demo/scans").status_code == 429
    # Limits are per client IP (last X-Forwarded-For hop on Cloud Run).
    assert upload(client, "/demo/scans", **{"X-Forwarded-For": "1.2.3.4"}).status_code == 200


def test_invalid_scan_id_is_rejected():
    client = TestClient(app)
    assert client.get("/v1/scans/..%2F..%2Fetc").status_code in (404, 422)
    assert client.get("/v1/scans/not-a-scan-id").status_code == 422
    assert client.get(f"/v1/scans/{'0' * 32}").status_code == 404


def test_care_guide_validation_and_mock():
    client = TestClient(app)
    scan_id = upload(client).json()["id"]
    assert client.get(f"/v1/scans/{scan_id}/care-guide?lang=xx").status_code == 422
    assert client.get(f"/v1/scans/{scan_id}/care-guide?country=ZZ").status_code == 422
    res = client.get(f"/v1/scans/{scan_id}/care-guide?lang=id&country=us")
    assert res.status_code == 200
    guide = res.json()
    assert (guide["language"], guide["country"], guide["provider"]) == ("id", "US", "mock")


def test_index_page():
    res = TestClient(app).get("/")
    assert res.status_code == 200
    assert "Kanopi Lens" in res.text
    assert res.headers["x-content-type-options"] == "nosniff"


def test_card_has_link_preview_tags():
    client = TestClient(app)
    scan = upload(client).json()
    card = client.get(f"/v1/scans/{scan['id']}/card").text
    assert 'property="og:title" content="Monstera &#x27;Thai Constellation&#x27;: ' in card
    assert f'property="og:image" content="http://testserver/v1/scans/{scan["id"]}/overlay.jpg"' in card
    assert 'name="twitter:card" content="summary_large_image"' in card
