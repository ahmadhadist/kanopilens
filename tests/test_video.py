import asyncio
import hashlib
import json
import os
import tempfile

import cv2
import httpx
import numpy as np
from fastapi.testclient import TestClient
from meta_sam_parser._mask_codec import _encode_raster

from app.main import app
from app.services.meta_client import MetaApiError, MetaModelClient
from app.services.video import best_views, prepare_video


def plant_video(seconds: float = 3, fps: int = 30, leaves: int = 3) -> bytes:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "plant.mp4")
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (640, 480))
        for f in range(int(seconds * fps)):
            bgr = np.full((480, 640, 3), (60, 90, 140), np.uint8)
            for i in range(leaves):
                center = (110 + i * 210 + f // 10, 240)  # slow pan, like walking around the plant
                cv2.ellipse(bgr, center, (70, 110), 0, 0, 360, (50, 130, 40), -1)
                cv2.ellipse(bgr, center, (25, 45), 0, 0, 360, (230, 240, 240), -1)
            writer.write(bgr)
        writer.release()
        with open(path, "rb") as fh:
            return fh.read()


def test_prepare_video_resamples_and_reencodes():
    prepared = prepare_video(plant_video(seconds=3, fps=30), target_fps=6, max_seconds=20, max_side=320)
    assert len(prepared.frames) == 18  # every 5th of 90 frames
    assert prepared.fps == 6.0 and prepared.duration_s == 3.0
    assert prepared.frames[0].shape == (240, 320, 3)
    with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
        f.write(prepared.mp4)
        f.flush()
        assert int(cv2.VideoCapture(f.name).get(cv2.CAP_PROP_FRAME_COUNT)) == 18


def test_prepare_video_caps_duration():
    prepared = prepare_video(plant_video(seconds=3, fps=30), target_fps=6, max_seconds=1, max_side=640)
    assert len(prepared.frames) == 6
    assert prepared.duration_s == 1.0


def test_best_view_is_largest_appearance():
    small = np.zeros((10, 10), bool)
    small[0:2, 0:2] = True
    large = np.zeros((10, 10), bool)
    large[0:6, 0:6] = True
    [(frame_index, mask)] = best_views({"7": {0: small, 4: large, 9: small}}, min_area=1)
    assert frame_index == 4 and mask.sum() == 36


def test_video_scan_end_to_end():
    client = TestClient(app)
    raw = plant_video()
    res = client.post(
        "/demo/scans",
        files={"image": ("plant.mp4", raw, "video/mp4")},
        data={"species_hint": "Monstera Thai Constellation"},
    )
    assert res.status_code == 200, res.text
    scan = res.json()
    assert scan["source"] == "video"
    proof = scan["video"]
    assert proof["sha256"] == hashlib.sha256(raw).hexdigest()
    assert proof["leaves_tracked"] == 3 and proof["tracking"] == "local-keyframe"
    assert all(leaf["frame_index"] is not None for leaf in scan["leaves"])
    assert 10 < scan["variegation"]["weighted_pct"] < 25

    video = client.get(f"/v1/scans/{scan['id']}/video")
    assert video.headers["content-type"] == "video/mp4" and video.content == raw
    assert client.get(f"/v1/scans/{scan['id']}/overlay.jpg").status_code == 200
    assert "Proof of Plant" in client.get(f"/v1/scans/{scan['id']}/card").text

    # Video arrival scans compare tracked leaf counts against the seller's video.
    arrival = client.post(
        f"/demo/scans/{scan['id']}/arrival", files={"image": ("arrival.mp4", plant_video(leaves=2), "video/mp4")}
    ).json()["arrival"]
    assert (arrival["leaf_count_before"], arrival["leaf_count_after"]) == (3, 2)


def test_photo_scan_has_no_video():
    client = TestClient(app)
    ok, jpg = cv2.imencode(".jpg", np.full((100, 100, 3), 128, np.uint8))
    scan = client.post("/demo/scans", files={"image": ("p.jpg", jpg.tobytes(), "image/jpeg")}).json()
    assert scan["source"] == "photo"
    assert client.get(f"/v1/scans/{scan['id']}/video").status_code == 404


def test_rejects_undecodable_video():
    res = TestClient(app).post("/demo/scans", files={"image": ("x.mp4", b"not a video", "video/mp4")})
    assert res.status_code == 400


def meta_client(handler) -> MetaModelClient:
    client = MetaModelClient("https://api.meta.ai/v1", "k", "sam-3.1", "muse-spark-1.3", poll_interval=0, video_timeout=5)
    client._http = httpx.AsyncClient(base_url="https://api.meta.ai/v1", transport=httpx.MockTransport(handler))
    return client


def test_meta_track_runs_in_background_and_groups_masks_by_object():
    payload = _encode_raster([1] * 8, 4, 2)
    text = (
        f"<0f>3<|box;x1=0;y1=0;x2=3;y2=1;w=10;h=8|><|mask;x=0;y=0;data=2,4,{payload}|>"
        f",9<|box;x1=5;y1=5;x2=8;y2=6;w=10;h=8|><|mask;x=0;y=0;data=2,4,{payload}|>\n"
        f"<2f>3<|box;x1=1;y1=1;x2=4;y2=2;w=10;h=8|><|mask;x=0;y=0;data=2,4,{payload}|>\n"
    )
    polls = iter([
        httpx.Response(404),  # stored result can lag behind the job
        httpx.Response(200, json={"id": "resp_1", "status": "in_progress"}),
        httpx.Response(200, json={"id": "resp_1", "status": "completed",
                                  "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}]}),
    ])
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/files"):
            seen["upload"] = request.content
            return httpx.Response(200, json={"id": "file_123"})
        if request.method == "POST":
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json={"id": "resp_1", "status": "queued", "background": True})
        assert request.url.path.endswith("/responses/resp_1")
        return next(polls)

    tracks = asyncio.run(meta_client(handler).track(b"mp4-bytes", "leaf", (8, 10)))

    assert b"user_data" in seen["upload"] and b"mp4-bytes" in seen["upload"]
    assert seen["body"]["background"] is True and "stream" not in seen["body"]
    assert seen["body"]["input"][0]["content"][1] == {"type": "input_video", "file_id": "file_123"}
    assert sorted(tracks) == ["3", "9"]
    assert sorted(tracks["3"]) == [0, 2]  # object 3 tracked across frames 0 and 2
    assert tracks["3"][2].shape == (8, 10) and tracks["3"][2][1:3, 1:5].all()


def test_large_video_goes_through_direct_upload(tmp_path):
    client = TestClient(app)
    raw = plant_video()

    ticket = client.post("/demo/uploads", json={"content_type": "video/mp4", "size": len(raw)}).json()
    assert ticket["upload_id"].endswith(".mp4") and ticket["upload_url"].startswith("/dev/uploads/")
    assert client.put(ticket["upload_url"], content=raw, headers=ticket["headers"]).status_code == 200

    res = client.post("/demo/scans", data={"upload_id": ticket["upload_id"], "species_hint": "Monstera"})
    assert res.status_code == 200, res.text
    scan = res.json()
    assert scan["source"] == "video" and scan["video"]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert client.get(f"/v1/scans/{scan['id']}/video").content == raw
    assert not (tmp_path / "uploads" / ticket["upload_id"]).exists()  # consumed upload is deleted
    # An upload can be used once.
    assert client.post("/demo/scans", data={"upload_id": ticket["upload_id"]}).status_code == 404


def test_upload_validation():
    client = TestClient(app)
    assert client.post("/demo/uploads", json={"content_type": "video/webm", "size": 10}).status_code == 415
    assert client.post("/demo/uploads", json={"content_type": "video/mp4", "size": 151 * 1024 * 1024}).status_code == 413
    assert client.post("/demo/scans", data={"upload_id": "../../etc/passwd"}).status_code == 422
    assert client.post("/demo/scans").status_code == 422  # neither image nor upload_id
    assert client.put(f"/dev/uploads/{'a' * 32}.exe", content=b"x").status_code == 404


def test_failed_tracking_job_is_retried_once():
    jobs = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/files"):
            return httpx.Response(200, json={"id": "file_1"})
        if request.method == "POST":
            jobs.append(1)
            return httpx.Response(200, json={"id": f"resp_{len(jobs)}", "status": "queued"})
        return httpx.Response(200, json={"status": "failed", "error": {"message": "internal"}})

    try:
        asyncio.run(meta_client(handler).track(b"mp4", "leaf", (8, 10)))
    except MetaApiError as e:
        assert e.status == 502 and "failed" in str(e)
    else:
        raise AssertionError("expected MetaApiError")
    assert len(jobs) == 2


def test_tracking_that_never_finishes_reports_504():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/files"):
            return httpx.Response(200, json={"id": "file_1"})
        if request.method == "POST":
            return httpx.Response(200, json={"id": "resp_1", "status": "queued"})
        return httpx.Response(200, json={"status": "in_progress"})

    client = meta_client(handler)
    client.video_timeout = 0.05
    try:
        asyncio.run(client.track(b"mp4", "leaf", (8, 10)))
    except MetaApiError as e:
        assert e.status == 504
    else:
        raise AssertionError("expected MetaApiError")
