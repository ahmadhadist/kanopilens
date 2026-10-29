"""Thin wrapper around Meta Model API (Responses API at https://api.meta.ai/v1).

Docs: https://dev.meta.ai/docs (SAM: /docs/sam/segmenting, /docs/sam/reading-segmentation;
images: /docs/image-understanding; JSON: /docs/structured-output).
"""

import asyncio
import base64
import os
from concurrent.futures import ProcessPoolExecutor
import json
import logging
import multiprocessing
import time

import cv2
import httpx
import numpy as np
from meta_sam_parser import (
    CompletedOutcome,
    SegmentationMaskRecord,
    decode_mask_to_raster,
    image_segmentation_format,
    video_segmentation_format,
)


logger = logging.getLogger(__name__)

RETRY_STATUSES = {429, 500, 502, 503, 504}


class MetaApiNotConfigured(RuntimeError):
    pass


class MetaApiError(RuntimeError):
    def __init__(self, status: int, body: str):
        super().__init__(f"Meta Model API returned {status}: {body[:500]}")
        self.status = status


def _data_url(image_jpeg: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(image_jpeg).decode()


def _output_text(response: dict) -> str:
    return "".join(
        part.get("text", "")
        for item in response.get("output", [])
        for part in item.get("content") or []
        if part.get("type") == "output_text"
    )


# (object id, frame index or None, box (left, top, right, bottom), box-local 0/1 raster)
DecodedMask = tuple[str, int | None, tuple[int, int, int, int], np.ndarray]

_WORKERS = os.cpu_count() or 1
_pool: ProcessPoolExecutor | None = None


def _decode_chunk(text: str, video: bool) -> list[DecodedMask]:
    """Parse and decode SAM output lines. Runs in a worker process: the official
    codec is pure Python and decoding a video's masks takes tens of seconds."""
    parser = (video_segmentation_format() if video else image_segmentation_format()).create_parser()
    parser.push(text)
    decoded = []
    for r in parser.finish(CompletedOutcome()).result.records:
        if isinstance(r, SegmentationMaskRecord):
            raster = np.frombuffer(decode_mask_to_raster(r.mask), np.uint8).reshape(r.mask.height, r.mask.width)
            b = r.bounds
            box = (int(b.left), int(b.top), int(b.right), int(b.bottom))
            decoded.append((r.object_id, r.frame.frame_index if r.frame else None, box, raster))
    return decoded


async def _decode(text: str, video: bool) -> list[DecodedMask]:
    """Each output line is one frame and stands on its own, so lines are spread over all CPUs."""
    global _pool
    if _pool is None:
        # spawn, not fork: forking a process that runs asyncio and gRPC threads can deadlock.
        _pool = ProcessPoolExecutor(max_workers=_WORKERS, mp_context=multiprocessing.get_context("spawn"))
    lines = [line for line in text.splitlines(keepends=True) if line.strip()]
    if not lines:
        return []
    workers = min(len(lines), _WORKERS)
    chunks = ["".join(lines[i::workers]) for i in range(workers)]
    loop = asyncio.get_running_loop()
    parts = await asyncio.gather(*(loop.run_in_executor(_pool, _decode_chunk, chunk, video) for chunk in chunks))
    return [mask for part in parts for mask in part]


def _place(box: tuple[int, int, int, int], raster: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """SAM masks are box-local rasters; scale each into its box on a full-frame canvas."""
    left, top, right, bottom = box
    local = cv2.resize(raster, (right - left, bottom - top), interpolation=cv2.INTER_NEAREST)
    full = np.zeros(shape, bool)
    full[top:bottom, left:right] = local.astype(bool)
    return full


class MetaModelClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        sam_model: str,
        vision_model: str,
        timeout: float = 120.0,
        video_timeout: float = 300.0,
        poll_interval: float = 3.0,
        max_retries: int = 2,
        backoff_s: float = 1.0,
    ):
        if not base_url or not api_key:
            raise MetaApiNotConfigured("META_API_BASE_URL and META_API_KEY must be set")
        self.sam_model = sam_model
        self.vision_model = vision_model
        self.video_timeout = video_timeout
        self.poll_interval = poll_interval
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self._http = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )

    async def _create_response(self, payload: dict) -> dict:
        """POST /responses, retrying rate limits, server errors and network failures."""
        for attempt in range(self.max_retries + 1):
            last = attempt == self.max_retries
            try:
                res = await self._http.post("/responses", json=payload)
            except httpx.TimeoutException as e:
                if last:
                    raise MetaApiError(504, "Meta Model API timed out") from e
                logger.warning("Meta API timeout, retrying", extra={"fields": {"attempt": attempt}})
            except httpx.TransportError as e:
                if last:
                    raise MetaApiError(502, f"Meta Model API unreachable: {e}") from e
                logger.warning("Meta API transport error, retrying", extra={"fields": {"attempt": attempt}})
            else:
                if res.status_code < 400:
                    return res.json()
                if res.status_code not in RETRY_STATUSES or last:
                    raise MetaApiError(res.status_code, res.text)
                logger.warning(
                    "Meta API returned retryable status",
                    extra={"fields": {"attempt": attempt, "status": res.status_code}},
                )
            await asyncio.sleep(self.backoff_s * 2**attempt)
        raise AssertionError("unreachable")

    async def segment(self, image_jpeg: bytes, text_prompt: str, shape: tuple[int, int]) -> list[np.ndarray]:
        """Return one full-frame boolean mask (shape = (H, W)) per instance of `text_prompt`."""
        response = await self._create_response({
            "model": self.sam_model,
            "input": [{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": text_prompt},
                    {"type": "input_image", "image_url": _data_url(image_jpeg)},
                ],
            }],
            "metadata": {"include_confidence": "true"},
        })
        return [_place(box, raster, shape) for _, _, box, raster in await _decode(_output_text(response), video=False)]

    async def upload_file(self, data: bytes, filename: str, content_type: str) -> str:
        """POST /files (purpose user_data); returns the file id for input_video."""
        try:
            res = await self._http.post(
                "/files",
                data={"purpose": "user_data"},
                files={"file": (filename, data, content_type)},
            )
        except httpx.TimeoutException as e:
            raise MetaApiError(504, "Meta Model API timed out uploading the video") from e
        except httpx.TransportError as e:
            raise MetaApiError(502, f"Meta Model API unreachable: {e}") from e
        if res.status_code >= 400:
            raise MetaApiError(res.status_code, res.text)
        return res.json()["id"]

    async def track(self, video_mp4: bytes, text_prompt: str, shape: tuple[int, int]) -> dict[str, dict[int, np.ndarray]]:
        """Track every instance of `text_prompt` through a video.

        Returns object id -> frame index -> full-frame boolean mask (shape = (H, W)).
        """
        file_id = await self.upload_file(video_mp4, "plant.mp4", "video/mp4")
        # A failed or timed-out tracking job is retried once.
        for attempt in range(2):
            try:
                text = await self._track_once(file_id, text_prompt)
                break
            except MetaApiError as e:
                if attempt or e.status not in RETRY_STATUSES:
                    raise
                logger.warning("SAM video tracking failed, retrying", extra={"fields": {"status": e.status}})
        tracks: dict[str, dict[int, np.ndarray]] = {}
        for object_id, frame_index, box, raster in await _decode(text, video=True):
            if frame_index is not None:
                tracks.setdefault(object_id, {})[frame_index] = _place(box, raster, shape)
        return tracks

    async def _track_once(self, file_id: str, text_prompt: str):
        """Run SAM video tracking in background mode and poll for the result.

        A streamed request keeps one connection silent until tracking is done; on
        Cloud Run such connections were reset by the peer after about 2.5 minutes.
        Background mode returns at once and the result is fetched by id.
        """
        created = await self._create_response({
            "model": self.sam_model,
            "input": [{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": text_prompt},
                    {"type": "input_video", "file_id": file_id},
                ],
            }],
            "background": True,
        })
        response_id = created["id"]
        deadline = time.monotonic() + self.video_timeout
        while time.monotonic() < deadline:
            await asyncio.sleep(self.poll_interval)
            try:
                res = await self._http.get(f"/responses/{response_id}")
            except httpx.TransportError:
                continue  # transient; keep polling until the deadline
            if res.status_code == 404 or res.status_code in RETRY_STATUSES:
                continue  # the stored result can briefly lag the job finishing
            if res.status_code >= 400:
                raise MetaApiError(res.status_code, res.text)
            body = res.json()
            status = body.get("status")
            if status == "completed":
                return _output_text(body)
            if status in ("failed", "cancelled", "incomplete"):
                raise MetaApiError(502, f"SAM video tracking {status}: {body.get('error') or body.get('incomplete_details')}")
        raise MetaApiError(504, "Meta Model API timed out tracking the video")

    async def generate_json(
        self, prompt: str, schema_name: str, schema: dict, images: list[bytes] = ()
    ) -> dict:
        """Ask a Muse model (optionally with JPEG images, in order) for schema-conforming JSON."""
        content = [{"type": "input_text", "text": prompt}]
        content += [{"type": "input_image", "image_url": _data_url(image)} for image in images]
        response = await self._create_response({
            "model": self.vision_model,
            "input": [{"role": "user", "content": content}],
            "text": {"format": {"type": "json_schema", "name": schema_name, "schema": schema, "strict": True}},
        })
        text = _output_text(response)
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise MetaApiError(502, f"Model returned invalid JSON: {text[:200]}") from e

    async def aclose(self) -> None:
        await self._http.aclose()
