import logging
import time
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, Path, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from app.config import Settings, get_settings
from app.deps import (
    client_ip,
    demo_rate_limit,
    demo_upload_rate_limit,
    get_care_limiter,
    get_care_writer,
    get_health_analyzer,
    get_repository,
    get_segmenter,
    get_storage,
    get_visual_comparer,
    require_api_key,
)
from app.schemas import CareGuide, Scan
from app.services import arrival
from app.services.arrival import VisualComparer
from app.services.care import COUNTRIES, LANGUAGES, CareGuideWriter
from app.services.grading import render_card_html
from app.services.health import HealthAnalyzer
from app.services.imaging import InvalidImage
from app.services.meta_client import MetaApiError
from app.services.pipeline import run_scan, run_video_scan
from app.services.repository import ScanRepository
from app.services.segmentation import Segmenter
from app.services.storage import Storage
from app.services.video import InvalidVideo

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/scans", tags=["scans"])
uploads_router = APIRouter(prefix="/v1/uploads", tags=["uploads"])
# Same-origin endpoint for the web demo: no API key, rate-limited per IP.
demo_router = APIRouter(prefix="/demo", include_in_schema=False)

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
VIDEO_TYPES = {"video/mp4", "video/quicktime"}
EXTENSIONS = {"image/jpeg": "jpg", "image/png": "png", "image/webp": "webp", "video/mp4": "mp4", "video/quicktime": "mov"}
CONTENT_TYPES = {ext: content_type for content_type, ext in EXTENSIONS.items()}
UPLOAD_TTL_S = 900

ScanId = Annotated[str, Path(pattern=r"^[0-9a-f]{32}$")]
UploadId = Annotated[
    str | None,
    Form(pattern=r"^[0-9a-f]{32}\.(jpg|png|webp|mp4|mov)$", description="From POST /v1/uploads, instead of `image`"),
]
MediaFile = Annotated[
    UploadFile | None,
    File(description="Photo (JPEG/PNG/WebP) or a short video (MP4/MOV, 15 s analysed); or send `upload_id`"),
]
SpeciesHint = Annotated[str | None, Form(max_length=120, description="e.g. Monstera deliciosa 'Thai Constellation'")]


def _upstream_error(e: MetaApiError) -> HTTPException:
    logger.error("Meta API call failed", extra={"fields": {"upstream_status": e.status, "error": str(e)}})
    return HTTPException(504 if e.status == 504 else 502, "Plant analysis service is temporarily unavailable")


class UploadRequest(BaseModel):
    content_type: str
    size: int


class UploadTicket(BaseModel):
    upload_id: str
    upload_url: str
    headers: dict[str, str]
    expires_in: int


def _issue_upload(body: UploadRequest, settings: Settings, storage: Storage) -> UploadTicket:
    ext = EXTENSIONS.get(body.content_type)
    if ext is None:
        raise HTTPException(415, f"Unsupported file type {body.content_type}")
    max_bytes = settings.max_direct_upload_mb * 1024 * 1024
    if not 0 < body.size <= max_bytes:
        raise HTTPException(413, f"Files up to {settings.max_direct_upload_mb} MB are accepted")
    upload_id = f"{uuid4().hex}.{ext}"
    url, headers = storage.upload_url(f"uploads/{upload_id}", body.content_type, max_bytes, UPLOAD_TTL_S)
    return UploadTicket(upload_id=upload_id, upload_url=url, headers=headers, expires_in=UPLOAD_TTL_S)


@uploads_router.post("", response_model=UploadTicket, dependencies=[Depends(require_api_key)])
async def create_upload(
    body: UploadRequest,
    settings: Settings = Depends(get_settings),
    storage: Storage = Depends(get_storage),
) -> UploadTicket:
    """Direct upload for large files (e.g. 4K phone videos): PUT the file to `upload_url`
    with `headers`, then pass `upload_id` to a scan endpoint instead of `image`."""
    return await run_in_threadpool(_issue_upload, body, settings, storage)


@demo_router.post("/uploads", response_model=UploadTicket, dependencies=[Depends(demo_upload_rate_limit)])
async def create_demo_upload(
    body: UploadRequest,
    settings: Settings = Depends(get_settings),
    storage: Storage = Depends(get_storage),
) -> UploadTicket:
    return await run_in_threadpool(_issue_upload, body, settings, storage)


@router.post("", response_model=Scan, dependencies=[Depends(require_api_key)])
async def create_scan(
    image: MediaFile = None,
    upload_id: UploadId = None,
    species_hint: SpeciesHint = None,
    settings: Settings = Depends(get_settings),
    segmenter: Segmenter = Depends(get_segmenter),
    health: HealthAnalyzer = Depends(get_health_analyzer),
    storage: Storage = Depends(get_storage),
    repo: ScanRepository = Depends(get_repository),
) -> Scan:
    return await _perform_scan(image, upload_id, species_hint, settings, segmenter, health, storage, repo)


@demo_router.post("/scans", response_model=Scan, dependencies=[Depends(demo_rate_limit)])
async def create_demo_scan(
    image: MediaFile = None,
    upload_id: UploadId = None,
    species_hint: SpeciesHint = None,
    settings: Settings = Depends(get_settings),
    segmenter: Segmenter = Depends(get_segmenter),
    health: HealthAnalyzer = Depends(get_health_analyzer),
    storage: Storage = Depends(get_storage),
    repo: ScanRepository = Depends(get_repository),
) -> Scan:
    return await _perform_scan(image, upload_id, species_hint, settings, segmenter, health, storage, repo)


@router.post("/{scan_id}/arrival", response_model=Scan, dependencies=[Depends(require_api_key)])
async def create_arrival_scan(
    scan_id: ScanId,
    image: MediaFile = None,
    upload_id: UploadId = None,
    settings: Settings = Depends(get_settings),
    segmenter: Segmenter = Depends(get_segmenter),
    health: HealthAnalyzer = Depends(get_health_analyzer),
    comparer: VisualComparer = Depends(get_visual_comparer),
    storage: Storage = Depends(get_storage),
    repo: ScanRepository = Depends(get_repository),
) -> Scan:
    """Arrival Check: scan the plant on arrival and compare it with the seller's scan `scan_id`."""
    baseline = await _load_baseline(scan_id, repo)
    return await _perform_scan(
        image, upload_id, baseline.species_hint, settings, segmenter, health, storage, repo, baseline, comparer
    )


@demo_router.post("/scans/{scan_id}/arrival", response_model=Scan, dependencies=[Depends(demo_rate_limit)])
async def create_demo_arrival_scan(
    scan_id: ScanId,
    image: MediaFile = None,
    upload_id: UploadId = None,
    settings: Settings = Depends(get_settings),
    segmenter: Segmenter = Depends(get_segmenter),
    health: HealthAnalyzer = Depends(get_health_analyzer),
    comparer: VisualComparer = Depends(get_visual_comparer),
    storage: Storage = Depends(get_storage),
    repo: ScanRepository = Depends(get_repository),
) -> Scan:
    baseline = await _load_baseline(scan_id, repo)
    return await _perform_scan(
        image, upload_id, baseline.species_hint, settings, segmenter, health, storage, repo, baseline, comparer
    )


async def _load_baseline(scan_id: str, repo: ScanRepository) -> Scan:
    baseline = await _load_scan(scan_id, repo)
    if baseline.arrival is not None:
        raise HTTPException(409, "Compare against the seller's original scan, not an arrival scan")
    return baseline


async def _read_media(
    image: UploadFile | None, upload_id: str | None, settings: Settings, storage: Storage
) -> tuple[bytes, str]:
    """The uploaded bytes and content type, sent inline or through a direct upload."""
    if (image is None) == (upload_id is None):
        raise HTTPException(422, "Send exactly one of `image` or `upload_id`")
    if upload_id is not None:
        raw = await run_in_threadpool(storage.get, f"uploads/{upload_id}")
        if raw is None:
            raise HTTPException(404, "Upload not found or not finished")
        return raw, CONTENT_TYPES[upload_id.rsplit(".", 1)[1]]

    content_type = image.content_type or ""
    is_video = content_type in VIDEO_TYPES
    if not is_video and content_type not in ALLOWED_TYPES:
        raise HTTPException(415, f"Unsupported file type {content_type}")
    raw = await image.read()
    limit_mb = settings.max_video_mb if is_video else settings.max_upload_mb
    if len(raw) > limit_mb * 1024 * 1024:
        raise HTTPException(
            413, f"{'Video' if is_video else 'Image'} larger than {limit_mb} MB; use POST /v1/uploads for large files"
        )
    return raw, content_type


async def _perform_scan(
    image: UploadFile | None,
    upload_id: str | None,
    species_hint: str | None,
    settings: Settings,
    segmenter: Segmenter,
    health: HealthAnalyzer,
    storage: Storage,
    repo: ScanRepository,
    baseline: Scan | None = None,
    comparer: VisualComparer | None = None,
) -> Scan:
    raw, content_type = await _read_media(image, upload_id, settings, storage)
    is_video = content_type in VIDEO_TYPES

    started = time.perf_counter()
    try:
        if is_video:
            scan, image_jpeg, overlay_jpeg = await run_video_scan(
                raw, content_type, species_hint or None, segmenter, health, settings
            )
        else:
            scan, image_jpeg, overlay_jpeg = await run_scan(raw, species_hint or None, segmenter, health, settings)
        if baseline is not None:
            before_overlay = await run_in_threadpool(storage.get, f"scans/{baseline.id}/overlay.jpg")
            if before_overlay is None:
                raise HTTPException(409, "The seller's scan images are no longer available")
            visual = await comparer.compare(before_overlay, overlay_jpeg, baseline.species_hint)
            scan.arrival = arrival.compare(baseline, scan, visual)
    except (InvalidImage, InvalidVideo) as e:
        raise HTTPException(400, str(e)) from e
    except MetaApiError as e:
        raise _upstream_error(e) from e

    await run_in_threadpool(storage.put, f"scans/{scan.id}/image.jpg", image_jpeg, "image/jpeg")
    await run_in_threadpool(storage.put, f"scans/{scan.id}/overlay.jpg", overlay_jpeg, "image/jpeg")
    if is_video:
        await run_in_threadpool(storage.put, f"scans/{scan.id}/video", raw, content_type)
    if upload_id is not None:
        await run_in_threadpool(storage.delete, f"uploads/{upload_id}")
    await run_in_threadpool(repo.save, scan)
    logger.info(
        "scan completed",
        extra={"fields": {
            "scan_id": scan.id,
            "leaves": len(scan.leaves),
            "variegation_pct": scan.variegation.weighted_pct if scan.variegation else None,
            "health_score": scan.health.score,
            "segmentation": scan.segmentation_provider,
            "source": scan.source,
            "health_provider": scan.health.provider,
            "arrival_of": baseline.id if baseline else None,
            "arrival_verdict": scan.arrival.verdict if scan.arrival else None,
            "latency_ms": round((time.perf_counter() - started) * 1000),
        }},
    )
    return scan


async def _load_scan(scan_id: str, repo: ScanRepository) -> Scan:
    scan = await run_in_threadpool(repo.get, scan_id)
    if scan is None:
        raise HTTPException(404, "Scan not found")
    return scan


@router.get("/{scan_id}", response_model=Scan)
async def get_scan(scan_id: ScanId, repo: ScanRepository = Depends(get_repository)) -> Scan:
    return await _load_scan(scan_id, repo)


@router.get("/{scan_id}/{kind}.jpg", response_class=Response)
async def get_scan_image(
    scan_id: ScanId,
    kind: str,
    storage: Storage = Depends(get_storage),
) -> Response:
    if kind not in {"image", "overlay"}:
        raise HTTPException(404, "Unknown image")
    data = await run_in_threadpool(storage.get, f"scans/{scan_id}/{kind}.jpg")
    if data is None:
        raise HTTPException(404, "Image not found")
    return Response(data, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=86400, immutable"})


@router.get("/{scan_id}/video", response_class=Response)
async def get_scan_video(
    scan_id: ScanId,
    repo: ScanRepository = Depends(get_repository),
    storage: Storage = Depends(get_storage),
) -> Response:
    """The original recording of a video scan; its SHA-256 matches `video.sha256`."""
    scan = await _load_scan(scan_id, repo)
    if scan.video is None:
        raise HTTPException(404, "Not a video scan")
    data = await run_in_threadpool(storage.get, f"scans/{scan_id}/video")
    if data is None:
        raise HTTPException(404, "Video not found")
    return Response(data, media_type=scan.video.content_type, headers={"Cache-Control": "public, max-age=86400, immutable"})


@router.get("/{scan_id}/card", response_class=HTMLResponse)
async def get_scan_card(
    scan_id: ScanId,
    request: Request,
    repo: ScanRepository = Depends(get_repository),
) -> HTMLResponse:
    scan = await _load_scan(scan_id, repo)
    overlay_url = str(request.url_for("get_scan_image", scan_id=scan.id, kind="overlay"))
    baseline_url = (
        str(request.url_for("get_scan_image", scan_id=scan.arrival.baseline_id, kind="overlay"))
        if scan.arrival
        else None
    )
    video_url = str(request.url_for("get_scan_video", scan_id=scan.id)) if scan.video else None
    return HTMLResponse(render_card_html(scan, overlay_url, baseline_url, video_url))


@router.get("/{scan_id}/care-guide", response_model=CareGuide)
async def get_care_guide(
    scan_id: ScanId,
    request: Request,
    lang: str = Query("en", description=f"One of: {', '.join(LANGUAGES)}"),
    country: str = Query("US", description=f"One of: {', '.join(COUNTRIES)}"),
    repo: ScanRepository = Depends(get_repository),
    writer: CareGuideWriter = Depends(get_care_writer),
) -> CareGuide:
    """Care guide for the scanned plant in the buyer's language, adapted to their country.

    Generated once per scan, language and country, then served from cache.
    """
    if lang not in LANGUAGES:
        raise HTTPException(422, f"Unsupported language {lang!r}")
    country = country.upper()
    if country not in COUNTRIES:
        raise HTTPException(422, f"Unsupported country {country!r}")

    scan = await _load_scan(scan_id, repo)
    key = f"{lang}-{country}"
    if key in scan.care_guides:
        return scan.care_guides[key]

    if not get_care_limiter().allow(client_ip(request)):
        raise HTTPException(429, "Care guide limit reached. Please try again in an hour.")
    try:
        guide = await writer.write(scan, lang, country)
    except MetaApiError as e:
        raise _upstream_error(e) from e
    if guide.provider != "mock":
        scan.care_guides[key] = guide
        await run_in_threadpool(repo.save, scan)
    return guide
