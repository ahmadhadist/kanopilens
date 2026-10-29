import time
from collections import defaultdict, deque
from functools import lru_cache

from fastapi import Header, HTTPException, Request

from app.config import get_settings
from app.services.arrival import MetaVisualComparer, MockVisualComparer, VisualComparer
from app.services.care import CareGuideWriter, MetaCareGuideWriter, MockCareGuideWriter
from app.services.health import HealthAnalyzer, MetaHealthAnalyzer, MockHealthAnalyzer
from app.services.meta_client import MetaModelClient
from app.services.repository import FirestoreRepository, MemoryRepository, ScanRepository
from app.services.segmentation import LocalSegmenter, MetaSamSegmenter, Segmenter
from app.services.storage import GcsStorage, LocalStorage, Storage


@lru_cache
def get_meta_client() -> MetaModelClient:
    s = get_settings()
    return MetaModelClient(
        s.meta_api_base_url,
        s.meta_api_key,
        s.meta_sam_model,
        s.meta_vision_model,
        timeout=s.meta_timeout_s,
        video_timeout=s.meta_video_timeout_s,
        max_retries=s.meta_max_retries,
    )


@lru_cache
def get_segmenter() -> Segmenter:
    if get_settings().segmentation_provider == "meta":
        return MetaSamSegmenter(get_meta_client())
    return LocalSegmenter()


@lru_cache
def get_health_analyzer() -> HealthAnalyzer:
    if get_settings().health_provider == "meta":
        return MetaHealthAnalyzer(get_meta_client())
    return MockHealthAnalyzer()


@lru_cache
def get_care_writer() -> CareGuideWriter:
    if get_settings().health_provider == "meta":
        return MetaCareGuideWriter(get_meta_client())
    return MockCareGuideWriter()


@lru_cache
def get_visual_comparer() -> VisualComparer:
    if get_settings().health_provider == "meta":
        return MetaVisualComparer(get_meta_client())
    return MockVisualComparer()


@lru_cache
def get_storage() -> Storage:
    s = get_settings()
    return GcsStorage(s.gcs_bucket) if s.storage_backend == "gcs" else LocalStorage(s.local_storage_dir)


@lru_cache
def get_repository() -> ScanRepository:
    s = get_settings()
    return FirestoreRepository(s.firestore_collection) if s.repository_backend == "firestore" else MemoryRepository()


async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    keys = get_settings().api_key_set
    if keys and x_api_key not in keys:
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key")


class RateLimiter:
    """In-memory sliding window, per Cloud Run instance. Enough to protect demo credits."""

    def __init__(self, limit: int, window_s: float):
        self.limit = limit
        self.window_s = window_s
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        hits = self._hits[key]
        while hits and now - hits[0] > self.window_s:
            hits.popleft()
        if len(hits) >= self.limit:
            return False
        hits.append(now)
        return True


def client_ip(request: Request) -> str:
    # Cloud Run appends the real client address last; earlier entries can be spoofed.
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[-1].strip()
    return request.client.host if request.client else "unknown"


@lru_cache
def get_demo_limiter() -> RateLimiter:
    return RateLimiter(get_settings().demo_scans_per_hour, 3600)


@lru_cache
def get_upload_limiter() -> RateLimiter:
    return RateLimiter(max(1, get_settings().demo_scans_per_hour * 2), 3600)


@lru_cache
def get_care_limiter() -> RateLimiter:
    return RateLimiter(get_settings().care_guides_per_hour, 3600)


async def demo_rate_limit(request: Request) -> None:
    if get_settings().demo_scans_per_hour <= 0:
        raise HTTPException(status_code=404, detail="Demo is disabled")
    if not get_demo_limiter().allow(client_ip(request)):
        raise HTTPException(status_code=429, detail="Demo limit reached. Please try again in an hour.")


async def demo_upload_rate_limit(request: Request) -> None:
    if get_settings().demo_scans_per_hour <= 0:
        raise HTTPException(status_code=404, detail="Demo is disabled")
    if not get_upload_limiter().allow(client_ip(request)):
        raise HTTPException(status_code=429, detail="Upload limit reached. Please try again in an hour.")
