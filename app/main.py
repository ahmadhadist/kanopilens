import logging
import re
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response

from app.api import scans
from app.config import get_settings
from app.deps import get_meta_client, get_storage
from app.logging_setup import setup_logging

logger = logging.getLogger(__name__)

INDEX_HTML = Path(__file__).parent / "static" / "index.html"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    setup_logging(settings.log_level, settings.log_json)
    if "meta" in (settings.segmentation_provider, settings.health_provider):
        get_meta_client()  # fail fast on missing credentials instead of on the first scan
    logger.info(
        "Kanopi Lens starting",
        extra={"fields": {
            "segmentation_provider": settings.segmentation_provider,
            "health_provider": settings.health_provider,
            "storage": settings.storage_backend,
            "repository": settings.repository_backend,
            "auth": bool(settings.api_key_set),
        }},
    )
    yield
    if get_meta_client.cache_info().currsize:
        await get_meta_client().aclose()


app = FastAPI(
    title="Kanopi Lens API",
    version="1.0.0",
    description=(
        "Objective health and variegation grading for aroids and other ornamental plants, "
        "built with Meta SAM 3.1 and Muse models."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in get_settings().cors_origins.split(",")],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(scans.router)
app.include_router(scans.uploads_router)
app.include_router(scans.demo_router)


@app.exception_handler(Exception)
async def unhandled_error(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error", extra={"fields": {"path": request.url.path}})
    return JSONResponse({"detail": "Internal server error"}, status_code=500)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    return response


@app.put("/dev/uploads/{name}", include_in_schema=False)
async def dev_upload(name: str, request: Request) -> Response:
    """Stands in for the signed Cloud Storage URL when STORAGE_BACKEND=local."""
    settings = get_settings()
    if settings.storage_backend != "local" or not re.fullmatch(r"[0-9a-f]{32}\.(jpg|png|webp|mp4|mov)", name):
        return Response(status_code=404)
    data = await request.body()
    if len(data) > settings.max_direct_upload_mb * 1024 * 1024:
        return Response(status_code=413)
    get_storage().put(f"uploads/{name}", data, request.headers.get("content-type", ""))
    return Response(status_code=200)


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(INDEX_HTML)


# Not /healthz: Cloud Run reserves some paths ending in "z".
@app.get("/health", include_in_schema=False)
async def healthz() -> dict:
    return {"status": "ok"}
