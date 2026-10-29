from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Meta Model API (https://dev.meta.ai/docs)
    meta_api_base_url: str = "https://api.meta.ai/v1"
    meta_api_key: str = ""
    meta_sam_model: str = "sam-3.1"
    meta_vision_model: str = "muse-spark-1.3"
    meta_timeout_s: float = 120.0
    # SAM video runs in background mode and is polled; clips usually finish in under a minute.
    meta_video_timeout_s: float = 300.0
    meta_max_retries: int = 2

    # "meta" uses Meta Model API; "local" / "mock" allow development without credits.
    segmentation_provider: str = "local"
    health_provider: str = "mock"  # also selects the care guide writer

    storage_backend: str = "local"  # local | gcs
    local_storage_dir: str = "./data"
    gcs_bucket: str = ""

    repository_backend: str = "memory"  # memory | firestore
    firestore_collection: str = "scans"

    # Comma-separated client API keys. Empty disables auth (local demo only).
    api_keys: str = ""
    cors_origins: str = "*"
    # Web demo scans allowed per IP per hour; 0 disables the demo endpoint.
    demo_scans_per_hour: int = 20
    # Newly generated (uncached) care guides allowed per IP per hour.
    care_guides_per_hour: int = 30

    max_upload_mb: int = 15
    # Cloud Run caps HTTP/1 request bodies at 32 MiB; larger files (4K phone video)
    # go straight to Cloud Storage through POST /v1/uploads.
    max_video_mb: int = 30
    max_direct_upload_mb: int = 150
    # SAM video takes roughly 3-4 s per frame, so 4 fps x 15 s (60 frames) keeps a
    # scan to a few minutes. Frames are also what SAM video bills for.
    video_max_seconds: float = 15.0
    video_fps: float = 4.0
    video_max_side: int = 960
    max_image_side: int = 2048
    min_leaf_area_ratio: float = 0.005
    # JSON overrides for variegation.Thresholds; keep calibrated values out of the repo.
    variegation_thresholds: str = ""

    log_level: str = "INFO"
    log_json: bool = False  # true on Cloud Run for structured Cloud Logging

    @property
    def api_key_set(self) -> set[str]:
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}


@lru_cache
def get_settings() -> Settings:
    return Settings()
