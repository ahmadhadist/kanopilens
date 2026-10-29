import pytest

from app import deps
from app.config import get_settings

CACHED = [
    get_settings,
    deps.get_meta_client,
    deps.get_segmenter,
    deps.get_health_analyzer,
    deps.get_care_writer,
    deps.get_visual_comparer,
    deps.get_storage,
    deps.get_repository,
    deps.get_demo_limiter,
    deps.get_care_limiter,
]


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    # Environment variables override .env, so tests never hit the real Meta API.
    monkeypatch.setenv("SEGMENTATION_PROVIDER", "local")
    monkeypatch.setenv("HEALTH_PROVIDER", "mock")
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    monkeypatch.setenv("REPOSITORY_BACKEND", "memory")
    monkeypatch.setenv("LOCAL_STORAGE_DIR", str(tmp_path))
    monkeypatch.setenv("API_KEYS", "")
    monkeypatch.setenv("DEMO_SCANS_PER_HOUR", "20")
    monkeypatch.setenv("VARIEGATION_THRESHOLDS", "")
    for fn in CACHED:
        fn.cache_clear()
    yield
    for fn in CACHED:
        fn.cache_clear()
