import asyncio
import hashlib
from datetime import UTC, datetime
from uuid import uuid4

import numpy as np

from app.config import Settings
from app.schemas import Scan, VideoProof
from app.services import focus, grading, imaging, variegation, video
from app.services.health import HealthAnalyzer
from app.services.segmentation import Segmenter


async def segment_photo(
    rgb: np.ndarray, image_jpeg: bytes, segmenter: Segmenter, settings: Settings
) -> list[np.ndarray]:
    """Leaf masks of the plant being graded; leaves and whole plants are segmented concurrently."""
    height, width = rgb.shape[:2]
    leaves, plants = await asyncio.gather(
        segmenter.segment_leaves(rgb, image_jpeg), segmenter.segment_plants(rgb, image_jpeg)
    )
    min_area = settings.min_leaf_area_ratio * height * width
    leaves = [m for m in (imaging.fit_mask(m, (height, width)) for m in leaves) if m.sum() >= min_area]
    plants = [imaging.fit_mask(m, (height, width)) for m in plants]
    return focus.filter_leaves(leaves, plants)


async def track_video(
    prepared: video.PreparedVideo, segmenter: Segmenter
) -> dict[str, dict[int, np.ndarray]]:
    """Leaf tracks of the plant being graded; leaves and whole plants are tracked concurrently."""
    frames = prepared.frames
    height, width = frames[0].shape[:2]
    raw_leaves, raw_plants = await asyncio.gather(
        segmenter.track_leaves(frames, prepared.mp4), segmenter.track_plants(frames, prepared.mp4)
    )

    def clean(raw):
        tracks = {
            oid: {i: imaging.fit_mask(m, (height, width)) for i, m in by_frame.items() if 0 <= i < len(frames)}
            for oid, by_frame in raw.items()
        }
        return {oid: by_frame for oid, by_frame in tracks.items() if by_frame}

    leaves = clean(raw_leaves)
    return focus.filter_tracks(leaves, clean(raw_plants), video.poster_frame(frames, leaves))


async def run_scan(
    raw: bytes,
    species_hint: str | None,
    segmenter: Segmenter,
    health: HealthAnalyzer,
    settings: Settings,
) -> tuple[Scan, bytes, bytes]:
    """Return the scan plus the processed image and overlay as JPEG bytes."""
    rgb = imaging.limit_size(imaging.decode_rgb(raw), settings.max_image_side)
    image_jpeg = imaging.encode_jpeg(rgb)
    height, width = rgb.shape[:2]

    masks = await segment_photo(rgb, image_jpeg, segmenter, settings)

    thresholds = variegation.for_species(variegation.thresholds_from_json(settings.variegation_thresholds), species_hint)
    leaves = variegation.measure_leaves(rgb, masks, thresholds)
    overlay_jpeg = imaging.encode_jpeg(grading.render_overlay(rgb, masks, leaves))
    health_summary = await health.analyze(overlay_jpeg, len(leaves), species_hint)

    scan = Scan(
        id=uuid4().hex,
        created_at=datetime.now(UTC),
        species_hint=species_hint,
        image_width=width,
        image_height=height,
        leaves=leaves,
        variegation=variegation.summarize(leaves, thresholds),
        health=health_summary,
        segmentation_provider=segmenter.name,
    )
    return scan, image_jpeg, overlay_jpeg


async def run_video_scan(
    raw: bytes,
    content_type: str,
    species_hint: str | None,
    segmenter: Segmenter,
    health: HealthAnalyzer,
    settings: Settings,
) -> tuple[Scan, bytes, bytes]:
    """Proof of Plant: track every leaf through a short video of the plant.

    Returns the scan plus a poster frame and the per-leaf gallery as JPEG bytes.
    """
    prepared = await asyncio.to_thread(
        video.prepare_video, raw, settings.video_fps, settings.video_max_seconds, settings.video_max_side
    )
    frames = prepared.frames
    height, width = frames[0].shape[:2]

    tracks = await track_video(prepared, segmenter)

    thresholds = variegation.for_species(variegation.thresholds_from_json(settings.variegation_thresholds), species_hint)
    views = video.best_views(tracks, settings.min_leaf_area_ratio * height * width)
    leaves = video.measure_views(frames, views, thresholds)
    gallery_jpeg = imaging.encode_jpeg(video.render_gallery(frames, views, leaves))
    poster_jpeg = imaging.encode_jpeg(frames[video.poster_frame(frames, tracks)])
    health_summary = await health.analyze(gallery_jpeg, len(leaves), species_hint)

    scan = Scan(
        id=uuid4().hex,
        created_at=datetime.now(UTC),
        species_hint=species_hint,
        image_width=width,
        image_height=height,
        leaves=leaves,
        variegation=variegation.summarize(leaves, thresholds),
        health=health_summary,
        segmentation_provider=segmenter.name,
        source="video",
        video=VideoProof(
            sha256=hashlib.sha256(raw).hexdigest(),
            content_type=content_type,
            duration_s=prepared.duration_s,
            frames_analyzed=len(frames),
            fps=prepared.fps,
            leaves_tracked=len(leaves),
            tracking=segmenter.video_tracking,
        ),
    )
    return scan, poster_jpeg, gallery_jpeg
