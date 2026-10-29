"""Video preparation and per-leaf selection for "Proof of Plant" video scans.

The upload is resampled once into the exact frames we analyse; the same frames
are re-encoded and sent to SAM, so tracked frame indices line up with the
pixels we measure.
"""

import math
import os
import tempfile
from dataclasses import dataclass

import cv2
import numpy as np

from app.schemas import LeafResult
from app.services import variegation
from app.services.grading import PALETTE

GALLERY_TILE = 320
GALLERY_COLUMNS = 3


class InvalidVideo(ValueError):
    pass


@dataclass
class PreparedVideo:
    frames: list[np.ndarray]  # RGB, all the same size
    fps: float
    duration_s: float
    mp4: bytes  # the resampled frames, MPEG-4 Part 2 in MP4 (accepted by SAM)


# Tracks: object id -> frame index -> full-frame boolean mask.
Tracks = dict[str, dict[int, np.ndarray]]


def prepare_video(raw: bytes, target_fps: float, max_seconds: float, max_side: int) -> PreparedVideo:
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "upload")
        with open(src, "wb") as f:
            f.write(raw)
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise InvalidVideo("Could not decode video")
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, round(src_fps / target_fps))
        max_source_frames = int(max_seconds * src_fps)

        frames = []
        index = 0
        while index < max_source_frames:
            if index % step:
                # grab() skips decoding, which matters for 4K phone video.
                if not cap.grab():
                    break
                index += 1
                continue
            ok, bgr = cap.read()
            if not ok:
                break
            h, w = bgr.shape[:2]
            scale = min(1.0, max_side / max(h, w))
            if scale < 1:
                bgr = cv2.resize(bgr, (round(w * scale) // 2 * 2, round(h * scale) // 2 * 2), interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            index += 1
        cap.release()
        if len(frames) < 2:
            raise InvalidVideo("Video is too short or could not be decoded")

        fps = src_fps / step
        out = os.path.join(tmp, "frames.mp4")
        h, w = frames[0].shape[:2]
        writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        with open(out, "rb") as f:
            mp4 = f.read()

    return PreparedVideo(frames=frames, fps=round(fps, 2), duration_s=round(index / src_fps, 1), mp4=mp4)


def best_views(tracks: Tracks, min_area: float) -> list[tuple[int, np.ndarray]]:
    """For each tracked leaf, the frame where it appears largest (closest to a frontal view)."""
    views = []
    for _, by_frame in sorted(tracks.items()):
        frame_index, mask = max(by_frame.items(), key=lambda item: int(item[1].sum()))
        if mask.sum() >= min_area:
            views.append((frame_index, mask))
    return views


def measure_views(
    frames: list[np.ndarray], views: list[tuple[int, np.ndarray]], t: variegation.Thresholds
) -> list[LeafResult]:
    """LeafResult.index is the position in `views`."""
    leaves = []
    for i, (frame_index, mask) in enumerate(views):
        measured = variegation.measure_leaves(frames[frame_index], [mask], t)
        if measured:
            leaves.append(measured[0].model_copy(update={"index": i, "frame_index": frame_index}))
    return leaves


def render_gallery(frames: list[np.ndarray], views: list[tuple[int, np.ndarray]], leaves: list[LeafResult]) -> np.ndarray:
    """One labelled tile per tracked leaf, cropped from its best frame."""
    if not leaves:
        return frames[len(frames) // 2]
    rows = math.ceil(len(leaves) / GALLERY_COLUMNS)
    cols = min(len(leaves), GALLERY_COLUMNS)
    canvas = np.full((rows * GALLERY_TILE, cols * GALLERY_TILE, 3), 24, np.uint8)
    for n, leaf in enumerate(leaves):
        frame_index, mask = views[leaf.index]
        frame = frames[frame_index].copy()
        frame[~mask] = (frame[~mask] * 0.3).astype(np.uint8)  # keep context, but make the tracked leaf stand out
        color = PALETTE[leaf.index % len(PALETTE)]
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours, -1, color, 3)

        x, y, w, h = leaf.bbox
        side = int(max(w, h) * 1.15)
        cx, cy = x + w // 2, y + h // 2
        fh, fw = frame.shape[:2]
        x0, y0 = max(0, cx - side // 2), max(0, cy - side // 2)
        crop = frame[y0:min(fh, y0 + side), x0:min(fw, x0 + side)]
        scale = GALLERY_TILE / max(crop.shape[:2])
        tile = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale))))

        r, c = divmod(n, GALLERY_COLUMNS)
        oy = r * GALLERY_TILE + (GALLERY_TILE - tile.shape[0]) // 2
        ox = c * GALLERY_TILE + (GALLERY_TILE - tile.shape[1]) // 2
        canvas[oy:oy + tile.shape[0], ox:ox + tile.shape[1]] = tile
        label = f"#{leaf.index} {leaf.variegation_pct:.0f}%"
        origin = (c * GALLERY_TILE + 10, r * GALLERY_TILE + 28)
        cv2.putText(canvas, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(canvas, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def poster_frame(frames: list[np.ndarray], tracks: Tracks) -> int:
    """The frame showing the most tracked leaves, used as the scan's main image."""
    counts = [0] * len(frames)
    for by_frame in tracks.values():
        for frame_index in by_frame:
            if 0 <= frame_index < len(counts):
                counts[frame_index] += 1
    return max(range(len(frames)), key=lambda i: counts[i]) if any(counts) else len(frames) // 2
