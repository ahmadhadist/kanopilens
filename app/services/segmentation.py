from typing import Protocol

import cv2
import numpy as np

from app.services.meta_client import MetaModelClient


class Segmenter(Protocol):
    name: str
    video_tracking: str

    async def segment_leaves(self, rgb: np.ndarray, image_jpeg: bytes) -> list[np.ndarray]: ...

    async def track_leaves(self, frames: list[np.ndarray], video_mp4: bytes) -> dict[str, dict[int, np.ndarray]]:
        """Object id -> frame index -> full-frame mask, for every leaf seen in the video."""
        ...

    async def segment_plants(self, rgb: np.ndarray, image_jpeg: bytes) -> list[np.ndarray]:
        """Whole-plant masks, used to tell the graded plant from the ones behind it."""
        ...

    async def track_plants(self, frames: list[np.ndarray], video_mp4: bytes) -> dict[str, dict[int, np.ndarray]]: ...


class MetaSamSegmenter:
    name = "meta-sam-3.1"
    video_tracking = "sam-3.1-video"

    def __init__(self, client: MetaModelClient):
        self.client = client

    async def segment_leaves(self, rgb: np.ndarray, image_jpeg: bytes) -> list[np.ndarray]:
        return await self.client.segment(image_jpeg, text_prompt="leaf", shape=rgb.shape[:2])

    async def track_leaves(self, frames: list[np.ndarray], video_mp4: bytes) -> dict[str, dict[int, np.ndarray]]:
        return await self.client.track(video_mp4, text_prompt="leaf", shape=frames[0].shape[:2])

    async def segment_plants(self, rgb: np.ndarray, image_jpeg: bytes) -> list[np.ndarray]:
        return await self.client.segment(image_jpeg, text_prompt="potted plant", shape=rgb.shape[:2])

    async def track_plants(self, frames: list[np.ndarray], video_mp4: bytes) -> dict[str, dict[int, np.ndarray]]:
        return await self.client.track(video_mp4, text_prompt="potted plant", shape=frames[0].shape[:2])


class LocalSegmenter:
    """Rough colour-based fallback for development without Meta credits.

    Assumes a plain, non-green background. Large white sectors on the leaf edge
    may be cut off; use SAM 3.1 for real results.
    """

    name = "local-hsv"
    # No tracking: leaves are counted on the single keyframe that shows the most.
    video_tracking = "local-keyframe"
    keyframes = 8

    async def segment_leaves(self, rgb: np.ndarray, image_jpeg: bytes) -> list[np.ndarray]:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        green = ((h >= 30) & (h <= 90) & (s >= 50) & (v >= 40)).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        # Closing bridges small variegated patches inside the green leaf.
        closed = cv2.morphologyEx(green, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        masks = []
        for contour in contours:
            canvas = np.zeros(green.shape, np.uint8)
            cv2.drawContours(canvas, [contour], -1, 1, thickness=cv2.FILLED)
            masks.append(canvas.astype(bool))
        return masks

    async def track_leaves(self, frames: list[np.ndarray], video_mp4: bytes) -> dict[str, dict[int, np.ndarray]]:
        step = max(1, len(frames) // self.keyframes)
        best_index, best_masks = 0, []
        for i in range(0, len(frames), step):
            masks = await self.segment_leaves(frames[i], b"")
            if len(masks) > len(best_masks):
                best_index, best_masks = i, masks
        return {f"k{n}": {best_index: mask} for n, mask in enumerate(best_masks)}

    async def segment_plants(self, rgb: np.ndarray, image_jpeg: bytes) -> list[np.ndarray]:
        return []  # no plant masks: every leaf is kept

    async def track_plants(self, frames: list[np.ndarray], video_mp4: bytes) -> dict[str, dict[int, np.ndarray]]:
        return {}
