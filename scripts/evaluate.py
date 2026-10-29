"""Evaluate Kanopi Lens against expert labels, and calibrate variegation thresholds.

    python -m scripts.evaluate run ~/kanopi-eval
    python -m scripts.evaluate calibrate ~/kanopi-eval

The data folder holds the photos or videos plus a labels.csv (see
eval/labels.example.csv). Leave a cell empty when it was not judged:

    file,species,leaves,variegation_pct,notes
    thai_01.jpg,Monstera Thai Constellation,6,32,
    albo_03.mp4,Monstera Albo,9,,walk-around video

`species` matters: pink only counts as variegation for pink cultivars.

`run` segments every labelled file with the configured provider
(SEGMENTATION_PROVIDER in .env) and caches the leaf masks in <data>/.cache, so
SAM is called once per file. It writes <data>/eval/results.csv and summary.json.

`calibrate` searches variegation thresholds on the cached masks only (no API
calls). It reports a cross-validated error, which is an honest estimate for new
plants, and writes <data>/eval/thresholds.json for the VARIEGATION_THRESHOLDS
secret. Keep that file and the data folder out of the public repo.
"""

import argparse
import asyncio
import csv
import itertools
import json
import statistics
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import cv2
import numpy as np

from app.config import Settings, get_settings
from app.deps import get_segmenter
from app.services import grading, imaging, pipeline, variegation, video
from app.services.meta_client import MetaApiError
from app.services.segmentation import Segmenter

PHOTO_EXT = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXT = {".mp4", ".mov"}

# Searched by `calibrate`; every other threshold keeps its current value.
GRID = {
    "white_max_s": [40, 50, 60, 70, 80],
    "white_min_v": [130, 150, 170, 190],
    "yellow_min_s": [50, 70, 90],
    "yellow_max_h": [30, 34, 38],
    "blown_min_v": [256, 250, 240],
}
PIXELS_PER_LEAF = 10_000  # random subsample for calibration; keeps the grid search fast


@dataclass
class Label:
    file: str
    species: str | None
    leaves: int | None
    variegation_pct: float | None


@dataclass
class Sample:
    """Cached segmentation of one file: the frames that contain leaves and one mask per leaf."""

    frames: np.ndarray  # (K, H, W, 3) RGB
    masks: np.ndarray  # (L, H, W) bool
    mask_frame: np.ndarray  # (L,) index into frames

    @property
    def leaf_count(self) -> int:
        return len(self.masks)


def _number(text: str | None, cast):
    text = (text or "").strip()
    return cast(float(text)) if text else None


def read_labels(data_dir: Path) -> list[Label]:
    path = data_dir / "labels.csv"
    if not path.exists():
        sys.exit(f"Missing {path}. Copy eval/labels.example.csv and fill it in.")
    with path.open(newline="", encoding="utf-8-sig") as f:
        return [
            Label(
                row["file"].strip(),
                (row.get("species") or "").strip() or None,
                _number(row.get("leaves"), int),
                _number(row.get("variegation_pct"), float),
            )
            for row in csv.DictReader(f)
            if (row.get("file") or "").strip()
        ]


async def segment_file(path: Path, segmenter: Segmenter, settings: Settings) -> Sample:
    """Segment a photo, or track leaves through a video, exactly like the API does."""
    if path.suffix.lower() in VIDEO_EXT:
        prepared = video.prepare_video(
            path.read_bytes(), settings.video_fps, settings.video_max_seconds, settings.video_max_side
        )
        frames = prepared.frames
        h, w = frames[0].shape[:2]
        tracks = await pipeline.track_video(prepared, segmenter)
        views = video.best_views(tracks, settings.min_leaf_area_ratio * h * w)
        used = sorted({i for i, _ in views})
        remap = {i: n for n, i in enumerate(used)}
        return Sample(
            frames=np.stack([frames[i] for i in used]) if used else np.zeros((0, h, w, 3), np.uint8),
            masks=np.stack([m for _, m in views]) if views else np.zeros((0, h, w), bool),
            mask_frame=np.array([remap[i] for i, _ in views], int),
        )

    rgb = imaging.limit_size(imaging.decode_rgb(path.read_bytes()), settings.max_image_side)
    h, w = rgb.shape[:2]
    masks = await pipeline.segment_photo(rgb, imaging.encode_jpeg(rgb), segmenter, settings)
    return Sample(
        frames=rgb[None],
        masks=np.stack(masks) if masks else np.zeros((0, h, w), bool),
        mask_frame=np.zeros(len(masks), int),
    )


def cache_path(data_dir: Path, file: str, segmenter: Segmenter) -> Path:
    return data_dir / ".cache" / f"{file}.{segmenter.name}.v2.npz"  # v2: main-plant filtering


def load_sample(path: Path) -> Sample:
    with np.load(path) as z:
        return Sample(frames=z["frames"], masks=z["masks"], mask_frame=z["mask_frame"])


def leaf_pixels(sample: Sample, max_pixels: int | None = None, seed: int = 0) -> list[tuple[np.ndarray, ...]]:
    """Per leaf: (h, s, v) of its pixels, optionally subsampled, plus the mask area."""
    rng = np.random.default_rng(seed)
    hsv_frames = [cv2.cvtColor(f, cv2.COLOR_RGB2HSV) for f in sample.frames]
    leaves = []
    for mask, frame_index in zip(sample.masks, sample.mask_frame):
        px = hsv_frames[frame_index][mask]
        if max_pixels and len(px) > max_pixels:
            px = px[rng.choice(len(px), max_pixels, replace=False)]
        leaves.append((px[:, 0], px[:, 1], px[:, 2], int(mask.sum())))
    return leaves


def predicted_pct(leaves: list[tuple[np.ndarray, ...]], t: variegation.Thresholds) -> float | None:
    """Area-weighted plant variegation, computed the same way as variegation.summarize."""
    pcts, areas = [], []
    for h, s, v, area in leaves:
        variegated, valid = variegation.classify_hsv(h, s, v, t)
        counted = int(valid.sum())
        if counted:
            pcts.append(100.0 * int(variegated.sum()) / counted)
            areas.append(area)
    if not pcts:
        return None
    return sum(p * a for p, a in zip(pcts, areas)) / sum(areas)


def error_summary(errors: list[float]) -> dict:
    if not errors:
        return {"n": 0}
    return {
        "n": len(errors),
        "mae": round(statistics.mean(errors), 2),
        "median_ae": round(statistics.median(errors), 2),
        "max_ae": round(max(errors), 2),
        "within_5_pts": round(sum(e <= 5 for e in errors) / len(errors), 3),
    }


def save_overlay(sample: Sample, thresholds: variegation.Thresholds, path: Path) -> None:
    """The same labelled overlay (photo) or leaf gallery (video) the API produces."""
    views = [(int(f), m) for f, m in zip(sample.mask_frame, sample.masks)]
    frames = list(sample.frames)
    leaves = video.measure_views(frames, views, thresholds)
    if len(frames) == 1:
        image = grading.render_overlay(frames[0], list(sample.masks), leaves)
    else:
        image = video.render_gallery(frames, views, leaves)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(imaging.encode_jpeg(image, quality=80))


async def cmd_run(data_dir: Path, overlays: bool = False) -> dict:
    settings = get_settings()
    segmenter = get_segmenter()
    thresholds = variegation.thresholds_from_json(settings.variegation_thresholds)
    if segmenter.name != "meta-sam-3.1":
        print(f"warning: segmenting with {segmenter.name}, not SAM 3.1. Set SEGMENTATION_PROVIDER=meta for real results.")

    rows, var_errors, leaf_errors, grade_hits = [], [], [], []
    for label in read_labels(data_dir):
        path = data_dir / label.file
        if path.suffix.lower() not in PHOTO_EXT | VIDEO_EXT or not path.exists():
            print(f"skip {label.file}: missing or unsupported file")
            continue
        cached = cache_path(data_dir, label.file, segmenter)
        if cached.exists():
            sample = load_sample(cached)
        else:
            print(f"segmenting {label.file} ...")
            try:
                sample = await segment_file(path, segmenter, settings)
            except MetaApiError as e:
                print(f"skip {label.file}: {e}")
                continue
            cached.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cached, frames=sample.frames, masks=sample.masks, mask_frame=sample.mask_frame)

        species_thresholds = variegation.for_species(thresholds, label.species)
        pct = predicted_pct(leaf_pixels(sample), species_thresholds)
        if overlays and sample.leaf_count:
            save_overlay(sample, species_thresholds, data_dir / "eval" / "overlays" / f"{label.file}.jpg")
        row = {
            "file": label.file,
            "leaves_expected": label.leaves,
            "leaves_predicted": sample.leaf_count,
            "variegation_expected": label.variegation_pct,
            "variegation_predicted": None if pct is None else round(pct, 1),
            "abs_error_pts": None,
            "grade_expected": None,
            "grade_predicted": None if pct is None else variegation.grade_for(pct, species_thresholds),
        }
        if label.variegation_pct is not None and pct is not None:
            row["abs_error_pts"] = round(abs(pct - label.variegation_pct), 1)
            row["grade_expected"] = variegation.grade_for(label.variegation_pct, thresholds)
            var_errors.append(abs(pct - label.variegation_pct))
            grade_hits.append(row["grade_expected"] == row["grade_predicted"])
        if label.leaves is not None:
            leaf_errors.append(abs(sample.leaf_count - label.leaves))
        rows.append(row)

    out = data_dir / "eval"
    out.mkdir(exist_ok=True)
    with (out / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["file"])
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "segmentation": segmenter.name,
        "variegation": error_summary(var_errors),
        "grade_agreement": round(sum(grade_hits) / len(grade_hits), 3) if grade_hits else None,
        "leaf_count": {
            "n": len(leaf_errors),
            "exact": round(sum(e == 0 for e in leaf_errors) / len(leaf_errors), 3) if leaf_errors else None,
            "mae": round(statistics.mean(leaf_errors), 2) if leaf_errors else None,
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    for row in rows:
        print(
            f"{row['file']:<28} leaves {row['leaves_predicted']:>3} (expected {row['leaves_expected'] or '-':>3})  "
            f"variegation {row['variegation_predicted'] if row['variegation_predicted'] is not None else '-':>5} "
            f"(expected {row['variegation_expected'] if row['variegation_expected'] is not None else '-':>5})"
        )
    print(json.dumps(summary, indent=2))
    print(f"wrote {out / 'results.csv'} and {out / 'summary.json'}")
    return summary


def cmd_calibrate(data_dir: Path, folds: int = 5) -> dict:
    settings = get_settings()
    segmenter = get_segmenter()
    base = variegation.thresholds_from_json(settings.variegation_thresholds)

    samples, targets, species = [], [], []
    for label in read_labels(data_dir):
        cached = cache_path(data_dir, label.file, segmenter)
        if label.variegation_pct is None or not cached.exists():
            continue
        leaves = leaf_pixels(load_sample(cached), PIXELS_PER_LEAF)
        if leaves:
            samples.append(leaves)
            targets.append(label.variegation_pct)
            species.append(label.species)
    if len(samples) < 3:
        sys.exit("Need at least 3 labelled files with cached masks. Run `run` first.")

    keys = list(GRID)
    combos = [dict(zip(keys, values)) for values in itertools.product(*GRID.values())]
    print(f"searching {len(combos)} threshold combinations on {len(samples)} labelled files ...")
    # errors[c, i]: absolute error of combination c on file i (100 if nothing was measurable).
    errors = np.empty((len(combos), len(samples)))
    for c, combo in enumerate(combos):
        t = replace(base, **combo)
        for i, leaves in enumerate(samples):
            pct = predicted_pct(leaves, variegation.for_species(t, species[i]))
            errors[c, i] = 100.0 if pct is None else abs(pct - targets[i])

    baseline = np.array([
        abs((predicted_pct(s, variegation.for_species(base, sp)) or 0) - y) for s, y, sp in zip(samples, targets, species)
    ])
    best = int(errors.mean(axis=1).argmin())

    # k-fold cross-validation: tune on the other folds, score on the held-out fold.
    k = min(folds, len(samples))
    held_out = np.empty(len(samples))
    for fold in range(k):
        test = np.arange(len(samples)) % k == fold
        chosen = int(errors[:, ~test].mean(axis=1).argmin())
        held_out[test] = errors[chosen, test]

    tuned = replace(base, **combos[best])
    overrides = {key: value for key, value in asdict(tuned).items() if getattr(variegation.DEFAULT_THRESHOLDS, key) != value}
    if "grade_limits" in overrides:
        overrides["grade_limits"] = list(overrides["grade_limits"])
    result = {
        "files": len(samples),
        "baseline_mae": round(float(baseline.mean()), 2),
        "tuned_mae_in_sample": round(float(errors[best].mean()), 2),
        "tuned_mae_cross_validated": round(float(held_out.mean()), 2),
        "thresholds": overrides,
    }
    out = data_dir / "eval"
    out.mkdir(exist_ok=True)
    (out / "thresholds.json").write_text(json.dumps(overrides))
    print(json.dumps(result, indent=2))
    print(f"wrote {out / 'thresholds.json'}. Quote the cross-validated MAE, not the in-sample one.")
    print("Store it as the VARIEGATION_THRESHOLDS secret; do not commit it.")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["run", "calibrate"])
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--overlays", action="store_true", help="run: also save labelled overlays to eval/overlays")
    args = parser.parse_args(argv)
    data_dir = args.data_dir.expanduser()
    if args.command == "run":
        asyncio.run(cmd_run(data_dir, overlays=args.overlays))
    else:
        cmd_calibrate(data_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
