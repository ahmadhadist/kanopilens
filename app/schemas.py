from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

Severity = Literal["low", "medium", "high"]
IssueCategory = Literal["pest", "disease", "nutrient", "environment", "physical", "other"]
VariegationGrade = Literal["none", "low", "medium", "high", "very_high"]
ArrivalVerdict = Literal["consistent", "minor_changes", "significant_changes"]

DISCLAIMER = (
    "Automated pre-screening only. Not a substitute for an expert inspection "
    "or an official phytosanitary certificate."
)


class HealthIssue(BaseModel):
    name: str
    category: IssueCategory
    severity: Severity
    confidence: float = Field(ge=0, le=1)
    leaf_index: int | None = None
    recommendation: str


class HealthSummary(BaseModel):
    score: int | None = Field(default=None, ge=0, le=100)
    issues: list[HealthIssue] = []
    notes: str = ""
    provider: str


class LeafResult(BaseModel):
    index: int
    bbox: tuple[int, int, int, int]  # x, y, w, h in processed-image (or frame) pixels
    area_px: int
    variegation_pct: float
    frame_index: int | None = None  # video scans: the frame where this leaf was measured


class VariegationSummary(BaseModel):
    weighted_pct: float  # weighted by leaf area
    mean_pct: float
    min_pct: float
    max_pct: float
    grade: VariegationGrade


class CareGuide(BaseModel):
    language: str  # ISO 639-1, e.g. "en"
    country: str  # ISO 3166-1 alpha-2, e.g. "US"
    summary: str
    light: str
    watering: str
    humidity: str
    temperature: str
    substrate: str
    feeding: str
    after_arrival: str
    watch_for: list[str]
    provider: str


class ArrivalDamage(BaseModel):
    type: str
    severity: Severity
    description: str


class VisualComparison(BaseModel):
    """Muse Spark's side-by-side reading of the seller and arrival photos."""

    same_plant: Literal["likely", "uncertain", "unlikely"]
    damage: list[ArrivalDamage] = []
    summary: str
    provider: str


class ArrivalCheck(BaseModel):
    """Comparison of a buyer's arrival scan against the seller's scan before shipping."""

    baseline_id: str
    baseline_created_at: datetime
    days_in_transit: float
    leaf_count_before: int
    leaf_count_after: int
    variegation_before: float | None
    variegation_after: float | None
    health_before: int | None
    health_after: int | None
    new_issues: list[str]
    verdict: ArrivalVerdict
    findings: list[str]
    visual: VisualComparison | None = None


class VideoProof(BaseModel):
    """Evidence that a video scan came from one continuous recording of a real plant."""

    sha256: str  # of the uploaded file, so the original recording can be verified later
    content_type: str
    duration_s: float
    frames_analyzed: int
    fps: float
    leaves_tracked: int
    tracking: str  # e.g. "sam-3.1-video", or "local-keyframe" without Meta


class Scan(BaseModel):
    id: str
    created_at: datetime
    species_hint: str | None = None
    image_width: int
    image_height: int
    leaves: list[LeafResult]
    variegation: VariegationSummary | None
    health: HealthSummary
    segmentation_provider: str
    source: Literal["photo", "video"] = "photo"
    video: VideoProof | None = None
    disclaimer: str = DISCLAIMER
    # Generated on demand and cached, keyed "<language>-<country>".
    care_guides: dict[str, CareGuide] = {}
    # Set when this scan is a buyer's arrival scan of an earlier seller scan.
    arrival: ArrivalCheck | None = None
