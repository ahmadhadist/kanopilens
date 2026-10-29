from datetime import UTC, datetime, timedelta

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.main import app
from app.schemas import (
    ArrivalDamage,
    HealthIssue,
    HealthSummary,
    LeafResult,
    Scan,
    VariegationSummary,
    VisualComparison,
)
from app.services.arrival import compare

T0 = datetime(2026, 11, 1, tzinfo=UTC)


def make_scan(leaves: int, variegation: float | None, score: int | None, issues=(), days: float = 0) -> Scan:
    return Scan(
        id=f"{int(days * 10):032x}",
        created_at=T0 + timedelta(days=days),
        image_width=100,
        image_height=100,
        leaves=[LeafResult(index=i, bbox=(0, 0, 1, 1), area_px=1, variegation_pct=variegation or 0) for i in range(leaves)],
        variegation=VariegationSummary(weighted_pct=variegation, mean_pct=variegation, min_pct=variegation,
                                       max_pct=variegation, grade="medium") if variegation is not None else None,
        health=HealthSummary(score=score, issues=list(issues), provider="test"),
        segmentation_provider="test",
    )


def issue(name: str, severity: str = "medium", category: str = "pest") -> HealthIssue:
    return HealthIssue(name=name, category=category, severity=severity, confidence=0.9, recommendation="treat")


def test_unchanged_plant_is_consistent():
    check = compare(make_scan(6, 30.0, 95), make_scan(6, 31.5, 92, days=4), None)
    assert check.verdict == "consistent"
    assert check.days_in_transit == 4.0
    assert check.findings == ["No significant change detected since the seller's scan."]


def test_one_hidden_leaf_is_minor():
    check = compare(make_scan(8, 30.0, None), make_scan(7, 30.0, None, days=3), None)
    assert check.verdict == "minor_changes"
    assert "1 fewer leaf visible" in check.findings[0]


def test_variegation_shift_flags_possible_swap():
    check = compare(make_scan(5, 45.0, None), make_scan(5, 20.0, None, days=3), None)
    assert check.verdict == "significant_changes"
    assert "Variegation differs by 25.0 points" in check.findings[0]


def test_new_high_severity_issue_and_health_drop():
    before = make_scan(5, 30.0, 95, [issue("Thrips damage", "low")])
    after = make_scan(5, 30.0, 60, [issue("thrips damage", "low"), issue("Root rot", "high")], days=5)
    check = compare(before, after, None)
    assert check.new_issues == ["Root rot"]  # same category, but more severe than before
    assert check.verdict == "significant_changes"
    assert any("Health score dropped from 95 to 60" in f for f in check.findings)


def test_visual_comparison_drives_verdict():
    visual = VisualComparison(
        same_plant="likely",
        damage=[ArrivalDamage(type="torn leaf", severity="medium", description="Leaf #2 torn at the margin.")],
        summary="Same plant; one leaf torn in transit.",
        provider="test",
    )
    check = compare(make_scan(5, 30.0, None), make_scan(5, 30.0, None, days=2), visual)
    assert check.verdict == "significant_changes"
    assert check.visual.summary.startswith("Same plant")

    unlikely = visual.model_copy(update={"same_plant": "unlikely", "damage": []})
    assert compare(make_scan(5, 30.0, None), make_scan(5, 30.0, None), unlikely).verdict == "significant_changes"


def photo(leaves: int) -> bytes:
    bgr = np.full((400, 600, 3), (60, 90, 140), np.uint8)
    for i in range(leaves):
        cv2.ellipse(bgr, (110 + i * 190, 200), (70, 90), 0, 0, 360, (50, 130, 40), -1)
    ok, buf = cv2.imencode(".jpg", bgr)
    return buf.tobytes()


def test_arrival_flow_end_to_end():
    client = TestClient(app)
    files = lambda n: {"image": ("plant.jpg", photo(n), "image/jpeg")}  # noqa: E731

    seller = client.post("/v1/scans", files=files(3), data={"species_hint": "Philodendron Pink Princess"}).json()
    res = client.post(f"/demo/scans/{seller['id']}/arrival", files=files(2))
    assert res.status_code == 200, res.text
    arrival = res.json()
    assert arrival["species_hint"] == "Philodendron Pink Princess"
    check = arrival["arrival"]
    assert (check["baseline_id"], check["leaf_count_before"], check["leaf_count_after"]) == (seller["id"], 3, 2)
    assert check["verdict"] == "significant_changes"  # 1 of 3 leaves is 33% of the plant
    assert check["visual"] is None  # mock provider

    card = client.get(f"/v1/scans/{arrival['id']}/card").text
    assert "Arrival Check" in card and f"/v1/scans/{seller['id']}/overlay.jpg" in card

    # An arrival scan cannot itself be used as a baseline.
    assert client.post(f"/v1/scans/{arrival['id']}/arrival", files=files(2)).status_code == 409
    assert client.post(f"/v1/scans/{'f' * 32}/arrival", files=files(2)).status_code == 404


def test_renamed_issue_is_not_new():
    # Seen live: the model called the same browning "tip burn" before and "necrosis" on arrival.
    before = make_scan(5, 60.0, 96, [issue("white tissue browning / tip burn", "low", "environment")])
    after = make_scan(5, 60.0, 95, [issue("white tissue necrosis", "low", "environment")], days=3)
    check = compare(before, after, None)
    assert check.new_issues == []
    assert check.verdict == "consistent"


def test_small_variegation_shift_is_a_note_once_muse_matches_the_plant():
    # Seen live: the same pothos photographed from another angle read 49.1% then 55.4%.
    same = VisualComparison(same_plant="likely", damage=[], summary="Same plant, no damage.", provider="test")
    check = compare(make_scan(7, 49.1, 100), make_scan(7, 55.4, 100, days=3), same)
    assert check.verdict == "consistent"
    assert "consistent with a different camera angle" in check.findings[0]
    # Without that confirmation the same shift is still flagged.
    assert compare(make_scan(7, 49.1, 100), make_scan(7, 55.4, 100), None).verdict == "minor_changes"
