"""Arrival Check: compare a buyer's arrival scan with the seller's scan before shipping.

Metric changes are judged by fixed rules so the verdict is reproducible; the
Muse visual comparison adds plant identity and shipping damage on top.
"""

from typing import Protocol

from app.schemas import ArrivalCheck, ArrivalVerdict, Scan, VisualComparison
from app.services.meta_client import MetaModelClient

# Variegation is fixed tissue and should not change in transit; a large shift
# points to a different plant or very different lighting.
VARIEGATION_MINOR_PTS = 4.0
VARIEGATION_SIGNIFICANT_PTS = 10.0
HEALTH_MINOR_DROP = 8
HEALTH_SIGNIFICANT_DROP = 20

VISUAL_PROMPT = """You are inspecting a rare plant ({species}) that was shipped to a buyer.

Image 1: the seller's photo before shipping.
Image 2: the buyer's photo on arrival.
Leaves are outlined and labelled in both images; the labels are assigned
independently per photo, so #1 in image 1 is not necessarily #1 in image 2.

1. same_plant: is image 2 likely the same individual plant as image 1? Judge by
   leaf shapes, fenestrations, the variegation pattern on each leaf, and leaf
   count and arrangement. Use "uncertain" if the angle makes it impossible to tell.
2. damage: list damage visible in image 2 that is not present in image 1, such
   as torn or creased leaves, broken petioles, bruising, rot, wilting or
   dehydration, yellowing, or cold damage. Ignore differences explained only by
   lighting or camera angle.
3. summary: one or two sentences for the buyer and seller."""

VISUAL_SCHEMA = {
    "type": "object",
    "properties": {
        "same_plant": {"type": "string", "enum": ["likely", "uncertain", "unlikely"]},
        "damage": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string"},
                    "severity": {"type": "string", "enum": ["low", "medium", "high"]},
                    "description": {"type": "string"},
                },
                "required": ["type", "severity", "description"],
                "additionalProperties": False,
            },
        },
        "summary": {"type": "string"},
    },
    "required": ["same_plant", "damage", "summary"],
    "additionalProperties": False,
}

_RANK = {"consistent": 0, "minor_changes": 1, "significant_changes": 2}
_SEVERITY = {"low": 0, "medium": 1, "high": 2}


class VisualComparer(Protocol):
    async def compare(self, before_overlay: bytes, after_overlay: bytes, species: str | None) -> VisualComparison | None: ...


class MetaVisualComparer:
    name = "meta-muse"

    def __init__(self, client: MetaModelClient):
        self.client = client

    async def compare(self, before_overlay: bytes, after_overlay: bytes, species: str | None) -> VisualComparison:
        prompt = VISUAL_PROMPT.format(species=species or "aroid")
        data = await self.client.generate_json(
            prompt, "arrival_comparison", VISUAL_SCHEMA, images=[before_overlay, after_overlay]
        )
        return VisualComparison(provider=self.name, **data)


class MockVisualComparer:
    async def compare(self, before_overlay: bytes, after_overlay: bytes, species: str | None) -> None:
        return None


def compare(before: Scan, after: Scan, visual: VisualComparison | None) -> ArrivalCheck:
    findings: list[tuple[ArrivalVerdict, str]] = []

    lost = len(before.leaves) - len(after.leaves)
    if lost > 0:
        level = "significant_changes" if lost >= 2 or lost >= len(before.leaves) / 4 else "minor_changes"
        findings.append((level, (
            f"{lost} fewer leaf{'' if lost == 1 else 's'} visible ({len(before.leaves)} before, "
            f"{len(after.leaves)} on arrival). A leaf may be hidden by the camera angle; check for detached leaves."
        )))

    var_before = before.variegation.weighted_pct if before.variegation else None
    var_after = after.variegation.weighted_pct if after.variegation else None
    if var_before is not None and var_after is not None:
        shift = abs(var_after - var_before)
        same_plant = visual is not None and visual.same_plant == "likely"
        if shift >= VARIEGATION_SIGNIFICANT_PTS:
            findings.append(("significant_changes", (
                f"Variegation differs by {shift:.1f} points ({var_before:.1f}% before, {var_after:.1f}% on arrival). "
                "Variegation does not change in transit, so this suggests a different plant or very different lighting."
            )))
        elif shift >= VARIEGATION_MINOR_PTS:
            # A photo shows the plant from one side, so a new camera angle shifts the visible
            # variegation a little. Once Muse has matched the plant itself, that is a note, not a change.
            level = "consistent" if same_plant else "minor_changes"
            reason = "consistent with a different camera angle" if same_plant else "check the lighting and camera angle"
            findings.append((level, (
                f"Variegation differs by {shift:.1f} points ({var_before:.1f}% before, {var_after:.1f}% on arrival), {reason}."
            )))

    health_before, health_after = before.health.score, after.health.score
    if health_before is not None and health_after is not None:
        drop = health_before - health_after
        if drop >= HEALTH_MINOR_DROP:
            level = "significant_changes" if drop >= HEALTH_SIGNIFICANT_DROP else "minor_changes"
            findings.append((level, f"Health score dropped from {health_before} to {health_after}."))

    # The model names the same problem differently between scans ("tip burn" vs
    # "necrosis"), so an issue only counts as new when its category was absent
    # before or has become more severe.
    worst_before: dict[str, int] = {}
    for i in before.health.issues:
        worst_before[i.category] = max(worst_before.get(i.category, -1), _SEVERITY[i.severity])
    new = [i for i in after.health.issues if _SEVERITY[i.severity] > worst_before.get(i.category, -1)]
    for issue in new:
        level = "significant_changes" if issue.severity == "high" else "minor_changes"
        findings.append((level, f"New on arrival: {issue.name} ({issue.severity})."))

    if visual is not None:
        if visual.same_plant == "unlikely":
            findings.append(("significant_changes", "The arrival photo does not appear to show the same plant."))
        elif visual.same_plant == "uncertain":
            findings.append(("minor_changes", "Could not confirm it is the same plant; retake from the seller's angle."))
        for d in visual.damage:
            level = "minor_changes" if d.severity == "low" else "significant_changes"
            findings.append((level, f"Shipping damage: {d.type} ({d.severity}). {d.description}"))

    verdict: ArrivalVerdict = max((level for level, _ in findings), key=_RANK.get, default="consistent")
    messages = [text for _, text in findings] or ["No significant change detected since the seller's scan."]

    return ArrivalCheck(
        baseline_id=before.id,
        baseline_created_at=before.created_at,
        days_in_transit=round((after.created_at - before.created_at).total_seconds() / 86400, 1),
        leaf_count_before=len(before.leaves),
        leaf_count_after=len(after.leaves),
        variegation_before=var_before,
        variegation_after=var_after,
        health_before=health_before,
        health_after=health_after,
        new_issues=[i.name for i in new],
        verdict=verdict,
        findings=messages,
        visual=visual,
    )
