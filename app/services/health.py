from typing import Protocol

from app.schemas import HealthIssue, HealthSummary
from app.services.meta_client import MetaModelClient

SEVERITY_PENALTY = {"low": 5, "medium": 15, "high": 30}

HEALTH_PROMPT = """You are an expert grower of tropical aroids (Monstera, Philodendron,
Anthurium, Alocasia, Syngonium and relatives).

The image shows a plant{species}. Each segmented leaf is outlined and labelled
"#<index> <variegation%>". There are {leaf_count} labelled leaves.

Inspect the plant for pests, diseases, nutrient deficiencies, environmental
stress and physical damage. Only report problems with visible evidence.
Natural variegation (white, cream, yellow or pink sectors) is NOT a problem.

For each issue give a short name (e.g. "thrips damage"), a confidence from 0
to 1, the leaf index it appears on (null if plant-wide) and one concrete care
action. Put one or two sentences on overall condition in "notes".
Return an empty "issues" list if the plant looks healthy."""

HEALTH_SCHEMA = {
    "type": "object",
    "properties": {
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "category": {"type": "string", "enum": ["pest", "disease", "nutrient", "environment", "physical", "other"]},
                    "severity": {"type": "string", "enum": ["low", "medium", "high"]},
                    "confidence": {"type": "number"},
                    "leaf_index": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                    "recommendation": {"type": "string"},
                },
                "required": ["name", "category", "severity", "confidence", "leaf_index", "recommendation"],
                "additionalProperties": False,
            },
        },
        "notes": {"type": "string"},
    },
    "required": ["issues", "notes"],
    "additionalProperties": False,
}


def score_from_issues(issues: list[HealthIssue]) -> int:
    penalty = sum(SEVERITY_PENALTY[i.severity] * i.confidence for i in issues)
    return max(0, round(100 - penalty))


class HealthAnalyzer(Protocol):
    name: str

    async def analyze(self, overlay_jpeg: bytes, leaf_count: int, species_hint: str | None) -> HealthSummary: ...


class MockHealthAnalyzer:
    name = "mock"

    async def analyze(self, overlay_jpeg: bytes, leaf_count: int, species_hint: str | None) -> HealthSummary:
        return HealthSummary(notes="Health scan not connected to Meta Model API yet.", provider=self.name)


class MetaHealthAnalyzer:
    name = "meta-muse"

    def __init__(self, client: MetaModelClient):
        self.client = client

    async def analyze(self, overlay_jpeg: bytes, leaf_count: int, species_hint: str | None) -> HealthSummary:
        prompt = HEALTH_PROMPT.format(
            species=f" (seller says: {species_hint})" if species_hint else "",
            leaf_count=leaf_count,
        )
        data = await self.client.generate_json(prompt, "plant_health", HEALTH_SCHEMA, images=[overlay_jpeg])
        issues = []
        for raw in data.get("issues", []):
            raw["confidence"] = min(max(float(raw.get("confidence", 0)), 0.0), 1.0)
            issue = HealthIssue.model_validate(raw)
            if issue.leaf_index is not None and not 0 <= issue.leaf_index < leaf_count:
                issue.leaf_index = None
            issues.append(issue)
        return HealthSummary(
            score=score_from_issues(issues),
            issues=issues,
            notes=data.get("notes", ""),
            provider=self.name,
        )
