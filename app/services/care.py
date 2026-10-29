"""Care guides for a scanned plant, localized to the buyer's language and country."""

from typing import Protocol

from app.schemas import CareGuide, Scan
from app.services.meta_client import MetaModelClient

LANGUAGES = {
    "en": "English",
    "id": "Bahasa Indonesia",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Simplified Chinese",
    "th": "Thai",
    "de": "German",
    "nl": "Dutch",
    "fr": "French",
    "es": "Spanish",
}

COUNTRIES = {
    "ID": "Indonesia",
    "US": "United States",
    "CA": "Canada",
    "GB": "United Kingdom",
    "DE": "Germany",
    "NL": "Netherlands",
    "FR": "France",
    "ES": "Spain",
    "JP": "Japan",
    "KR": "South Korea",
    "TW": "Taiwan",
    "SG": "Singapore",
    "MY": "Malaysia",
    "TH": "Thailand",
    "AU": "Australia",
    "AE": "United Arab Emirates",
}

CARE_FIELDS = ["summary", "light", "watering", "humidity", "temperature", "substrate", "feeding", "after_arrival"]

CARE_SCHEMA = {
    "type": "object",
    "properties": {
        **{name: {"type": "string"} for name in CARE_FIELDS},
        "watch_for": {"type": "array", "items": {"type": "string"}},
    },
    "required": [*CARE_FIELDS, "watch_for"],
    "additionalProperties": False,
}

CARE_PROMPT = """You are an expert grower of tropical aroids writing a care guide for a buyer.

Plant: {species}
Variegation: {variegation}
Current condition: {condition}

The buyer lives in {country}. Adapt the advice to typical indoor conditions
there (seasons, heating or air conditioning, humidity, available light).
If the plant is highly variegated, explain how to protect the variegated
tissue. Address every listed health issue.

Write every field in {language}, one to three practical sentences each:
- summary: overall care priorities for this specific plant
- light, watering, humidity, temperature, substrate, feeding
- after_arrival: how to unpack and acclimatize the plant after shipping
- watch_for: three to five short warning signs to monitor"""


def build_prompt(scan: Scan, language: str, country: str) -> str:
    v = scan.variegation
    issues = "; ".join(f"{i.name} ({i.severity})" for i in scan.health.issues)
    return CARE_PROMPT.format(
        species=scan.species_hint or "unspecified aroid",
        variegation=f"{v.weighted_pct:.0f}% of leaf area ({v.grade})" if v else "unknown",
        condition=issues or (scan.health.notes or "no visible issues"),
        country=COUNTRIES[country],
        language=LANGUAGES[language],
    )


class CareGuideWriter(Protocol):
    name: str

    async def write(self, scan: Scan, language: str, country: str) -> CareGuide: ...


class MetaCareGuideWriter:
    name = "meta-muse"

    def __init__(self, client: MetaModelClient):
        self.client = client

    async def write(self, scan: Scan, language: str, country: str) -> CareGuide:
        data = await self.client.generate_json(build_prompt(scan, language, country), "care_guide", CARE_SCHEMA)
        return CareGuide(language=language, country=country, provider=self.name, **data)


class MockCareGuideWriter:
    name = "mock"

    async def write(self, scan: Scan, language: str, country: str) -> CareGuide:
        placeholder = "Care guide not connected to Meta Model API yet."
        return CareGuide(
            language=language,
            country=country,
            provider=self.name,
            watch_for=[],
            **{name: placeholder for name in CARE_FIELDS},
        )
