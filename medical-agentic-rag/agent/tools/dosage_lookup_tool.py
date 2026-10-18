"""Dosage *reference* retrieval.

This tool retrieves stored reference text only. It deliberately cannot produce a
personalised dose: it performs no arithmetic, applies no patient parameters, and
returns an explicit machine-readable flag
(``personalized_recommendation: false``) that the synthesizer prompt and the
guardrail layer both rely on.
"""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agent.state import Evidence, TrustTier
from agent.tools.base import BaseTool
from config.settings import Settings, get_settings
from stores.structured_store import StructuredStore, build_structured_store

ALLOWED_POPULATIONS = {"adult", "paediatric", "pediatric", "geriatric", "any"}

DEFERRAL_NOTICE = (
    "Reference information only. Individualised dosing depends on indication, "
    "renal and hepatic function, age, weight, comorbidities and co-medication, and "
    "must be determined by a qualified prescriber."
)


class DosageLookupArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    drug: str = Field(min_length=2, max_length=64)
    population: str | None = Field(default=None, max_length=32)
    indication: str | None = Field(default=None, max_length=120)

    @field_validator("drug")
    @classmethod
    def _clean_drug(cls, value: str) -> str:
        cleaned = " ".join(value.strip().split())
        if not cleaned:
            raise ValueError("drug must not be blank")
        return cleaned

    @field_validator("population")
    @classmethod
    def _check_population(cls, value: str | None) -> str | None:
        if value is None:
            return None
        lowered = value.strip().lower()
        if lowered not in ALLOWED_POPULATIONS:
            raise ValueError(f"population must be one of {sorted(ALLOWED_POPULATIONS)}")
        return lowered


class DosageLookupTool(BaseTool):
    """Retrieve stored dosage reference text - never a personalised recommendation."""

    name: ClassVar[str] = "dosage_lookup"
    description: ClassVar[str] = (
        "Retrieve stored dosage REFERENCE information for a drug from the curated "
        "structured store. Returns reference text only and never computes or suggests "
        "a dose for a specific person. Use for general 'what does the reference say' "
        "questions, not for patient-specific dosing decisions."
    )
    args_model: ClassVar[type[BaseModel]] = DosageLookupArgs

    def __init__(
        self, store: StructuredStore | None = None, settings: Settings | None = None
    ) -> None:
        self._settings = settings or get_settings()
        self._store = store

    @property
    def store(self) -> StructuredStore:
        if self._store is None:
            self._store = build_structured_store(self._settings)
        return self._store

    def run(self, args: BaseModel) -> tuple[list[Evidence], dict[str, Any]]:
        assert isinstance(args, DosageLookupArgs)
        rows = self.store.get_dosage_reference(args.drug, args.population, args.indication)

        if not rows:
            return [], {
                "found": False,
                "drug": args.drug,
                "personalized_recommendation": False,
                "data_status": "demo",
                "message": (
                    "No dosage reference record for this drug in the curated store. "
                    "Do not generate or estimate a dose."
                ),
                "deferral_notice": DEFERRAL_NOTICE,
            }

        evidence: list[Evidence] = []
        records: list[dict[str, Any]] = []
        for row in rows:
            evidence.append(
                Evidence(
                    text=(
                        f"Dosage reference for {row['drug']} "
                        f"({row.get('population') or 'unspecified population'}"
                        f"{', ' + row['indication'] if row.get('indication') else ''}): "
                        f"{row['reference_text']}"
                    ),
                    source=row.get("source") or "structured medical database",
                    title=f"Dosage reference: {row['drug']}",
                    url=row.get("source_url"),
                    publication_date=row.get("last_reviewed"),
                    evidence_level="structured_database_record",
                    document_type="dosage_reference",
                    section=row.get("indication"),
                    document_id=f"dosage:{row['drug']}",
                    chunk_id=f"dosage:{row['id']}",
                    score=1.0,
                    trust_tier=TrustTier.UNKNOWN,
                    data_status=row.get("data_status", "demo"),
                    extra={"retrieval": "structured_store", "route": row.get("route")},
                )
            )
            records.append(
                {
                    "drug": row["drug"],
                    "population": row.get("population"),
                    "indication": row.get("indication"),
                    "route": row.get("route"),
                    "reference_text": row["reference_text"],
                    "source": row.get("source"),
                    "data_status": row.get("data_status", "demo"),
                }
            )

        return evidence, {
            "found": True,
            "drug": args.drug,
            "records": records,
            "personalized_recommendation": False,
            "data_status": records[0]["data_status"],
            "deferral_notice": DEFERRAL_NOTICE,
        }
