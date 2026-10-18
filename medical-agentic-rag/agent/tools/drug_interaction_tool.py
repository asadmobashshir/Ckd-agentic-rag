"""Structured pairwise drug-interaction lookup.

Hard rule: a miss is reported as ``found: false``. Absence of a row means
*absence from this database*, never "no interaction exists", and the downstream
prompt states that explicitly so the model cannot upgrade a miss into a claim.
"""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agent.state import Evidence, TrustTier
from agent.tools.base import BaseTool
from config.settings import Settings, get_settings
from stores.structured_store import StructuredStore, build_structured_store

_DRUG_NAME_MAX = 64


class DrugInteractionArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    drug_a: str = Field(min_length=2, max_length=_DRUG_NAME_MAX)
    drug_b: str = Field(min_length=2, max_length=_DRUG_NAME_MAX)

    @field_validator("drug_a", "drug_b")
    @classmethod
    def _clean(cls, value: str) -> str:
        cleaned = " ".join(value.strip().split())
        if not cleaned:
            raise ValueError("drug name must not be blank")
        if any(ch in cleaned for ch in ";\n\r\t"):
            raise ValueError("drug name contains illegal characters")
        return cleaned


class DrugInteractionTool(BaseTool):
    """Look up a documented interaction between exactly two drugs."""

    name: ClassVar[str] = "drug_interaction"
    description: ClassVar[str] = (
        "Look up a documented interaction between exactly two named drugs in the "
        "structured medical database. Use whenever a question involves two or more "
        "medicines taken together. Returns found=false when the pair is not present "
        "in the database - which does NOT mean the combination is safe."
    )
    args_model: ClassVar[type[BaseModel]] = DrugInteractionArgs

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
        assert isinstance(args, DrugInteractionArgs)
        row = self.store.find_interaction(args.drug_a, args.drug_b)

        if row is None:
            known_a = self.store.get_drug(args.drug_a) is not None
            known_b = self.store.get_drug(args.drug_b) is not None
            return [], {
                "found": False,
                "drug_a": args.drug_a,
                "drug_b": args.drug_b,
                "drug_a_in_database": known_a,
                "drug_b_in_database": known_b,
                "data_status": "demo",
                "message": (
                    "No interaction record exists in this database for this pair. "
                    "This is an absence of data, NOT evidence that the combination is "
                    "safe. Do not state or imply that no interaction exists."
                ),
            }

        text = (
            f"Interaction between {row['drug_a']} and {row['drug_b']}: {row['interaction']}"
            f"\nSeverity: {row.get('severity') or 'not recorded'}"
            f"\nMechanism: {row.get('mechanism') or 'not recorded'}"
        )
        evidence = Evidence(
            text=text,
            source=row.get("source") or "structured medical database",
            title=f"Drug interaction record: {row['drug_a']} + {row['drug_b']}",
            url=row.get("source_url"),
            publication_date=row.get("last_reviewed"),
            evidence_level="structured_database_record",
            document_type="structured_record",
            document_id=f"interaction:{row['drug_a']}+{row['drug_b']}",
            chunk_id=f"interaction:{row['id']}",
            score=1.0,
            trust_tier=TrustTier.UNKNOWN,
            data_status=row.get("data_status", "demo"),
            extra={"retrieval": "structured_store", "severity": row.get("severity")},
        )
        return [evidence], {
            "found": True,
            "drug_a": row["drug_a"],
            "drug_b": row["drug_b"],
            "interaction": row["interaction"],
            "severity": row.get("severity"),
            "mechanism": row.get("mechanism"),
            "management": row.get("management"),
            "source": row.get("source"),
            "data_status": row.get("data_status", "demo"),
        }
