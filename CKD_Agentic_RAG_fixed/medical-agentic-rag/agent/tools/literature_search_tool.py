"""Peer-reviewed literature search via the public NCBI E-utilities API.

Uses the official ESearch + ESummary endpoints (no scraping). Kept separate from
:mod:`agent.tools.web_search_tool` because literature metadata (PMID, journal,
authors, publication date) is structured and carries a different trust tier.

Abstracts are fetched only when explicitly requested and are truncated; PubMed
abstract text is subject to publisher copyright, so full text is not stored.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agent.state import Evidence, TrustTier, utc_now_iso
from agent.tools.base import BaseTool
from config.settings import Settings, get_settings

logger = logging.getLogger(__name__)

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PUBMED_URL = "https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
ABSTRACT_CHAR_LIMIT = 900


class LiteratureSearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=3, max_length=256)
    max_results: int = Field(default=5, ge=1, le=20)
    include_abstract: bool = Field(default=True)

    @field_validator("query")
    @classmethod
    def _clean(cls, value: str) -> str:
        return " ".join(value.strip().split())


class LiteratureSearchTool(BaseTool):
    """Search PubMed for peer-reviewed articles matching a query."""

    name: ClassVar[str] = "literature_search"
    description: ClassVar[str] = (
        "Search PubMed (NCBI E-utilities) for peer-reviewed biomedical literature. "
        "Use for questions about research findings, study evidence, or 'what does "
        "recent literature say'. Returns article titles, PMIDs, journals, publication "
        "dates, authors and canonical PubMed URLs."
    )
    args_model: ClassVar[type[BaseModel]] = LiteratureSearchArgs
    requires_network: ClassVar[bool] = True

    def __init__(self, settings: Settings | None = None, http_client: Any = None) -> None:
        self._settings = settings or get_settings()
        self._http = http_client  # injectable for tests

    # ------------------------------------------------------------------ http
    def _get(self, path: str, params: dict[str, Any]) -> str:
        params = {k: v for k, v in params.items() if v is not None}
        if self._settings.ncbi_api_key:
            params["api_key"] = self._settings.ncbi_api_key
        if self._settings.ncbi_email:
            params["email"] = self._settings.ncbi_email
        params.setdefault("tool", "medical-agentic-rag")

        if self._http is not None:
            return self._http(path, params)

        import httpx  # noqa: PLC0415

        url = f"{EUTILS_BASE}/{path}"
        with httpx.Client(timeout=self._settings.request_timeout) as client:
            response = client.get(url, params=params)
            response.raise_for_status()
            return response.text

    # --------------------------------------------------------------- parsing
    @staticmethod
    def _parse_ids(xml_text: str) -> list[str]:
        root = ET.fromstring(xml_text)
        return [node.text for node in root.findall(".//IdList/Id") if node.text]

    @staticmethod
    def _parse_summaries(xml_text: str) -> list[dict[str, Any]]:
        root = ET.fromstring(xml_text)
        articles: list[dict[str, Any]] = []
        for doc in root.findall(".//DocSum"):
            record: dict[str, Any] = {"pmid": None, "authors": []}
            id_node = doc.find("Id")
            if id_node is not None:
                record["pmid"] = id_node.text
            for item in doc.findall("Item"):
                name = item.get("Name")
                if name == "Title":
                    record["title"] = item.text
                elif name == "FullJournalName":
                    record["journal"] = item.text
                elif name == "Source" and not record.get("journal"):
                    record["journal"] = item.text
                elif name == "PubDate":
                    record["publication_date"] = item.text
                elif name == "AuthorList":
                    record["authors"] = [a.text for a in item.findall("Item") if a.text]
                elif name == "DOI":
                    record["doi"] = item.text
            if record.get("pmid"):
                articles.append(record)
        return articles

    @staticmethod
    def _parse_abstracts(xml_text: str) -> dict[str, str]:
        root = ET.fromstring(xml_text)
        abstracts: dict[str, str] = {}
        for article in root.findall(".//PubmedArticle"):
            pmid_node = article.find(".//PMID")
            if pmid_node is None or not pmid_node.text:
                continue
            parts = [
                (node.text or "").strip()
                for node in article.findall(".//Abstract/AbstractText")
                if (node.text or "").strip()
            ]
            if parts:
                abstracts[pmid_node.text] = " ".join(parts)[:ABSTRACT_CHAR_LIMIT]
        return abstracts

    # ------------------------------------------------------------- execution
    def run(self, args: BaseModel) -> tuple[list[Evidence], dict[str, Any]]:
        assert isinstance(args, LiteratureSearchArgs)
        if not self._settings.enable_network_tools:
            return [], {
                "results": 0,
                "skipped": True,
                "message": "Network tools are disabled (ENABLE_NETWORK_TOOLS=false).",
            }

        limit = min(args.max_results, self._settings.literature_max_results)
        try:
            search_xml = self._get(
                "esearch.fcgi",
                {"db": "pubmed", "term": args.query, "retmax": limit, "retmode": "xml", "sort": "relevance"},
            )
            pmids = self._parse_ids(search_xml)
        except Exception as exc:  # noqa: BLE001
            logger.warning("PubMed search failed: %s", exc)
            return [], {
                "results": 0,
                "error": f"PubMed unavailable: {exc}",
                "message": "No literature retrieved. Do not substitute recalled citations.",
            }

        if not pmids:
            return [], {"results": 0, "query": args.query, "message": "No PubMed records matched."}

        try:
            summary_xml = self._get(
                "esummary.fcgi", {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"}
            )
            articles = self._parse_summaries(summary_xml)
        except Exception as exc:  # noqa: BLE001
            logger.warning("PubMed summary fetch failed: %s", exc)
            return [], {"results": 0, "error": f"PubMed summary unavailable: {exc}"}

        abstracts: dict[str, str] = {}
        if args.include_abstract:
            try:
                fetch_xml = self._get(
                    "efetch.fcgi",
                    {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml", "rettype": "abstract"},
                )
                abstracts = self._parse_abstracts(fetch_xml)
            except Exception as exc:  # noqa: BLE001 - abstracts are optional
                logger.info("Abstract fetch failed (continuing with metadata): %s", exc)

        timestamp = utc_now_iso()
        evidence: list[Evidence] = []
        records: list[dict[str, Any]] = []

        for article in articles:
            pmid = article["pmid"]
            title = article.get("title") or "Untitled article"
            abstract = abstracts.get(pmid)
            authors = article.get("authors") or []
            author_line = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")
            body = f"{title}"
            if author_line:
                body += f" — {author_line}"
            if article.get("journal"):
                body += f" ({article['journal']})"
            if abstract:
                body += f"\nAbstract excerpt: {abstract}"

            evidence.append(
                Evidence(
                    text=body,
                    source=article.get("journal") or "PubMed",
                    title=title,
                    url=PUBMED_URL.format(pmid=pmid),
                    publication_date=article.get("publication_date"),
                    retrieved_at=timestamp,
                    evidence_level="peer_reviewed_literature",
                    document_type="journal_article",
                    document_id=f"pmid:{pmid}",
                    chunk_id=f"pmid:{pmid}",
                    trust_tier=TrustTier.B,
                    data_status="live",
                    extra={
                        "retrieval": "literature_search",
                        "pmid": pmid,
                        "authors": authors,
                        "doi": article.get("doi"),
                        "has_abstract": bool(abstract),
                    },
                )
            )
            records.append(
                {
                    "pmid": pmid,
                    "title": title,
                    "journal": article.get("journal"),
                    "publication_date": article.get("publication_date"),
                    "authors": authors,
                    "url": PUBMED_URL.format(pmid=pmid),
                }
            )

        return evidence, {"results": len(evidence), "query": args.query, "articles": records}
