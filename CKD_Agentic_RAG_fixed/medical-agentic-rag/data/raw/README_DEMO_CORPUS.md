---
title: "About this CKD demo corpus"
source: "DEMO corpus"
url: null
publication_date: null
document_type: "demo_corpus_note"
evidence_level: "demo_material"
data_status: "demo"
---

# DEMO CONTENT — NOT FOR CLINICAL USE

Every file in `data/raw/` is a **small, synthetic demonstration corpus** about
**Chronic Kidney Disease (CKD)**, written so the ingestion, chunking, embedding,
indexing and retrieval code has something to operate on out of the box.

- It is **not** a curated or validated medical knowledge base.
- It contains **no citations to real papers** and no invented URLs.
- It contains **no numeric drug doses** anywhere, by design.
- Statements are deliberately general and non-actionable.
- `source` is recorded as `DEMO corpus`; `url` and `publication_date` are `null`
  because no real source exists — the pipeline stores `null` rather than
  inventing metadata.

This assistant is specialised for CKD. Questions about unrelated conditions are
outside its supported domain and will not trigger retrieval.

Replace this directory with properly licensed source documents before doing
anything beyond software testing. Always consult a qualified clinician for
decisions about kidney care or medicines.
