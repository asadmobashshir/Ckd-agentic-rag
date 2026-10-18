# CKD Medical Agentic RAG

An evidence-grounded **Chronic Kidney Disease** information assistant built as a
genuine tool-using agent: the model chooses which retrieval tools to run, reads
their real output, and writes an answer that cites only evidence a tool actually
returned.

> **Research prototype — NOT medical advice.** It does not diagnose, interpret
> test results, or prescribe. The bundled corpus is synthetic demonstration data
> containing no numeric drug doses. Always consult a qualified clinician.

---

## What "agentic" means here

A conventional RAG pipeline runs the same retrieval for every question. This one
runs a real tool-calling loop:

```
question + tool specs  ->  model
model requests tools   ->  validated against Pydantic schemas
tools execute in Python
ACTUAL tool output     ->  back to the model as role="tool" messages
model may call more tools because of what it just read
model writes the final answer, citing the evidence ids it was shown
```

The loop is bounded by `MAX_AGENT_ITERATIONS` (default 5) and the final turn
withholds tools so the model must produce prose.

Guarantees:

* A tool is reported as "used" only if it **actually executed**. Calls rejected
  at validation are reported separately, never as used.
* An identical `(tool, arguments)` pair executes **once**; repeats are answered
  from cache.
* Citations resolve to real retrieved evidence; unmatched `[S#]` markers are
  stripped before display.

## Providers

Both providers are implemented and both do **native tool calling**:

| Provider | Key | Default model |
|---|---|---|
| Groq | `GROQ_API_KEY` | `llama-3.3-70b-versatile` |
| Google Gemini | `GEMINI_API_KEY` | `gemini-2.5-flash` |

`LLM_PROVIDER=auto` (the default) uses whichever key is present, preferring Groq.

**Model ids are retired upstream regularly.** Verify yours against your own
account rather than against documentation:

```bash
python scripts/diagnostics.py --list-models
```

If the configured id is rejected, the client automatically falls back through
`*_MODEL_FALLBACKS` and logs the switch, so a retired model does not take the
app down.

## Quick start

```bash
cd medical-agentic-rag
pip install -r requirements.txt
cp .env.example .env            # add GROQ_API_KEY (or GEMINI_API_KEY)
python -m ingestion.run_ingestion --reset
python scripts/diagnostics.py
streamlit run streamlit_app.py
```

## Deploying to Streamlit Community Cloud

1. Push to GitHub.
2. Create an app with main file `medical-agentic-rag/streamlit_app.py`.
3. **Settings → Secrets**, using `.streamlit/secrets.toml.example` as the
   template:

   ```toml
   LLM_PROVIDER = "auto"
   GROQ_API_KEY = "gsk_your_key"
   GROQ_MODEL = "llama-3.3-70b-versatile"
   ```

4. Save, then **Reboot app** — secrets are only re-read on restart.

The app builds its own vector index on first run (`data/processed/` is
gitignored). Secrets are loaded into the environment *before* `Settings` is
constructed, then the settings cache is cleared; building `Settings` first would
cache a key-less object for the life of the process.

### Dependencies

`requirements.txt` is deliberately light — no PyTorch. `sentence-transformers`
is opt-in via `requirements-embeddings.txt` because it repeatedly exceeded the
free-tier build limit. Retrieval is hybrid (dense + BM25), so without the
semantic model the BM25 half still routes every benchmark query to the correct
document; only paraphrase matching degrades. The status panel states which
backend is live.

## Retrieval and the "0.23 similarity" question

Retrieval returns the correct top-1 document on **9/9** benchmark queries while
raw cosine averages ≈0.30. That is not a weak match: with the lexical hashing
backend, a correct match simply *lands* in the 0.22–0.55 range. Reading it on a
semantic-cosine scale is a units error.

Confidence therefore calibrates per backend, with the mapping documented in
`safety/confidence.py`, and the rationale says plainly that semantic embeddings
are disabled rather than claiming "similarity was weak". Scores are not inflated
— lexical strength is capped below a semantic match on purpose.

## Using real data instead of the demo corpus

The bundled corpus is synthetic. `docs/REAL_DATA_SOURCES.md` lists real CKD
sources that can replace it (NIDDK, MedlinePlus, CDC, KDIGO, NICE, openFDA
labelling, PMC open-access), what each licence actually permits, the front
matter to write, and the three things that break when you swap it in.

Provenance is a closed vocabulary — `demo`, `guideline`, `regulatory`,
`reference`, `literature`, `licensed`, `live`, `unknown` — so a typo fails
loudly and unlabelled content stays `demo`.

## Testing

```bash
pytest -q          # 205 tests, fully offline, no API key or network
pytest -m live     # opt-in: real round trip against the configured model
```

The offline suite covers LLM generation, unavailable models, missing keys, tool
execution, multi-step tool calls, invalid arguments, tool failure, empty
retrieval, weak evidence, conflicting sources, citation validation, the
iteration ceiling, safe model failure, prompt injection, and app startup.

## Safety

* Emergency and self-harm requests short-circuit **before** any model call.
* Personalised dosing requests are answered with general reference evidence
  only; numeric doses are redacted by the output guardrail.
* A structured-store miss is reported as *absence of data*, never as evidence
  that a combination is safe.
* Retrieved text is treated as data; instruction-like content in it is
  neutralised before it reaches the model.
* Tool arguments are schema-validated before execution; the model cannot cause
  an arbitrary call.
* Demo provenance (`data_status: demo`) is carried to the UI on every citation.

## Layout

```
medical-agentic-rag/
├── streamlit_app.py        # Streamlit entry point (view layer only)
├── agent/
│   ├── tool_loop.py        # native tool-calling loop  <-- the agentic core
│   ├── orchestrator.py     # pipeline coordination
│   ├── planner.py          # legacy plan-then-execute fallback
│   └── tools/              # registry + 5 validated tools
├── llm/                    # groq_client, gemini_client, factory
├── safety/                 # guardrails, verifier, confidence, citations
├── stores/                 # vector (Qdrant/NumPy), BM25, SQLite
├── ingestion/              # chunk -> tag -> embed -> index
└── scripts/diagnostics.py  # component health + --list-models
```
