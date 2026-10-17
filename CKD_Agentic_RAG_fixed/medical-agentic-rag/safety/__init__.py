"""Safety layer: guardrails, claim verification, confidence and citations.

No re-exports: :mod:`ingestion.metadata_tagger` imports ``safety.source_trust``
during ingestion, and pulling the whole safety package in at that point would
drag the agent package into a partially initialised state. Import submodules
directly, e.g.::

    from safety.guardrails import GuardrailEngine
"""
