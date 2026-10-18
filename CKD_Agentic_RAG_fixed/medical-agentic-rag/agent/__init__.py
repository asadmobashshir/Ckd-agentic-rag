"""Agent package: planning, orchestration, synthesis and state.

Intentionally free of re-exports. Eagerly importing :mod:`agent.orchestrator`
here creates an import cycle (``ingestion`` -> ``safety`` -> ``agent.state`` ->
``agent`` -> ``agent.orchestrator`` -> ``agent.synthesizer`` -> ``safety``).
Import the concrete module you need, e.g.::

    from agent.orchestrator import build_orchestrator
"""
