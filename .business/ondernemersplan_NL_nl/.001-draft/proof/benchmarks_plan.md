# Benchmarks Plan (Concept)

## Doel & Reikwijdte
- Meten van latency/throughput/kosten per 1K tokens voor representatieve prompts.

## Methodiek
- Dataset: [N] prompts per use‑case; herhaalbaarheid met vaste seed/instellingen.
- Metingen: p50/p95 latency, tokens/s, fouten.
- Omgeving: template‑profiel, hardwareprofiel gedocumenteerd.

## Acceptatiegrenzen
- p50 ≤ 300 ms, foutpercentage ≤ [X]%, kosten/1K tokens ≤ [€X].

## Rapportage
- Charts + SSE-transcripten + configuraties (redacted).
