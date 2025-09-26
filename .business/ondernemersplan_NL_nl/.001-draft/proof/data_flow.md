# Dataflow & Classificatie (Concept)

## Overzicht (tekstueel)
- Client → Orchestratord (SSE) → Adapter (vLLM/TGI/llama.cpp) → GPU.
- Observability: logs/metrics/traces; logging minimal en doelgebonden.

## Classificatie
- Publiek: marketing/presentaties.
- Intern: configuraties/runbooks.
- Vertrouwelijk: klantconfig/logextracten (redacted).

## Retentie
- Zie dataretentiebeleid; verwijder/anonimiseer conform schema.
