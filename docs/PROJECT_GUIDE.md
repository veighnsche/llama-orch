# Project Guide

This repo uses a contract-first, stub-first, TDD approach. Start with `TODO.md`, `specs/orchestrator-spec.md`, and `docs/workflow.md`.

- Contracts: `contracts/openapi/*.yaml` (OpenAPI-first)
- Types: generated into `contracts/api-types`, server stubs in `orchestratord`
- Config Schema: `contracts/config-schema` emits JSON Schema
- Adapters: `worker-adapters/*` (stubs only in this phase)
- Tools: `xtask` for regen and CI helpers
