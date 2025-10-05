# OpenAPI Upgrade Plan (Spec‑First, Local‑Only)
This document describes how we evolve the OpenAPI workflow in this repo with local, repo‑scoped tooling. YAML under `contracts/openapi/` remains the source of truth.
- Law: `contracts/openapi/*.yaml` (design‑first). Code MUST follow the spec.
- Determinism: pins and vendored tools to keep outputs stable across machines.
- Developer UX: single entry via `cargo xtask` for lint → bundle → diff → generate.
References:
- Repo guideline: see `.specs/` and `AGENTS.md` (spec‑first; no backwards compat pre‑1.0.0; determinism by default).
## Goals
- **Validate** OpenAPI (spec compliance + style) locally and in CI.
- **Bundle** spec(s) for consumers and SDK generation.
- **Detect breakage** across branches/PRs.
- **Generate Rust types** for the `contracts-api-types` crate from the spec.
- Keep everything **local**: no global installs; no pnpm; Java is OK.
## Tooling (local, vendored)
- **Redocly CLI** (lint + bundle): `@redocly/cli`
- **Spectral** (linter): `@stoplight/spectral-cli`
- **IBM OpenAPI Validator** (Spectral rules + IBM best practices): `ibm-openapi-validator` and `@ibm-cloud/openapi-ruleset`
- **oasdiff** (breaking‑change diff): vendored Linux binary
- **OpenAPI Generator** (models + optional Axum server stubs): vendored JAR, run with local Java 11+
We keep these isolated in `tools/openapi/` with pinned versions.
## Directory layout
```
contracts/
  openapi/
    data.yaml
    control.yaml
    catalog.yaml
    artifacts.yaml
    meta.yaml
    sessions.yaml
    OPENAPI_UPGRADE.md  <-- this file
    _bundles/           <-- bundled outputs (git-ignored OK)
tools/
  openapi/
    package.json        <-- local devDeps only
    package-lock.json
    vendor/
      openapi-generator-cli.jar  <-- pinned
      oasdiff                  <-- pinned linux binary (chmod +x)
    scripts/
      download-openapi-generator.sh
      download-oasdiff.sh
      lint.sh
      bundle.sh
      validate.sh
      diff.sh
      generate-models.sh
      generate-axum-stubs.sh (optional)
```
## Version pins (initial)
- Redocly CLI: `^2.2.1` (as of 2025‑09‑30)
- Spectral CLI: `^6` (latest stable 6.x)
- IBM OpenAPI Validator: latest (kept in package‑lock)
- OpenAPI Generator JAR: `7.15.0`
- oasdiff: latest stable 2.x (binary release; pin exact URL/sha256 in script)
Update pins periodically; changes should be deliberate and captured in a PR.
## Install steps (one‑time)
```bash
# 1) Node dev tools locally (no global install)
cd tools/openapi
npm ci
# 2) Vendor OpenAPI Generator (Java 11+ required)
./scripts/download-openapi-generator.sh
# 3) Vendor oasdiff binary
./scripts/download-oasdiff.sh
```
## Day‑to‑day commands
All paths are repo‑root unless stated otherwise.
- **Lint (spec correctness + style)**
  ```bash
  npx --yes @redocly/cli lint contracts/openapi/*.yaml
  npx --yes @stoplight/spectral-cli lint contracts/openapi/*.yaml
  npx --yes ibm-openapi-validator contracts/openapi/*.yaml
  ```
- **Bundle (produce distributable single files)**
  ```bash
  mkdir -p contracts/openapi/_bundles
  npx --yes @redocly/cli bundle contracts/openapi/data.yaml     -o contracts/openapi/_bundles/data.bundle.yaml
  npx --yes @redocly/cli bundle contracts/openapi/control.yaml  -o contracts/openapi/_bundles/control.bundle.yaml
  npx --yes @redocly/cli bundle contracts/openapi/catalog.yaml  -o contracts/openapi/_bundles/catalog.bundle.yaml
  npx --yes @redocly/cli bundle contracts/openapi/artifacts.yaml -o contracts/openapi/_bundles/artifacts.bundle.yaml
  npx --yes @redocly/cli bundle contracts/openapi/meta.yaml     -o contracts/openapi/_bundles/meta.bundle.yaml
  npx --yes @redocly/cli bundle contracts/openapi/sessions.yaml -o contracts/openapi/_bundles/sessions.bundle.yaml
  ```
- **Diff (breaking changes vs main)**
  ```bash
  # Example for data.yaml
  git show origin/main:contracts/openapi/data.yaml > /tmp/data.main.yaml
  tools/openapi/vendor/oasdiff breaking /tmp/data.main.yaml contracts/openapi/data.yaml --fail-on-changes
  ```
- **Generate Rust models for contracts‑api‑types**
  ```bash
  # Output to temp directory, then copy into contracts/api-types
  java -jar tools/openapi/vendor/openapi-generator-cli.jar generate \
    -g rust \
    -i contracts/openapi/data.yaml \
    -o /tmp/oa-data-rust \
    --additional-properties=packageName=contracts_api_types,library=serde
  # Repeat for control.yaml if desired; aggregate into:
  # contracts/api-types/src/generated.rs (data)
  # contracts/api-types/src/generated_control.rs (control)
  ```
- **Optional: Generate Axum stubs for drift checks**
  ```bash
  java -jar tools/openapi/vendor/openapi-generator-cli.jar generate \
    -g rust-axum \
    -i contracts/openapi/data.yaml \
    -o contracts/generated/rust-axum/data
  ```
## xtask integration
Expose a single cargo entry point and wire into the dev loop:
- `cargo xtask openapi:lint` → run Redocly + Spectral + IBM validator
- `cargo xtask openapi:bundle` → bundle all YAMLs to `_bundles/`
- `cargo xtask openapi:diff` → breakage check vs `origin/main`
- `cargo xtask regen-openapi` → call generator to refresh `contracts/api-types/src/generated*.rs`
- `cargo xtask dev:loop` → include the above (lint → bundle → diff → regen), then fmt/clippy/tests
Notes:
- We already validate YAML parse via `openapiv3` in `xtask::regen_openapi`; keep that as a fast early guard, and add the linters for richer feedback.
- Generated files should be formatted (`cargo fmt`) and stable (only rewrite on change).
## CI gates
- Run `openapi:lint`, `openapi:bundle`, and `openapi:diff` in PRs.
- Fail PR when:
  - Lint errors occur (any tool)
  - Breaking changes detected by `oasdiff`
  - Bundling fails
Artifacts to upload on CI failure:
- Linter outputs (JSON/STDOUT)
- Bundled spec(s)
- oasdiff report
## Policy and conventions
- The spec in `contracts/openapi/*.yaml` is normative; code MUST conform.
- Keep **operationId** stable and meaningful (used by clients and codegen).
- Prefer small, focused YAMLs per domain (data/control/catalog/etc.) and bundle for consumers.
- Treat version upgrades of tools as visible changes; pin and document.
- Any behavior change in spec must ship with tests and  per `.docs/testing/` guidance.
## Troubleshooting
- Redocly fails on refs: ensure relative $refs resolve from the YAML file; prefer bundling frequently.
- Spectral/IBM complaints: start with Redocly fixes, then address style rules or adjust ruleset if needed.
- oasdiff false positives: verify that intended changes are non‑breaking; otherwise, version accordingly.
- OpenAPI Generator schema quirks: adjust `--additional-properties` or post‑process generated models before copying into `contracts/api-types`.
## Future options (non‑normative)
- Add `utoipa` or `aide` annotations to Axum handlers purely for local doc preview (Swagger UI/Redoc) while keeping YAML as the law.
- Add a master index YAML that $refs domain YAMLs for a single bundling entry.
---
Maintainers: keep this document updated when bumping tool versions or adjusting workflows.
