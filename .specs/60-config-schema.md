# Config Schema SPEC — Validation & Generation (v1.0)

Status: Stable (draft)
Applies to: `contracts/config-schema/`
Conformance language: RFC‑2119

## 0) Scope & Versioning

Requirements are versioned as `OC-CONFIG-6xxx`.

## 1) Validation

- [OC-CONFIG-6001] Config MUST be strictly validated; unknown fields rejected (strict) or logged (compat) per mode.
- [OC-CONFIG-6002] Examples in tests MUST validate without errors.

## 2) Generation

- [OC-CONFIG-6010] Schema generation MUST be deterministic and idempotent across runs.

## 3) Engine Provisioning & Model Fetcher Fields

- [OC-CONFIG-6020] The schema MUST define engine provisioning modes under `engine`/`pool` configuration: `provisioning.mode: external|source|container|package|binary`.
- [OC-CONFIG-6021] The schema MUST support engine identification/version pinning: `engine.id` (llamacpp|vllm|tgi|triton), `engine.version` (string), and engine‑specific version labels.
- [OC-CONFIG-6022] Source mode fields MUST include `engine.source.git.repo` (URL), `engine.source.git.ref` (tag/branch/sha), `engine.source.submodules: bool`, and build fields `engine.source.build.cmake_flags: [string]`, `engine.source.build.generator: string`, `engine.source.cache_dir: path`.
- [OC-CONFIG-6023] Container mode fields MUST include `engine.container.image` and `engine.container.tag`.
- [OC-CONFIG-6024] Package mode fields MUST include `engine.package.name` and MUST honor deployment policy `allow_package_installs: bool`.
- [OC-CONFIG-6025] Binary mode fields MUST include `engine.binary.url` and `engine.binary.checksum` (required) and MUST honor `allow_binary_downloads: bool`.
- [OC-CONFIG-6026] Model fetcher fields MUST include `model.ref` (HF path, local path, URL, S3/OCI), `model.cache_dir`, and optional verification digests.
- [OC-CONFIG-6027] Arch/CachyOS deployments MAY set `allow_package_installs: bool` to enable pacman/AUR usage by provisioners.

## 4) Traceability

- Code: [contracts/config-schema/src/lib.rs](../contracts/config-schema/src/lib.rs)
- Tests: [contracts/config-schema/tests/validate_examples.rs](../contracts/config-schema/tests/validate_examples.rs)

## Refinement Opportunities

- Add schema `x-examples` illustrating each provisioning mode across engines.
- Consider JSON Schema `oneOf` branches per `provisioning.mode` to improve validation messages.
- Tighten formats for `engine.binary.checksum` (e.g., `sha256:<hex>`) and `engine.source.git.repo` (URL allowlist).
