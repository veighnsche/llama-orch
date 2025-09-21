# M1 — SDK Design and MVP

## Scope

This milestone delivers the **minimal viable SDK** (`llama-orch-sdk`) that can connect to a running **llama-orch server** and issue typed requests/responses. The SDK is a low-level, server-facing library.

The SDK **MUST**:

* Provide typed request/response models for llama-orch APIs.
* Support synchronous and streaming interactions.
* Handle serialization/deserialization and error codes deterministically.
* Remain agnostic of Blueprints, Utils, and CLI-generated files.

This milestone proves that the SDK can serve as the **single entry point** for all future consumers (Utils, CLI, Blueprints).

## Non-Goals

* No Blueprint applets (those are in M2).
* No file system helpers, prompt builders, or orchestration logic.
* No CLI integration or codegen for TS/JS/RS.
* No advanced features like retries, rate limiting, or caching.
* No artifact proof-bundle emission (those are later).

## Exit Criteria

* A minimal SDK crate exists at `consumers/llama-orch-sdk`.
* SDK includes:

  * Typed models for at least one control-plane API (`/v1/capabilities`) and one data-plane API (e.g., `/v1/completions`).
  * Client functions for:

    * **Non-streaming calls** (request → response).
    * **Streaming calls** (tokens, metrics, end events).
  * Error type(s) covering HTTP errors, transport errors, and llama-orch error codes.
* End-to-end test: SDK can call a running llama-orch instance, fetch `/v1/capabilities`, and run a completion with deterministic seed.
* Unit tests with JSON fixtures for both success and failure paths.
* CI workflow builds `llama-orch-sdk` for:

  * `x86_64-unknown-linux-gnu`
  * `wasm32-unknown-unknown` (compilation only; no live network tests).

## Required Artifacts

* `proof-bundle/manifest.json` (recording SDK version, test hashes).
* `proof-bundle/fixtures/` containing JSON request/response pairs used in contract tests.
* CI logs proving SDK builds and passes tests on both targets.

## SDK MVP API Surface

* **Types:**

  * `CapabilitiesResponse` (models, engines, pools).
  * `CompletionRequest`, `CompletionResponse`.
  * `Error` enum (network, parse, server).
* **Functions:**

  * `get_capabilities(client) -> CapabilitiesResponse`.
  * `create_completion(client, CompletionRequest) -> CompletionResponse`.
  * `stream_completion(client, CompletionRequest) -> Stream<Item=CompletionChunk>`.

## Risks & Controls

* **Risk:** Over-expanding scope into applets → Controlled by strict milestone separation (M2 handles applets).
* **Risk:** Drift between SDK and server schema → Controlled by JSON fixtures from actual server responses.
* **Risk:** WASM target gaps (no SSE in browser envs) → Controlled by `cfg` gates and feature flags.

## Rollback

* Remove `consumers/llama-orch-sdk`.
* Consumers (Utils, CLI) revert to no-op stubs until SDK reintroduced.
* Proof-bundle artifacts serve as audit trail of attempt.

## Changelog

* `2025-09-21 — Initial draft of SDK-only M1 milestone.`
