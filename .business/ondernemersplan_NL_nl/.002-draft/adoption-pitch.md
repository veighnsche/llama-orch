# Adoption Pitch (Open Source Developers)

## Headline

**Build with open, deterministic AI tools — no black boxes required**

## Short Pitch

I maintain an **open toolkit** of SDKs, utils, and deterministic applets designed for developers who want to build their own AI systems.
Everything is open source, inspectable, and designed to integrate cleanly into your workflows — whether you host locally, in the cloud, or in your own lab.

---

## Why it matters for OSS developers

Most AI stacks are frustrating:

* They hide internals behind closed APIs.
* They make reproducibility impossible.
* Tooling is scattered across half-maintained repos.

My toolkit and specs solve this by giving you:

* **Deterministic applets** → reproducible outputs from code review to summarization.
* **Language support** → TypeScript, JavaScript, Rust (Python/Mojo coming soon).
* **Specs + contracts** → everything defined in RFC-style docs, so you know what to expect.
* **Extensibility** → fork it, extend it, or plug it into your own orchestrator.

---

## What you get

### SDK & Utils

* Clean abstractions for building agentic AI tools.
* OpenAI adapter included — use existing clients immediately.
* Strong typing and predictable error handling.

### Deterministic Applets

* Prebuilt helpers: summarizer, code review, doc QA, automation bots.
* Verified for reproducibility — same input, same output.
* Easy to extend or combine into your own workflows.

### Open Specs & Core

* **orchestrator-core**: queueing, scheduling, determinism, observability.
* **contracts**: OpenAPI definitions for control/data plane.
* **proof-first ethos**: metrics, logs, and test harnesses included.

---

## Differentiation (for devs)

* **Truly open source** → everything inspectable, no hidden binaries.
* **Deterministic by design** → reproducibility guaranteed.
* **Practical defaults** → curated models and applets that “just work.”
* **Fork-friendly** → use the toolkit with my infra, or spin up your own.

---

## One-liner

**“I give developers an open toolkit to build deterministic AI tools — portable, reproducible, and yours to control.”**
