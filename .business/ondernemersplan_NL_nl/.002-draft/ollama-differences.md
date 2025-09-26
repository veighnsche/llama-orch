# Ollama vs You

## 1. **Scope & Positioning**

* **Ollama**: A local inference wrapper → “run models on your laptop with one command.”
* **You**: An **open, spec-driven orchestration layer + toolkit** → queueing, scheduling, determinism, proofs, SDK + applets. Not just local inference, but the *governed infra* around it.

---

## 2. **Determinism & Proofs**

* **Ollama**: Runs models, but doesn’t guarantee deterministic replay, job IDs, or metrics contracts. Logs are basic.
* **You**: **Proof-first** → every job comes with logs, metrics, and reproducibility guarantees (same input → same output).

---

## 3. **Toolkit vs CLI**

* **Ollama**: Provides a CLI + simple API to run models.
* **You**: Deliver a **developer toolkit** (SDK + utils + deterministic applets) across TS/JS/Rust, with contracts and OpenAI adapter. The focus isn’t just “run a model” but “build reproducible AI tools inside IT workflows.”

---

## 4. **Open Specs & Contracts**

* **Ollama**: Closed roadmap, internal design choices, no RFC-2119 specs.
* **You**: Public, open specs (RFC-2119 style) for orchestrator-core, APIs, determinism, and observability. OSS developers can extend, fork, or replace pieces because the contracts are documented.

---

## 5. **Multi-GPU & Scheduling**

* **Ollama**: Primarily single-machine, consumer developer laptops.
* **You**: **orchestrator-core** supports multiple GPUs, scheduling, queues, budgets, deterministic tie-breakers — scaling from a dev box to multi-GPU infra.

---

## 6. **Applets**

* **Ollama**: “Run a model, then you figure out how to wire it.”
* **You**: Ship **curated deterministic applets** (summarizer, code review, doc QA) as reference implementations. IT teams don’t start from scratch; they start from working, reproducible building blocks.

---

## 7. **Target Audience**

* **Ollama**: Hobbyists, developers who want a fast local LLM runtime.
* **You**: IT teams, agencies, and OSS developers who need **infrastructure-grade guarantees** — determinism, compliance, reproducibility — whether they self-host or use your taps.

---

✅ **Short framing line you can use**:
*“Ollama runs models locally. I open-source the orchestration layer, the specs, and the toolkit that make those models reliable, auditable, and production-ready.”*
