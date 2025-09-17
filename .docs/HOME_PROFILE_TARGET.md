# Home Lab Reference Environment

We validate every release of the home profile against the setup described below. If a change fails here, it is not ready to ship.

## Hardware

- **Workstation (orchestrator host)**
  - NVIDIA GeForce RTX 3090 (24 GB VRAM)
  - NVIDIA GeForce RTX 3060 (12 GB VRAM)
  - NVMe SSD for model/artifact storage (≥ 1 TB suggested)
- **Developer Box (CLI + tools)**
  - Runs `llama-orch-cli`, pact/BDD suites, and automation agents.
  - Network access to the workstation via LAN or SSH tunnel.

## Topology & Configuration

- Orchestrator services, adapters, and artifact storage run on the workstation.
- Default bind address is `127.0.0.1`; remote control is achieved through an SSH tunnel (`ssh -L` or `ssh -R`) or an explicit bind override guarded by firewall rules.
- The developer box exports `LLORCH_API_TOKEN` and points the CLI to `http://127.0.0.1:port` locally (tunnel) or `http://workstation:port` when on the same LAN.

## Workload Model

- Target use case: multiple auto-coder agents running concurrently, editing large repositories, and queueing work onto both GPUs.
- Determinism is mandatory for reproducible diffs; every orchestration loop relies on SSE `metrics` frames, streaming tokens, and backpressure headers.
- Artifact registry stores plan snapshots, diffs, traces, and evaluation notes locally on the workstation.

## Validation Expectations

- Mixed-VRAM scheduling (24 GB + 12 GB) MUST function without manual per-model pinning.
- Pool drain/reload, catalog updates, and artifact uploads MUST succeed end-to-end in this environment.
- Optional features (tooling policy hook, session budgets) SHOULD be toggled at least once per release to verify sane defaults.
- All smoke/BDD/determinism suites MUST run against this setup before a release tag is cut.

Other home labs are free to deviate (different GPUs, storage, or network layout), but support commitments and release gates reference this environment.
