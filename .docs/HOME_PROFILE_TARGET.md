# Home Profile Reference Environment

This document captures the concrete home-lab deployment that we treat as the primary validation target for the Home Profile specification.

## Hardware

- **Workstation (llama-orch host)**
  - NVIDIA GeForce RTX 3090 (24 GB VRAM)
  - NVIDIA GeForce RTX 3060 (12 GB VRAM)
  - Mixed-generation GPUs are expected to operate concurrently under the same host scheduler.
- **Dev Box (CLI and tooling)**
  - Runs `llama-orch-cli` and associated automation (auto coder agents).
  - Connects to the workstation over the LAN or an SSH tunnel when remote access is needed.

## Topology

- The workstation runs the `llama-orch` services and hosts all models/artifacts.
- The dev box uses the CLI to submit workloads to the workstation.
- Default configuration assumes loopback binding on the workstation; remote access is achieved via an SSH tunnel or explicit bind override when necessary.

## Workload Goals

- Primary workload is an auto-coder/agent loop that demands deterministic sampling and fast turnaround for multi-file edits.
- Streaming responses with `metrics` frames, correlation IDs, and backpressure headers are required to keep the CLI’s orchestration loop informed.
- Artifact storage is used for plan snapshots, diffs, and traces during iteration.

## Constraints & Testing Notes

- All validation for Home Profile v2.1 must run successfully against this hardware pairing before broader release.
- GPU scheduling must gracefully handle asymmetric VRAM sizes without manual per-model pinning.
- Any optional features (e.g., policy enforcement hooks, budgets) should be exercised in this environment to ensure sane defaults.

This setup represents our best-effort testing equipment for the Home Profile; other home-lab deployments should remain compatible but are considered out-of-scope for mandatory validation.
