# llama-orch-cli

Developer tool (Dev Tool CLI) that orchestrates the Spec → Contract → Tests → Code loop over the llama‑orch backend.

Start with the expectations/spec here:

- See `DEV_TOOL_SPEC.md` for what this CLI expects from the backend to deliver a high‑performance, multi‑agent, multi‑GPU experience.
- And the focused checklist: `feature-requirements.md` (CLI → backend requirements)

## Current State

The binary is a study stub. Running it will only print a message pointing to `feature-requirements.md` and `DEV_TOOL_SPEC.md`.

Planned Dev Tool surface (backed by `DEV_TOOL_SPEC.md`):

- `llama-orch plan` — Generate/update plan artifacts from specs.
- `llama-orch contract sync` — Update Pact(s) from OpenAPI and CLI needs.
- `llama-orch tests plan` — Propose BDD/property tests from specs and gaps.
- `llama-orch impl apply` — Propose diffs and run checks; open a PR.
- `llama-orch loop run --agents auto` — Run multi‑agent loop with budgets, saturating GPUs via backend pools.
- `llama-orch status` — Show pool capacity, queue predictions, agent progress.

### Detailed behavior (High / Mid / Low)

- High-level
  - A developer-facing CLI intended to drive and validate the Spec → Contract → Tests → Code loop against a running llama‑orch backend.
  - Current implementation is a stub; it prints guidance and exits successfully.

- Mid-level
  - Main binary: `src/main.rs` prints a static multiline string with pointers to `feature-requirements.md` and `DEV_TOOL_SPEC.md`.
  - No subcommands, arguments, or network calls are implemented yet.

- Low-level (from `src/main.rs`)
  - `fn main()` uses `println!` to emit the stub message and exits.

## Install/Build

Within the repository workspace:

```sh
cargo build -p llama-orch-cli
```

The binary name is `llama-orch`.

### Running locally

```sh
cargo run -p llama-orch-cli
```

Expected output is the stub help text pointing to `feature-requirements.md` and `DEV_TOOL_SPEC.md`.

### Config & Env

- None yet. Future versions will read configuration for backend endpoints and credentials, and surface flags per the Dev Tool spec.

## Notes

- This repository currently captures requirements and expectations for the Dev Tool CLI; the actual command surface will be implemented after backend capability decisions (see `feature-requirements.md`).
