# Config Schema Plan — Home Profile

## Objectives
- Encode all home profile settings in `contracts/config-schema/src/lib.rs` with clear defaults.
- Generate deterministic `contracts/schemas/config.schema.json` via `cargo xtask regen-schema`.

## Required Fields
- Pools: id, engine, model, devices, optional `tensor_split`, preload flag.
- Queue: capacity, full policy (`reject`|`drop-lru`).
- Sessions: TTL, turn limit, optional budgets.
- Artifacts: storage path, retention options, size limits.
- Tooling policy: enable/disable, plugin path.
- Network: bind address (default loopback), optional advertised address for dev box.

## Tasks
1. Audit existing schema for legacy multi-tenant/fairness/preemption fields and deprecate them.
2. Document defaults in `.docs/HOME_PROFILE.md` and provide sample config under `examples/home-profile/` (to be created).
3. Add schema tests validating example files (`cargo test -p contracts-config-schema`).
4. Update regeneration instructions in `.docs/PROJECT_GUIDE.md` if paths change.

## Risks
- Ensure schema remains backwards-compatible with any existing home experiments (document migrations if needed).
- Keep regeneration deterministic to avoid noise in PRs.
