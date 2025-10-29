# Queen UI Dev Proxy vs Static Serving — Full Plan and Reference (2025-10-29)

Status: Plan + references (no code changes). Audience: engineers working on queen UI, keeper, and dev workflow.

---

## Purpose

Capture the end-to-end findings about:
- Radix Select crash in RhaiIDE
- "CORS request did not succeed" from dev UI on 7834
- Why SDK connects to 7833 by default
- Dev proxy behavior (proxy root to Vite 7834 in debug)
- Keeper install flow crashing Turbo dev servers
- A concrete plan: keep the dev proxy vs revert to static serving in debug

---

## Quick Links

- Investigation report with evidence and line references:
  - .docs/debug/2025-10-29_queen-ui_cors_select_sdk_install_investigation.md

---

## TL;DR

- Select crash = empty string values in RhaiIDE Select → fix values.
- CORS error from 7834 = queen at 7833 was down; browser reports network failure as CORS.
- SDK targets 7833 by design; dev UI at 7834 is separate.
- Dev proxy is implemented and correct; root (/) proxies to Vite 7834 in debug.
- Keeper "Install Queen" triggers `cargo build` which triggers Vite build via queen build.rs, competing with Turbo dev → can crash dev servers.

---

## Findings (summary)

- **Radix Select crash (RhaiIDE)**
  - Empty string values are passed to Select items/selection.
  - Must use non-empty strings (e.g., sentinel like "new") or rely on placeholder.

- **CORS symptom at 7834**
  - Happens when 7833 is not running; CORS layer is permissive when queen is up.
  - Debug builds proxy root to Vite 7834; API routes (including SSE) are on 7833.

- **SDK connects to 7833**
  - Default base URL in hooks is 7833 by design; dev UI (7834) targets the API (7833).

- **Keeper install destabilizes dev**
  - Install path triggers cargo build for queen → queen build.rs runs `pnpm exec vite build`.
  - Competes with Turbo/Vite dev servers; high contention and potential crashes.

---

## Evidence (files and lines)

- CORS layer and router merge: bin/10_queen_rbee/src/main.rs:117–121, 142–145
- SSE route: bin/10_queen_rbee/src/main.rs:133 → http::handle_heartbeat_stream
- SSE handler: bin/10_queen_rbee/src/http/heartbeat_stream.rs
- Dev proxy: bin/10_queen_rbee/src/http/static_files.rs (debug branch proxies root to 7834)
- RhaiIDE Select: bin/10_queen_rbee/ui/app/src/components/RhaiIDE.tsx (empty string values)
- Hooks base URLs: queen-rbee-react src/hooks/useHeartbeat.ts (line 35), useRhaiScripts.ts (line 57)
- Keeper iframe guard: bin/00_rbee_keeper/ui/src/pages/QueenPage.tsx
- Install flow → build.rs triggers Vite build: bin/00_rbee_keeper/src/tauri_commands.rs → handlers/queen.rs → daemon-lifecycle/build.rs → bin/10_queen_rbee/build.rs (runs `pnpm exec vite build`)

---

## Root Causes

- Select uses empty string values → Radix runtime error.
- Queen API down when UI attempts SSE → browser shows CORS-like fetch failure.
- Hardcoded 7833 base URL ensures UI always targets queen API.
- Queen build.rs triggers a Vite build during cargo build, conflicting with dev servers.

---

## Reproduction

- Select crash: open RhaiIDE in 7834 → render-time error due to empty string value.
- CORS message: stop queen, open 7834 → SSE attempts to 7833 fail with "CORS request did not succeed".
- Turbo crash: run `turbo dev`, click "Install Queen" in Keeper → cargo build → Vite build from build.rs → dev graph may crash.

---

## Plan A — Keep Dev Proxy (preferred)

Timebox: 2–3 hours

- [ ] Fix RhaiIDE Select: remove empty string values; use sentinel (e.g., "new") or placeholder-only for new.
- [ ] SSE Health Gate: check `/health` before creating EventSource; if queen is down, show offline and suppress noisy logs.
- [ ] build.rs Guard: gate `pnpm exec vite build` during dev installs (e.g., `RBEE_SKIP_UI_BUILD=1`) or make it release-only.

Acceptance:
- Visiting 7834: no Select crash; SSE connects when queen is up; silent/offline when down.
- Keeper iframe to 7833: stable when queen is running.
- Keeper Install: no Turbo dev instability.

---

## Plan B — Revert Dev Proxy (serve static files in debug)

If Plan A fails in timebox.

- [ ] static_files.rs (debug): replace proxy fallback with filesystem serving of `ui/app/dist/` (via ServeDir) and SPA fallback to `index.html`.
- [ ] UI build loop: introduce `pnpm run build:watch` for `@rbee/queen-rbee-ui` to continuously rebuild dist.
- [ ] Dev loop:
  - Terminal 1: `pnpm -F @rbee/queen-rbee-ui run build:watch`
  - Terminal 2: `cargo run -p queen-rbee`
  - Refresh 7833 to see changes (no HMR).
- [ ] Keeper: iframe 7833; note that manual refresh is expected.
- [ ] build.rs: keep `vite build` or respect `RBEE_SKIP_UI_BUILD=1` to avoid duplicate effort.

Acceptance:
- 7833 reliably serves the UI (debug and release) from dist.
- No dependency on 7834 in dev for the iframe path.
- Install via Keeper does not crash Turbo dev.

---

## Decision Gate

- Attempt Plan A for 2–3 hours. If any acceptance item fails, execute Plan B.

---

## Developer Notes

- UI iteration speed with Plan B remains acceptable using `vite build --watch` (incremental), albeit without HMR.
- Update documentation (PORT_CONFIGURATION.md) to remove stale 8500 references and clarify dev expectations:
  - 7834 is optional (pure UI dev with HMR).
  - Keeper iframe always targets 7833.

---

## Next Steps

1) Implement Plan A tasks; verify acceptance.
2) If issues persist, flip to Plan B and communicate the updated workflow in docs.

