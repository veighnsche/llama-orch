# Queen UI CORS, Radix Select, SDK Port, and Turbo Crash — Deep Investigation (2025-10-29)

Status: Analysis only (no code changes). Scope covers Queen UI (Vite on 7834), queen-rbee API (7833), Keeper GUI embedding Queen, SDK behavior, and the install flow from Keeper.

---

## Findings

- **Radix Select crash in RhaiIDE**
  - File: `bin/10_queen_rbee/ui/app/src/components/RhaiIDE.tsx` lines 121–140
  - Error: `A <Select.Item /> must have a value prop that is not an empty string.`
  - Cause: Two places use empty string values for Select items/selection:
    - `value={currentScript?.id || ""}`
    - Map: `<SelectItem key={script.id} value={script.id || ""}>`
    - New item: `<SelectItem value="">New Script</SelectItem>`
  - Radix/SHAD UI reserves empty string to clear the selection. Items must not use `""`. This throws synchronously during render, independent of backend availability.

- **Why visiting http://localhost:7834 shows a "CORS request did not succeed"**
  - 7834 = Queen UI Vite dev server. The UI tries to connect to queen-rbee at 7833 via SDK heartbeat SSE.
  - SDK logs (from browser):
    - `🐝 [SDK] Connecting to SSE: http://localhost:7833/v1/heartbeats/stream`
    - Then the browser reports a CORS network failure.
  - Root cause: queen-rbee on 7833 was not running. In Firefox and some browsers, a network failure during cross-origin fetch is surfaced as "CORS request did not succeed" even when there are no CORS headers to check.
  - When queen is running, CORS is configured to allow the UI:
    - Code: `bin/10_queen_rbee/src/main.rs` lines 117–121 add `CorsLayer::new().allow_origin(Any).allow_methods(Any).allow_headers(Any)`.
    - This layer is applied to the merged API + static router (line 144). SSE is a GET request and is covered.

- **Why the SDK connects to :7833 even if queen is off**
  - By design. Defaults:
    - `useHeartbeat(baseUrl = 'http://localhost:7833')` in `queen-rbee-react` (file: `src/hooks/useHeartbeat.ts`, line 35).
    - `useRhaiScripts(baseUrl = 'http://localhost:7833')` in `queen-rbee-react` (file: `src/hooks/useRhaiScripts.ts`, line 57).
  - The Queen UI is for 7833 (API + prod UI). In dev, the UI runs on 7834 but still targets the API on 7833.
  - If queen isn’t running, EventSource fails and the SDK logs a warning. That is expected.

- **Dev environment routing to Vite**
  - Requirement in PORT_CONFIGURATION.md is implemented:
    - `bin/10_queen_rbee/src/http/static_files.rs` uses `#[cfg(debug_assertions)]` to proxy root `/` to `http://localhost:7834` (fallback handler). In release, it serves embedded dist/.
    - Merged in `main.rs` so API routes take priority; UI is a fallback (root path), aligned with the doc.

- **Keeper GUI page behavior (Queen iframe)**
  - `bin/00_rbee_keeper/ui/src/pages/QueenPage.tsx`: If queen isn’t running, the page shows an alert with "Start Queen". Only when running does it iframe `http://localhost:7833`.
  - Therefore, going directly to `http://localhost:7834` bypasses Keeper’s guard and will attempt SSE to 7833 regardless of queen state.

- **Install Queen from Keeper makes turbo dev crash (likely cause)**
  - Keeper’s Tauri command `queenInstall` calls `daemon-lifecycle::install_daemon` for `queen-rbee` (file: `bin/00_rbee_keeper/src/tauri_commands.rs`, lines 156–172 → delegates to handlers/queen.rs).
  - `install_daemon` calls `build_daemon` (file: `bin/99_shared_crates/daemon-lifecycle/src/install.rs`, lines 160–178; build logic in `src/build.rs`).
  - `build_daemon` runs `cargo build` for `queen-rbee`. The queen crate has a `build.rs` that runs `pnpm exec vite build` to produce UI assets prior to embedding (file: `bin/10_queen_rbee/build.rs`). This runs ALWAYS, not only in release.
  - If you have Turbo dev servers already running (e.g., `turbo dev` starting Vite at 7834), a nested `pnpm exec vite build` within `cargo build` competes with dev watchers and repository-wide builds:
    - High CPU/memory leading to Turbo node process crash.
    - Potential lock-contention on node_modules/.pnpm store or Vite caches.
    - Sudden rebuilds of shared packages (storybook/ui) triggered by the build, rippling through turbo graph.
  - This explains "turbo dev server crashes when installing Queen via Keeper" without needing a bug in Turbo itself.

- **Port documentation inconsistencies**
  - `PORT_CONFIGURATION.md` shows Quick Reference with queen-rbee at 7833 (correct), but the Detailed Information section still references 8500 in code blocks. The code in `main.rs` defaults to 7833, so the doc is partially stale. This can confuse expectations when reading logs vs docs.

---

## Evidence and code references

- Queen server CORS and routes:
  - `bin/10_queen_rbee/src/main.rs` lines 117–121, 123–145
- UI dev proxy:
  - `bin/10_queen_rbee/src/http/static_files.rs` lines 49–63 and 65–126
- Queen UI heartbeat hook default URL:
  - `bin/10_queen_rbee/ui/packages/queen-rbee-react/src/hooks/useHeartbeat.ts` line 35
- Queen UI RHAI hook default URL:
  - `bin/10_queen_rbee/ui/packages/queen-rbee-react/src/hooks/useRhaiScripts.ts` line 57
- RhaiIDE Select with empty values causing crash:
  - `bin/10_queen_rbee/ui/app/src/components/RhaiIDE.tsx` lines 121–140
- SDK EventSource behavior and logs:
  - `bin/10_queen_rbee/ui/packages/queen-rbee-sdk/src/heartbeat.rs` lines 59–140
- Keeper Queen page behavior:
  - `bin/00_rbee_keeper/ui/src/pages/QueenPage.tsx` (iframe only when running)
- Keeper Tauri commands (install flow):
  - `bin/00_rbee_keeper/src/tauri_commands.rs` lines 156–172 (queen_install)
  - Delegation: `bin/00_rbee_keeper/src/handlers/queen.rs`
- Build pipeline that triggers Vite build inside cargo build:
  - `bin/10_queen_rbee/build.rs` (always runs `pnpm exec vite build`)

---

## Root causes

- **Radix Select crash:** Rendering `<SelectItem value="">` and mapping items with `value={id || ""}` violates component contract. Must not use empty string for item values.
- **"CORS request did not succeed" at 7834:** Queen API at 7833 is down; the browser reports cross-origin network failure as CORS. Not a headers/policy issue.
- **SDK connects to 7833 when queen is off:** Expected by design; default base URL is 7833. The UI should tolerate queen being offline.
- **Turbo dev crash during Install:** Installing queen from Keeper triggers `cargo build` for the queen crate, which triggers `pnpm exec vite build` via queen `build.rs`. Running a full Vite build concurrently with Turbo/Vite dev servers can destabilize or crash the dev graph process.

---

## Reproduction steps

- Radix Select:
  1. Open Queen UI at 7834.
  2. Navigate to RHAI IDE panel.
  3. Observe immediate error due to `<SelectItem value="">`.

- CORS error at 7834:
  1. Ensure queen-rbee daemon (7833) is NOT running.
  2. Open 7834; SDK attempts SSE to 7833.
  3. Browser shows: "CORS request did not succeed" and SSE warning logs.

- Turbo crash during "Install Queen":
  1. Start Turbo dev: `turbo dev` (starting Vite servers including 7834).
  2. In Keeper GUI, click "Install Queen".
  3. Keeper calls Tauri → daemon-lifecycle → cargo build → queen `build.rs` → `pnpm exec vite build`.
  4. Observe Turbo dev process instability/crash logs.

---

## Proposed fixes (no code changes yet)

- **RhaiIDE Select (critical UI fix):**
  - Never use `""` as a value for `<SelectItem>`.
  - Options:
    - Use a sentinel like `"new"` for "New Script" and handle it in `selectScript` (`if id === 'new' → createNewScript()`).
    - Do not render a `<SelectItem>` for new when `currentScript?.id` is empty; use the placeholder only.
    - Ensure mapped script items only render when `script.id` is a non-empty string.

- **Heartbeat/SSE robustness:**
  - Add a pre-check to hit `/health` before starting EventSource; if queen is down, suppress the SSE start and show an offline state. This avoids scary console warnings during development.
  - Alternatively, debounce retry and lower log verbosity in SDK for the dev UI.

- **Dev UX for API base URL:**
  - Centralize base URL selection (env var or `/v1/info` discovery) to avoid hardcoding `http://localhost:7833` in multiple hooks. Keep 7833 as default.

- **Avoid Turbo crashes on Install:**
  - Make `build.rs` skip `pnpm exec vite build` when a dev server is running or when a skip flag is set (e.g., env `RBEE_SKIP_UI_BUILD=1`).
  - Or restrict UI build to release builds only; in debug builds prefer the dev proxy (no need to embed dist).
  - As a short-term dev workaround: stop Turbo dev before clicking Install, or perform Install from a clean shell (not during dev).

- **Documentation hygiene:**
  - Update `PORT_CONFIGURATION.md` Detailed section to consistently reflect 7833 (remove 8500 remnants).

---

## Verification plan (after fixes)

- Select:
  - Load RhaiIDE with no scripts and with scripts. Confirm no render-time errors; placeholder works; sentinel "new" path triggers `createNewScript`.
- CORS/SSE:
  - With queen down: UI should show offline state; no CORS errors; no noisy console logs.
  - With queen up: SSE connects; heartbeat events flow; CORS headers present.
- Install flow:
  - With Turbo dev running: clicking Install no longer triggers a nested Vite build (or it’s gated by a flag) and dev servers remain stable.

---

## Answers to your specific questions

- "Why CORS error at localhost:7834?"
  - Because the UI on 7834 attempted to connect to 7833 while queen was not running. The browser surfaced that network failure as a CORS error message. When queen is up, CORS headers are permissive and SSE works.

- "Why is the SDK connecting to :7833 while the queen isn't even on?"
  - Default base URL is 7833 by design (dev UI ↔ API split). It will attempt to connect on page load; it should tolerate queen being offline.

- "IF DEV ENVIRONMENT reroute the UI to the Vite dev server?"
  - Implemented. In debug builds, queen’s root path proxies to `http://localhost:7834`. In production, queen serves embedded `dist/`.

- "Radix Select error cause?"
  - Empty string `""` values in `<SelectItem>` and `value` prop. Items must have non-empty values; use a sentinel or skip rendering items without IDs.

- "Keeper install causes Turbo dev server to crash?"
  - The install flow triggers `cargo build` which triggers queen `build.rs`, which runs `pnpm exec vite build` concurrently with your dev servers. That contention is the likely crash trigger.

---

## Next steps (recommended order)

1. Fix RhaiIDE Select values (no empty strings).
2. Add health check gate before starting SSE in dev.
3. Add a skip flag or conditional to `build.rs` to avoid Vite build during dev-installs.
4. Update PORT_CONFIGURATION.md to remove 8500 remnants.

s