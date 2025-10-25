# Folder Structure: bin/ and frontend/apps/ Parity

**TEAM-293: Aligned folder naming between binaries and UIs**

## Design Principle

**Parity:** Each binary in `bin/` has a corresponding UI in `frontend/apps/` with the same numbered prefix.

## Complete Folder Structure

```
bin/
├── 00_rbee_keeper/
│   └── ui/                      # Desktop GUI (Tauri) - no app subfolder
│       ├── src/
│       ├── src-tauri/
│       ├── package.json
│       └── vite.config.ts
│
├── 10_queen_rbee/
│   └── ui/
│       ├── app/                 # Queen web UI
│       │   ├── src/
│       │   ├── package.json
│       │   └── vite.config.ts
│       └── packages/
│           ├── queen-rbee-sdk/  # HTTP client for queen
│           │   ├── src/
│           │   ├── package.json
│           │   └── tsconfig.json
│           └── queen-rbee-react/ # React hooks for queen
│               ├── src/
│               ├── package.json
│               └── tsconfig.json
│
├── 20_rbee_hive/
│   └── ui/
│       ├── app/                 # Hive web UI
│       │   ├── src/
│       │   ├── package.json
│       │   └── vite.config.ts
│       └── packages/
│           ├── rbee-hive-sdk/   # HTTP client for hive
│           │   ├── src/
│           │   ├── package.json
│           │   └── tsconfig.json
│           └── rbee-hive-react/  # React hooks for hive
│               ├── src/
│               ├── package.json
│               └── tsconfig.json
│
└── 30_llm_worker_rbee/
    └── ui/
        ├── app/                 # Worker web UI
        │   ├── src/
        │   ├── package.json
        │   └── vite.config.ts
        └── packages/
            ├── llm-worker-sdk/  # HTTP client for worker
            │   ├── src/
            │   ├── package.json
            │   └── tsconfig.json
            └── llm-worker-react/ # React hooks for worker
                ├── src/
                ├── package.json
                └── tsconfig.json

frontend/
├── apps/
│   ├── commercial/              # Marketing site
│   ├── user-docs/               # Documentation site
│   └── web-ui/                  # DEPRECATED
│
└── packages/
    ├── rbee-ui/                 # Shared Storybook
    ├── rbee-sdk/                # DEPRECATED
    ├── rbee-react/              # DEPRECATED
    └── tailwind-config/         # Shared Tailwind

## Numbering Convention

| Prefix | Component Type | Examples |
|--------|---------------|----------|
| **00_** | Keeper/Orchestrator | `00_rbee_keeper` |
| **10_** | Queen/Scheduler | `10_queen_rbee` |
| **20_** | Hive/Manager | `20_rbee_hive` |
| **30_** | Workers | `30_llm_worker_rbee`, `30_comfy_worker_rbee` |
| **40_** | Future: Adapters | `40_vllm_adapter` |
| **50_** | Future: Tools | `50_monitoring_tool` |

**Note:** Workers all use `30_` prefix since they're at the same hierarchical level.

## Mapping: Binary → UI

| Binary | UI | Relationship |
|--------|-----|-------------|
| `bin/00_rbee_keeper/` | `bin/00_rbee_keeper/ui/` | Tauri app (no app subfolder) |
| `bin/10_queen_rbee/` | `bin/10_queen_rbee/ui/app/` | Binary serves UI as static files |
| `bin/20_rbee_hive/` | `bin/20_rbee_hive/ui/app/` | Binary serves UI as static files |
| `bin/30_llm_worker_rbee/` | `bin/30_llm_worker_rbee/ui/app/` | Binary serves UI as static files |

## Benefits

### 1. Clear Correspondence
```bash
# Want to find the UI for a binary?
ls bin/10_queen_rbee/src     # Binary source
ls bin/10_queen_rbee/ui      # UI source

# Everything for one component in one place
```

### 2. Easy Navigation
```bash
# Jump between binary and UI
cd bin/10_queen_rbee/src     # Binary
cd ../ui/app                 # UI
```

### 3. Consistent Naming
- No more `ui-queen-rbee` vs `queen-rbee` confusion
- No more `GUI` vs `ui` vs `web-ui` inconsistency
- Clear hierarchy: 00 → 10 → 20 → 30

### 4. Scalable
```bash
# Add new worker type?
bin/30_whisper_worker_rbee/
  ├── src/                # Binary
  └── ui/app/             # UI

# Add new component?
bin/40_new_component/
  ├── src/                # Binary
  └── ui/app/             # UI
```

## Migration from Old Structure

### Before (Inconsistent)
```
bin/00_rbee_keeper/GUI/              # ❌ Nested in binary
frontend/apps/web-ui/                # ❌ Generic name
frontend/apps/00_rbee_keeper/        # ❌ Separate from binary
frontend/apps/10_queen_rbee/         # ❌ Separate from binary
```

### After (Consistent)
```
bin/00_rbee_keeper/ui/               # ✅ Co-located with binary
bin/10_queen_rbee/ui/app/            # ✅ Co-located with binary
bin/20_rbee_hive/ui/app/             # ✅ Co-located with binary
bin/30_llm_worker_rbee/ui/app/       # ✅ Co-located with binary
```

## Tauri Configuration

**File:** `bin/00_rbee_keeper/tauri.conf.json`

```json
{
  "build": {
    "devPath": "http://localhost:5173",
    "distDir": "../../../frontend/apps/00_rbee_keeper/dist"
  }
}
```

**Note:** Tauri binary points to UI in `frontend/apps/00_rbee_keeper/`

## pnpm-workspace.yaml

```yaml
packages:
  # Commercial & Docs (unchanged)
  - frontend/apps/commercial
  - frontend/apps/user-docs
  
  # rbee UIs (numbered to match bin/)
  - frontend/apps/00_rbee_keeper
  - frontend/apps/10_queen_rbee
  - frontend/apps/20_rbee_hive
  - frontend/apps/30_llm_worker_rbee
  - frontend/apps/30_comfy_worker_rbee
  - frontend/apps/30_vllm_worker_rbee
  
  # Shared packages (unchanged)
  - frontend/packages/rbee-ui
  - frontend/packages/rbee-sdk
  - frontend/packages/rbee-react
  - frontend/packages/tailwind-config
```

## Static File Serving

Each binary (except keeper) serves its UI:

```rust
// bin/10_queen_rbee/src/main.rs
use tower_http::services::ServeDir;

let app = Router::new()
    .route("/api/*", /* API routes */)
    .nest_service("/ui", ServeDir::new("../../../frontend/apps/10_queen_rbee/dist"));
```

**Production URLs:**
- Queen: `http://localhost:7833/ui`
- Hive: `http://localhost:7835/ui`
- Worker: `http://localhost:8080/ui`

## Development Workflow

### Start Keeper GUI
```bash
# Terminal 1: Frontend dev server
cd frontend/apps/00_rbee_keeper
pnpm dev

# Terminal 2: Tauri
cd bin/00_rbee_keeper
cargo tauri dev
```

### Start Queen UI
```bash
# Terminal 1: Frontend dev server
cd frontend/apps/10_queen_rbee
pnpm dev

# Terminal 2: Queen binary
cd bin/10_queen_rbee
cargo run
```

### Build All UIs
```bash
# From root
pnpm --filter "frontend/apps/*_*" build
```

## File Checklist

### Migration Steps

1. **Move keeper GUI:**
   ```bash
   mv bin/00_rbee_keeper/GUI frontend/apps/00_rbee_keeper
   ```

2. **Rename queen UI:**
   ```bash
   mv frontend/apps/web-ui frontend/apps/10_queen_rbee
   ```

3. **Create hive UI:**
   ```bash
   mkdir frontend/apps/20_rbee_hive
   ```

4. **Create worker UIs:**
   ```bash
   mkdir frontend/apps/30_llm_worker_rbee
   mkdir frontend/apps/30_comfy_worker_rbee
   mkdir frontend/apps/30_vllm_worker_rbee
   ```

5. **Update tauri.conf.json:**
   ```json
   "distDir": "../../../frontend/apps/00_rbee_keeper/dist"
   ```

6. **Update pnpm-workspace.yaml:**
   ```yaml
   - frontend/apps/00_rbee_keeper
   - frontend/apps/10_queen_rbee
   - frontend/apps/20_rbee_hive
   - frontend/apps/30_*_worker_rbee
   ```

7. **Update package.json names:**
   ```json
   "@rbee/00-keeper-gui"
   "@rbee/10-queen-ui"
   "@rbee/20-hive-ui"
   "@rbee/30-llm-worker-ui"
   ```

## Verification

```bash
# Check structure
ls -la bin/
ls -la frontend/apps/

# Verify parity
for dir in bin/*/; do
  num=$(basename "$dir" | cut -d_ -f1)
  echo "Binary: $dir"
  echo "UI:     frontend/apps/$(basename "$dir")"
  echo
done
```

## Expected Output

```
✅ bin/00_rbee_keeper → frontend/apps/00_rbee_keeper
✅ bin/10_queen_rbee → frontend/apps/10_queen_rbee
✅ bin/20_rbee_hive → frontend/apps/20_rbee_hive
✅ bin/30_llm_worker_rbee → frontend/apps/30_llm_worker_rbee
✅ Clear 1:1 mapping
✅ Consistent numbering
✅ Easy to navigate
```

---

**Status:** 📋 STRUCTURE DEFINED  
**Benefit:** Clear parity between binaries and UIs, easy to understand and maintain
