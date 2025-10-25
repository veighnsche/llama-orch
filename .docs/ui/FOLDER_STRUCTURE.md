# Folder Structure: bin/ and frontend/apps/ Parity

**TEAM-293: Aligned folder naming between binaries and UIs**

## Design Principle

**Parity:** Each binary in `bin/` has a corresponding UI in `frontend/apps/` with the same numbered prefix.

## Structure

```
/home/vince/Projects/llama-orch/

├── bin/                                    # Rust binaries
│   ├── 00_rbee_keeper/                     # Keeper binary (Tauri)
│   │   ├── src/
│   │   ├── Cargo.toml
│   │   └── tauri.conf.json                 # Points to ../../../frontend/apps/00_rbee_keeper
│   │
│   ├── 10_queen_rbee/                      # Queen binary
│   │   ├── src/
│   │   └── Cargo.toml
│   │
│   ├── 20_rbee_hive/                       # Hive binary
│   │   ├── src/
│   │   └── Cargo.toml
│   │
│   └── 30_llm_worker_rbee/                 # Worker binary
│       ├── src/
│       └── Cargo.toml
│
└── frontend/apps/                          # React UIs
    ├── 00_rbee_keeper/                     # Keeper UI (Tauri frontend)
    │   ├── src/
    │   ├── package.json
    │   ├── vite.config.ts
    │   └── index.html
    │
    ├── 10_queen_rbee/                      # Queen UI (static)
    │   ├── src/
    │   ├── package.json
    │   └── vite.config.ts
    │
    ├── 20_rbee_hive/                       # Hive UI (static)
    │   ├── src/
    │   ├── package.json
    │   └── vite.config.ts
    │
    ├── 30_llm_worker_rbee/                 # LLM Worker UI (static)
    │   ├── src/
    │   ├── package.json
    │   └── vite.config.ts
    │
    ├── 30_comfy_worker_rbee/               # ComfyUI Worker UI (static)
    │   └── ...
    │
    ├── 30_vllm_worker_rbee/                # vLLM Worker UI (static)
    │   └── ...
    │
    ├── commercial/                         # Commercial website (unchanged)
    └── user-docs/                          # User docs (unchanged)
```

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
| `bin/00_rbee_keeper/` | `frontend/apps/00_rbee_keeper/` | Tauri app (binary uses UI) |
| `bin/10_queen_rbee/` | `frontend/apps/10_queen_rbee/` | Binary serves UI as static files |
| `bin/20_rbee_hive/` | `frontend/apps/20_rbee_hive/` | Binary serves UI as static files |
| `bin/30_llm_worker_rbee/` | `frontend/apps/30_llm_worker_rbee/` | Binary serves UI as static files |

## Benefits

### 1. Clear Correspondence
```bash
# Want to find the UI for a binary?
ls bin/10_queen_rbee        # Binary
ls frontend/apps/10_queen_rbee  # UI

# Same number = related components
```

### 2. Easy Navigation
```bash
# Jump between binary and UI
cd bin/10_queen_rbee
cd ../../../frontend/apps/10_queen_rbee
```

### 3. Consistent Naming
- No more `ui-queen-rbee` vs `queen-rbee` confusion
- No more `GUI` vs `ui` vs `web-ui` inconsistency
- Clear hierarchy: 00 → 10 → 20 → 30

### 4. Scalable
```bash
# Add new worker type?
bin/30_whisper_worker_rbee/
frontend/apps/30_whisper_worker_rbee/

# Add new component?
bin/40_new_component/
frontend/apps/40_new_component/
```

## Migration from Old Structure

### Before (Inconsistent)
```
bin/00_rbee_keeper/GUI/              # ❌ Nested in binary
frontend/apps/web-ui/                # ❌ Generic name
frontend/apps/ui-rbee-hive/          # ❌ Different naming
frontend/apps/ui-llm-worker-rbee/    # ❌ Different naming
```

### After (Consistent)
```
frontend/apps/00_rbee_keeper/        # ✅ Top-level, numbered
frontend/apps/10_queen_rbee/         # ✅ Numbered, clear
frontend/apps/20_rbee_hive/          # ✅ Numbered, clear
frontend/apps/30_llm_worker_rbee/    # ✅ Numbered, clear
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
