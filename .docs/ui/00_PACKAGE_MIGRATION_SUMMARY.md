# Package Migration Summary

**TEAM-293: From generic to specialized SDKs**

## What Changed

### ❌ DELETED (Generic Packages)

```
frontend/packages/rbee-sdk/          # Generic SDK (tried to do everything)
frontend/packages/rbee-react/        # Generic React hooks
```

**Why deleted:**
- One package tried to handle all binaries (queen + hive + workers)
- Tight coupling between unrelated APIs
- Version conflicts
- Unclear ownership

### ✅ CREATED (Specialized Packages)

```
frontend/packages/
├── 10_queen_rbee/
│   ├── queen-rbee-sdk/              # HTTP client for queen API
│   └── queen-rbee-react/            # React hooks for queen
│
├── 20_rbee_hive/
│   ├── rbee-hive-sdk/               # HTTP client for hive API
│   └── rbee-hive-react/             # React hooks for hive
│
└── 30_workers/
    ├── llm-worker-sdk/              # HTTP client for LLM worker API
    ├── llm-worker-react/            # React hooks for LLM worker
    ├── comfy-worker-sdk/            # HTTP client for ComfyUI worker API
    ├── comfy-worker-react/          # React hooks for ComfyUI worker
    ├── vllm-worker-sdk/             # HTTP client for vLLM worker API
    └── vllm-worker-react/           # React hooks for vLLM worker
```

## Key Rules

### 1. Keeper Exception

**rbee-keeper does NOT have SDK packages.**

**Why:**
- No HTTP API (only CLI)
- Uses Tauri commands directly

### 2. SDKs are HTTP-only

**No WASM, no Rust compilation.**

### 3. One Binary = One SDK

| Binary | SDK Package | React Package |
|--------|-------------|---------------|
| `bin/00_rbee_keeper/` | ❌ None | ❌ None (uses Tauri) |
| `bin/10_queen_rbee/` | `@rbee/queen-rbee-sdk` | `@rbee/queen-rbee-react` |
| `bin/20_rbee_hive/` | `@rbee/rbee-hive-sdk` | `@rbee/rbee-hive-react` |
| `bin/30_llm_worker_rbee/` | `@rbee/llm-worker-sdk` | `@rbee/llm-worker-react` |

---

**Status:** 📋 MIGRATION DEFINED  
**See:** `PACKAGE_STRUCTURE.md` for complete details
