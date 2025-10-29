# TEAM-268: Model Catalog Operations

**Phase:** 2 of 9  
**Estimated Effort:** 16-20 hours  
**Prerequisites:** TEAM-267 complete  
**Blocks:** TEAM-269 (Model Provisioner)

---

## 🎯 Mission

Implement the operation handlers for model catalog in rbee-hive's job_router. Wire up ModelList, ModelGet, and ModelDelete operations.

**Deliverables:**
1. ✅ ModelList operation working
2. ✅ ModelGet operation working
3. ✅ ModelDelete operation working
4. ✅ Narration events for all operations
5. ✅ Integration with job_router.rs
6. ✅ Unit tests

---

## 📁 Files to Modify

```
bin/20_rbee_hive/
├── src/
│   ├── job_router.rs       ← Replace TODO stubs
│   └── main.rs             ← Initialize ModelCatalog
└── Cargo.toml              ← Add model-catalog dependency
```

---

## 🏗️ Implementation Guide

### Step 1: Add Dependency (Cargo.toml)

```toml
[dependencies]
rbee-hive-model-catalog = { path = "../25_rbee_hive_crates/model-catalog" }
```

### Step 2: Initialize in main.rs

```rust
// TEAM-268: Initialize model catalog
use rbee_hive_model_catalog::ModelCatalog;

let model_catalog = Arc::new(ModelCatalog::new());

let job_state = http::jobs::HiveState {
    registry: job_registry,
    model_catalog: model_catalog.clone(),  // TEAM-268: Add to state
};
```

### Step 3: Update JobState (job_router.rs)

```rust
#[derive(Clone)]
pub struct JobState {
    pub registry: Arc<JobRegistry<String>>,
    pub model_catalog: Arc<ModelCatalog>,  // TEAM-268: Added
    // TODO: Add worker_registry when implemented
}
```

### Step 4: Implement ModelList

```rust
Operation::ModelList { hive_id } => {
    // TEAM-268: Implemented model list
    NARRATE
        .action("model_list_start")
        .job_id(&job_id)
        .context(&hive_id)
        .human("📋 Listing models on hive '{}'")
        .emit();

    let models = state.model_catalog.list();
    
    NARRATE
        .action("model_list_result")
        .job_id(&job_id)
        .context(models.len().to_string())
        .human("Found {} model(s)")
        .emit();
    
    // Format as JSON table
    if models.is_empty() {
        NARRATE
            .action("model_list_empty")
            .job_id(&job_id)
            .human("No models found")
            .emit();
    } else {
        for model in &models {
            NARRATE
                .action("model_list_entry")
                .job_id(&job_id)
                .context(&model.id)
                .context(&model.name)
                .context(&format!("{:.2} GB", model.size_bytes as f64 / 1_000_000_000.0))
                .human("  {} | {} | {}")
                .emit();
        }
    }
}
```

### Step 5: Implement ModelGet

```rust
Operation::ModelGet { hive_id, id } => {
    // TEAM-268: Implemented model get
    NARRATE
        .action("model_get_start")
        .job_id(&job_id)
        .context(&hive_id)
        .context(&id)
        .human("🔍 Getting model '{}' on hive '{}'")
        .emit();

    match state.model_catalog.get(&id) {
        Ok(model) => {
            NARRATE
                .action("model_get_found")
                .job_id(&job_id)
                .context(&model.id)
                .context(&model.name)
                .context(&model.path.display().to_string())
                .human("✅ Model: {} | Name: {} | Path: {}")
                .emit();
            
            // Emit model details as JSON
            let json = serde_json::to_string_pretty(&model)
                .unwrap_or_else(|_| "Failed to serialize".to_string());
            
            NARRATE
                .action("model_get_details")
                .job_id(&job_id)
                .human(&json)
                .emit();
        }
        Err(e) => {
            NARRATE
                .action("model_get_error")
                .job_id(&job_id)
                .context(&id)
                .human("❌ Model '{}' not found: {}")
                .emit();
            return Err(e);
        }
    }
}
```

### Step 6: Implement ModelDelete

```rust
Operation::ModelDelete { hive_id, id } => {
    // TEAM-268: Implemented model delete
    NARRATE
        .action("model_delete_start")
        .job_id(&job_id)
        .context(&hive_id)
        .context(&id)
        .human("🗑️  Deleting model '{}' on hive '{}'")
        .emit();

    match state.model_catalog.remove(&id) {
        Ok(model) => {
            NARRATE
                .action("model_delete_catalog")
                .job_id(&job_id)
                .context(&id)
                .human("✅ Removed '{}' from catalog")
                .emit();
            
            // TODO: TEAM-269 will add actual file deletion
            NARRATE
                .action("model_delete_files")
                .job_id(&job_id)
                .context(&model.path.display().to_string())
                .human("⚠️  Files still on disk at: {} (deletion TODO)")
                .emit();
        }
        Err(e) => {
            NARRATE
                .action("model_delete_error")
                .job_id(&job_id)
                .context(&id)
                .human("❌ Failed to delete model '{}': {}")
                .emit();
            return Err(e);
        }
    }
}
```

---

## ✅ Acceptance Criteria

- [ ] ModelCatalog initialized in main.rs
- [ ] JobState includes model_catalog field
- [ ] ModelList operation implemented with narration
- [ ] ModelGet operation implemented with narration
- [ ] ModelDelete operation implemented with narration
- [ ] All operations emit proper events
- [ ] `cargo check --bin rbee-hive` passes
- [ ] Manual testing shows operations work

---

## 🧪 Testing Commands

```bash
# Check compilation
cargo check --bin rbee-hive

# Run rbee-hive
cargo run --bin rbee-hive -- --port 8600

# In another terminal, test operations:
# (Requires rbee-keeper or curl)
curl -X POST http://localhost:8600/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{"operation": "model_list", "hive_id": "localhost"}'
```

---

## 📝 Handoff Checklist

Create `TEAM_268_HANDOFF.md` with:

- [ ] All 3 operations working
- [ ] Example narration output
- [ ] Known limitations (file deletion TODO)
- [ ] Notes for TEAM-269

---

**TEAM-268: Wire it up! Make those operations sing! 🎵**
