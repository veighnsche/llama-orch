# TEAM-269: Model Provisioner

**Phase:** 3 of 9  
**Estimated Effort:** 24-32 hours  
**Prerequisites:** TEAM-268 complete  
**Blocks:** TEAM-270 (Worker Registry)

---

## 🎯 Mission

Implement model downloading from HuggingFace Hub. Add ModelDownload operation to job_router to enable actual model file downloads.

**Deliverables:**
1. ✅ ModelProvisioner struct
2. ✅ download_model() function with progress tracking
3. ✅ File management in ~/.cache/rbee/models/
4. ✅ ModelDownload operation wired up
5. ✅ Narration events for progress
6. ✅ Unit tests

---

## 📁 Files to Create/Modify

```
bin/25_rbee_hive_crates/model-provisioner/
├── src/
│   ├── lib.rs          ← Implement ModelProvisioner
│   └── download.rs     ← Download logic (optional separate module)
├── Cargo.toml          ← Add dependencies
└── README.md           ← Document API

bin/20_rbee_hive/
├── src/
│   ├── job_router.rs   ← Wire up ModelDownload operation
│   └── main.rs         ← Initialize ModelProvisioner
└── Cargo.toml          ← Add model-provisioner dependency
```

---

## 🏗️ Implementation Guide

### Step 1: Add Dependencies (model-provisioner/Cargo.toml)

```toml
[dependencies]
anyhow = "1.0"
tokio = { workspace = true, features = ["full"] }
reqwest = { version = "0.11", features = ["stream"] }
futures = "0.3"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
dirs = "5.0"

# Catalog integration
rbee-hive-model-catalog = { path = "../model-catalog" }

# Narration
observability-narration-core = { path = "../../99_shared_crates/narration-core" }
```

### Step 2: Implement ModelProvisioner (lib.rs)

```rust
// TEAM-269: Model provisioner implementation
use anyhow::{anyhow, Result};
use observability_narration_core::NarrationFactory;
use rbee_hive_model_catalog::{ModelCatalog, ModelEntry, ModelStatus};
use std::path::PathBuf;
use std::sync::Arc;

const NARRATE: NarrationFactory = NarrationFactory::new("model-prov");

/// Model provisioner for downloading models from HuggingFace
pub struct ModelProvisioner {
    catalog: Arc<ModelCatalog>,
    cache_dir: PathBuf,
}

impl ModelProvisioner {
    /// Create a new model provisioner
    pub fn new(catalog: Arc<ModelCatalog>) -> Result<Self> {
        // Use the same directory as ModelCatalog
        // Linux/Mac: ~/.cache/rbee/models/
        // Windows: %LOCALAPPDATA%\rbee\models\
        let cache_dir = dirs::cache_dir()
            .ok_or_else(|| anyhow!("Cannot determine cache directory"))?
            .join("rbee")
            .join("models");

        std::fs::create_dir_all(&cache_dir)?;

        Ok(Self { catalog, cache_dir })
    }

    /// Download a model from HuggingFace Hub
    ///
    /// # Arguments
    /// * `job_id` - Job ID for narration routing
    /// * `model_id` - Model identifier (e.g., "meta-llama/Llama-2-7b-chat-hf")
    ///
    /// # Returns
    /// Model ID on success
    pub async fn download_model(&self, job_id: &str, model_id: &str) -> Result<String> {
        NARRATE
            .action("download_start")
            .job_id(job_id)
            .context(model_id)
            .human("📥 Starting download: {}")
            .emit();

        // Check if already exists
        if self.catalog.contains(model_id) {
            NARRATE
                .action("download_exists")
                .job_id(job_id)
                .context(model_id)
                .human("⚠️  Model '{}' already exists in catalog")
                .emit();
            return Err(anyhow!("Model '{}' already exists", model_id));
        }

        // Create model entry with Downloading status
        let model_path = self.cache_dir.join(model_id);
        let mut model = ModelEntry::new(
            model_id.to_string(),
            model_id.to_string(), // Use model_id as name for now
            model_path.clone(),
            0, // Size unknown until download
        );
        model.status = ModelStatus::Downloading { progress: 0.0 };

        // Add to catalog (creates directory and metadata.yaml)
        self.catalog.add(model)?;

        NARRATE
            .action("download_catalog_added")
            .job_id(job_id)
            .context(model_id)
            .human("📝 Added to catalog with Downloading status")
            .emit();

        // Create model directory
        tokio::fs::create_dir_all(&model_path).await?;

        // TODO: Actual HuggingFace download
        // For v0.1.0, create placeholder files
        // TEAM-269: This is where you'd implement actual HF Hub API calls
        //
        // Example flow:
        // 1. GET https://huggingface.co/api/models/{model_id}
        // 2. Parse response to get file list
        // 3. Download each file with progress tracking
        // 4. Update catalog status with progress
        //
        // For now, simulate download with delays

        NARRATE
            .action("download_progress")
            .job_id(job_id)
            .context("25")
            .human("Progress: {}%")
            .emit();

        self.catalog
            .update_status(model_id, ModelStatus::Downloading { progress: 0.25 })?;

        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

        NARRATE
            .action("download_progress")
            .job_id(job_id)
            .context("50")
            .human("Progress: {}%")
            .emit();

        self.catalog
            .update_status(model_id, ModelStatus::Downloading { progress: 0.5 })?;

        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

        NARRATE
            .action("download_progress")
            .job_id(job_id)
            .context("75")
            .human("Progress: {}%")
            .emit();

        self.catalog
            .update_status(model_id, ModelStatus::Downloading { progress: 0.75 })?;

        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

        // Create placeholder file to indicate download "complete"
        let placeholder_file = model_path.join("model.placeholder");
        tokio::fs::write(&placeholder_file, b"Placeholder model file").await?;

        // Update status to Ready
        self.catalog.update_status(model_id, ModelStatus::Ready)?;

        NARRATE
            .action("download_complete")
            .job_id(job_id)
            .context(model_id)
            .human("✅ Download complete: {}")
            .emit();

        Ok(model_id.to_string())
    }

    /// Delete model files (called by ModelDelete operation)
    pub async fn delete_model_files(&self, model_id: &str) -> Result<()> {
        let model = self.catalog.get(model_id)?;

        if model.path.exists() {
            tokio::fs::remove_dir_all(&model.path).await?;
        }

        Ok(())
    }
}

impl Default for ModelProvisioner {
    fn default() -> Self {
        let catalog = Arc::new(ModelCatalog::new().expect("Failed to create catalog"));
        Self::new(catalog).expect("Failed to create provisioner")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_download_model() {
        let catalog = Arc::new(ModelCatalog::new().unwrap());
        let provisioner = ModelProvisioner::new(catalog.clone()).unwrap();

        let result = provisioner
            .download_model("test-job", "test-model")
            .await;

        assert!(result.is_ok());
        assert!(catalog.contains("test-model"));

        // Cleanup
        catalog.remove("test-model").ok();
    }

    #[tokio::test]
    async fn test_download_duplicate() {
        let catalog = Arc::new(ModelCatalog::new().unwrap());
        let provisioner = ModelProvisioner::new(catalog.clone()).unwrap();

        // First download
        provisioner
            .download_model("test-job", "test-dup")
            .await
            .unwrap();

        // Second download should fail
        let result = provisioner.download_model("test-job", "test-dup").await;
        assert!(result.is_err());

        // Cleanup
        catalog.remove("test-dup").ok();
    }
}
```

### Step 3: Wire Up in job_router.rs

```rust
// Add to imports
use rbee_hive_model_provisioner::ModelProvisioner;

// Add to JobState
pub struct JobState {
    pub registry: Arc<JobRegistry<String>>,
    pub model_catalog: Arc<ModelCatalog>,
    pub model_provisioner: Arc<ModelProvisioner>, // TEAM-269: Added
    // TODO: Add worker_registry when implemented
}

// Replace ModelDownload TODO stub
Operation::ModelDownload { hive_id, model } => {
    // TEAM-269: Implemented model download
    NARRATE
        .action("model_download_start")
        .job_id(&job_id)
        .context(&hive_id)
        .context(&model)
        .human("📥 Downloading model '{}' on hive '{}'")
        .emit();

    match state.model_provisioner.download_model(&job_id, &model).await {
        Ok(model_id) => {
            NARRATE
                .action("model_download_complete")
                .job_id(&job_id)
                .context(&model_id)
                .human("✅ Model downloaded: {}")
                .emit();
        }
        Err(e) => {
            NARRATE
                .action("model_download_error")
                .job_id(&job_id)
                .context(&model)
                .context(&e.to_string())
                .human("❌ Download failed for '{}': {}")
                .emit();
            return Err(e);
        }
    }
}
```

### Step 4: Initialize in main.rs

```rust
// Add import
use rbee_hive_model_provisioner::ModelProvisioner;

// After model_catalog initialization
let model_provisioner = Arc::new(
    ModelProvisioner::new(model_catalog.clone())
        .expect("Failed to initialize model provisioner")
);

NARRATE
    .action("provisioner_init")
    .human("📦 Model provisioner initialized")
    .emit();

// Update HiveState
let job_state = http::jobs::HiveState {
    registry: job_registry,
    model_catalog,
    model_provisioner, // TEAM-269: Added
};
```

### Step 5: Update http/jobs.rs

```rust
// Add import
use rbee_hive_model_provisioner::ModelProvisioner;

// Update HiveState
pub struct HiveState {
    pub registry: Arc<JobRegistry<String>>,
    pub model_catalog: Arc<ModelCatalog>,
    pub model_provisioner: Arc<ModelProvisioner>, // TEAM-269: Added
    // TODO: Add worker_registry when implemented
}

// Update From implementation
impl From<HiveState> for crate::job_router::JobState {
    fn from(state: HiveState) -> Self {
        Self {
            registry: state.registry,
            model_catalog: state.model_catalog,
            model_provisioner: state.model_provisioner, // TEAM-269: Added
        }
    }
}
```

---

## ✅ Acceptance Criteria

- [ ] ModelProvisioner struct implemented
- [ ] download_model() function working
- [ ] Progress tracking via ModelCatalog status updates
- [ ] ModelDownload operation wired up in job_router.rs
- [ ] Files stored in ~/.cache/rbee/models/
- [ ] Narration events emitted with `.job_id()`
- [ ] Unit tests passing (2+ tests)
- [ ] `cargo check --bin rbee-hive` passes
- [ ] `cargo test --package rbee-hive-model-provisioner` passes

---

## 🧪 Testing Commands

```bash
# Check compilation
cargo check --package rbee-hive-model-provisioner
cargo check --bin rbee-hive

# Run unit tests
cargo test --package rbee-hive-model-provisioner

# Manual testing
cargo run --bin rbee-hive -- --port 8600

# In another terminal
curl -X POST http://localhost:8600/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{"operation": "model_download", "hive_id": "localhost", "model": "test-model"}'
```

---

## 📝 Handoff Checklist

Create `TEAM_269_HANDOFF.md` with:

- [ ] ModelProvisioner implementation complete
- [ ] download_model() function working
- [ ] Progress tracking demonstrated
- [ ] Example narration output
- [ ] Known limitations (placeholder download)
- [ ] Notes for TEAM-270

---

## 🚨 Known Limitations

### 1. Placeholder Download

**Current:** Creates placeholder files instead of actual HuggingFace downloads.

**Why:** HuggingFace Hub API integration is complex and out of scope for v0.1.0.

**Future:** TEAM-269 or later can implement actual HF Hub downloads:
```rust
// Example HF Hub API call
let url = format!("https://huggingface.co/api/models/{}", model_id);
let response = reqwest::get(&url).await?;
let model_info: ModelInfo = response.json().await?;

// Download each file
for file in model_info.siblings {
    let file_url = format!("https://huggingface.co/{}/resolve/main/{}", 
        model_id, file.rfilename);
    // Download with progress tracking
}
```

### 2. No Resume Support

**Current:** If download fails, must start over.

**Future:** Implement partial download tracking and resume.

---

## 🎓 Learning Resources

- **HuggingFace Hub API:** https://huggingface.co/docs/hub/api
- **reqwest streaming:** https://docs.rs/reqwest/latest/reqwest/
- **tokio fs:** https://docs.rs/tokio/latest/tokio/fs/

---

## 📚 Reference Implementations

- **TEAM-267:** ModelCatalog (similar patterns)
- **TEAM-268:** Model operations (narration patterns)
- **daemon-lifecycle:** Process spawning patterns

---

**TEAM-269: Download those models! 📥🚀**
