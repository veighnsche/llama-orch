# Checkpoint 2: Model Catalog System

**Created by:** TEAM-022  
**Checkpoint:** CP2  
**Duration:** Week 2 (5 days)  
**Status:** Pending  
**Depends On:** CP1 Complete

---

## Objective

Implement the model catalog system:
1. Define catalog JSON schema
2. Implement catalog management in pool-core
3. Wire catalog commands in rbee-hive
4. Create catalogs on all pools
5. Register all test models

**Why This Second:** Need catalog before we can download models (catalog tracks what's downloaded).

---

## Work Units

### WU2.1: Design Catalog Schema (Day 1)

**Location:** `bin/shared-crates/pool-core/src/catalog.rs`

**Tasks:**
1. Define JSON schema
2. Implement serde types
3. Add validation logic
4. Write schema documentation

**Catalog Format:**
```json
{
  "version": "1.0",
  "pool_id": "mac.home.arpa",
  "updated_at": "2025-10-09T15:00:00Z",
  "models": [
    {
      "id": "tinyllama",
      "name": "TinyLlama 1.1B Chat",
      "path": ".test-models/tinyllama",
      "format": "safetensors",
      "size_gb": 2.2,
      "architecture": "llama",
      "downloaded": true,
      "backends": ["cpu", "metal", "cuda"],
      "metadata": {
        "repo": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "context_length": 2048,
        "vocab_size": 32000,
        "quantization": "Q4_K_M"
      }
    }
  ]
}
```

**Rust Types:**
```rust
// TEAM-022: Model catalog types
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCatalog {
    pub version: String,
    pub pool_id: String,
    pub updated_at: String,
    pub models: Vec<ModelEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEntry {
    pub id: String,
    pub name: String,
    pub path: PathBuf,
    pub format: String,
    pub size_gb: f64,
    pub architecture: String,
    pub downloaded: bool,
    pub backends: Vec<String>,
    pub metadata: serde_json::Value,
}

impl ModelCatalog {
    pub fn new(pool_id: String) -> Self {
        Self {
            version: "1.0".to_string(),
            pool_id,
            updated_at: chrono::Utc::now().to_rfc3339(),
            models: Vec::new(),
        }
    }
    
    pub fn load(path: &Path) -> Result<Self>;
    pub fn save(&self, path: &Path) -> Result<()>;
    pub fn add_model(&mut self, entry: ModelEntry) -> Result<()>;
    pub fn remove_model(&mut self, id: &str) -> Result<()>;
    pub fn find_model(&self, id: &str) -> Option<&ModelEntry>;
    pub fn find_model_mut(&mut self, id: &str) -> Option<&mut ModelEntry>;
}
```

**Success Criteria:**
- [ ] Schema documented in .specs
- [ ] Rust types compile
- [ ] Serde serialization works
- [ ] Validation logic implemented

---

### WU2.2: Implement Catalog Management (Day 2)

**Location:** `bin/shared-crates/pool-core/src/catalog.rs`

**Tasks:**
1. Implement load/save
2. Implement add/remove
3. Implement search/filter
4. Add unit tests

**Implementation:**
```rust
// TEAM-022: Catalog persistence
impl ModelCatalog {
    pub fn load(path: &Path) -> Result<Self> {
        if !path.exists() {
            anyhow::bail!("Catalog not found: {}", path.display());
        }
        let content = std::fs::read_to_string(path)?;
        let catalog: Self = serde_json::from_str(&content)?;
        Ok(catalog)
    }
    
    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }
    
    pub fn add_model(&mut self, entry: ModelEntry) -> Result<()> {
        if self.find_model(&entry.id).is_some() {
            anyhow::bail!("Model {} already exists", entry.id);
        }
        self.models.push(entry);
        self.updated_at = chrono::Utc::now().to_rfc3339();
        Ok(())
    }
    
    pub fn remove_model(&mut self, id: &str) -> Result<()> {
        let index = self.models.iter()
            .position(|m| m.id == id)
            .ok_or_else(|| anyhow::anyhow!("Model {} not found", id))?;
        self.models.remove(index);
        self.updated_at = chrono::Utc::now().to_rfc3339();
        Ok(())
    }
}
```

**Unit Tests:**
```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_catalog_create() {
        let catalog = ModelCatalog::new("test-pool".to_string());
        assert_eq!(catalog.pool_id, "test-pool");
        assert_eq!(catalog.models.len(), 0);
    }
    
    #[test]
    fn test_catalog_add_model() {
        let mut catalog = ModelCatalog::new("test-pool".to_string());
        let entry = ModelEntry {
            id: "test-model".to_string(),
            // ... other fields
        };
        catalog.add_model(entry).unwrap();
        assert_eq!(catalog.models.len(), 1);
    }
    
    #[test]
    fn test_catalog_save_load() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("catalog.json");
        
        let mut catalog = ModelCatalog::new("test-pool".to_string());
        // Add test model
        catalog.save(&path).unwrap();
        
        let loaded = ModelCatalog::load(&path).unwrap();
        assert_eq!(loaded.pool_id, catalog.pool_id);
    }
}
```

**Success Criteria:**
- [ ] Load/save works
- [ ] Add/remove works
- [ ] All unit tests pass
- [ ] Error handling is robust

---

### WU2.3: Wire Catalog Commands in rbee-hive (Day 3)

**Location:** `bin/rbee-hive/src/commands/models.rs`

**Tasks:**
1. Implement `rbee-hive models catalog`
2. Implement `rbee-hive models register`
3. Implement `rbee-hive models unregister`
4. Add colored output

**Commands:**
```rust
// TEAM-022: Model catalog commands
use pool_core::catalog::{ModelCatalog, ModelEntry};
use colored::Colorize;

pub fn handle_catalog_command() -> Result<()> {
    let catalog_path = PathBuf::from(".test-models/catalog.json");
    
    let catalog = match ModelCatalog::load(&catalog_path) {
        Ok(c) => c,
        Err(_) => {
            println!("{}", "No catalog found. Create one with 'rbee-hive models register'".yellow());
            return Ok(());
        }
    };
    
    println!("\n{}", format!("Model Catalog for {}", catalog.pool_id).bold());
    println!("{}", "=".repeat(80));
    println!("{:<15} {:<30} {:<12} {:<10}", 
        "ID".bold(), "Name".bold(), "Downloaded".bold(), "Size".bold());
    println!("{}", "-".repeat(80));
    
    for model in &catalog.models {
        let status = if model.downloaded { "✅".green() } else { "❌".red() };
        println!("{:<15} {:<30} {:<12} {:.1} GB", 
            model.id, model.name, status, model.size_gb);
    }
    
    println!("{}", "=".repeat(80));
    println!("Total models: {}\n", catalog.models.len());
    
    Ok(())
}

pub fn handle_register_command(
    id: String,
    name: String,
    repo: String,
    architecture: String,
) -> Result<()> {
    let catalog_path = PathBuf::from(".test-models/catalog.json");
    
    let mut catalog = ModelCatalog::load(&catalog_path)
        .unwrap_or_else(|_| {
            let pool_id = hostname::get()
                .unwrap()
                .to_string_lossy()
                .to_string();
            ModelCatalog::new(pool_id)
        });
    
    let entry = ModelEntry {
        id: id.clone(),
        name,
        path: PathBuf::from(format!(".test-models/{}", id)),
        format: "safetensors".to_string(),
        size_gb: 0.0,  // Will be updated after download
        architecture,
        downloaded: false,
        backends: vec!["cpu".to_string(), "metal".to_string(), "cuda".to_string()],
        metadata: serde_json::json!({
            "repo": repo,
        }),
    };
    
    catalog.add_model(entry)?;
    catalog.save(&catalog_path)?;
    
    println!("{}", format!("✅ Model {} registered", id).green());
    
    Ok(())
}
```

**Success Criteria:**
- [ ] `rbee-hive models catalog` displays catalog
- [ ] `rbee-hive models register` adds model
- [ ] `rbee-hive models unregister` removes model
- [ ] Output is colored and formatted

---

### WU2.4: Create Catalogs on All Pools (Day 4)

**Tasks:**
1. Create catalog on mac.home.arpa
2. Create catalog on workstation.home.arpa
3. Create catalog on blep.home.arpa (if needed)
4. Verify catalog persistence

**Test Models to Register:**
1. **TinyLlama** (already have)
2. **Qwen 0.5B** (smallest)
3. **Phi-3 Mini** (medium)
4. **Mistral 7B** (largest)

**Commands to Run:**
```bash
# On mac.home.arpa
rbee-hive models register tinyllama \
    --name "TinyLlama 1.1B Chat" \
    --repo "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
    --architecture llama

rbee-hive models register qwen-0.5b \
    --name "Qwen2.5 0.5B Instruct" \
    --repo "Qwen/Qwen2.5-0.5B-Instruct" \
    --architecture qwen

rbee-hive models register phi3 \
    --name "Phi-3 Mini 4K Instruct" \
    --repo "microsoft/Phi-3-mini-4k-instruct" \
    --architecture phi

rbee-hive models register mistral \
    --name "Mistral 7B Instruct v0.2" \
    --repo "mistralai/Mistral-7B-Instruct-v0.2" \
    --architecture mistral

# Verify
rbee-hive models catalog
```

**Expected Output:**
```
Model Catalog for mac.home.arpa
================================================================================
ID              Name                           Downloaded   Size      
--------------------------------------------------------------------------------
tinyllama       TinyLlama 1.1B Chat            ✅           2.2 GB
qwen-0.5b       Qwen2.5 0.5B Instruct          ❌           0.0 GB
phi3            Phi-3 Mini 4K Instruct         ❌           0.0 GB
mistral         Mistral 7B Instruct v0.2       ❌           0.0 GB
================================================================================
Total models: 4
```

**Success Criteria:**
- [ ] Catalog created on mac.home.arpa
- [ ] Catalog created on workstation.home.arpa
- [ ] All 4 models registered
- [ ] Catalogs persist across restarts

---

### WU2.5: Remote Catalog Access via rbee-keeper (Day 5)

**Location:** `bin/rbee-keeper/src/commands/pool.rs`

**Tasks:**
1. Implement `llorch pool models catalog --host <host>`
2. Implement `llorch pool models register --host <host>`
3. Test remote catalog access
4. Document usage

**Implementation:**
```rust
// TEAM-022: Remote catalog commands
use crate::ssh::execute_remote_command;

pub fn handle_remote_catalog(host: &str) -> Result<()> {
    let command = "cd ~/Projects/llama-orch && rbee-hive models catalog";
    let output = execute_remote_command(host, command)?;
    println!("{}", output);
    Ok(())
}

pub fn handle_remote_register(
    host: &str,
    id: &str,
    name: &str,
    repo: &str,
    architecture: &str,
) -> Result<()> {
    let command = format!(
        "cd ~/Projects/llama-orch && rbee-hive models register {} --name '{}' --repo '{}' --architecture {}",
        id, name, repo, architecture
    );
    let output = execute_remote_command(host, &command)?;
    println!("{}", output);
    Ok(())
}
```

**Test Commands:**
```bash
# From blep (orchestrator host)
llorch pool models catalog --host mac.home.arpa
llorch pool models catalog --host workstation.home.arpa

# Register model remotely
llorch pool models register qwen-0.5b \
    --host mac.home.arpa \
    --name "Qwen2.5 0.5B Instruct" \
    --repo "Qwen/Qwen2.5-0.5B-Instruct" \
    --architecture qwen
```

**Success Criteria:**
- [ ] Remote catalog viewing works
- [ ] Remote model registration works
- [ ] SSH errors are handled gracefully
- [ ] Output is identical to local execution

---

## Checkpoint Gate: CP2 Verification

**Before proceeding to CP3, verify:**

### Catalog System
- [ ] Catalog JSON schema is documented
- [ ] Catalog load/save works
- [ ] Catalog add/remove works
- [ ] Unit tests pass

### Local Commands
- [ ] `rbee-hive models catalog` works
- [ ] `rbee-hive models register` works
- [ ] `rbee-hive models unregister` works
- [ ] Output is formatted and colored

### Remote Commands
- [ ] `llorch pool models catalog --host mac` works
- [ ] `llorch pool models register --host mac` works
- [ ] SSH connectivity is stable

### Catalogs Created
- [ ] Catalog exists on mac.home.arpa
- [ ] Catalog exists on workstation.home.arpa
- [ ] All 4 models registered on each pool
- [ ] Catalogs persist across restarts

### Code Quality
- [ ] `cargo fmt --all` clean
- [ ] `cargo clippy --all` clean
- [ ] Team signatures added
- [ ] Documentation updated

---

## Deliverables

**Code:**
- `bin/shared-crates/pool-core/src/catalog.rs` (enhanced)
- `bin/rbee-hive/src/commands/models.rs` (catalog commands)
- `bin/rbee-keeper/src/commands/pool.rs` (remote catalog)

**Catalogs:**
- `.test-models/catalog.json` on mac.home.arpa
- `.test-models/catalog.json` on workstation.home.arpa

**Documentation:**
- Catalog schema spec
- Command usage examples
- Remote access guide

---

## Dependencies

**Additional Cargo.toml dependencies:**
```toml
# pool-core
chrono = { version = "0.4", features = ["serde"] }
hostname = "0.4"

# rbee-hive
tempfile = "3.0"  # For tests
```

---

## Risk Mitigation

**Risk 1:** Catalog corruption  
**Mitigation:** Atomic writes, backup before save

**Risk 2:** SSH command escaping issues  
**Mitigation:** Proper shell escaping, test with special characters

**Risk 3:** Catalog schema evolution  
**Mitigation:** Version field, migration plan

---

## Next Checkpoint

After CP2 gate passes, proceed to `03_CP3_AUTOMATION.md`.

---

**Status:** Ready to start after CP1  
**Estimated Duration:** 5 days  
**Blocking:** CP1 must be complete
