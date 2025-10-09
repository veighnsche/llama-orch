# Feature Toggles Specification

**Status**: Normative  
**Version**: 1.0  
**Date**: 2025-10-09

---

## Overview

Feature toggles allow runtime configuration of optional features without recompilation. Critical for:
1. **EU Compliance** - Enable/disable audit logging overhead
2. **Update Methods** - Git vs binary updates
3. **Development vs Production** - Different behaviors per environment

---

## Toggle 1: EU Audit Mode

### Environment Variable

```bash
LLORCH_EU_AUDIT=true   # Enable EU compliance features
LLORCH_EU_AUDIT=false  # Disable (default, homelab mode)
```

### When Enabled (EU Compliance Mode)

**Audit Logging:**
```rust
// All API calls logged to immutable audit log
POST /v2/tasks
  → audit_logger.log(AuditEvent {
      timestamp: Utc::now(),
      user_id: "tenant-123",
      action: "job_submit",
      resource: "job-xyz",
      metadata: { model: "llama3", prompt_hash: "abc123" },
      ip_address: "192.168.1.100",
      correlation_id: "req-456",
  });
```

**GDPR Endpoints:**
```rust
// Data export
GET /gdpr/export?user_id=tenant-123
  → Returns all data for user (jobs, logs, metadata)
  → Includes audit trail
  → Format: JSON

// Data deletion
POST /gdpr/delete
  { "user_id": "tenant-123", "reason": "user_request" }
  → Soft delete (mark as deleted, retain audit)
  → Hard delete after retention period
  → Audit log entry for deletion
```

**Data Residency:**
```rust
// Enforce EU-only workers
fn select_worker(job: &Job) -> Result<Worker> {
    let workers = registry.list_workers()?;
    
    if env::var("LLORCH_EU_AUDIT")? == "true" {
        // Filter to EU-only workers
        workers.retain(|w| w.region == "EU");
        
        if workers.is_empty() {
            return Err("No EU workers available");
        }
    }
    
    // Select worker
    workers.first().cloned()
}
```

**Consent Tracking:**
```rust
// Track user consent
POST /gdpr/consent
  {
    "user_id": "tenant-123",
    "consent_type": "data_processing",
    "granted": true,
    "timestamp": "2025-10-09T12:00:00Z"
  }
  → Stored in audit log
  → Required before processing
```

**Retention Policies:**
```rust
// Auto-delete old data
if env::var("LLORCH_EU_AUDIT")? == "true" {
    // Delete jobs older than 90 days
    let cutoff = Utc::now() - Duration::days(90);
    db.execute("DELETE FROM jobs WHERE created_at < ?", [cutoff])?;
    
    // Audit log never deleted (immutable)
}
```

### When Disabled (Homelab Mode)

**No Audit Logging:**
```rust
// No overhead
POST /v2/tasks
  → Process job
  → No audit log entry
  → Faster
```

**No GDPR Endpoints:**
```rust
// Endpoints return 404
GET /gdpr/export
  → 404 Not Found

POST /gdpr/delete
  → 404 Not Found
```

**No Data Residency Checks:**
```rust
// Use any worker
fn select_worker(job: &Job) -> Result<Worker> {
    let workers = registry.list_workers()?;
    workers.first().cloned()  // No filtering
}
```

**No Retention Policies:**
```rust
// Keep data forever (or until manual cleanup)
// No auto-deletion
```

### Implementation

**In rbees-orcd:**
```rust
// src/main.rs
use std::env;

#[tokio::main]
async fn main() -> Result<()> {
    let eu_audit_enabled = env::var("LLORCH_EU_AUDIT")
        .unwrap_or_else(|_| "false".to_string()) == "true";
    
    let mut app = Router::new()
        .route("/v2/tasks", post(submit_task));
    
    if eu_audit_enabled {
        // Add audit middleware
        let audit_logger = AuditLogger::new(
            env::var("LLORCH_AUDIT_LOG_PATH")
                .unwrap_or_else(|_| "/var/log/llorch/audit.log".to_string())
        )?;
        
        app = app.layer(AuditMiddleware::new(audit_logger));
        
        // Add GDPR endpoints
        app = app
            .route("/gdpr/export", get(gdpr_export))
            .route("/gdpr/delete", post(gdpr_delete))
            .route("/gdpr/consent", post(gdpr_consent));
        
        info!("EU audit mode ENABLED");
        info!("Audit log: {}", env::var("LLORCH_AUDIT_LOG_PATH")?);
    } else {
        info!("EU audit mode DISABLED (homelab mode)");
    }
    
    // Start server
    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .await?;
    
    Ok(())
}
```

**In audit-logging crate:**
```rust
// bin/shared-crates/audit-logging/src/lib.rs

pub struct AuditLogger {
    log_path: PathBuf,
    enabled: bool,
}

impl AuditLogger {
    pub fn new(log_path: impl Into<PathBuf>) -> Result<Self> {
        let enabled = env::var("LLORCH_EU_AUDIT")
            .unwrap_or_else(|_| "false".to_string()) == "true";
        
        if enabled {
            let log_path = log_path.into();
            // Create log file if doesn't exist
            if !log_path.exists() {
                fs::create_dir_all(log_path.parent().unwrap())?;
                fs::File::create(&log_path)?;
            }
            
            Ok(Self { log_path, enabled })
        } else {
            // No-op logger
            Ok(Self {
                log_path: PathBuf::new(),
                enabled: false,
            })
        }
    }
    
    pub fn log(&self, event: AuditEvent) -> Result<()> {
        if !self.enabled {
            return Ok(());  // No-op
        }
        
        // Append to log (immutable, append-only)
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.log_path)?;
        
        let json = serde_json::to_string(&event)?;
        writeln!(file, "{}", json)?;
        
        Ok(())
    }
}
```

---

## Toggle 2: Update Method

### Environment Variable

```bash
LLORCH_UPDATE_METHOD=git      # Update via git pull (default, current)
LLORCH_UPDATE_METHOD=binary   # Update via binary download (future)
```

### When Git (Current)

**Update process:**
```bash
# rbees-ctl update
cd ~/Projects/llama-orch
git pull origin main
git submodule update --init --recursive
cargo build --release --bin rbees-orcd
cargo build --release --bin rbees-pool
cargo build --release --bin rbees-ctl
```

**Advantages:**
- ✅ Always latest code
- ✅ Can modify locally
- ✅ See commit history
- ✅ Easy rollback (git checkout)

**Disadvantages:**
- ❌ Requires Rust toolchain
- ❌ Slow (cargo build)
- ❌ Large disk usage (target/ dir)

### When Binary (Future)

**Update process:**
```bash
# rbees-ctl update
curl -L https://releases.llama-orch.com/latest/rbees-orcd -o /tmp/rbees-orcd
chmod +x /tmp/rbees-orcd
mv /tmp/rbees-orcd ~/.local/bin/rbees-orcd

curl -L https://releases.llama-orch.com/latest/rbees-pool -o /tmp/rbees-pool
chmod +x /tmp/rbees-pool
mv /tmp/rbees-pool ~/.local/bin/rbees-pool

curl -L https://releases.llama-orch.com/latest/rbees-ctl -o /tmp/rbees-ctl
chmod +x /tmp/rbees-ctl
mv /tmp/rbees-ctl ~/.local/bin/rbees-ctl
```

**Advantages:**
- ✅ Fast (no compilation)
- ✅ Small download
- ✅ No Rust toolchain needed
- ✅ Stable releases

**Disadvantages:**
- ❌ Can't modify locally
- ❌ Requires release infrastructure
- ❌ Version management needed

### Implementation

**In rbees-ctl:**
```rust
// src/commands/update.rs

pub async fn update() -> Result<()> {
    let method = env::var("LLORCH_UPDATE_METHOD")
        .unwrap_or_else(|_| "git".to_string());
    
    match method.as_str() {
        "git" => update_via_git().await,
        "binary" => update_via_binary().await,
        _ => Err(anyhow!("Invalid LLORCH_UPDATE_METHOD: {}", method)),
    }
}

async fn update_via_git() -> Result<()> {
    println!("Updating via git...");
    
    // Find repo root
    let repo_root = env::var("LLORCH_REPO_ROOT")
        .unwrap_or_else(|_| "~/Projects/llama-orch".to_string());
    
    // Git pull
    Command::new("git")
        .current_dir(&repo_root)
        .args(&["pull", "origin", "main"])
        .status()?;
    
    // Update submodules
    Command::new("git")
        .current_dir(&repo_root)
        .args(&["submodule", "update", "--init", "--recursive"])
        .status()?;
    
    // Build binaries
    println!("Building binaries...");
    Command::new("cargo")
        .current_dir(&repo_root)
        .args(&["build", "--release", "--bin", "rbees-orcd"])
        .status()?;
    
    Command::new("cargo")
        .current_dir(&repo_root)
        .args(&["build", "--release", "--bin", "rbees-pool"])
        .status()?;
    
    Command::new("cargo")
        .current_dir(&repo_root)
        .args(&["build", "--release", "--bin", "rbees-ctl"])
        .status()?;
    
    println!("✅ Update complete");
    Ok(())
}

async fn update_via_binary() -> Result<()> {
    println!("Updating via binary download...");
    
    let base_url = env::var("LLORCH_RELEASE_URL")
        .unwrap_or_else(|_| "https://releases.llama-orch.com".to_string());
    
    let binaries = vec!["rbees-orcd", "rbees-pool", "rbees-ctl"];
    
    for binary in binaries {
        println!("Downloading {}...", binary);
        
        let url = format!("{}/latest/{}", base_url, binary);
        let response = reqwest::get(&url).await?;
        
        if !response.status().is_success() {
            return Err(anyhow!("Failed to download {}: {}", binary, response.status()));
        }
        
        let bytes = response.bytes().await?;
        
        // Write to temp file
        let temp_path = format!("/tmp/{}", binary);
        fs::write(&temp_path, bytes)?;
        
        // Make executable
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(&temp_path)?.permissions();
            perms.set_mode(0o755);
            fs::set_permissions(&temp_path, perms)?;
        }
        
        // Move to bin dir
        let bin_dir = env::var("LLORCH_BIN_DIR")
            .unwrap_or_else(|_| "~/.local/bin".to_string());
        let dest_path = format!("{}/{}", bin_dir, binary);
        fs::rename(&temp_path, &dest_path)?;
        
        println!("✅ Updated {}", binary);
    }
    
    println!("✅ Update complete");
    Ok(())
}
```

---

## Toggle 3: Development Mode

### Environment Variable

```bash
LLORCH_DEV_MODE=true   # Development mode (verbose logging, debug endpoints)
LLORCH_DEV_MODE=false  # Production mode (default)
```

### When Enabled (Development)

**Verbose Logging:**
```rust
if env::var("LLORCH_DEV_MODE")? == "true" {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Debug)
        .init();
} else {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();
}
```

**Debug Endpoints:**
```rust
if env::var("LLORCH_DEV_MODE")? == "true" {
    app = app
        .route("/debug/state", get(debug_state))
        .route("/debug/workers", get(debug_workers))
        .route("/debug/jobs", get(debug_jobs));
}
```

**Relaxed Validation:**
```rust
if env::var("LLORCH_DEV_MODE")? == "true" {
    // Allow test API keys
    if api_key.starts_with("test_") {
        return Ok(());
    }
}
```

### When Disabled (Production)

**Minimal Logging:**
```rust
// Only INFO and above
env_logger::Builder::from_default_env()
    .filter_level(log::LevelFilter::Info)
    .init();
```

**No Debug Endpoints:**
```rust
// /debug/* routes return 404
```

**Strict Validation:**
```rust
// No test API keys allowed
if api_key.starts_with("test_") {
    return Err("Test API keys not allowed in production");
}
```

---

## Configuration File

### .llorch.toml

```toml
# Feature toggles
[features]
eu_audit = false          # Enable EU compliance features
update_method = "git"     # "git" or "binary"
dev_mode = false          # Enable development mode

# EU Audit settings (if enabled)
[eu_audit]
log_path = "/var/log/llorch/audit.log"
retention_days = 90
data_residency = "EU"

# Update settings
[update]
repo_root = "~/Projects/llama-orch"
release_url = "https://releases.llama-orch.com"
bin_dir = "~/.local/bin"

# Development settings (if dev_mode enabled)
[development]
log_level = "debug"
debug_endpoints = true
allow_test_keys = true
```

### Loading Config

```rust
// src/config.rs

#[derive(Debug, Deserialize)]
pub struct Config {
    pub features: Features,
    pub eu_audit: Option<EuAuditConfig>,
    pub update: UpdateConfig,
    pub development: Option<DevelopmentConfig>,
}

#[derive(Debug, Deserialize)]
pub struct Features {
    #[serde(default)]
    pub eu_audit: bool,
    
    #[serde(default = "default_update_method")]
    pub update_method: String,
    
    #[serde(default)]
    pub dev_mode: bool,
}

fn default_update_method() -> String {
    "git".to_string()
}

impl Config {
    pub fn load() -> Result<Self> {
        // Try config file first
        if let Ok(config_path) = env::var("LLORCH_CONFIG_PATH") {
            let contents = fs::read_to_string(config_path)?;
            return Ok(toml::from_str(&contents)?);
        }
        
        // Try default locations
        let default_paths = vec![
            ".llorch.toml",
            "~/.config/llorch/config.toml",
            "/etc/llorch/config.toml",
        ];
        
        for path in default_paths {
            if let Ok(contents) = fs::read_to_string(path) {
                return Ok(toml::from_str(&contents)?);
            }
        }
        
        // Use defaults
        Ok(Self::default())
    }
    
    pub fn eu_audit_enabled(&self) -> bool {
        // Environment variable overrides config
        env::var("LLORCH_EU_AUDIT")
            .map(|v| v == "true")
            .unwrap_or(self.features.eu_audit)
    }
    
    pub fn update_method(&self) -> &str {
        // Environment variable overrides config
        env::var("LLORCH_UPDATE_METHOD")
            .as_deref()
            .unwrap_or(&self.features.update_method)
    }
    
    pub fn dev_mode_enabled(&self) -> bool {
        // Environment variable overrides config
        env::var("LLORCH_DEV_MODE")
            .map(|v| v == "true")
            .unwrap_or(self.features.dev_mode)
    }
}
```

---

## Summary

### Feature Toggles

1. **LLORCH_EU_AUDIT** (true/false)
   - Enables: Audit logging, GDPR endpoints, data residency, retention
   - Default: false (homelab mode)
   - Use case: EU compliance for B2B customers

2. **LLORCH_UPDATE_METHOD** (git/binary)
   - git: Update via git pull + cargo build
   - binary: Update via binary download
   - Default: git (current)
   - Use case: Future binary releases

3. **LLORCH_DEV_MODE** (true/false)
   - Enables: Verbose logging, debug endpoints, relaxed validation
   - Default: false (production)
   - Use case: Development and debugging

### Configuration Precedence

1. Environment variables (highest)
2. Config file (.llorch.toml)
3. Built-in defaults (lowest)

### Implementation Status

- [ ] EU audit toggle (rbees-orcd, audit-logging crate)
- [ ] Update method toggle (rbees-ctl)
- [ ] Development mode toggle (all binaries)
- [ ] Config file support (.llorch.toml)
- [ ] Documentation (README, website)

---

**Version**: 1.0  
**Status**: Normative (MUST implement)  
**Last Updated**: 2025-10-09

---

**End of Feature Toggles Specification**
