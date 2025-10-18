# Component: Beehive Registry (SQLite)

**Location:** `bin/queen-rbee/src/beehive_registry.rs`  
**Type:** Persistent storage  
**Language:** Rust  
**Created by:** TEAM-043  
**Status:** ✅ IMPLEMENTED

## Overview

Persistent SQLite database tracking registered rbee-hive nodes with SSH connection details, capabilities, and status. Survives queen-rbee restarts.

## Database Schema

```sql
CREATE TABLE IF NOT EXISTS beehives (
    node_name TEXT PRIMARY KEY,
    ssh_host TEXT NOT NULL,
    ssh_port INTEGER NOT NULL DEFAULT 22,
    ssh_user TEXT NOT NULL,
    ssh_key_path TEXT,
    git_repo_url TEXT NOT NULL,
    git_branch TEXT NOT NULL,
    install_path TEXT NOT NULL,
    last_connected_unix INTEGER,
    status TEXT NOT NULL DEFAULT 'unknown',
    backends TEXT,              -- JSON array: ["cuda", "metal", "cpu"]
    devices TEXT                -- JSON object: {"cuda": 2, "metal": 1}
)
```

## Data Model

```rust
pub struct BeehiveNode {
    pub node_name: String,           // PRIMARY KEY
    pub ssh_host: String,            // Hostname or IP
    pub ssh_port: u16,               // SSH port (default 22)
    pub ssh_user: String,            // SSH username
    pub ssh_key_path: Option<String>, // Path to SSH private key
    pub git_repo_url: String,        // Git repo for installation
    pub git_branch: String,          // Git branch
    pub install_path: String,        // Installation directory on remote
    pub last_connected_unix: Option<i64>, // Last successful connection
    pub status: String,              // "online", "offline", "error"
    pub backends: Option<String>,    // TEAM-052: JSON capabilities
    pub devices: Option<String>,     // TEAM-052: JSON device counts
}
```

## API Methods

### Core Operations
```rust
// Create/open database
pub async fn new(db_path: Option<PathBuf>) -> Result<Self>

// Add or update node
pub async fn add_node(&self, node: BeehiveNode) -> Result<()>

// Get node by name
pub async fn get_node(&self, node_name: &str) -> Result<Option<BeehiveNode>>

// List all nodes
pub async fn list_nodes(&self) -> Result<Vec<BeehiveNode>>

// Remove node
pub async fn remove_node(&self, node_name: &str) -> Result<bool>

// Update status
pub async fn update_status(&self, node_name: &str, status: &str) -> Result<()>

// Update last connected timestamp
pub async fn update_last_connected(&self, node_name: &str) -> Result<()>
```

### Capability Management (TEAM-052)
```rust
// Update backend capabilities
pub async fn update_capabilities(
    &self,
    node_name: &str,
    backends: Vec<String>,
    devices: HashMap<String, u32>
) -> Result<()>
```

## Storage Location

**Default:** `~/.rbee/beehives.db`  
**Override:** Via constructor parameter

## Lifecycle

1. **Initialization:**
   - Creates `~/.rbee/` directory if missing
   - Opens/creates SQLite database
   - Creates table if not exists

2. **Registration:**
   - Node added via `rbee-keeper setup add-node`
   - Stores SSH credentials, git info
   - Initial status: "unknown"

3. **Connection:**
   - Queen-rbee connects via SSH
   - Updates `last_connected_unix`
   - Updates status: "online"

4. **Capability Discovery (TEAM-052):**
   - Query remote hive for GPU info
   - Store backends (cuda, metal, cpu)
   - Store device counts

5. **Health Monitoring:**
   - Periodic SSH connection tests
   - Update status: "online" / "offline" / "error"

6. **Removal:**
   - Delete from database
   - SSH credentials removed

## Usage Example

```rust
// Initialize registry
let registry = BeehiveRegistry::new(None).await?;

// Add node
let node = BeehiveNode {
    node_name: "gpu-server-1".to_string(),
    ssh_host: "192.168.1.100".to_string(),
    ssh_port: 22,
    ssh_user: "rbee".to_string(),
    ssh_key_path: Some("/home/user/.ssh/id_rsa".to_string()),
    git_repo_url: "https://github.com/user/llama-orch.git".to_string(),
    git_branch: "main".to_string(),
    install_path: "/opt/rbee".to_string(),
    last_connected_unix: None,
    status: "unknown".to_string(),
    backends: None,
    devices: None,
};

registry.add_node(node).await?;

// Update capabilities
registry.update_capabilities(
    "gpu-server-1",
    vec!["cuda".to_string(), "cpu".to_string()],
    HashMap::from([
        ("cuda".to_string(), 2),
        ("cpu".to_string(), 1),
    ])
).await?;

// List all nodes
let nodes = registry.list_nodes().await?;
```

## Integration Points

### rbee-keeper
- `setup add-node` → Calls `add_node()`
- `setup list-nodes` → Calls `list_nodes()`
- `setup remove-node` → Calls `remove_node()`

### queen-rbee
- Reads registry on startup
- Connects to registered hives
- Updates status and capabilities
- Uses SSH credentials for remote management

## Maturity Assessment

**Status:** ✅ **PRODUCTION READY**

**Strengths:**
- ✅ Persistent storage (survives restarts)
- ✅ Complete CRUD operations
- ✅ SSH credential management
- ✅ Capability tracking (TEAM-052)
- ✅ Async/await support
- ✅ Error handling with anyhow

**Limitations:**
- ⚠️ No encryption for SSH keys in DB (stored as paths)
- ⚠️ No connection pooling
- ⚠️ No migration system for schema changes
- ⚠️ No backup/restore functionality

**Recommended Improvements:**
1. Encrypt sensitive data (SSH keys)
2. Add database migrations
3. Add backup/restore commands
4. Add connection pooling for concurrent access
5. Add audit logging (who added/removed nodes)

## Related Components

- **SSH Module** (`queen-rbee/src/ssh.rs`) - Uses credentials from registry
- **Worker Registry** - Tracks workers on registered hives
- **HTTP API** (`queen-rbee/src/http/beehives.rs`) - Exposes registry via REST

## Testing

```bash
# Unit tests
cargo test -p queen-rbee beehive_registry

# Integration test
rbee-keeper setup add-node --name test-node --host localhost
rbee-keeper setup list-nodes
rbee-keeper setup remove-node --name test-node
```

---

**Created by:** TEAM-043  
**Enhanced by:** TEAM-052 (capabilities)  
**Last Updated:** 2025-10-18  
**Maturity:** ✅ Production Ready (with noted limitations)
