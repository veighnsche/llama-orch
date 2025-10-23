# SSH Config Import Feature - Implementation Proposal

## Goal

Import SSH hosts from `~/.ssh/config` into `~/.config/rbee/hives.conf` with interactive prompts for missing rbee-specific fields.

---

## Current State

### What We Have
- ✅ `hives.conf` parser (SSH config style)
- ✅ HiveEntry struct with required fields
- ✅ 3-file operation pattern

### What We're Missing
- ❌ SSH config parser (reads `~/.ssh/config`)
- ❌ Interactive prompt system
- ❌ Merge logic (don't overwrite existing hives)
- ❌ CLI command to trigger import

---

## Gap: SSH Config vs Hives Config

### SSH Config Fields (Standard)
```
Host workstation
    HostName 192.168.178.29
    User vince
    Port 22                    # Optional, defaults to 22
    IdentityFile ~/.ssh/id_ed25519
```

### Hives Config Fields (Required)
```
Host workstation
    HostName 192.168.178.29
    User vince
    Port 22
    HivePort 8081              # ← MISSING in SSH config
    BinaryPath /path/to/rbee   # ← Optional, but useful
```

**Problem:** `HivePort` is rbee-specific and doesn't exist in SSH config.

---

## Implementation Strategy

### Option A: Interactive Import (RECOMMENDED)

**Flow:**
1. Parse `~/.ssh/config`
2. Filter hosts (skip wildcards, localhost, etc.)
3. For each host, prompt user:
   - "Import host 'workstation' (192.168.178.29)? [y/N]"
   - "HivePort for 'workstation': [8081]"
   - "BinaryPath (optional): [/usr/local/bin/rbee-hive]"
4. Merge with existing `hives.conf` (don't overwrite)
5. Write updated `hives.conf`

**Pros:**
- ✅ User controls what gets imported
- ✅ Validates HivePort before writing
- ✅ Safe (doesn't overwrite existing)

**Cons:**
- ❌ Interactive (can't automate)
- ❌ Requires terminal interaction

### Option B: Batch Import with Defaults

**Flow:**
1. Parse `~/.ssh/config`
2. Import all hosts with default HivePort (8081)
3. Write to `hives.conf`
4. User edits manually if needed

**Pros:**
- ✅ Fast, no interaction
- ✅ Can be automated

**Cons:**
- ❌ All hosts get same HivePort (wrong!)
- ❌ User must manually fix conflicts
- ❌ Dangerous (could import too much)

### Option C: Hybrid (Interactive + Config File)

**Flow:**
1. Parse `~/.ssh/config`
2. Look for rbee-specific comments:
   ```
   Host workstation
       HostName 192.168.178.29
       User vince
       # rbee:HivePort=8082
       # rbee:BinaryPath=/custom/path
   ```
3. If found, import automatically
4. If not found, prompt interactively

**Pros:**
- ✅ Best of both worlds
- ✅ Can annotate SSH config once

**Cons:**
- ❌ More complex
- ❌ Pollutes SSH config with comments

---

## Recommended Approach: Option A (Interactive)

### Why?
- HivePort is critical and host-specific
- User knows which hosts are rbee hives
- Safe (user confirms each import)
- Follows Unix philosophy (do one thing well)

---

## Technical Implementation

### 1. New Module: `ssh_config_parser`

**Location:** `bin/99_shared_crates/rbee-config/src/ssh_config_parser.rs`

```rust
pub struct SshHost {
    pub alias: String,
    pub hostname: String,
    pub port: u16,
    pub user: String,
}

pub fn parse_ssh_config(path: &Path) -> Result<Vec<SshHost>>;
```

**Complexity:** ~200 LOC
- Parse SSH config (subset: Host, HostName, Port, User)
- Ignore unknown directives
- Handle comments, wildcards

### 2. New Operation: `HiveImportSsh`

**Location:** `bin/99_shared_crates/rbee-operations/src/lib.rs`

```rust
pub enum Operation {
    // ... existing ...
    
    HiveImportSsh {
        /// Path to SSH config (defaults to ~/.ssh/config)
        ssh_config_path: Option<String>,
        /// Default HivePort for all imports
        default_hive_port: u16,
    },
}
```

### 3. Handler: `execute_hive_import_ssh`

**Location:** `bin/15_queen_rbee_crates/hive-lifecycle/src/import_ssh.rs`

```rust
pub async fn execute_hive_import_ssh(
    request: HiveImportSshRequest,
    config: Arc<RbeeConfig>,
    job_id: &str,
) -> Result<()> {
    // 1. Parse SSH config
    let ssh_hosts = parse_ssh_config(&ssh_config_path)?;
    
    // 2. Filter (skip localhost, wildcards)
    let candidates = filter_importable_hosts(ssh_hosts);
    
    // 3. Interactive prompts (via narration)
    for host in candidates {
        // Prompt user via SSE
        // Get HivePort input
        // Create HiveEntry
    }
    
    // 4. Merge with existing hives.conf
    // 5. Write updated hives.conf
}
```

**Problem:** Interactive prompts over SSE are complex!

### 4. CLI Command

**Location:** `bin/00_rbee_keeper/src/main.rs`

```rust
pub enum HiveAction {
    // ... existing ...
    
    ImportSsh {
        /// Path to SSH config
        #[arg(long, default_value = "~/.ssh/config")]
        ssh_config: String,
        
        /// Default HivePort
        #[arg(long, default_value = "8081")]
        default_port: u16,
    },
}
```

---

## Challenge: Interactive Prompts

### Problem
Current architecture:
- rbee-keeper → queen-rbee (HTTP/SSE)
- queen-rbee executes operations asynchronously
- No bidirectional communication for prompts

### Solutions

#### Solution 1: Client-Side Import (RECOMMENDED)

**Move import logic to rbee-keeper:**
```
rbee-keeper:
  1. Parse ~/.ssh/config locally
  2. Prompt user interactively (stdin/stdout)
  3. Build complete hives.conf entries
  4. Send batch update to queen-rbee
```

**Pros:**
- ✅ Simple (no SSE bidirectional needed)
- ✅ Fast (no network round-trips)
- ✅ Works offline

**Cons:**
- ❌ Breaks pattern (business logic in client)
- ❌ Can't import on remote queen

#### Solution 2: SSE Bidirectional (Complex)

**Extend SSE for prompts:**
```
queen → client: "Prompt: HivePort for 'workstation'?"
client → queen: POST /v1/jobs/{job_id}/input {"value": "8082"}
queen → client: "Received: 8082"
```

**Pros:**
- ✅ Keeps logic in queen
- ✅ Works for remote queen

**Cons:**
- ❌ Very complex
- ❌ Requires new /input endpoint
- ❌ SSE not designed for bidirectional

#### Solution 3: Wizard Mode (Multi-Step)

**Break into multiple operations:**
```
1. rbee hive scan-ssh → Lists importable hosts
2. rbee hive import workstation --hive-port 8082
3. rbee hive import infra --hive-port 8083
```

**Pros:**
- ✅ Fits existing architecture
- ✅ No bidirectional needed
- ✅ Scriptable

**Cons:**
- ❌ Tedious for many hosts
- ❌ Not a single command

---

## Recommended Implementation: Client-Side Import

### Why?
- Simplest to implement
- No architectural changes needed
- Fast and reliable
- Fits Unix philosophy (local tools)

### Trade-off
- Import logic lives in rbee-keeper (not queen)
- Acceptable because:
  - SSH config is local to user's machine
  - Import is a one-time setup operation
  - No server state needed

---

## Implementation Steps

### Phase 1: SSH Parser (2-3 hours)
1. Add `ssh_config_parser.rs` to `rbee-config`
2. Parse subset of SSH config (Host, HostName, Port, User)
3. Unit tests with real SSH config examples

### Phase 2: Client-Side Import (3-4 hours)
1. Add `import_ssh` module to `rbee-keeper`
2. Interactive prompts using `dialoguer` crate
3. Merge logic (don't overwrite existing)
4. Write to `~/.config/rbee/hives.conf`

### Phase 3: CLI Command (1 hour)
1. Add `ImportSsh` to `HiveAction`
2. Wire up in `handle_command()`
3. Manual testing

### Phase 4: Polish (1-2 hours)
1. Error handling
2. Dry-run mode (`--dry-run`)
3. Documentation

**Total Estimate:** 7-10 hours

---

## Alternative: Quick & Dirty Script

If you just want to import YOUR config once:

```bash
#!/bin/bash
# quick_import.sh

cat ~/.ssh/config | grep -A 3 "^Host " | while read line; do
    if [[ $line =~ ^Host\ (.+) ]]; then
        alias="${BASH_REMATCH[1]}"
        echo "Host $alias"
    elif [[ $line =~ HostName\ (.+) ]]; then
        echo "    HostName ${BASH_REMATCH[1]}"
    elif [[ $line =~ User\ (.+) ]]; then
        echo "    User ${BASH_REMATCH[1]}"
        echo "    HivePort 8081"  # Default
        echo ""
    fi
done >> ~/.config/rbee/hives.conf
```

**Then manually edit** `~/.config/rbee/hives.conf` to fix HivePorts.

---

## Recommendation

**For one-time import:** Use quick script + manual edit (30 minutes)

**For production feature:** Implement client-side import (7-10 hours)

**Don't implement:** Server-side interactive import (too complex, 20+ hours)

---

## Questions for You

1. **How many hosts** do you need to import? (If <5, manual is faster)
2. **One-time or recurring?** (One-time → script, recurring → feature)
3. **Client-side OK?** (Import logic in rbee-keeper vs queen-rbee)
4. **Interactive or batch?** (Prompt per host vs default HivePort)

Let me know your preference and I'll implement accordingly!
