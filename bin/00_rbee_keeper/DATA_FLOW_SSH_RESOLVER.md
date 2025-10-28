# SSH Resolver Data Flow

**TEAM-333: Documentation of data transformation pipeline**

---

## The Split Point

```
~/.ssh/config (raw file)
    ↓
    ↓ [PARSE]
    ↓
ssh_resolver::parse_ssh_config()
    ↓
    ↓ Returns: HashMap<String, SshConfig>
    ↓ Pure SSH config data (no transformations)
    ↓
    ├─────────────────────────────────────────┐
    ↓                                         ↓
    ↓                                         ↓
[SSH Operations]                      [Tauri Display]
    ↓                                         ↓
resolve_ssh_config(alias)             ssh_list()
    ↓                                         ↓
Returns: SshConfig                    Returns: Vec<SshTarget>
(for SSH commands)                    (for UI display)
    ↓                                         ↓
Used by:                              Used by:
- hive install                        - React table
- hive start/stop                     - UI components
- worker spawn                        - Display only
```

---

## Layer 1: SSH Resolver (Pure Data)

**File:** `src/ssh_resolver.rs`

**Function:** `parse_ssh_config(path: &PathBuf) -> Result<HashMap<String, SshConfig>>`

**Purpose:** Parse SSH config file, return raw data

**Input:**
```text
~/.ssh/config:

Host workstation
    HostName 192.168.178.29
    User vince
    Port 22

Host workstation.home.arpa
    HostName 192.168.178.29
    User vince
    Port 22

Host mac
    HostName 192.168.178.15
    User vinceliem
    Port 22
```

**Output:**
```rust
HashMap {
    "workstation" => SshConfig {
        hostname: "192.168.178.29",
        user: "vince",
        port: 22,
    },
    "workstation.home.arpa" => SshConfig {
        hostname: "192.168.178.29",
        user: "vince",
        port: 22,
    },
    "mac" => SshConfig {
        hostname: "192.168.178.15",
        user: "vinceliem",
        port: 22,
    },
}
```

**Characteristics:**
- ✅ No deduplication
- ✅ No localhost injection
- ✅ No transformations
- ✅ Just parses the file
- ✅ Returns ALL entries as-is

**Used by:**
- `resolve_ssh_config(alias)` - For SSH operations
- `ssh_list()` - For Tauri display (with transformations)

---

## Layer 2A: SSH Operations (resolve_ssh_config)

**File:** `src/ssh_resolver.rs`

**Function:** `resolve_ssh_config(alias: &str) -> Result<SshConfig>`

**Purpose:** Resolve a single host alias to SSH config for operations

**Input:** `"workstation"`

**Output:**
```rust
SshConfig {
    hostname: "192.168.178.29",
    user: "vince",
    port: 22,
}
```

**Special case:** `"localhost"` → Returns `SshConfig::localhost()` (no SSH)

**Used by:**
- Hive install/start/stop commands
- Worker spawn commands
- Any operation that needs to SSH to a host

**Characteristics:**
- ✅ Single host lookup
- ✅ Localhost bypass (no SSH config needed)
- ✅ Returns exact match from config
- ✅ No transformations

---

## Layer 2B: Tauri Display (ssh_list)

**File:** `src/tauri_commands.rs`

**Function:** `ssh_list() -> Result<Vec<SshTarget>, String>`

**Purpose:** Transform SSH config data for UI display

**Input:** HashMap from `parse_ssh_config()`

**Transformations:**
1. **Deduplicate by `hostname:port@user`**
   - Multiple aliases → same IP = ONE entry
   - Keep shortest alias as primary
   - Longer alias becomes subtitle

2. **Convert to SshTarget**
   - Add `host_subtitle` field
   - Add `status` field (Unknown/Online/Offline)
   - Sort alphabetically

3. **NO localhost injection**
   - Only show what's in SSH config
   - If user wants localhost, add to config

**Output:**
```rust
Vec<SshTarget> [
    SshTarget {
        host: "mac",
        host_subtitle: None,
        hostname: "192.168.178.15",
        user: "vinceliem",
        port: 22,
        status: Unknown,
    },
    SshTarget {
        host: "workstation",
        host_subtitle: Some("workstation.home.arpa"),
        hostname: "192.168.178.29",
        user: "vince",
        port: 22,
        status: Unknown,
    },
]
```

**Used by:**
- React `SshHivesTable` component
- UI display only
- NOT used for SSH operations

**Characteristics:**
- ✅ Deduplicates by hostname
- ✅ Keeps shortest alias
- ✅ Adds UI-specific fields
- ✅ NO localhost injection
- ✅ Sorted alphabetically

---

## Why This Split Exists

### Different Data Needs

**SSH Operations need:**
- Exact alias lookup (`"workstation"` → config)
- Localhost bypass (no SSH for local)
- Raw SSH config data
- No deduplication (aliases are intentional)

**Tauri Display needs:**
- Deduplicated list (no duplicate IPs)
- Shortest alias (better UX)
- Status indicators (UI feedback)
- Sorted list (better UX)
- NO localhost (user can add to config if needed)

### The Transform Happens at the Last Moment

```
parse_ssh_config()     ← Pure data (no transforms)
        ↓
        ├─→ resolve_ssh_config()  ← For operations (no transforms)
        │
        └─→ ssh_list()            ← For display (transforms here!)
                ↓
                ├─ Deduplicate
                ├─ Add UI fields
                ├─ Sort
                └─ Return Vec<SshTarget>
```

**Why last moment?**
- Keep `parse_ssh_config()` pure (reusable)
- Transform only for display (separation of concerns)
- Operations get raw data (no surprises)
- UI gets clean data (better UX)

---

## Example: Adding Localhost to SSH Config

**If user wants localhost in the list:**

```bash
# Add to ~/.ssh/config
Host localhost
    HostName 127.0.0.1
    User vince
    Port 22
```

**Result:**
- Shows up in `ssh_list()` automatically
- Works with `resolve_ssh_config("localhost")`
- No special cases needed
- Consistent with other entries

**Why not inject it?**
- ❌ Artificial data (not in config)
- ❌ Inconsistent (special case)
- ❌ Confusing (where did it come from?)
- ✅ User controls what's in the list
- ✅ Config is source of truth

---

## Summary

| Layer | Function | Purpose | Transforms |
|-------|----------|---------|------------|
| **1. Parser** | `parse_ssh_config()` | Parse file | None |
| **2A. Operations** | `resolve_ssh_config()` | Lookup alias | None |
| **2B. Display** | `ssh_list()` | UI data | Dedupe, sort, UI fields |

**Key principle:** Transform data at the last moment, keep intermediate layers pure.

---

**TEAM-333 | Oct 28, 2025**
