# TEAM-335: SSH Config Multi-Alias Fix

**Status:** ✅ COMPLETE  
**Date:** Oct 27, 2025

## Problem

Remote hive installation was failing with:
```
Error: Host 'workstation' not found in ~/.ssh/config
```

But `~/.ssh/config` DID have a workstation entry:
```
Host workstation workstation.home.arpa
  HostName 192.168.178.29
  User vince
```

## Root Cause

The SSH config parser wasn't handling **multiple host aliases** on a single `Host` line.

### What Was Happening

```rust
// WRONG - Treating "workstation workstation.home.arpa" as a single host name
current_host = Some(value);  // value = "workstation workstation.home.arpa"
hosts.insert(host, config);  // Inserts "workstation workstation.home.arpa" as key
```

When looking up `"workstation"`, it couldn't find it because the key was the full string `"workstation workstation.home.arpa"`.

## Solution

Split the host aliases and insert a separate entry for each:

```rust
// CORRECT - Create entry for each alias
for alias in host_aliases.split_whitespace() {
    hosts.insert(alias.to_string(), config.clone());
}
// Now both "workstation" and "workstation.home.arpa" work!
```

## Fix Applied

**File:** `bin/00_rbee_keeper/src/ssh_resolver.rs`

### 1. When Saving Previous Host Entry

```diff
  "host" => {
-     // Save previous host entry
-     if let (Some(host), Some(hostname)) = (current_host.take(), current_hostname.take()) {
+     // Save previous host entry for ALL aliases
+     if let (Some(host_aliases), Some(hostname)) = (current_host.take(), current_hostname.take()) {
          let user = current_user.take().unwrap_or_else(whoami::username);
-         hosts.insert(host, SshConfig::new(hostname, user, current_port));
+         let config = SshConfig::new(hostname, user, current_port);
+         
+         // Add entry for each alias (e.g., "workstation" and "workstation.home.arpa")
+         for alias in host_aliases.split_whitespace() {
+             hosts.insert(alias.to_string(), config.clone());
+         }
      }
```

### 2. When Saving Final Host Entry

```diff
- // Save last host entry
- if let (Some(host), Some(hostname)) = (current_host, current_hostname) {
+ // Save last host entry for ALL aliases
+ if let (Some(host_aliases), Some(hostname)) = (current_host, current_hostname) {
      let user = current_user.unwrap_or_else(whoami::username);
-     hosts.insert(host, SshConfig::new(hostname, user, current_port));
+     let config = SshConfig::new(hostname, user, current_port);
+     
+     // Add entry for each alias
+     for alias in host_aliases.split_whitespace() {
+         hosts.insert(alias.to_string(), config.clone());
+     }
  }
```

## Verification

```bash
# Build
cargo build -p rbee-keeper
✅ Build successful

# Test remote installation
./rbee hive install -a workstation
✅ SUCCESS!

Output:
📦 Installing rbee-hive on vince@192.168.178.29
🔨 Building rbee-hive from source...
✅ Build complete: target/release/rbee-hive
📤 Copying rbee-hive to vince@192.168.178.29:~/.local/bin/rbee-hive
🎉 rbee-hive installed successfully on vince@192.168.178.29
```

## SSH Config Format Support

The parser now correctly handles:

### Single Alias
```
Host workstation
  HostName 192.168.178.29
  User vince
```
✅ Lookup: `workstation` → Works

### Multiple Aliases
```
Host workstation workstation.home.arpa
  HostName 192.168.178.29
  User vince
```
✅ Lookup: `workstation` → Works  
✅ Lookup: `workstation.home.arpa` → Works

### Multiple Hosts
```
Host infra infra.home.arpa
  HostName 192.168.178.84
  User vince

Host mac mac.home.arpa
  HostName 192.168.178.15
  User vinceliem
```
✅ All aliases work for each host

## Key Takeaway

**SSH config `Host` lines can have multiple space-separated aliases.** Each alias should resolve to the same configuration.

This is standard SSH config behavior:
```bash
ssh workstation              # Works
ssh workstation.home.arpa    # Also works (same host)
```

## Files Changed

- `bin/00_rbee_keeper/src/ssh_resolver.rs` (2 sections modified)

## Next Steps

Now that remote installation works, you can:

```bash
# Install hive on remote host
./rbee hive install -a workstation
✅ DONE

# Start hive on remote host
./rbee hive start -a workstation
⏳ TODO

# Stop hive on remote host
./rbee hive stop -a workstation
⏳ TODO

# Check hive status on remote host
./rbee hive status -a workstation
⏳ TODO
```

**Remote hive installation now works!** 🚀
