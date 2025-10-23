# RUSSH Migration - Quick Start

**TL;DR:** Replace shell `ssh`/`scp` with pure Rust `russh` library

---

## Why?

✅ Better error handling  
✅ Cross-platform (Windows support)  
✅ Connection pooling  
✅ Easier testing  
✅ No external dependencies  

---

## What Changes?

### Before (Shell Commands)
```rust
tokio::process::Command::new("ssh")
    .arg("-p").arg("22")
    .arg("user@host")
    .arg("echo test")
    .output()
    .await?
```

### After (Pure Rust)
```rust
let mut client = RbeeSSHClient::connect("host", 22, "user").await?;
let (stdout, stderr, code) = client.exec("echo test").await?;
client.close().await?;
```

---

## Files to Change

1. **Add:** `hive-lifecycle/src/russh_client.rs` (new module)
2. **Update:** `hive-lifecycle/src/ssh_helper.rs` (replace shell commands)
3. **Update:** `hive-lifecycle/Cargo.toml` (add dependencies)
4. **Update:** `hive-lifecycle/src/lib.rs` (export new module)

---

## Dependencies to Add

```toml
russh = "0.44"
russh-keys = "0.44"
russh-sftp = "2.0"
```

---

## Estimated Time

**4-6 hours** for complete migration

---

## Testing

```bash
# Should work exactly the same as before
./rbee hive install -a workstation
./rbee hive start -a workstation
./rbee hive stop -a workstation
```

---

## See Full Guide

📄 `bin/.plan/RUSSH_MIGRATION_GUIDE.md`
