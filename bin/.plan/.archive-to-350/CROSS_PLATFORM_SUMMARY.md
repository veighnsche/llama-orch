# Cross-Platform Support Summary

**Issue:** rbee-config hardcodes Linux paths  
**Solution:** Use `dirs` crate for cross-platform directory resolution  
**Effort:** 3-4 hours

---

## 🎯 The Problem

Current code in `rbee-config/src/lib.rs`:

```rust
let home = std::env::var("HOME")?;  // ❌ Doesn't exist on Windows
let config_dir = PathBuf::from(home).join(".config").join("rbee");  // ❌ Linux-only
```

---

## ✅ The Solution

Use the `dirs` crate:

```rust
use dirs;

let config_dir = dirs::config_dir()  // ✅ Cross-platform!
    .ok_or_else(|| anyhow!("Cannot determine config directory"))?
    .join("rbee");
```

---

## 📁 Directory Mapping

| Purpose | Linux | macOS | Windows |
|---------|-------|-------|---------|
| **Config** | `~/.config/rbee/` | `~/Library/Application Support/rbee/` | `%APPDATA%\rbee\` |
| **Cache** | `~/.cache/rbee/` | `~/Library/Caches/rbee/` | `%LOCALAPPDATA%\rbee\` |
| **Data** | `~/.local/share/rbee/` | `~/Library/Application Support/rbee/` | `%LOCALAPPDATA%\rbee\` |

---

## 🛠️ Implementation (3 Steps)

### 1. Add Dependency

```toml
# Cargo.toml
[dependencies]
dirs = "5.0"
```

### 2. Update config_dir()

```rust
pub fn config_dir() -> Result<PathBuf> {
    let base = dirs::config_dir()
        .ok_or_else(|| ConfigError::InvalidConfig(
            "Cannot determine config directory".to_string()
        ))?;
    
    let config_dir = base.join("rbee");
    std::fs::create_dir_all(&config_dir)?;
    Ok(config_dir)
}
```

### 3. Add cache_dir() and data_dir()

```rust
pub fn cache_dir() -> Result<PathBuf> {
    let base = dirs::cache_dir()
        .ok_or_else(|| ConfigError::InvalidConfig(
            "Cannot determine cache directory".to_string()
        ))?;
    
    let cache_dir = base.join("rbee");
    std::fs::create_dir_all(&cache_dir)?;
    Ok(cache_dir)
}

pub fn data_dir() -> Result<PathBuf> {
    let base = dirs::data_local_dir()
        .ok_or_else(|| ConfigError::InvalidConfig(
            "Cannot determine data directory".to_string()
        ))?;
    
    let data_dir = base.join("rbee");
    std::fs::create_dir_all(&data_dir)?;
    Ok(data_dir)
}
```

---

## ✅ Benefits

- ✅ Works on Linux, macOS, Windows
- ✅ Follows platform conventions
- ✅ No breaking changes for Linux users
- ✅ Simple implementation (3-4 hours)
- ✅ Already used in model catalog (consistent!)

---

## 📊 Impact

### Components Already Cross-Platform

✅ **Model Catalog** - Already uses `dirs::cache_dir()`  
✅ **Model Provisioner** - Uses same directory as catalog

### Components That Need Updates

🔧 **rbee-config** - Needs this fix (3-4 hours)

---

## 🧪 Testing

```bash
# Linux
cargo test --package rbee-config
ls ~/.config/rbee/

# macOS
cargo test --package rbee-config
ls ~/Library/Application\ Support/rbee/

# Windows
cargo test --package rbee-config
dir %APPDATA%\rbee\
```

---

## 📝 Full Details

See: `CROSS_PLATFORM_CONFIG_PLAN.md` for complete implementation guide

---

**Ready to implement! 🐝**
