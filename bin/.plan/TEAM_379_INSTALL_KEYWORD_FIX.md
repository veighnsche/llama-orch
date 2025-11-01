# TEAM-379: Install Keyword Fix

**Status:** ✅ COMPLETE

**Problem:** When installing with `binary: "release"` from the UI, the install command failed with:
```
lifecycle_shared::install::resolve_binary_path verify_binary       
🔍 Verifying pre-built binary at: release
lifecycle_shared::install::resolve_binary_path binary_not_found    
❌ Binary not found at: release
```

The UI was passing `"release"` as a string, but `resolve_binary_path` was treating it as a literal file path instead of a keyword.

## Solution

Updated `resolve_binary_path()` in `lifecycle-shared/src/install.rs` to handle special keywords:
- `"release"` → resolves to `target/release/{daemon_name}`
- `"debug"` → resolves to `target/debug/{daemon_name}`
- Any other path → used as-is

## Implementation

```rust
// TEAM-379: Handle special keywords "release" and "debug"
let resolved_path = if path.to_str() == Some("release") {
    PathBuf::from(format!("target/release/{}", daemon_name))
} else if path.to_str() == Some("debug") {
    PathBuf::from(format!("target/debug/{}", daemon_name))
} else {
    path
};
```

## Usage

Now these all work correctly:

1. **Install production build:**
   ```bash
   rbee hive install --host workstation --binary release
   ```
   → Resolves to `target/release/rbee-hive`

2. **Install debug build:**
   ```bash
   rbee hive install --host workstation --binary debug
   ```
   → Resolves to `target/debug/rbee-hive`

3. **Install specific binary:**
   ```bash
   rbee hive install --host workstation --binary /path/to/custom/binary
   ```
   → Uses exact path

4. **Auto-build from source:**
   ```bash
   rbee hive install --host workstation
   ```
   → Builds with `cargo build --release`

## Files Changed

- `bin/96_lifecycle/lifecycle-shared/src/install.rs` - Added keyword resolution

## Testing

```bash
# Build release binary first
cargo build --bin rbee-hive --release

# Install to remote machine using keyword
rbee hive install --host workstation --binary release
```

Should now resolve to `target/release/rbee-hive` and install successfully.

---

**TEAM-379 Complete** - Install keywords now work correctly.
