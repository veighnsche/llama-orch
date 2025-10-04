# 🚨 EMERGENCY: VERSION AUDIT & UPDATE PLAN

**Date**: 2025-10-04  
**Severity**: HIGH  
**Status**: ACTION REQUIRED

---

## ⚠️ CRITICAL ISSUE

**We are NOT using the latest stable versions across the entire stack.**

This is causing:
- Build compatibility issues (CUDA 13 architecture support)
- Potential security vulnerabilities
- Missing performance improvements
- Inconsistent behavior across environments
- Technical debt accumulation

---

## 📊 CURRENT VERSION INVENTORY

### System Toolchain (CachyOS)
```
✅ Rust:      1.90.0 (stable, 2025-09-14) - LATEST STABLE
✅ Cargo:     1.90.0 (2025-07-30)
✅ CMake:     4.1.1
✅ CUDA:      13.0.1 (2025-08-20) - LATEST
⚠️  Rustup:   1.28.2 (system-managed via pacman)
✅ Node.js:   20.19.5 LTS (Iron)
✅ npm:       11.6.1
✅ pnpm:      10.17.1
✅ Bun:       1.2.23
```

### Rust Dependencies (Cargo.toml)
```toml
[workspace.dependencies]
anyhow = "1"                    # ⚠️  UNPINNED - could be outdated
thiserror = "1"                 # ⚠️  UNPINNED
serde = "1"                     # ⚠️  UNPINNED
serde_json = "1"                # ⚠️  UNPINNED
serde_yaml = "0.9"              # ⚠️  UNPINNED
schemars = "0.8"                # ⚠️  UNPINNED
axum = "0.7"                    # ⚠️  UNPINNED
tokio = "1"                     # ⚠️  UNPINNED
tracing = "0.1"                 # ⚠️  UNPINNED
tracing-subscriber = "0.3"      # ⚠️  UNPINNED
reqwest = "0.12"                # ⚠️  UNPINNED
futures = "0.3"                 # ⚠️  UNPINNED
http = "1"                      # ⚠️  UNPINNED
hyper = "1"                     # ⚠️  UNPINNED
bytes = "1"                     # ⚠️  UNPINNED
uuid = "1"                      # ⚠️  UNPINNED
clap = "4"                      # ⚠️  UNPINNED
sha2 = "0.10"                   # ⚠️  UNPINNED
hmac = "0.12"                   # ⚠️  UNPINNED
subtle = "2.5"                  # ⚠️  UNPINNED
hkdf = "0.12"                   # ⚠️  UNPINNED
walkdir = "2"                   # ⚠️  UNPINNED
regex = "1"                     # ⚠️  UNPINNED
insta = "1"                     # ⚠️  UNPINNED
proptest = "1"                  # ⚠️  UNPINNED
wiremock = "0.6"                # ⚠️  UNPINNED
openapiv3 = "1"                 # ⚠️  UNPINNED
jsonschema = "0.17"             # ⚠️  UNPINNED
once_cell = "1"                 # ⚠️  UNPINNED
chrono = "0.4"                  # ⚠️  UNPINNED
```

**PROBLEM**: All dependencies use loose version constraints (e.g., `"1"` instead of `"1.0.123"`).  
This means:
- ❌ No reproducible builds
- ❌ Unknown which exact versions are in use
- ❌ Could be using outdated patch versions with known bugs
- ❌ `cargo update` could introduce breaking changes

### CUDA/CMake
```
✅ CUDA 13.0.1 - Latest stable
✅ CMake 4.1.1 - Latest
❌ CMAKE_CUDA_ARCHITECTURES includes compute_70 (FIXED but indicates version awareness issue)
```

---

## 🎯 IMMEDIATE ACTION PLAN

### Phase 1: AUDIT (30 minutes)
**Priority: CRITICAL**

1. **Generate Cargo.lock analysis**
   ```bash
   cargo tree --workspace --depth 1 > DEPENDENCY_TREE.txt
   cargo outdated --workspace --root-deps-only > OUTDATED_DEPS.txt
   ```

2. **Check for security advisories**
   ```bash
   cargo audit
   cargo deny check advisories
   ```

3. **Document exact versions in use**
   ```bash
   cargo tree --workspace --edges normal --format "{p} = {f}" | sort -u > EXACT_VERSIONS.txt
   ```

### Phase 2: UPDATE STRATEGY (1 hour)
**Priority: HIGH**

#### Option A: Conservative (RECOMMENDED)
- Pin all dependencies to current working versions
- Update only security-critical packages
- Test thoroughly before broader updates

#### Option B: Aggressive
- Update all dependencies to latest stable
- Risk of breaking changes
- Requires extensive testing

**RECOMMENDATION**: Start with Option A, then plan Option B for next sprint.

### Phase 3: IMPLEMENTATION (2-4 hours)

#### Step 1: Install cargo-outdated
```bash
cargo install cargo-outdated cargo-audit cargo-deny
```

#### Step 2: Pin Current Versions
```bash
# Generate exact versions from Cargo.lock
cargo metadata --format-version 1 | jq -r '.packages[] | "\(.name) = \"\(.version)\""' > current_versions.txt
```

#### Step 3: Update Cargo.toml
Replace all `"1"` with exact versions like `"1.0.123"`

#### Step 4: Security Updates
```bash
cargo audit fix
```

#### Step 5: Test Everything
```bash
cargo test --workspace
cargo build --workspace --release
```

---

## 📋 DEPENDENCY UPDATE CHECKLIST

### Critical Dependencies (Update First)
- [ ] `tokio` - Async runtime (security critical)
- [ ] `hyper` - HTTP implementation (security critical)
- [ ] `axum` - Web framework (security critical)
- [ ] `reqwest` - HTTP client (security critical)
- [ ] `serde` - Serialization (security critical)
- [ ] `tracing` - Observability
- [ ] `clap` - CLI parsing

### Secondary Dependencies
- [ ] `anyhow` - Error handling
- [ ] `thiserror` - Error derives
- [ ] `uuid` - UUID generation
- [ ] `chrono` - Time handling
- [ ] `regex` - Pattern matching
- [ ] `sha2` / `hmac` - Cryptography
- [ ] `bytes` - Byte utilities

### Development Dependencies
- [ ] `insta` - Snapshot testing
- [ ] `proptest` - Property testing
- [ ] `wiremock` - HTTP mocking

### CUDA/Build Dependencies
- [ ] `cmake` crate version in worker-orcd/Cargo.toml
- [ ] Check for CUDA-related Rust crates

---

## 🔍 SPECIFIC CONCERNS

### 1. Rust Toolchain
**Current**: 1.90.0 (stable, 2025-09-14)  
**Status**: ✅ Latest stable  
**Action**: None required

### 2. CUDA Compatibility
**Current**: CUDA 13.0.1  
**Issue**: Code was written assuming older CUDA versions (compute_70 support)  
**Action**: ✅ FIXED - Removed compute_70 from CMakeLists.txt

### 3. Dependency Pinning
**Current**: All deps use loose constraints  
**Issue**: Non-reproducible builds, unknown versions  
**Action**: ⚠️  URGENT - Pin all versions

### 4. Security Advisories
**Status**: ❓ UNKNOWN - Need to run `cargo audit`  
**Action**: ⚠️  URGENT - Run audit immediately

---

## 🚀 RECOMMENDED IMMEDIATE ACTIONS

### RIGHT NOW (Next 10 minutes)
```bash
# 1. Check for security issues
cargo audit

# 2. Check for outdated dependencies
cargo install cargo-outdated
cargo outdated --workspace --root-deps-only

# 3. Generate dependency report
cargo tree --workspace > DEPENDENCY_REPORT.txt
```

### TODAY (Next 2 hours)
1. **Pin all workspace dependencies to exact versions**
2. **Run full test suite to establish baseline**
3. **Document any breaking changes**
4. **Create update plan for outdated deps**

### THIS WEEK
1. **Update security-critical dependencies**
2. **Update build tooling (cmake crate, etc.)**
3. **Test on all target platforms**
4. **Update CI/CD to enforce version pinning**

---

## 📝 VERSION PINNING TEMPLATE

Replace in `Cargo.toml`:
```toml
# ❌ BEFORE (Loose)
[workspace.dependencies]
tokio = { version = "1", features = ["full"] }
axum = { version = "0.7", features = ["macros", "json"] }

# ✅ AFTER (Pinned)
[workspace.dependencies]
tokio = { version = "=1.41.0", features = ["full"] }
axum = { version = "=0.7.9", features = ["macros", "json"] }
```

**Note**: Use `=` prefix for exact version pinning.

---

## 🎯 SUCCESS CRITERIA

- [ ] All dependencies pinned to exact versions
- [ ] Zero security advisories from `cargo audit`
- [ ] All tests passing
- [ ] Build succeeds on all platforms
- [ ] CI/CD enforces version constraints
- [ ] Documentation updated with version requirements
- [ ] `Cargo.lock` committed to git

---

## 📚 REFERENCES

- [Cargo Version Specifiers](https://doc.rust-lang.org/cargo/reference/specifying-dependencies.html)
- [cargo-outdated](https://github.com/kbknapp/cargo-outdated)
- [cargo-audit](https://github.com/rustsec/rustsec/tree/main/cargo-audit)
- [Semantic Versioning](https://semver.org/)

---

## 🔥 EMERGENCY CONTACTS

**If builds break after updates:**
1. Check `DEPENDENCY_REPORT.txt` for exact versions that worked
2. Revert to pinned versions from baseline
3. Update one dependency at a time
4. Test after each update

---

**CREATED BY**: Cascade (AI Assistant)  
**DATE**: 2025-10-04  
**URGENCY**: 🚨 HIGH - Address within 24 hours  
**IMPACT**: Build stability, security, reproducibility
