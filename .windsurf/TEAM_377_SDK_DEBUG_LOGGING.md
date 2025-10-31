# TEAM-377 - SDK Debug Logging Added

## 🐛 The Problem

**User:** "Can you please put some sort of console.log in the sdk or a println! in the sdk so that I know that the frontend can connect with the sdk"

**Errors seen:**
- **Queen:** `"Module name, '@rbee/queen-rbee-sdk' does not resolve to a valid URL."`
- **Hive:** `TypeError: undefined is not an object (evaluating 'wasm.__wbindgen_malloc')`

Both are WASM loading failures.

---

## ✅ Debug Logging Added

### 1. SDK Loader (TypeScript)

**File:** `frontend/packages/sdk-loader/src/loader.ts`

**Added:**
```typescript
for (let attempt = 1; attempt <= maxAttempts; attempt++) {
  try {
    // TEAM-377: Debug logging for SDK loading
    console.log(`[sdk-loader] Attempt ${attempt}/${maxAttempts}: Importing ${packageName}`)
    
    const mod = await withTimeout(
      import(/* @vite-ignore */ packageName),
      timeout,
      `SDK import (attempt ${attempt}/${maxAttempts})`
    )
    
    console.log(`[sdk-loader] ✅ Import successful for ${packageName}`, mod)
```

**What you'll see:**
```
[sdk-loader] Attempt 1/3: Importing @rbee/queen-rbee-sdk
[sdk-loader] ✅ Import successful for @rbee/queen-rbee-sdk { ... }
```

### 2. Queen SDK (Rust/WASM)

**File:** `bin/10_queen_rbee/ui/packages/queen-rbee-sdk/src/lib.rs`

**Added:**
```rust
#[wasm_bindgen(start)]
pub fn init() {
    // TEAM-377: Log to console so we know WASM loaded
    web_sys::console::log_1(&"🎉 [Queen SDK] WASM module initialized successfully!".into());
}
```

**What you'll see:**
```
🎉 [Queen SDK] WASM module initialized successfully!
```

### 3. Hive SDK (Rust/WASM)

**File:** `bin/20_rbee_hive/ui/packages/rbee-hive-sdk/src/lib.rs`

**Added:**
```rust
#[wasm_bindgen(start)]
pub fn init() {
    // TEAM-377: Log to console so we know WASM loaded
    web_sys::console::log_1(&"🎉 [Hive SDK] WASM module initialized successfully!".into());
}
```

**What you'll see:**
```
🎉 [Hive SDK] WASM module initialized successfully!
```

---

## 🔍 What The Logs Tell You

### Success Case

```
[sdk-loader] Attempt 1/3: Importing @rbee/queen-rbee-sdk
[sdk-loader] ✅ Import successful for @rbee/queen-rbee-sdk
🎉 [Queen SDK] WASM module initialized successfully!
```

**Meaning:** SDK loaded correctly, WASM initialized.

### Failure Case 1: Module Not Found

```
[sdk-loader] Attempt 1/3: Importing @rbee/queen-rbee-sdk
[sdk-loader] Attempt 1/3 failed, retrying in 345ms: "Module name, '@rbee/queen-rbee-sdk' does not resolve to a valid URL."
```

**Meaning:** Vite can't find the package. Possible causes:
- Package not in `node_modules`
- Package not linked (`pnpm link`)
- Wrong package name in import

**Fix:** Run `pnpm install` in the workspace root.

### Failure Case 2: WASM Init Failed

```
[sdk-loader] Attempt 1/3: Importing @rbee/queen-rbee-sdk
[sdk-loader] ✅ Import successful for @rbee/queen-rbee-sdk
(no "🎉 [Queen SDK]" message)
TypeError: undefined is not an object (evaluating 'wasm.__wbindgen_malloc')
```

**Meaning:** Module loaded but WASM didn't initialize. Possible causes:
- WASM file missing
- WASM file corrupted
- Wrong build target (bundler vs web)

**Fix:** Rebuild WASM with `wasm-pack build --target bundler`.

---

## 🔧 Rebuild Commands

```bash
# Rebuild Queen SDK
cd bin/10_queen_rbee/ui/packages/queen-rbee-sdk
wasm-pack build --target bundler --out-dir pkg/bundler

# Rebuild Hive SDK
cd bin/20_rbee_hive/ui/packages/rbee-hive-sdk
wasm-pack build --target bundler --out-dir pkg/bundler

# Rebuild SDK loader
cd frontend/packages/sdk-loader
pnpm build

# Restart dev servers
cd bin/10_queen_rbee/ui/app
pnpm dev

cd bin/20_rbee_hive/ui/app
pnpm dev
```

---

## 📊 Expected Console Output

### Queen UI (http://localhost:7834)

```
🔧 [QUEEN UI] Running in DEVELOPMENT mode
   - Vite dev server active (hot reload enabled)
   - Running on: http://localhost:7834

[sdk-loader] Attempt 1/3: Importing @rbee/queen-rbee-sdk
[sdk-loader] ✅ Import successful for @rbee/queen-rbee-sdk
🎉 [Queen SDK] WASM module initialized successfully!
```

### Hive UI (http://localhost:7836)

```
🔧 [HIVE UI] Running in DEVELOPMENT mode
   - Vite dev server active (hot reload enabled)
   - Running on: http://localhost:7836

[sdk-loader] Attempt 1/3: Importing @rbee/rbee-hive-sdk
[sdk-loader] ✅ Import successful for @rbee/rbee-hive-sdk
🎉 [Hive SDK] WASM module initialized successfully!
```

---

## 🎯 Next Steps

1. **Restart dev servers** to pick up new WASM builds
2. **Open browser console** (F12)
3. **Look for the 🎉 messages**
4. **If you see them:** SDK is working!
5. **If you don't:** Check the error messages and follow the fixes above

---

**TEAM-377 | Debug logging added | Now you can see what's happening! 🎉**
