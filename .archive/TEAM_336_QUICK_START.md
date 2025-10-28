# TEAM-336: Narration Panel - Quick Start

## ✅ Verification

```bash
./verify-narration-setup.sh
```

**Expected:** All checks pass ✅

---

## 🚀 Run It

```bash
cargo run --bin rbee-keeper
```

---

## 🔍 What to Look For

### 1. Panel Visible
- Right side of window
- 320px wide
- Header says "Narration"
- Two buttons: "Test" and "Clear"

### 2. Browser Console (F12)
```
[NarrationPanel] Setting up listener for 'narration' events
```

### 3. Click "Test" Button
- Should see 4 events appear in panel
- Console shows 4 "Received event" messages

---

## ❌ Not Working?

### Quick Checks

1. **Panel not visible?**
   ```bash
   grep "NarrationPanel" bin/00_rbee_keeper/ui/src/components/Shell.tsx
   ```
   Should see: `import { NarrationPanel }` and `<NarrationPanel />`

2. **No console logs?**
   ```bash
   cd bin/00_rbee_keeper/ui && pnpm build
   ```

3. **Test button does nothing?**
   ```javascript
   // In browser console:
   const { invoke } = await import('@tauri-apps/api/core');
   await invoke('test_narration');
   ```

---

## 📚 Full Documentation

- **Complete guide:** `TEAM_336_COMPLETE_SUMMARY.md`
- **Debugging:** `TEAM_336_NARRATION_NOT_WORKING.md`
- **Implementation:** `TEAM_336_TRACING_CONSOLIDATION.md`

---

## 🎯 Expected Result

After clicking "Test", you should see:

```
┌─────────────────────────────────────────┐
│ Narration                  [Test][Clear]│
├─────────────────────────────────────────┤
│ 14:43:25              [INFO]            │
│ 🎯 Test narration event from Tauri...  │
│                                         │
│ 14:43:25              [INFO]            │
│ This is a tracing::info! event          │
│                                         │
│ 14:43:25              [WARN]            │
│ This is a tracing::warn! event          │
│                                         │
│ 14:43:25              [ERROR]           │
│ This is a tracing::error! event         │
├─────────────────────────────────────────┤
│ 4 entries                               │
└─────────────────────────────────────────┘
```

---

**TEAM-336** | Quick Start | 2025-10-28
