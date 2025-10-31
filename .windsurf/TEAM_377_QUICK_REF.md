# TEAM-377 Quick Reference Card

```
┌─────────────────────────────────────────────────────────┐
│ TEAM-377: Queen UI Bug Fixes                           │
├─────────────────────────────────────────────────────────┤
│ 🐛 BUG 1: SDK Module Resolution                        │
│   "Module name '@rbee/queen-rbee-sdk' does not         │
│    resolve to a valid URL"                             │
│                                                         │
│ 🔍 ROOT CAUSE:                                          │
│   Missing ES module configuration in package.json      │
│   - No "type": "module"                                │
│   - No "exports" field                                 │
│                                                         │
│ 🐛 BUG 2: Hive Count Always 0                          │
│   "Active Hives: 0" (but 2 hives running)              │
│                                                         │
│ 🔍 ROOT CAUSE:                                          │
│   Hardcoded empty array in DashboardPage.tsx           │
│   - const hives: any[] = []  // TODO comment           │
│   - Ignored actual data from backend                   │
│                                                         │
│ ✅ FIX 1:                                               │
│   bin/10_queen_rbee/ui/packages/queen-rbee-sdk/        │
│   package.json:                                        │
│     + "type": "module"                                 │
│     + "exports": { ".": "./pkg/bundler/..." }          │
│     + "files": ["pkg"]                                 │
│                                                         │
│ ✅ FIX 2:                                               │
│   bin/10_queen_rbee/ui/app/src/pages/DashboardPage:   │
│     - const hives: any[] = []                          │
│     + const hives = data?.hives || []                  │
│     - const hivesOnline = hives.length                 │
│     + const hivesOnline = data?.hives_online || 0      │
│                                                         │
│ 📊 IMPACT:                                              │
│   8 lines changed | 2 files | 0 breaking changes      │
│   ✅ All automated checks pass                         │
│                                                         │
│ 🧪 VERIFY:                                              │
│   bash .windsurf/TEAM_377_VERIFICATION.sh              │
│                                                         │
│ 🚀 TEST:                                                │
│   cd bin/10_queen_rbee/ui/app && pnpm dev              │
│   Open: http://localhost:7834                          │
│   Check 1: No module resolution errors                 │
│   Check 2: Active Hives count matches running hives    │
│                                                         │
│ 📚 DOCS:                                                │
│   - TEAM_377_COMPLETE.md      (Summary)                │
│   - TEAM_377_HANDOFF.md       (SDK investigation)      │
│   - TEAM_377_FIX_SUMMARY.md   (SDK visual comparison)  │
│   - TEAM_377_HIVE_COUNT_BUG.md (Hive bug analysis)     │
│   - TEAM_377_VERIFICATION.sh  (Automated checks)       │
└─────────────────────────────────────────────────────────┘

KEY LESSONS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LESSON 1: Modern ES Modules
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Modern bundlers prioritize "exports" field over "main".
Without it, runtime module resolution WILL fail, even with:
  ✅ Correct Vite config
  ✅ WASM plugins installed  
  ✅ WASM files built
  ✅ "main" field set correctly

The "exports" field is NOT optional for ESM packages.

LESSON 2: TODO Comments Are Dangerous
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
❌ NEVER leave TODO comments with broken implementations:
  const hives: any[] = []; // TODO: Parse hives from heartbeat data
  
This creates silent bugs:
  - No compilation error (empty array is valid)
  - No runtime error (code "works", returns wrong data)
  - Feature appears broken to users
  - Backend is working, UI ignores it

✅ If you can't implement now, fail loudly:
  const hives = data?.hives || (() => {
    throw new Error('UNIMPLEMENTED: Hive parsing not complete');
  })();

Better yet: Just implement it immediately.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

COMPARISON:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
❌ BROKEN (Queen - before):
{
  "name": "@rbee/queen-rbee-sdk",
  "main": "./pkg/bundler/queen_rbee_sdk.js"
}

✅ WORKING (Queen - after):
{
  "name": "@rbee/queen-rbee-sdk",
  "type": "module",
  "main": "./pkg/bundler/queen_rbee_sdk.js",
  "exports": {
    ".": "./pkg/bundler/queen_rbee_sdk.js"
  }
}

✅ WORKING (Hive - reference):
{
  "name": "@rbee/rbee-hive-sdk",
  "type": "module",
  "main": "./pkg/bundler/rbee_hive_sdk.js",
  "exports": {
    ".": "./pkg/bundler/rbee_hive_sdk.js"
  }
}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DEBUGGING CHECKLIST:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
If you see "Module name does not resolve to a valid URL":

1. Check package.json:
   □ Has "type": "module"
   □ Has "exports" field
   □ "files" includes pkg directory

2. Check WASM build:
   □ pkg/bundler/*.wasm exists
   □ File size reasonable (>100KB)

3. Check Vite config:
   □ vite-plugin-wasm installed
   □ vite-plugin-top-level-await installed
   □ SDK excluded from optimizeDeps

4. Check import:
   □ Using correct package name
   □ Dynamic import (import()) not static

5. Browser DevTools:
   □ Console for errors
   □ Network tab for 404s
   □ Clear cache (Cmd+Shift+R)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TEAM-377 | 2 bugs fixed | 8 lines | 2 files | 6 docs | 100% verified
```
