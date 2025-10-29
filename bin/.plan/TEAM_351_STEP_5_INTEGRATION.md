# TEAM-351 Step 5: Integrate All Packages

**Estimated Time:** 15-20 minutes  
**Priority:** CRITICAL  
**Previous Step:** TEAM_351_STEP_4_DEV_UTILS.md  
**Next Step:** TEAM_352 (Queen Migration)

---

## Mission

Integrate all 4 packages into the pnpm workspace and verify everything works together.

**Why This Matters:**
- Makes packages available to all UIs
- Validates package dependencies
- Ensures Rust constants are generated
- Confirms the foundation is solid

---

## Deliverables Checklist

- [ ] pnpm-workspace.yaml updated
- [ ] All packages installed
- [ ] All packages build successfully
- [ ] Rust constants generated
- [ ] Test imports work
- [ ] No circular dependencies

---

## Step 1: Update pnpm Workspace

Edit `frontend/pnpm-workspace.yaml`:

```yaml
packages:
  - packages/*
  - packages/shared-config
  - packages/narration-client
  - packages/iframe-bridge
  - packages/dev-utils
  - apps/*
  - bin/*/ui/packages/*
  - bin/*/ui/app
```

---

## Step 2: Install All Packages

```bash
cd frontend
pnpm install
```

**Expected output:**
```
Progress: resolved X, reused Y, downloaded Z, added A
```

**Verify no errors.**

---

## Step 3: Build All Packages in Order

```bash
# Build in dependency order
cd packages/shared-config
pnpm build

cd ../narration-client
pnpm build

cd ../iframe-bridge
pnpm build

cd ../dev-utils
pnpm build
```

**Each should output:**
```
dist/
├── *.js
└── *.d.ts
```

---

## Step 4: Generate Rust Constants

```bash
cd packages/shared-config
pnpm generate:rust
```

**Expected output:**
```
✅ Generated Rust constants at: /path/to/frontend/shared-constants.rs
```

**Verify file exists:**
```bash
ls -la ../../shared-constants.rs
```

**File should contain:**
```rust
pub const QUEEN_DEV_PORT: u16 = 7834;
pub const QUEEN_PROD_PORT: u16 = 7833;
pub const HIVE_DEV_PORT: u16 = 7836;
// ... etc
```

---

## Step 5: Test Package Imports

Create a test file:

```bash
cd frontend
cat > test-packages.ts << 'EOF'
// Test all package imports
import { getIframeUrl, getAllowedOrigins, PORTS } from '@rbee/shared-config'
import { SERVICES, createStreamHandler } from '@rbee/narration-client'
import { createMessageSender, createMessageReceiver } from '@rbee/iframe-bridge'
import { logStartupMode, isDevelopment } from '@rbee/dev-utils'

console.log('✅ All imports successful!')

// Test shared-config
console.log('\n📦 @rbee/shared-config')
console.log('  Queen URL (dev):', getIframeUrl('queen', true))
console.log('  Allowed origins:', getAllowedOrigins())
console.log('  Ports:', Object.keys(PORTS))

// Test narration-client
console.log('\n📦 @rbee/narration-client')
console.log('  Services:', Object.keys(SERVICES))
const handler = createStreamHandler(SERVICES.queen)
console.log('  Handler type:', typeof handler)

// Test iframe-bridge
console.log('\n📦 @rbee/iframe-bridge')
const sender = createMessageSender({ targetOrigin: '*' })
console.log('  Sender type:', typeof sender)

// Test dev-utils
console.log('\n📦 @rbee/dev-utils')
console.log('  Is development:', typeof isDevelopment)
console.log('  Log function:', typeof logStartupMode)

console.log('\n✅ All packages working correctly!')
EOF
```

Run the test:

```bash
npx tsx test-packages.ts
```

**Expected output:**
```
✅ All imports successful!

📦 @rbee/shared-config
  Queen URL (dev): http://localhost:7834
  Allowed origins: [ 'http://localhost:7833', 'http://localhost:7834', ... ]
  Ports: [ 'keeper', 'queen', 'hive', 'worker' ]

📦 @rbee/narration-client
  Services: [ 'queen', 'hive', 'worker' ]
  Handler type: function

📦 @rbee/iframe-bridge
  Sender type: function

📦 @rbee/dev-utils
  Is development: function
  Log function: function

✅ All packages working correctly!
```

---

## Step 6: Verify Package Dependencies

Check that packages can import each other if needed:

```bash
cd packages/narration-client
pnpm list
```

Should show no missing dependencies.

---

## Step 7: Clean Up Test File

```bash
cd frontend
rm test-packages.ts
```

---

## Verification Checklist

### Package Structure
- [ ] `frontend/packages/shared-config/dist/` exists
- [ ] `frontend/packages/narration-client/dist/` exists
- [ ] `frontend/packages/iframe-bridge/dist/` exists
- [ ] `frontend/packages/dev-utils/dist/` exists

### Rust Integration
- [ ] `frontend/shared-constants.rs` exists
- [ ] Contains all port constants
- [ ] File is auto-generated (has timestamp)

### Workspace Integration
- [ ] `pnpm install` succeeds
- [ ] All packages in workspace
- [ ] No circular dependencies
- [ ] Test imports work

### Build System
- [ ] All packages build without errors
- [ ] TypeScript compilation succeeds
- [ ] No missing dependencies

---

## Troubleshooting

### Issue: Package not found

```bash
cd frontend
pnpm install
```

### Issue: Build fails

```bash
# Clean and rebuild
cd packages/shared-config
rm -rf dist node_modules
pnpm install
pnpm build
```

### Issue: Rust constants not generated

```bash
cd packages/shared-config
chmod +x scripts/generate-rust.js
node scripts/generate-rust.js
```

### Issue: Import errors in test

Make sure all packages are built:
```bash
cd frontend/packages
for dir in */; do
  cd "$dir"
  pnpm build
  cd ..
done
```

---

## Final Verification

Run this command to verify everything:

```bash
cd frontend

# Check all packages exist
ls -la packages/*/dist/

# Check Rust constants
cat shared-constants.rs | grep "pub const"

# Verify workspace
pnpm list --depth=0
```

**Expected:**
- All 4 packages listed
- Rust file has 10+ constants
- No errors

---

## Summary

You've created:

1. ✅ **@rbee/shared-config** - Port configuration (single source of truth)
2. ✅ **@rbee/narration-client** - Narration handling (~100 LOC saved per UI)
3. ✅ **@rbee/iframe-bridge** - iframe communication
4. ✅ **@rbee/dev-utils** - Environment utilities

**Total:** 4 packages, ~400 LOC of reusable code

**Savings:** ~360 LOC across 3 UIs (Queen, Hive, Worker)

---

## Next Steps

✅ **TEAM-351 Complete!**

**Handoff to TEAM-352:**
- All shared packages created and working
- Rust constants generated
- Ready for Queen UI migration
- Pattern validated

**Next:** `TEAM_352_QUEEN_MIGRATION_PHASE.md` - Migrate Queen to use shared packages

---

## Create Handoff Document

```bash
cat > bin/.plan/TEAM_351_HANDOFF.md << 'EOF'
# TEAM-351 Handoff

## Mission Complete

Created 4 shared packages for zero-duplication UI development.

## Deliverables

### Packages Created
1. @rbee/shared-config (port configuration)
2. @rbee/narration-client (narration handling)
3. @rbee/iframe-bridge (iframe communication)
4. @rbee/dev-utils (environment utilities)

### Integration
- All packages in pnpm workspace
- All packages build successfully
- Rust constants generated
- Test imports verified

## Metrics

- **Packages:** 4
- **Total LOC:** ~400
- **Estimated savings:** ~360 LOC across 3 UIs
- **Time spent:** 2-3 days

## Files Created

- frontend/packages/shared-config/*
- frontend/packages/narration-client/*
- frontend/packages/iframe-bridge/*
- frontend/packages/dev-utils/*
- frontend/shared-constants.rs

## Next Team: TEAM-352

**Mission:** Migrate Queen UI to use these packages

**Prerequisites:**
- Read TEAM_351 step documents
- Understand shared package APIs
- Review Queen UI current implementation

**Expected outcome:**
- Queen uses all shared packages
- ~110 LOC removed from Queen
- Pattern validated for Hive/Worker

## Success Criteria

✅ All 4 packages created  
✅ All packages build  
✅ Rust constants generated  
✅ Workspace integrated  
✅ Test imports work  
✅ Ready for TEAM-352

---

**TEAM-351: Foundation complete!** 🏗️
EOF
```

---

**TEAM-351: All packages integrated and ready!** ✅
