#!/bin/bash
# TEAM-336: Verify narration panel setup
# Run this script to check if all pieces are in place

set -e

echo "🔍 TEAM-336 Narration Panel Setup Verification"
echo "=============================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

check_pass() {
    echo -e "${GREEN}✅ PASS:${NC} $1"
}

check_fail() {
    echo -e "${RED}❌ FAIL:${NC} $1"
    exit 1
}

check_warn() {
    echo -e "${YELLOW}⚠️  WARN:${NC} $1"
}

echo "📁 Checking file existence..."
echo ""

# Check Rust files
if [ -f "bin/00_rbee_keeper/src/tracing_init.rs" ]; then
    check_pass "tracing_init.rs exists"
else
    check_fail "tracing_init.rs missing"
fi

if [ -f "bin/00_rbee_keeper/src/main.rs" ]; then
    check_pass "main.rs exists"
else
    check_fail "main.rs missing"
fi

if [ -f "bin/00_rbee_keeper/src/tauri_commands.rs" ]; then
    check_pass "tauri_commands.rs exists"
else
    check_fail "tauri_commands.rs missing"
fi

# Check TypeScript files
if [ -f "bin/00_rbee_keeper/ui/src/components/NarrationPanel.tsx" ]; then
    check_pass "NarrationPanel.tsx exists"
else
    check_fail "NarrationPanel.tsx missing"
fi

if [ -f "bin/00_rbee_keeper/ui/src/components/Shell.tsx" ]; then
    check_pass "Shell.tsx exists"
else
    check_fail "Shell.tsx missing"
fi

if [ -f "bin/00_rbee_keeper/ui/src/generated/bindings.ts" ]; then
    check_pass "bindings.ts exists"
else
    check_fail "bindings.ts missing"
fi

echo ""
echo "🔧 Checking code integration..."
echo ""

# Check Shell.tsx imports NarrationPanel
if grep -q "import.*NarrationPanel" bin/00_rbee_keeper/ui/src/components/Shell.tsx; then
    check_pass "Shell.tsx imports NarrationPanel"
else
    check_fail "Shell.tsx does NOT import NarrationPanel"
fi

# Check Shell.tsx renders NarrationPanel
if grep -q "<NarrationPanel" bin/00_rbee_keeper/ui/src/components/Shell.tsx; then
    check_pass "Shell.tsx renders <NarrationPanel />"
else
    check_fail "Shell.tsx does NOT render <NarrationPanel />"
fi

# Check main.rs calls init_gui_tracing
if grep -q "init_gui_tracing" bin/00_rbee_keeper/src/main.rs; then
    check_pass "main.rs calls init_gui_tracing()"
else
    check_fail "main.rs does NOT call init_gui_tracing()"
fi

# Check test_narration command registered
if grep -q "test_narration" bin/00_rbee_keeper/src/main.rs; then
    check_pass "test_narration command registered"
else
    check_fail "test_narration command NOT registered"
fi

# Check NarrationEvent in bindings
if grep -q "NarrationEvent" bin/00_rbee_keeper/ui/src/generated/bindings.ts; then
    check_pass "NarrationEvent type in bindings.ts"
else
    check_fail "NarrationEvent type NOT in bindings.ts"
fi

# Check NarrationPanel listens to events
if grep -q 'listen.*"narration"' bin/00_rbee_keeper/ui/src/components/NarrationPanel.tsx; then
    check_pass "NarrationPanel listens to 'narration' events"
else
    check_fail "NarrationPanel does NOT listen to events"
fi

# Check tracing_init.rs emits events
if grep -q 'emit.*"narration"' bin/00_rbee_keeper/src/tracing_init.rs; then
    check_pass "tracing_init.rs emits 'narration' events"
else
    check_fail "tracing_init.rs does NOT emit events"
fi

echo ""
echo "📦 Checking dependencies..."
echo ""

# Check Cargo.toml has required deps
if grep -q "chrono.*serde" bin/00_rbee_keeper/Cargo.toml; then
    check_pass "chrono dependency with serde feature"
else
    check_warn "chrono dependency might be missing serde feature"
fi

if grep -q "tracing-subscriber.*registry" bin/00_rbee_keeper/Cargo.toml; then
    check_pass "tracing-subscriber with registry feature"
else
    check_warn "tracing-subscriber might be missing registry feature"
fi

echo ""
echo "🔨 Building project..."
echo ""

# Try to build
if cargo check --bin rbee-keeper 2>&1 | grep -q "Finished"; then
    check_pass "cargo check succeeds"
else
    check_fail "cargo check failed - see errors above"
fi

echo ""
echo "🧪 Regenerating TypeScript bindings..."
echo ""

# Regenerate bindings
if cargo test --package rbee-keeper --lib export_typescript_bindings 2>&1 | grep -q "test result: ok"; then
    check_pass "TypeScript bindings regenerated"
else
    check_fail "TypeScript bindings generation failed"
fi

echo ""
echo "=============================================="
echo -e "${GREEN}✅ ALL CHECKS PASSED!${NC}"
echo ""
echo "Next steps:"
echo "1. Run: cargo run --bin rbee-keeper"
echo "2. Open browser DevTools (F12)"
echo "3. Look for 'Narration' panel on right side"
echo "4. Click 'Test' button"
echo "5. Check console for:"
echo "   - [NarrationPanel] Setting up listener"
echo "   - [NarrationPanel] Received event: ..."
echo ""
echo "If events still don't appear, see:"
echo "  TEAM_336_NARRATION_NOT_WORKING.md"
echo ""
