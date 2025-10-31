#!/bin/bash
# TEAM-377: Verification script for Queen SDK fix

set -e

echo "============================================"
echo "TEAM-377 SDK FIX VERIFICATION"
echo "============================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check 1: Verify package.json has required fields
echo "📋 Check 1: Verifying package.json configuration..."
PACKAGE_JSON="bin/10_queen_rbee/ui/packages/queen-rbee-sdk/package.json"

if [ ! -f "$PACKAGE_JSON" ]; then
    echo -e "${RED}❌ FAIL: package.json not found${NC}"
    exit 1
fi

HAS_TYPE=$(cat "$PACKAGE_JSON" | jq -r '.type // "missing"')
HAS_EXPORTS=$(cat "$PACKAGE_JSON" | jq -r '.exports // "missing"')

if [ "$HAS_TYPE" = "module" ]; then
    echo -e "${GREEN}✅ PASS: type: \"module\" found${NC}"
else
    echo -e "${RED}❌ FAIL: type: \"module\" missing (found: $HAS_TYPE)${NC}"
    exit 1
fi

if [ "$HAS_EXPORTS" != "missing" ]; then
    echo -e "${GREEN}✅ PASS: exports field found${NC}"
else
    echo -e "${RED}❌ FAIL: exports field missing${NC}"
    exit 1
fi

echo ""

# Check 2: Verify WASM build exists
echo "📦 Check 2: Verifying WASM build output..."
WASM_FILE="bin/10_queen_rbee/ui/packages/queen-rbee-sdk/pkg/bundler/queen_rbee_sdk_bg.wasm"

if [ -f "$WASM_FILE" ]; then
    WASM_SIZE=$(ls -lh "$WASM_FILE" | awk '{print $5}')
    echo -e "${GREEN}✅ PASS: WASM file exists ($WASM_SIZE)${NC}"
else
    echo -e "${RED}❌ FAIL: WASM file missing${NC}"
    echo -e "${YELLOW}⚠️  Run: cd bin/10_queen_rbee/ui/packages/queen-rbee-sdk && pnpm build${NC}"
    exit 1
fi

echo ""

# Check 3: Compare with working Hive SDK
echo "🔄 Check 3: Comparing with working Hive SDK..."

QUEEN_TYPE=$(cat "bin/10_queen_rbee/ui/packages/queen-rbee-sdk/package.json" | jq -r '.type // "missing"')
HIVE_TYPE=$(cat "bin/20_rbee_hive/ui/packages/rbee-hive-sdk/package.json" | jq -r '.type // "missing"')

QUEEN_EXPORTS=$(cat "bin/10_queen_rbee/ui/packages/queen-rbee-sdk/package.json" | jq -r '.exports // "missing"')
HIVE_EXPORTS=$(cat "bin/20_rbee_hive/ui/packages/rbee-hive-sdk/package.json" | jq -r '.exports // "missing"')

if [ "$QUEEN_TYPE" = "$HIVE_TYPE" ]; then
    echo -e "${GREEN}✅ PASS: type field matches Hive SDK${NC}"
else
    echo -e "${RED}❌ FAIL: type field doesn't match (Queen: $QUEEN_TYPE, Hive: $HIVE_TYPE)${NC}"
    exit 1
fi

if [ "$QUEEN_EXPORTS" != "missing" ] && [ "$HIVE_EXPORTS" != "missing" ]; then
    echo -e "${GREEN}✅ PASS: Both SDKs have exports field${NC}"
else
    echo -e "${RED}❌ FAIL: exports field mismatch${NC}"
    exit 1
fi

echo ""

# Check 4: Verify Vite config has WASM plugins
echo "⚙️  Check 4: Verifying Vite configuration..."
VITE_CONFIG="bin/10_queen_rbee/ui/app/vite.config.ts"

if grep -q "vite-plugin-wasm" "$VITE_CONFIG"; then
    echo -e "${GREEN}✅ PASS: vite-plugin-wasm found${NC}"
else
    echo -e "${RED}❌ FAIL: vite-plugin-wasm missing${NC}"
    exit 1
fi

if grep -q "vite-plugin-top-level-await" "$VITE_CONFIG"; then
    echo -e "${GREEN}✅ PASS: vite-plugin-top-level-await found${NC}"
else
    echo -e "${RED}❌ FAIL: vite-plugin-top-level-await missing${NC}"
    exit 1
fi

if grep -q "exclude.*queen-rbee-sdk" "$VITE_CONFIG"; then
    echo -e "${GREEN}✅ PASS: SDK excluded from optimizeDeps${NC}"
else
    echo -e "${RED}❌ FAIL: SDK not excluded from optimizeDeps${NC}"
    exit 1
fi

echo ""

# Summary
echo "============================================"
echo -e "${GREEN}✅ ALL CHECKS PASSED${NC}"
echo "============================================"
echo ""
echo "Next steps:"
echo "1. Start dev server: cd bin/10_queen_rbee/ui/app && pnpm dev"
echo "2. Open browser: http://localhost:7834"
echo "3. Check console: Should see no module resolution errors"
echo "4. Verify: Connection status shows 'Connected' after heartbeat"
echo ""
echo "If errors persist, check:"
echo "- Browser DevTools Console for specific errors"
echo "- Network tab for 404s on WASM files"
echo "- Clear browser cache (Cmd+Shift+R or Ctrl+Shift+R)"
echo ""
