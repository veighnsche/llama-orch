#!/bin/bash
# Audit script to find false positive patterns in the codebase
# Run from project root: ./audit_false_positives.sh

set -e

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  AUDIT FOR FALSE POSITIVE PATTERNS                             ║"
echo "║  Finding tests/code that claim to do X but actually do Y       ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Colors for output
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# 1. Tests with "real" or "actual" in name (RED FLAG)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1. Tests with 'real' or 'actual' in name (RED FLAG):"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
REAL_TESTS=$(grep -r "real\|actual" xtask/tests/ --include="*.rs" -i 2>/dev/null | grep "fn test" || true)
if [ -z "$REAL_TESTS" ]; then
    echo -e "${GREEN}✓ None found${NC}"
else
    echo -e "${RED}⚠️  FOUND:${NC}"
    echo "$REAL_TESTS"
fi
echo ""

# 2. SSH tests NOT using RbeeSSHClient
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "2. SSH tests NOT using RbeeSSHClient:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
FOUND_ISSUE=0
for file in $(find xtask/tests -name "*ssh*.rs" 2>/dev/null); do
    if [ -f "$file" ]; then
        if ! grep -q "RbeeSSHClient" "$file"; then
            echo -e "${RED}⚠️  $file - No RbeeSSHClient found!${NC}"
            FOUND_ISSUE=1
        else
            echo -e "${GREEN}✓ $file - Uses RbeeSSHClient${NC}"
        fi
    fi
done
if [ $FOUND_ISSUE -eq 0 ] && [ -z "$(find xtask/tests -name "*ssh*.rs" 2>/dev/null)" ]; then
    echo -e "${YELLOW}  No SSH test files found${NC}"
fi
echo ""

# 3. SSH tests using docker exec (WRONG)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "3. SSH tests using docker exec (WRONG):"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
for file in $(find xtask/tests -name "*ssh*.rs" 2>/dev/null); do
    if [ -f "$file" ]; then
        DOCKER_EXEC=$(grep -n "docker exec\|Command::new.*docker" "$file" 2>/dev/null || true)
        if [ ! -z "$DOCKER_EXEC" ]; then
            echo -e "${RED}⚠️  $file uses docker exec:${NC}"
            echo "$DOCKER_EXEC"
        fi
    fi
done
echo -e "${GREEN}✓ Check complete${NC}"
echo ""

# 4. Query functions returning hardcoded empty values
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "4. Query functions returning hardcoded empty values:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
SUSPICIOUS=$(grep -r "fn query" --include="*.rs" -A 3 2>/dev/null | grep -B 3 "Vec::new()" | grep "fn query" || true)
if [ -z "$SUSPICIOUS" ]; then
    echo -e "${GREEN}✓ None found${NC}"
else
    echo -e "${RED}⚠️  FOUND suspicious query functions:${NC}"
    echo "$SUSPICIOUS"
fi
echo ""

# 5. TODOs in critical code
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "5. TODOs in critical code:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ -d "bin/99_shared_crates/daemon-sync/src" ]; then
    DAEMON_TODOS=$(grep -r "TODO" bin/99_shared_crates/daemon-sync/src/ --include="*.rs" 2>/dev/null | wc -l)
    echo "  daemon-sync: $DAEMON_TODOS TODOs"
    if [ $DAEMON_TODOS -gt 0 ]; then
        echo -e "${YELLOW}  Listing daemon-sync TODOs:${NC}"
        grep -r "TODO" bin/99_shared_crates/daemon-sync/src/ --include="*.rs" 2>/dev/null | head -10
    fi
fi
if [ -d "xtask/tests" ]; then
    TEST_TODOS=$(grep -r "TODO" xtask/tests/ --include="*.rs" 2>/dev/null | wc -l)
    echo "  tests: $TEST_TODOS TODOs"
fi
echo ""

# 6. Unimplemented functions
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "6. Unimplemented functions:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
UNIMPL=$(grep -r "unimplemented!\|todo!" --include="*.rs" 2>/dev/null | grep -v "test\|#\[" | wc -l)
echo "  Found: $UNIMPL unimplemented functions"
if [ $UNIMPL -gt 0 ]; then
    echo -e "${YELLOW}  Listing unimplemented:${NC}"
    grep -r "unimplemented!\|todo!" --include="*.rs" 2>/dev/null | grep -v "test\|#\[" | head -10
fi
echo ""

# 7. Files with "mock" or "fake" in name
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "7. Files with 'mock' or 'fake' in name:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
MOCK_FILES=$(find . -name "*mock*.rs" -o -name "*fake*.rs" 2>/dev/null | grep -v target || true)
if [ -z "$MOCK_FILES" ]; then
    echo -e "${GREEN}✓ None found${NC}"
else
    echo -e "${YELLOW}  Found:${NC}"
    echo "$MOCK_FILES"
fi
echo ""

# Summary
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  MANUAL REVIEW REQUIRED                                        ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Review these files manually:"
echo "  1. bin/99_shared_crates/daemon-sync/src/query.rs"
echo "  2. xtask/tests/docker/ssh_tests.rs"
echo "  3. xtask/src/chaos/*.rs"
echo "  4. xtask/src/integration/*.rs"
echo ""
echo "For each file, verify:"
echo "  - Does it do what the name claims?"
echo "  - Does it use the real implementation (SSH, HTTP, etc)?"
echo "  - Does it return actual values or hardcoded ones?"
echo "  - Are there TODOs in critical paths?"
echo ""
echo "See .docs/AUDIT_FALSE_POSITIVES.md for detailed audit guide"
