#!/bin/bash
# TEAM-135: Scaffolding verification script

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEAM-135 SCAFFOLDING VERIFICATION"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

cd /home/vince/Projects/llama-orch/bin

PASS=0
FAIL=0

# Check shared crates
echo "✓ Checking shared crates..."
for crate in daemon-lifecycle rbee-http-client rbee-types; do
  if [ ! -d "shared-crates/$crate" ]; then
    echo "  ❌ Missing: shared-crates/$crate"
    FAIL=$((FAIL + 1))
  else
    echo "  ✅ Found: shared-crates/$crate"
    PASS=$((PASS + 1))
  fi
done

# Check binaries
echo ""
echo "✓ Checking binaries..."
for binary in rbee-keeper queen-rbee rbee-hive llm-worker-rbee; do
  if [ ! -f "$binary/src/main.rs" ]; then
    echo "  ❌ Missing: $binary/src/main.rs"
    FAIL=$((FAIL + 1))
  else
    echo "  ✅ Found: $binary/src/main.rs"
    PASS=$((PASS + 1))
  fi
done

# Check NO CLI in rbee-hive
echo ""
echo "✓ Checking rbee-hive has NO CLI crate..."
if [ -d "rbee-hive-crates/cli" ]; then
  echo "  ❌ VIOLATION: rbee-hive-crates/cli/ should NOT exist!"
  FAIL=$((FAIL + 1))
else
  echo "  ✅ Correct: No CLI in rbee-hive"
  PASS=$((PASS + 1))
fi

# Check SSH in queen-rbee-crates (NOT shared-crates)
echo ""
echo "✓ Checking SSH location..."
if [ -d "queen-rbee-crates/ssh-client" ]; then
  echo "  ✅ Correct: SSH in queen-rbee-crates"
  PASS=$((PASS + 1))
else
  echo "  ❌ Missing: queen-rbee-crates/ssh-client"
  FAIL=$((FAIL + 1))
fi

if [ -d "shared-crates/rbee-ssh-client" ]; then
  echo "  ❌ VIOLATION: SSH should NOT be in shared-crates!"
  FAIL=$((FAIL + 1))
else
  echo "  ✅ Correct: No SSH in shared-crates"
  PASS=$((PASS + 1))
fi

# Check backend in llm-worker binary
echo ""
echo "✓ Checking backend location..."
if [ -d "llm-worker-rbee/src/backend" ]; then
  echo "  ✅ Correct: backend in llm-worker binary"
  PASS=$((PASS + 1))
else
  echo "  ❌ Missing: llm-worker-rbee/src/backend"
  FAIL=$((FAIL + 1))
fi

if [ -d "llm-worker-rbee-crates/backend" ]; then
  echo "  ❌ VIOLATION: backend should NOT be in crates!"
  FAIL=$((FAIL + 1))
else
  echo "  ✅ Correct: No backend in crates"
  PASS=$((PASS + 1))
fi

# Count crates
echo ""
echo "✓ Counting crates..."
SHARED_COUNT=$(find shared-crates -maxdepth 1 -type d -name "daemon-lifecycle" -o -name "rbee-http-client" -o -name "rbee-types" | wc -l)
KEEPER_COUNT=$(find rbee-keeper-crates -maxdepth 1 -type d | tail -n +2 | wc -l)
QUEEN_COUNT=$(find queen-rbee-crates -maxdepth 1 -type d | tail -n +2 | wc -l)
HIVE_COUNT=$(find rbee-hive-crates -maxdepth 1 -type d | tail -n +2 | wc -l)
WORKER_COUNT=$(find llm-worker-rbee-crates -maxdepth 1 -type d | tail -n +2 | wc -l)

echo "  Shared crates: $SHARED_COUNT (expected: 3)"
echo "  rbee-keeper crates: $KEEPER_COUNT (expected: 3)"
echo "  queen-rbee crates: $QUEEN_COUNT (expected: 6)"
echo "  rbee-hive crates: $HIVE_COUNT (expected: 7)"
echo "  llm-worker-rbee crates: $WORKER_COUNT (expected: 3)"

TOTAL=$((SHARED_COUNT + KEEPER_COUNT + QUEEN_COUNT + HIVE_COUNT + WORKER_COUNT))
echo "  Total new crates: $TOTAL (expected: 22)"

if [ $TOTAL -eq 22 ]; then
  echo "  ✅ Correct crate count"
  PASS=$((PASS + 1))
else
  echo "  ❌ Incorrect crate count"
  FAIL=$((FAIL + 1))
fi

# Check .bak preservation
echo ""
echo "✓ Checking .bak preservation..."
for bak in rbee-keeper.bak queen-rbee.bak rbee-hive.bak llm-worker-rbee.bak; do
  if [ -d "$bak" ]; then
    echo "  ✅ Preserved: $bak"
    PASS=$((PASS + 1))
  else
    echo "  ❌ Missing: $bak"
    FAIL=$((FAIL + 1))
  fi
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "RESULTS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ✅ Passed: $PASS"
echo "  ❌ Failed: $FAIL"
echo ""

if [ $FAIL -eq 0 ]; then
  echo "🎉 ALL CHECKS PASSED!"
  exit 0
else
  echo "❌ SOME CHECKS FAILED"
  exit 1
fi
