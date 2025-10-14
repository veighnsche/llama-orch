#!/bin/bash
set -e

echo "=========================================="
echo "CLEAN REINSTALL SCRIPT"
echo "=========================================="
echo ""

echo "Step 1/10: Killing any running dev servers..."
pkill -f "next dev" || true
echo "✓ Dev servers stopped"
echo ""

echo "Step 2/10: Removing .next build directories..."
find . -name ".next" -type d -exec rm -rf {} + 2>/dev/null || true
echo "✓ .next directories removed"
echo ""

echo "Step 3/10: Removing .turbo cache directories..."
find . -name ".turbo" -type d -exec rm -rf {} + 2>/dev/null || true
echo "✓ .turbo directories removed"
echo ""

echo "Step 4/10: Removing dist directories..."
find . -name "dist" -type d -exec rm -rf {} + 2>/dev/null || true
echo "✓ dist directories removed"
echo ""

echo "Step 5/10: Removing node_modules (this may take a while)..."
find . -name "node_modules" -type d | while read dir; do
    echo "  Removing $dir"
    rm -rf "$dir"
done
echo "✓ All node_modules removed"
echo ""

echo "Step 6/10: Removing root node_modules..."
rm -rf ../node_modules
echo "✓ Root node_modules removed"
echo ""

echo "Step 7/10: Removing pnpm-lock.yaml..."
rm -f pnpm-lock.yaml
rm -f ../pnpm-lock.yaml
echo "✓ Lock files removed"
echo ""

echo "Step 8/10: Installing dependencies (this will take several minutes)..."
pnpm install
echo "✓ Dependencies installed"
echo ""

echo "Step 9/10: Building UI package..."
pnpm --filter @rbee/ui run build
echo "✓ UI package built"
echo ""

echo "Step 10/10: Building commercial app..."
pnpm --filter @rbee/commercial run build
echo "✓ Commercial app built"
echo ""

echo "=========================================="
echo "CLEAN REINSTALL COMPLETE!"
echo "=========================================="
echo ""
echo "You can now start the dev server with:"
echo "  pnpm --filter @rbee/commercial run dev"
echo ""
