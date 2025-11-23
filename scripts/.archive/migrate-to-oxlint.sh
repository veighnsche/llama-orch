#!/bin/bash
# TEAM-452: Migrate from ESLint to oxlint across the entire repo
# ESLint is deprecated, oxlint is the modern replacement

set -e

echo "🔄 Migrating from ESLint to oxlint..."
echo ""

# List of packages to migrate
PACKAGES=(
  "bin/00_rbee_keeper/ui"
  "bin/10_queen_rbee/ui/app"
  "bin/20_rbee_hive/ui/app"
  "bin/30_llm_worker_rbee/ui/app"
  "bin/31_sd_worker_rbee/ui/app"
  "bin/80-hono-worker-catalog"
  "frontend/packages/env-config"
)

for pkg in "${PACKAGES[@]}"; do
  echo "📦 Migrating $pkg..."
  
  # Update package.json - replace eslint with oxlint in lint script
  if [ -f "$pkg/package.json" ]; then
    sed -i.bak 's/"lint": "eslint \./"lint": "oxlint \./' "$pkg/package.json"
    
    # Remove eslint dependencies and add oxlint
    # This is complex, so we'll do it manually for each package
    echo "  ⚠️  Manual step needed: Update dependencies in $pkg/package.json"
  fi
  
  # Remove eslint config files
  rm -f "$pkg/.eslintrc"* "$pkg/eslint.config."* 2>/dev/null || true
  
  # Create oxlintrc.json if it doesn't exist
  if [ ! -f "$pkg/oxlintrc.json" ]; then
    cat > "$pkg/oxlintrc.json" << 'EOF'
{
  "$schema": "https://raw.githubusercontent.com/oxc-project/oxc/main/npm/oxlint/configuration_schema.json",
  "rules": {
    "typescript": "warn",
    "react": "warn",
    "correctness": "warn",
    "suspicious": "warn",
    "perf": "warn"
  }
}
EOF
    echo "  ✅ Created oxlintrc.json"
  fi
  
  echo ""
done

echo "✅ Migration script complete!"
echo ""
echo "⚠️  Next steps:"
echo "1. Manually update package.json dependencies in each package"
echo "2. Remove: @eslint/js, eslint, eslint-*, typescript-eslint, @repo/eslint-config"
echo "3. Add: oxlint@^0.17.3"
echo "4. Run: pnpm install"
echo "5. Test: pnpm lint in each package"
