#!/bin/bash

# Script to migrate IconPlate and IconCardHeader consumers to use rendered icon components
# with default w-6 h-6 classes
#
# Usage:
#   ./migrate-icon-components.sh           # Dry run (preview changes)
#   ./migrate-icon-components.sh --apply   # Apply changes
#
# This script will:
#   1. Change LucideIcon types to React.ReactNode
#   2. Convert icon={IconName} to icon={<IconName className="w-6 h-6" />}
#   3. Clean up unused LucideIcon imports
#   4. Create backups of all modified files

set -e

DRY_RUN=true
if [[ "$1" == "--apply" ]]; then
  DRY_RUN=false
fi

UI_SRC="/home/vince/Projects/llama-orch/frontend/packages/rbee-ui/src"

echo "🔍 Finding all files that use IconPlate or IconCardHeader..."

# Find all TypeScript/TSX files that import or use IconPlate or IconCardHeader
# Exclude the component definition files themselves and stories
FILES=$(grep -rl "IconPlate\|IconCardHeader" "$UI_SRC" \
  --include="*.tsx" \
  --include="*.ts" \
  --exclude="IconPlate.tsx" \
  --exclude="IconCardHeader.tsx" \
  --exclude="*.stories.tsx" \
  --exclude="*.spec.tsx" \
  --exclude="*.test.tsx" \
  | sort -u)

echo "📝 Found $(echo "$FILES" | wc -l) files to process"
echo ""

if $DRY_RUN; then
  echo "🔍 DRY RUN MODE - No files will be modified"
  echo "   Run with --apply to make changes"
  echo ""
fi

# Backup directory
BACKUP_DIR="/tmp/icon-migration-backup-$(date +%Y%m%d-%H%M%S)"
if ! $DRY_RUN; then
  mkdir -p "$BACKUP_DIR"
fi

for file in $FILES; do
  echo "Processing: $file"
  
  if $DRY_RUN; then
    # Show what would change
    echo "  Would update:"
    grep -n "icon=\{[A-Z][a-zA-Z0-9]*\}" "$file" 2>/dev/null | head -3 || echo "    (no direct icon props found)"
    grep -n ":\s*LucideIcon" "$file" 2>/dev/null | head -3 || echo "    (no LucideIcon types found)"
    continue
  fi
  
  # Create backup
  cp "$file" "$BACKUP_DIR/$(basename "$file")"
  
  # Create a temporary file for processing
  temp_file=$(mktemp)
  
  # Process the file with Perl for complex regex replacements
  perl -i -pe '
    # Step 1: Change LucideIcon type to React.ReactNode in interfaces/types
    s/:\s*LucideIcon(\s*[;,\}])/: React.ReactNode$1/g;
    s/\?:\s*LucideIcon(\s*[;,\}])/?:  React.ReactNode$1/g;
    
    # Step 2: Rename icon prop to Icon in function destructuring for JSX rendering
    # Pattern: { icon, -> { icon: Icon,
    s/\{\s*icon,/{ icon: Icon,/g;
    s/\{\s*icon:/{ icon: Icon:/g;
    
    # Step 3: Update IconPlate/IconCardHeader usage - icon={SomeIcon} to icon={<SomeIcon className="w-6 h-6" />}
    # Only match direct icon component references (PascalCase identifiers)
    # Avoid matching already-rendered components (those with < or >)
    s/\bicon=\{([A-Z][a-zA-Z0-9]*)\}(?!\s*\/>)/icon={<$1 className="w-6 h-6" \/>}/g;
    
    # Step 4: Handle dynamic icon props like stat.icon, item.icon, etc.
    # Pattern: icon={variable.icon} -> icon={<variable.icon className="w-6 h-6" />}
    s/\bicon=\{([a-z][a-zA-Z0-9]*\.icon)\}/icon={<$1 className="w-6 h-6" \/>}/g;
    
  ' "$file"
  
  # Step 5: Clean up LucideIcon imports if no longer needed
  # Check if LucideIcon is still referenced in the file (excluding type definitions)
  if ! grep -q "LucideIcon" "$file" 2>/dev/null; then
    # Remove the LucideIcon import line
    perl -i -pe '
      s/import type \{ LucideIcon \} from .lucide-react.\n//g;
      s/, LucideIcon//g;
      s/LucideIcon,\s*//g;
    ' "$file"
  fi
  
  echo "  ✓ Updated type definitions and icon props"
done

echo ""
if $DRY_RUN; then
  echo "✅ Dry run complete!"
  echo ""
  echo "📋 Summary of what would change:"
  echo "   - Update LucideIcon → React.ReactNode in type definitions"
  echo "   - Convert icon={IconName} → icon={<IconName className=\"w-6 h-6\" />}"
  echo "   - Clean up unused LucideIcon imports"
  echo ""
  echo "🔍 Files with dynamic icons (need manual review):"
  grep -l "icon={.*\.icon}" $FILES 2>/dev/null || echo "   None found"
  echo ""
  echo "▶️  To apply changes, run: $0 --apply"
else
  echo "✅ Migration complete!"
  echo ""
  echo "📦 Backups saved to: $BACKUP_DIR"
  echo ""
  echo "⚠️  MANUAL REVIEW REQUIRED:"
  echo "   1. Check files with dynamic icon props (e.g., stat.icon, item.icon)"
  echo "   2. Verify icon imports are still needed"
  echo "   3. Update story files manually if needed"
  echo "   4. Run: pnpm biome check --write"
  echo ""
  echo "🔍 Files that may need manual review (dynamic icons):"
  grep -l "icon={.*\.icon}" $FILES 2>/dev/null || echo "   None found"
  echo ""
  echo "📋 Summary of changes:"
  echo "   - Updated LucideIcon → React.ReactNode in type definitions"
  echo "   - Converted icon={IconName} → icon={<IconName className=\"w-6 h-6\" />}"
  echo "   - Backups created in: $BACKUP_DIR"
fi
