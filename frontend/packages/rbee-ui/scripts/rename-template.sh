#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# rename-template.sh
# ============================================================================
# Removes the "Template" suffix from a template directory and updates all
# references throughout the codebase.
#
# Usage:
#   ./rename-template.sh <TemplateName>
#
# Example:
#   ./rename-template.sh ProvidersSecurityTemplate
#
# This will rename:
#   - ProvidersSecurityTemplate/ → ProvidersSecurity/
#   - ProvidersSecurityTemplate.tsx → ProvidersSecurity.tsx
#   - ProvidersSecurityTemplateProps → ProvidersSecurityProps
#   - All imports and usages
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATES_DIR="$(cd "$SCRIPT_DIR/../src/templates" && pwd)"
PAGES_DIR="$(cd "$SCRIPT_DIR/../src/pages" && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# Helper Functions
# ============================================================================

log_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

log_success() {
    echo -e "${GREEN}✓${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

log_error() {
    echo -e "${RED}✗${NC} $1"
}

# ============================================================================
# Validation
# ============================================================================

if [ $# -ne 1 ]; then
    log_error "Usage: $0 <TemplateName>"
    log_info "Example: $0 ProvidersSecurityTemplate"
    exit 1
fi

TEMPLATE_NAME="$1"

# Validate template name ends with "Template"
if [[ ! "$TEMPLATE_NAME" =~ Template$ ]]; then
    log_error "Template name must end with 'Template'"
    log_info "Got: $TEMPLATE_NAME"
    exit 1
fi

# Extract new name (remove "Template" suffix)
NEW_NAME="${TEMPLATE_NAME%Template}"

log_info "Template rename operation"
log_info "  From: ${TEMPLATE_NAME}"
log_info "  To:   ${NEW_NAME}"
echo ""

# ============================================================================
# Preflight Checks
# ============================================================================

log_info "Running preflight checks..."

PREFLIGHT_FAILED=0

# Check 1: Template directory exists
TEMPLATE_DIR="$TEMPLATES_DIR/$TEMPLATE_NAME"
if [ ! -d "$TEMPLATE_DIR" ]; then
    log_error "Template directory does not exist: $TEMPLATE_DIR"
    PREFLIGHT_FAILED=1
else
    log_success "Template directory exists"
fi

# Check 2: New directory doesn't already exist
NEW_DIR="$TEMPLATES_DIR/$NEW_NAME"
if [ -d "$NEW_DIR" ]; then
    log_error "Target directory already exists: $NEW_DIR"
    PREFLIGHT_FAILED=1
else
    log_success "Target directory does not exist"
fi

# Check 3: Required files exist
REQUIRED_FILES=(
    "$TEMPLATE_NAME.tsx"
    "$TEMPLATE_NAME.stories.tsx"
    "index.ts"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$TEMPLATE_DIR/$file" ]; then
        log_error "Required file missing: $file"
        PREFLIGHT_FAILED=1
    else
        log_success "Found: $file"
    fi
done

# Check 4: Git working directory is clean
if ! git diff-index --quiet HEAD -- 2>/dev/null; then
    log_warning "Git working directory has uncommitted changes"
    log_warning "Consider committing or stashing changes before proceeding"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Aborted by user"
        exit 1
    fi
else
    log_success "Git working directory is clean"
fi

if [ $PREFLIGHT_FAILED -eq 1 ]; then
    log_error "Preflight checks failed"
    exit 1
fi

log_success "All preflight checks passed"
echo ""

# ============================================================================
# Confirmation
# ============================================================================

log_warning "This will:"
echo "  1. Rename directory: $TEMPLATE_NAME → $NEW_NAME"
echo "  2. Rename files: ${TEMPLATE_NAME}.tsx → ${NEW_NAME}.tsx"
echo "  3. Update component name: ${TEMPLATE_NAME} → ${NEW_NAME}"
echo "  4. Update type: ${TEMPLATE_NAME}Props → ${NEW_NAME}Props"
echo "  5. Update all imports and usages in:"
echo "     - src/templates/index.ts"
echo "     - src/pages/*/*.tsx"
echo "     - src/pages/*/*.ts"
echo ""

read -p "Proceed with rename? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    log_info "Aborted by user"
    exit 0
fi

# ============================================================================
# Rename Operations
# ============================================================================

log_info "Starting rename operations..."

# Step 1: Rename directory
log_info "Renaming directory..."
mv "$TEMPLATE_DIR" "$NEW_DIR"
log_success "Directory renamed"

# Step 2: Rename files
log_info "Renaming files..."
mv "$NEW_DIR/${TEMPLATE_NAME}.tsx" "$NEW_DIR/${NEW_NAME}.tsx"
mv "$NEW_DIR/${TEMPLATE_NAME}.stories.tsx" "$NEW_DIR/${NEW_NAME}.stories.tsx"
log_success "Files renamed"

# Step 3: Update component file
log_info "Updating component file..."
sed -i "s/${TEMPLATE_NAME}Props/${NEW_NAME}Props/g" "$NEW_DIR/${NEW_NAME}.tsx"
sed -i "s/export function ${TEMPLATE_NAME}(/export function ${NEW_NAME}(/g" "$NEW_DIR/${NEW_NAME}.tsx"
sed -i "s/ \* ${TEMPLATE_NAME} -/ \* ${NEW_NAME} -/g" "$NEW_DIR/${NEW_NAME}.tsx"
sed -i "s/ \* <${TEMPLATE_NAME}/ \* <${NEW_NAME}/g" "$NEW_DIR/${NEW_NAME}.tsx"
log_success "Component file updated"

# Step 4: Update stories file
log_info "Updating stories file..."
sed -i "s/import { ${TEMPLATE_NAME} } from '.\/${TEMPLATE_NAME}'/import { ${NEW_NAME} } from '.\/${NEW_NAME}'/g" "$NEW_DIR/${NEW_NAME}.stories.tsx"
sed -i "s/title: 'Templates\/${TEMPLATE_NAME}'/title: 'Templates\/${NEW_NAME}'/g" "$NEW_DIR/${NEW_NAME}.stories.tsx"
sed -i "s/component: ${TEMPLATE_NAME}/component: ${NEW_NAME}/g" "$NEW_DIR/${NEW_NAME}.stories.tsx"
sed -i "s/Meta<typeof ${TEMPLATE_NAME}>/Meta<typeof ${NEW_NAME}>/g" "$NEW_DIR/${NEW_NAME}.stories.tsx"
sed -i "s/ \* ${TEMPLATE_NAME} as/ \* ${NEW_NAME} as/g" "$NEW_DIR/${NEW_NAME}.stories.tsx"
sed -i "s/<${TEMPLATE_NAME} /<${NEW_NAME} /g" "$NEW_DIR/${NEW_NAME}.stories.tsx"
log_success "Stories file updated"

# Step 5: Update index.ts
log_info "Updating index.ts..."
sed -i "s/export \* from '.\/${TEMPLATE_NAME}'/export \* from '.\/${NEW_NAME}'/g" "$NEW_DIR/index.ts"
log_success "Index file updated"

# Step 6: Update templates barrel export
log_info "Updating templates/index.ts..."
TEMPLATES_INDEX="$TEMPLATES_DIR/index.ts"
sed -i "s/export \* from '.\/${TEMPLATE_NAME}'/export \* from '.\/${NEW_NAME}'/g" "$TEMPLATES_INDEX"
log_success "Templates index updated"

# Step 7: Update page imports and usages
log_info "Updating page files..."
find "$PAGES_DIR" -type f \( -name "*.tsx" -o -name "*.ts" \) -exec sed -i \
    -e "s/${TEMPLATE_NAME}Props/${NEW_NAME}Props/g" \
    -e "s/import { ${TEMPLATE_NAME} }/import { ${NEW_NAME} }/g" \
    -e "s/import {\\([^}]*\\)${TEMPLATE_NAME}\\([^}]*\\)}/import {\\1${NEW_NAME}\\2}/g" \
    -e "s/<${TEMPLATE_NAME} /<${NEW_NAME} /g" \
    {} \;
log_success "Page files updated"

log_success "All rename operations completed"
echo ""

# ============================================================================
# Post-Rename Checks
# ============================================================================

log_info "Running post-rename checks..."

POSTCHECK_FAILED=0

# Check 1: New directory exists
if [ ! -d "$NEW_DIR" ]; then
    log_error "New directory does not exist: $NEW_DIR"
    POSTCHECK_FAILED=1
else
    log_success "New directory exists"
fi

# Check 2: Old directory is gone
if [ -d "$TEMPLATE_DIR" ]; then
    log_error "Old directory still exists: $TEMPLATE_DIR"
    POSTCHECK_FAILED=1
else
    log_success "Old directory removed"
fi

# Check 3: New files exist
NEW_FILES=(
    "$NEW_NAME.tsx"
    "$NEW_NAME.stories.tsx"
    "index.ts"
)

for file in "${NEW_FILES[@]}"; do
    if [ ! -f "$NEW_DIR/$file" ]; then
        log_error "Expected file missing: $file"
        POSTCHECK_FAILED=1
    else
        log_success "Found: $file"
    fi
done

# Check 4: No old references in new files
log_info "Checking for old references in renamed files..."
OLD_REFS_COUNT=$(grep -r "${TEMPLATE_NAME}" "$NEW_DIR" 2>/dev/null | wc -l || echo "0")
if [ "$OLD_REFS_COUNT" -gt 0 ]; then
    log_warning "Found $OLD_REFS_COUNT references to old name in new directory:"
    grep -rn "${TEMPLATE_NAME}" "$NEW_DIR" || true
    log_warning "Manual review recommended"
else
    log_success "No old references found in new directory"
fi

# Check 5: Templates index updated
if grep -q "from '\./${TEMPLATE_NAME}'" "$TEMPLATES_INDEX" 2>/dev/null; then
    log_error "Old export still exists in templates/index.ts"
    POSTCHECK_FAILED=1
else
    log_success "Templates index correctly updated"
fi

# Check 6: New export exists
if grep -q "from '\./${NEW_NAME}'" "$TEMPLATES_INDEX" 2>/dev/null; then
    log_success "New export exists in templates/index.ts"
else
    log_error "New export missing from templates/index.ts"
    POSTCHECK_FAILED=1
fi

# Check 7: TypeScript compilation (if available)
if command -v tsc &> /dev/null; then
    log_info "Running TypeScript check..."
    if (cd "$SCRIPT_DIR/.." && tsc --noEmit 2>&1 | grep -i "error" > /dev/null); then
        log_warning "TypeScript errors detected - manual review recommended"
    else
        log_success "TypeScript check passed"
    fi
else
    log_warning "TypeScript not available - skipping type check"
fi

if [ $POSTCHECK_FAILED -eq 1 ]; then
    log_error "Some post-rename checks failed"
    log_warning "Review the errors above and fix manually if needed"
    exit 1
fi

log_success "All post-rename checks passed"
echo ""

# ============================================================================
# Summary
# ============================================================================

echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
log_success "Rename completed successfully!"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
log_info "Summary:"
echo "  • Directory:  $TEMPLATE_NAME → $NEW_NAME"
echo "  • Component:  $TEMPLATE_NAME → $NEW_NAME"
echo "  • Props Type: ${TEMPLATE_NAME}Props → ${NEW_NAME}Props"
echo ""
log_info "Next steps:"
echo "  1. Review changes: git diff"
echo "  2. Test the application"
echo "  3. Commit changes: git add . && git commit -m 'refactor: rename $TEMPLATE_NAME to $NEW_NAME'"
echo ""
