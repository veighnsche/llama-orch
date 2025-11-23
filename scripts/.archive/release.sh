#!/bin/bash
# Simple release helper using existing tools (pnpm, cargo-workspaces)
# Created by: TEAM-451
# Rule Zero: Use existing tools > custom scripts

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

if [ -z "$1" ]; then
    echo "Usage: ./scripts/release.sh [patch|minor|major]"
    echo ""
    echo "  patch: 0.1.0 → 0.1.1 (bug fixes)"
    echo "  minor: 0.1.0 → 0.2.0 (new features)"
    echo "  major: 0.1.0 → 1.0.0 (breaking changes)"
    exit 1
fi

BUMP_TYPE="$1"

# Validate bump type
case "$BUMP_TYPE" in
    patch|minor|major) ;;
    *)
        echo "Invalid bump type: $BUMP_TYPE"
        echo "Must be: patch, minor, or major"
        exit 1
        ;;
esac

echo -e "${BLUE}🔄 Bumping version ($BUMP_TYPE)...${NC}"
echo ""

# Check if cargo-workspaces is installed
if ! command -v cargo-workspaces &> /dev/null; then
    echo -e "${YELLOW}⚠️  cargo-workspaces not found${NC}"
    echo ""
    echo "Install it:"
    echo "  cargo install cargo-workspaces"
    echo ""
    exit 1
fi

# Bump JavaScript packages
echo -e "${BLUE}📦 Bumping JavaScript packages...${NC}"
pnpm -r version "$BUMP_TYPE" --no-git-tag-version

# Bump Rust crates
echo -e "${BLUE}🦀 Bumping Rust crates...${NC}"
cargo workspaces version --no-git-commit --yes "$BUMP_TYPE"

# Get new version
NEW_VERSION=$(grep '^version = ' Cargo.toml | head -1 | cut -d'"' -f2)

echo ""
echo -e "${GREEN}✅ Version bumped to $NEW_VERSION${NC}"
echo ""
echo "Next steps:"
echo -e "  ${BLUE}git add .${NC}"
echo -e "  ${BLUE}git commit -m \"chore: bump version to $NEW_VERSION\"${NC}"
echo -e "  ${BLUE}git push origin development${NC}"
echo -e "  ${BLUE}gh pr create --base production --head development --title \"Release v$NEW_VERSION\"${NC}"
