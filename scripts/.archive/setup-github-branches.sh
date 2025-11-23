#!/bin/bash
# Setup GitHub development and production branches
# Created by: TEAM-451
# 
# Usage:
#   ./scripts/setup-github-branches.sh           # Create branches
#   ./scripts/setup-github-branches.sh --protect # Create + configure protection

set -e

REPO="rbee-keeper/rbee"
DEV_BRANCH="development"
PROD_BRANCH="production"
PROTECT_BRANCHES=false

# Parse arguments
if [[ "$1" == "--protect" ]]; then
    PROTECT_BRANCHES=true
fi

echo "🔧 Setting up GitHub branches..."

# Authenticate
if ! gh auth status &>/dev/null; then
    echo "📝 Authenticating with GitHub..."
    gh auth login
fi

# Get current branch
CURRENT_BRANCH=$(git branch --show-current)
echo "📍 Current branch: $CURRENT_BRANCH"

# Create development branch
if git show-ref --verify --quiet "refs/heads/$DEV_BRANCH"; then
    echo "✅ $DEV_BRANCH exists"
else
    echo "🌿 Creating $DEV_BRANCH..."
    git checkout -b "$DEV_BRANCH"
    git push -u origin "$DEV_BRANCH"
    git checkout "$CURRENT_BRANCH"
fi

# Create production branch
if git show-ref --verify --quiet "refs/heads/$PROD_BRANCH"; then
    echo "✅ $PROD_BRANCH exists"
else
    echo "🌿 Creating $PROD_BRANCH..."
    git checkout -b "$PROD_BRANCH"
    git push -u origin "$PROD_BRANCH"
    git checkout "$CURRENT_BRANCH"
fi

# Set default branch
echo "🔄 Setting default branch to $DEV_BRANCH..."
gh repo edit "$REPO" --default-branch "$DEV_BRANCH" 2>/dev/null || echo "⚠️  Skipped (needs admin access)"

# Configure branch protection if requested
if [[ "$PROTECT_BRANCHES" == true ]]; then
    echo "🛡️  Configuring branch protection..."
    
    # Production: strict protection
    gh api repos/"$REPO"/branches/"$PROD_BRANCH"/protection \
        --method PUT \
        -f required_pull_request_reviews[required_approving_review_count]=1 \
        -f required_pull_request_reviews[dismiss_stale_reviews]=true \
        -f enforce_admins=true \
        -f allow_force_pushes=false \
        -f allow_deletions=false \
        -f required_conversation_resolution=true \
        2>/dev/null && echo "✅ Production protected" || echo "⚠️  Production protection failed (needs admin)"
    
    # Development: allow force push
    gh api repos/"$REPO"/branches/"$DEV_BRANCH"/protection \
        --method PUT \
        -f allow_force_pushes=true \
        -f allow_deletions=false \
        2>/dev/null && echo "✅ Development configured" || echo "⚠️  Development config failed (needs admin)"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "📋 Branches:"
echo "   • $DEV_BRANCH (default) - Active development"
echo "   • $PROD_BRANCH - Production releases"
echo ""
if [[ "$PROTECT_BRANCHES" != true ]]; then
    echo "💡 To configure branch protection, run:"
    echo "   ./scripts/setup-github-branches.sh --protect"
    echo ""
    echo "   Or configure manually at:"
    echo "   https://github.com/$REPO/settings/branches"
    echo ""
fi
