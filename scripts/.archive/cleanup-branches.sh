#!/bin/bash
# Cleanup old branches
# Created by: TEAM-451

set -e

echo "🧹 Cleaning up old branches..."
echo ""

# List of branches to delete
OLD_BRANCHES=(
    "commercial-site-updates"
    "fix/team-117-ambiguous-steps"
    "fix/team-122-panics-final"
    "mac-compat"
    "stakeholder-story"
)

echo "Branches to delete:"
for branch in "${OLD_BRANCHES[@]}"; do
    echo "  - $branch"
done
echo ""

read -p "Delete these branches? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

echo ""
echo "Deleting remote branches..."
for branch in "${OLD_BRANCHES[@]}"; do
    echo "  Deleting origin/$branch..."
    git push origin --delete "$branch" 2>/dev/null || echo "    (already deleted or doesn't exist)"
done

echo ""
echo "✅ Cleanup complete!"
echo ""
echo "Remaining branches:"
git branch -a | grep -E "development|production|main"
