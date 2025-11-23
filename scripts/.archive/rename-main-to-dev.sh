#!/bin/bash
# Rename main branch to development
# Created by: TEAM-451

set -e

echo "🔄 Renaming main → development..."
echo ""
echo "⚠️  This will:"
echo "   1. Rename local 'main' to 'development'"
echo "   2. Delete remote 'main' branch"
echo "   3. Make 'development' the default branch"
echo ""
read -p "Continue? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# Ensure we're on main
git checkout main
git pull origin main

# Merge main into development (in case there are differences)
git checkout development
git merge main --no-edit

# Push development
git push origin development

# Delete remote main
echo "🗑️  Deleting remote main branch..."
git push origin --delete main

# Delete local main
git branch -d main

echo ""
echo "✅ Done! Main branch renamed to development."
echo ""
echo "⚠️  Update GitHub default branch if not already set:"
echo "   gh repo edit rbee-keeper/rbee --default-branch development"
