#!/usr/bin/env bash
set -euo pipefail

# Archive MD files into .archive subfolders
# Edge cases:
# - Skip README.md, LICENSE.md, CONTRIBUTING.md, CODEOWNERS, and other root-level docs
# - Skip folders that are entirely MD files (already dedicated doc folders)
# - Only archive in folders with mixed file types

# Files to never archive (case-insensitive matching)
SKIP_FILES=(
    "README.md"
    "LICENSE.md"
    "CONTRIBUTING.md"
    "CODEOWNERS"
    "CHANGELOG.md"
    "CODE_OF_CONDUCT.md"
)

# Convert to lowercase for comparison
declare -A SKIP_MAP
for file in "${SKIP_FILES[@]}"; do
    SKIP_MAP["${file,,}"]=1
done

# Check if a file should be skipped
should_skip() {
    local filename="$1"
    local basename=$(basename "$filename")
    local lowercase="${basename,,}"
    
    [[ -n "${SKIP_MAP[$lowercase]:-}" ]]
}

# Check if a directory contains ONLY .md files (and subdirs)
is_pure_md_folder() {
    local dir="$1"
    local has_non_md=0
    local has_md=0
    
    # Check immediate children only (not recursive)
    while IFS= read -r -d '' file; do
        local basename=$(basename "$file")
        
        # Skip hidden files/dirs
        [[ "$basename" == .* ]] && continue
        
        if [[ -f "$file" ]]; then
            if [[ "$file" == *.md ]]; then
                has_md=1
            else
                has_non_md=1
                break
            fi
        fi
    done < <(find "$dir" -maxdepth 1 -print0 2>/dev/null)
    
    # Pure MD folder = has MD files AND no non-MD files
    [[ $has_md -eq 1 && $has_non_md -eq 0 ]]
}

# Process a directory
process_directory() {
    local dir="$1"
    local basename=$(basename "$dir")
    
    # Skip .archive directories themselves
    [[ "$basename" == ".archive" ]] && return
    
    # Skip hidden directories (but not . or ..)
    [[ "$basename" == .* ]] && [[ "$basename" != "." ]] && [[ "$basename" != ".." ]] && return
    
    # Skip if this is a pure MD folder
    if is_pure_md_folder "$dir"; then
        echo "⏭️  Skipping pure MD folder: $dir"
        return
    fi
    
    # Find MD files in this directory (not subdirs)
    local md_files=()
    while IFS= read -r -d '' file; do
        local basename=$(basename "$file")
        
        # Skip if it's in a skip list
        if should_skip "$file"; then
            echo "⏭️  Skipping protected file: $file"
            continue
        fi
        
        md_files+=("$file")
    done < <(find "$dir" -maxdepth 1 -type f -name "*.md" -print0 2>/dev/null)
    
    # If we found MD files to archive, create .archive and move them
    if [[ ${#md_files[@]} -gt 0 ]]; then
        local archive_dir="$dir/.archive"
        mkdir -p "$archive_dir"
        
        echo "📁 Processing: $dir"
        for md_file in "${md_files[@]}"; do
            local basename=$(basename "$md_file")
            echo "  📄 Moving: $basename → .archive/"
            mv "$md_file" "$archive_dir/"
        done
    fi
    
    # Recursively process subdirectories
    while IFS= read -r -d '' subdir; do
        process_directory "$subdir"
    done < <(find "$dir" -mindepth 1 -maxdepth 1 -type d -print0 2>/dev/null)
}

# Main execution
main() {
    local root_dir="${1:-.}"
    
    echo "🚀 Starting MD file archival from: $root_dir"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    process_directory "$root_dir"
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "✅ Archival complete!"
}

main "$@"
