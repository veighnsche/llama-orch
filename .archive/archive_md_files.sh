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
    local has_non_md_non_helper=0
    local has_md=0
    local has_code_subdirs=0
    
    # Helper files that don't count as "real" files (config/build files)
    local helper_patterns=(
        "Cargo.toml"
        "Cargo.lock"
        "package.json"
        "package-lock.json"
        "pnpm-lock.yaml"
        ".gitignore"
        ".editorconfig"
        "tsconfig.json"
        "*.toml"
        "*.lock"
        "*.log"
        "*.sh"
        "*.py"
        "*.js"
    )
    
    # Check immediate children only (not recursive)
    while IFS= read -r -d '' item; do
        local basename=$(basename "$item")
        
        # Skip hidden files/dirs
        [[ "$basename" == .* ]] && continue
        
        if [[ -f "$item" ]]; then
            if [[ "$item" == *.md ]]; then
                has_md=1
            else
                # Check if it's a helper file
                local is_helper=0
                for pattern in "${helper_patterns[@]}"; do
                    if [[ "$basename" == $pattern ]]; then
                        is_helper=1
                        break
                    fi
                done
                
                # If not a helper file, it's a real non-MD file
                if [[ $is_helper -eq 0 ]]; then
                    has_non_md_non_helper=1
                    break
                fi
            fi
        elif [[ -d "$item" ]]; then
            # Check if subdirectory contains code files
            # Common code directories: src, tests, lib, bin, etc.
            local subdir_basename=$(basename "$item")
            if [[ "$subdir_basename" =~ ^(src|tests|test|lib|bin|pkg|dist|build)$ ]]; then
                has_code_subdirs=1
            fi
        fi
    done < <(find "$dir" -maxdepth 1 -print0 2>/dev/null)
    
    # Pure MD folder = has MD files AND no non-helper files AND no code subdirs
    # If it has code subdirs, it's NOT a pure MD folder (needs archival)
    [[ $has_md -eq 1 && $has_non_md_non_helper -eq 0 && $has_code_subdirs -eq 0 ]]
}

# Process a directory
process_directory() {
    local dir="$1"
    local basename=$(basename "$dir")
    
    # Skip .archive directories themselves
    [[ "$basename" == ".archive" ]] && return
    
    # Skip hidden directories (but not . or ..)
    [[ "$basename" == .* ]] && [[ "$basename" != "." ]] && [[ "$basename" != ".." ]] && return
    
    # Check if this is a pure MD folder
    local is_pure_md=0
    if is_pure_md_folder "$dir"; then
        is_pure_md=1
        echo "⏭️  Skipping pure MD folder: $dir"
        # Don't return yet - we still need to recurse into subdirs
    fi
    
    # Only process MD files if this is NOT a pure MD folder
    if [[ $is_pure_md -eq 0 ]]; then
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
