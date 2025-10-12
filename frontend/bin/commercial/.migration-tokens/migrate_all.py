#!/usr/bin/env python3
"""
Automated migration script to convert all remaining components to use primitives.
This script applies systematic transformations to migrate section wrappers to SectionContainer.
"""

import os
import re
from pathlib import Path

# Base directory
BASE_DIR = Path("/home/vince/Projects/llama-orch/frontend/bin/commercial/components")

# Directories to process
DIRS_TO_PROCESS = [
    "developers",
    "enterprise",
    "providers",
    "pricing",
    "features",
    "use-cases"
]

def add_import_if_missing(content: str, import_name: str) -> str:
    """Add import to primitives if not already present."""
    primitives_import_pattern = r'from\s+"@/components/primitives"'
    
    if re.search(primitives_import_pattern, content):
        # Import already exists, add to it if needed
        existing_import_match = re.search(
            r'import\s+{([^}]+)}\s+from\s+"@/components/primitives"',
            content
        )
        if existing_import_match:
            imports = existing_import_match.group(1)
            if import_name not in imports:
                new_imports = imports.strip() + f", {import_name}"
                content = content.replace(
                    f'import {{{imports}}} from "@/components/primitives"',
                    f'import {{{new_imports}}} from "@/components/primitives"'
                )
    else:
        # Add new import after other imports
        import_insert_pos = 0
        for match in re.finditer(r'^import\s+.*$', content, re.MULTILINE):
            import_insert_pos = match.end()
        
        if import_insert_pos > 0:
            content = (
                content[:import_insert_pos] +
                f'\nimport {{ {import_name} }} from "@/components/primitives"' +
                content[import_insert_pos:]
            )
    
    return content

def migrate_section_container(content: str) -> tuple[str, bool]:
    """
    Migrate section wrapper to SectionContainer primitive.
    Returns (new_content, was_modified).
    """
    modified = False
    
    # Pattern 1: section with title in h2
    pattern1 = re.compile(
        r'<section className="py-24 bg-(secondary|background|card)">\s*'
        r'<div className="container mx-auto px-4">\s*'
        r'<div className="max-w-4xl mx-auto text-center mb-16">\s*'
        r'<h2 className="text-4xl lg:text-5xl font-bold text-foreground mb-6 text-balance">\s*'
        r'([^<]+)\s*'
        r'</h2>',
        re.DOTALL
    )
    
    match = pattern1.search(content)
    if match:
        bg_variant = match.group(1)
        title = match.group(2).strip()
        
        # Add import
        content = add_import_if_missing(content, "SectionContainer")
        
        # Replace opening
        replacement = f'<SectionContainer\n      title="{title}"\n      bgVariant="{bg_variant}"\n    >'
        content = pattern1.sub(replacement, content, count=1)
        
        # Replace closing (find the matching closing tags)
        closing_pattern = r'</div>\s*</div>\s*</section>'
        content = re.sub(closing_pattern, '</SectionContainer>', content, count=1)
        
        modified = True
    
    return content, modified

def process_file(filepath: Path) -> bool:
    """Process a single file. Returns True if modified."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Apply migrations
        content, modified = migrate_section_container(content)
        
        if modified:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Migrated: {filepath.relative_to(BASE_DIR)}")
            return True
        else:
            print(f"⏭️  Skipped: {filepath.relative_to(BASE_DIR)} (no changes needed)")
            return False
            
    except Exception as e:
        print(f"❌ Error processing {filepath}: {e}")
        return False

def main():
    """Main migration function."""
    print("🚀 Starting automated migration to primitives...\n")
    
    total_files = 0
    modified_files = 0
    
    for dir_name in DIRS_TO_PROCESS:
        dir_path = BASE_DIR / dir_name
        if not dir_path.exists():
            print(f"⚠️  Directory not found: {dir_path}")
            continue
        
        print(f"\n📁 Processing directory: {dir_name}")
        print("=" * 60)
        
        tsx_files = list(dir_path.glob("*.tsx"))
        for filepath in tsx_files:
            total_files += 1
            if process_file(filepath):
                modified_files += 1
    
    print("\n" + "=" * 60)
    print(f"✨ Migration complete!")
    print(f"   Total files processed: {total_files}")
    print(f"   Files modified: {modified_files}")
    print(f"   Files unchanged: {total_files - modified_files}")

if __name__ == "__main__":
    main()
