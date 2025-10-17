#!/usr/bin/env python3
"""
Fix duplicate import names in template story files by adding aliases.
"""

import re
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
TEMPLATES_DIR = BASE_DIR / "src" / "templates"

# Map of conflicting imports and which page should be aliased
CONFLICTS = {
    "EmailCapture.stories.tsx": {
        "ResearchPage": ["emailCaptureContainerProps", "emailCaptureProps"],
    },
    "PricingTemplate.stories.tsx": {
        "HomePage": ["pricingTemplateContainerProps", "pricingTemplateProps"],
    },
    "HowItWorks.stories.tsx": {
        "HomePage": ["howItWorksContainerProps", "howItWorksProps"],
    },
    "SolutionTemplate.stories.tsx": {
        "HomePage": ["solutionTemplateContainerProps", "solutionTemplateProps"],
    },
    "ProblemTemplate.stories.tsx": {
        "HomePage": ["problemTemplateProps"],
    },
}


def fix_file(stories_file: Path, page_to_alias: str, props_to_alias: list):
    """Fix duplicate imports in a single file"""
    content = stories_file.read_text()
    
    # Find the import line for the page
    pattern = rf"import\s+\{{([^}}]+)\}}\s+from\s+['\"]@rbee/ui/pages/{page_to_alias}['\"]"
    match = re.search(pattern, content)
    
    if not match:
        print(f"  ⚠️  Could not find import from {page_to_alias}")
        return False
    
    import_content = match.group(1)
    import_names = [n.strip() for n in import_content.split(',')]
    
    # Build new import with aliases
    new_imports = []
    aliases = {}
    
    for name in import_names:
        name = name.strip()
        if name in props_to_alias:
            # Create alias: emailCaptureProps -> homeEmailCaptureProps
            page_prefix = page_to_alias.replace("Page", "").lower()
            alias = f"{page_prefix}{name[0].upper()}{name[1:]}"
            new_imports.append(f"{name} as {alias}")
            aliases[name] = alias
        else:
            new_imports.append(name)
    
    new_import_line = f"import {{ {', '.join(new_imports)} }} from '@rbee/ui/pages/{page_to_alias}'"
    
    # Replace the import line
    content = content.replace(match.group(0), new_import_line)
    
    # Replace all usages of the old names with aliases
    for old_name, new_name in aliases.items():
        # Replace in story code (but not in import lines)
        # Look for {...oldName} patterns
        content = re.sub(
            rf'\{{\.\.\.{old_name}\}}',
            f'{{...{new_name}}}',
            content
        )
        # Look for args: oldName patterns
        content = re.sub(
            rf'\bargs:\s*{old_name}\b',
            f'args: {new_name}',
            content
        )
    
    stories_file.write_text(content)
    print(f"  ✓ Fixed {page_to_alias} imports with aliases: {aliases}")
    return True


def main():
    print("🔧 Fixing Duplicate Imports in Template Stories")
    print("=" * 60)
    
    for filename, conflicts in CONFLICTS.items():
        print(f"\n{filename}")
        stories_file = None
        
        # Find the file
        for f in TEMPLATES_DIR.glob(f"*/{filename}"):
            stories_file = f
            break
        
        if not stories_file:
            print(f"  ⚠️  File not found")
            continue
        
        for page, props in conflicts.items():
            fix_file(stories_file, page, props)
    
    print(f"\n{'='*60}")
    print("✅ Done!")


if __name__ == "__main__":
    main()
