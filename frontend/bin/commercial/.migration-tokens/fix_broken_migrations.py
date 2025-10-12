#!/usr/bin/env python3
"""Fix broken migrations from the automated script."""

import re
from pathlib import Path

files_to_fix = [
    "/home/vince/Projects/llama-orch/frontend/bin/commercial/components/features/error-handling.tsx",
    "/home/vince/Projects/llama-orch/frontend/bin/commercial/components/features/intelligent-model-management.tsx",
    "/home/vince/Projects/llama-orch/frontend/bin/commercial/components/features/multi-backend-gpu.tsx",
    "/home/vince/Projects/llama-orch/frontend/bin/commercial/components/features/real-time-progress.tsx",
    "/home/vince/Projects/llama-orch/frontend/bin/commercial/components/features/security-isolation.tsx",
    "/home/vince/Projects/llama-orch/frontend/bin/commercial/components/pricing/pricing-comparison.tsx",
    "/home/vince/Projects/llama-orch/frontend/bin/commercial/components/pricing/pricing-faq.tsx",
    "/home/vince/Projects/llama-orch/frontend/bin/commercial/components/use-cases/use-cases-industry.tsx",
    "/home/vince/Projects/llama-orch/frontend/bin/commercial/components/use-cases/use-cases-primary.tsx",
]

def fix_file(filepath: str):
    """Fix a single file."""
    path = Path(filepath)
    content = path.read_text()
    
    # Pattern 1: Remove orphaned </div> after SectionContainer opening and extract subtitle
    pattern1 = re.compile(
        r'(<SectionContainer\s+title="[^"]+"\s+bgVariant="[^"]+"\s*>\s*)'
        r'<p className="text-xl text-muted-foreground[^"]*">\s*([^<]+)\s*</p>\s*</div>\s*'
        r'(<div className="max-w-)',
        re.DOTALL
    )
    
    match = pattern1.search(content)
    if match:
        section_opening = match.group(1)
        subtitle_text = match.group(2).strip()
        div_opening = match.group(3)
        
        # Extract title and bgVariant from section_opening
        title_match = re.search(r'title="([^"]+)"', section_opening)
        bg_match = re.search(r'bgVariant="([^"]+)"', section_opening)
        
        if title_match and bg_match:
            title = title_match.group(1)
            bg_variant = bg_match.group(1)
            
            replacement = (
                f'<SectionContainer\n'
                f'      title="{title}"\n'
                f'      bgVariant="{bg_variant}"\n'
                f'      subtitle="{subtitle_text}"\n'
                f'    >\n'
                f'      <{div_opening}'
            )
            
            content = pattern1.sub(replacement, content, count=1)
    
    # Pattern 2: Remove orphaned </div> without subtitle
    pattern2 = re.compile(
        r'(<SectionContainer\s+title="[^"]+"\s+bgVariant="[^"]+"\s*>\s*)'
        r'</div>\s*'
        r'(<div className="max-w-)',
        re.DOTALL
    )
    
    match2 = pattern2.search(content)
    if match2:
        section_opening = match2.group(1)
        div_opening = match2.group(2)
        
        # Extract title and bgVariant
        title_match = re.search(r'title="([^"]+)"', section_opening)
        bg_match = re.search(r'bgVariant="([^"]+)"', section_opening)
        
        if title_match and bg_match:
            title = title_match.group(1)
            bg_variant = bg_match.group(1)
            
            replacement = (
                f'<SectionContainer\n'
                f'      title="{title}"\n'
                f'      bgVariant="{bg_variant}"\n'
                f'    >\n'
                f'      <{div_opening}'
            )
            
            content = pattern2.sub(replacement, content, count=1)
    
    # Pattern 3: Fix closing - add missing </div> before </SectionContainer>
    pattern3 = re.compile(
        r'(</div>\s*</div>\s*)</SectionContainer>',
        re.DOTALL
    )
    
    if not re.search(r'</div>\s*</div>\s*</div>\s*</SectionContainer>', content):
        # Only add if not already there
        content = pattern3.sub(r'\1</div>\n      </SectionContainer>', content)
    
    path.write_text(content)
    print(f"✅ Fixed: {path.name}")

for filepath in files_to_fix:
    try:
        fix_file(filepath)
    except Exception as e:
        print(f"❌ Error fixing {filepath}: {e}")

print("\n✨ All files fixed!")
