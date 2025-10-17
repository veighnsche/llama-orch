#!/usr/bin/env python3
"""
Automatically add template stories for new pages.

This script reads page component files, identifies which templates they use,
and adds corresponding stories to each template's story file.
"""

import re
import os
from pathlib import Path
from typing import Dict, List, Tuple

# Base directory
BASE_DIR = Path(__file__).parent.parent
PAGES_DIR = BASE_DIR / "src" / "pages"
TEMPLATES_DIR = BASE_DIR / "src" / "templates"

# Pages that need stories added
NEW_PAGES = [
    "EducationPage",
    "CommunityPage",
    "CompliancePage",
    "DevOpsPage",
    "ResearchPage",
    "SecurityPage",
    "PrivacyPage",
]

# Template name mappings (component name -> story file name)
TEMPLATE_MAPPINGS = {
    "HeroTemplate": "HeroTemplate",
    "EmailCapture": "EmailCapture",
    "ProblemTemplate": "ProblemTemplate",
    "SolutionTemplate": "SolutionTemplate",
    "PricingTemplate": "PricingTemplate",
    "EnterpriseSecurity": "EnterpriseSecurity",
    "HowItWorks": "HowItWorks",
    "UseCasesTemplate": "UseCasesTemplate",
    "TestimonialsTemplate": "TestimonialsTemplate",
    "CardGridTemplate": "CardGridTemplate",
    "FAQTemplate": "FAQTemplate",
    "CTATemplate": "CTATemplate",
}


def extract_templates_from_page(page_name: str) -> List[Tuple[str, str, str]]:
    """
    Extract template usage from a page component.
    Returns list of (template_name, props_name, container_props_name)
    """
    page_file = PAGES_DIR / page_name / f"{page_name}.tsx"
    
    if not page_file.exists():
        print(f"⚠️  Page file not found: {page_file}")
        return []
    
    content = page_file.read_text()
    templates = []
    
    # Find all TemplateContainer + Template pairs
    # Pattern: <TemplateContainer {...someContainerProps}>\n        <SomeTemplate {...someProps} />
    pattern = r'<TemplateContainer\s+\{\.\.\.(\w+)\}>\s*<(\w+)\s+\{\.\.\.(\w+)\}'
    
    for match in re.finditer(pattern, content):
        container_props = match.group(1)
        template_name = match.group(2)
        props_name = match.group(3)
        
        if template_name in TEMPLATE_MAPPINGS:
            templates.append((template_name, props_name, container_props))
    
    return templates


def get_story_name(page_name: str, template_name: str) -> str:
    """Generate story name like OnEducationPage or OnEducationProblem"""
    # Remove 'Page' suffix from page name
    page_base = page_name.replace("Page", "")
    
    # Remove 'Template' suffix from template name
    template_base = template_name.replace("Template", "")
    
    # Special cases
    if template_name == "EmailCapture":
        return f"On{page_base}Page"
    elif template_name == "ProblemTemplate":
        return f"On{page_base}Problem"
    elif template_name == "SolutionTemplate":
        return f"On{page_base}Solution"
    elif template_name == "PricingTemplate":
        # Check if props name gives us a hint
        return f"On{page_base}CourseLevels" if "CourseLevels" in page_base else f"On{page_base}Pricing"
    elif template_name == "EnterpriseSecurity":
        return f"On{page_base}Curriculum" if page_base == "Education" else f"On{page_base}Security"
    elif template_name == "HowItWorks":
        return f"On{page_base}LabExercises" if page_base == "Education" else f"On{page_base}HowItWorks"
    elif template_name == "UseCasesTemplate":
        return f"On{page_base}StudentTypes" if page_base == "Education" else f"On{page_base}UseCases"
    elif template_name == "TestimonialsTemplate":
        return f"On{page_base}Testimonials"
    elif template_name == "CardGridTemplate":
        return f"On{page_base}ResourcesGrid" if page_base == "Education" else f"On{page_base}Grid"
    elif template_name == "FAQTemplate":
        return f"On{page_base}FAQ"
    elif template_name == "CTATemplate":
        return f"On{page_base}CTA"
    else:
        return f"On{page_base}{template_base}"


def get_story_description(page_name: str, template_name: str) -> str:
    """Generate story description"""
    page_base = page_name.replace("Page", "")
    
    descriptions = {
        "EducationPage": {
            "EmailCapture": "Educator resources focus\n * - Curriculum guides and teaching materials\n * - Free for educators messaging",
            "ProblemTemplate": "The learning gap in distributed systems\n * - Theoretical-only education challenges\n * - No real infrastructure access",
            "SolutionTemplate": "Real production infrastructure for learning\n * - Hands-on with production code\n * - BDD testing and real patterns",
            "PricingTemplate": "Structured curriculum levels\n * - Beginner, Intermediate, Advanced modules\n * - Progressive learning path",
            "EnterpriseSecurity": "Six core curriculum modules\n * - Foundations to Production\n * - Comprehensive coverage",
            "HowItWorks": "Step-by-step hands-on labs\n * - Deploy workers, orchestrate, monitor\n * - Write BDD tests",
            "UseCasesTemplate": "Student types and learning paths\n * - CS Student, Career Switcher, Researcher\n * - Different goals and outcomes",
            "TestimonialsTemplate": "Student success stories\n * - Real outcomes and job placements\n * - Portfolio projects",
            "CardGridTemplate": "Learning resources\n * - Documentation, examples, tutorials\n * - Community support",
            "FAQTemplate": "Common questions about learning\n * - Prerequisites, GPU requirements\n * - Completion time",
            "CTATemplate": "Build real skills CTA\n * - Join students learning distributed AI\n * - Free for education",
        }
    }
    
    return descriptions.get(page_name, {}).get(template_name, f"{page_base} page usage")


def add_import_to_stories(stories_file: Path, page_name: str, props_name: str, container_props: str) -> bool:
    """Add import statement to template stories file"""
    if not stories_file.exists():
        print(f"⚠️  Stories file not found: {stories_file}")
        return False
    
    content = stories_file.read_text()
    
    # Check if already imported
    if f"from '@rbee/ui/pages/{page_name}'" in content:
        print(f"  ✓ Import already exists for {page_name}")
        return True
    
    # Find the last import from pages
    import_pattern = r"(import .+ from '@rbee/ui/pages/\w+')\n"
    matches = list(re.finditer(import_pattern, content))
    
    if not matches:
        # No page imports yet, add after TemplateContainer import
        insert_pattern = r"(import { TemplateContainer } from '@rbee/ui/molecules')\n"
        match = re.search(insert_pattern, content)
        if match:
            insert_pos = match.end()
        else:
            print(f"  ⚠️  Could not find insertion point")
            return False
    else:
        # Insert after last page import
        insert_pos = matches[-1].end()
    
    new_import = f"import {{ {container_props}, {props_name} }} from '@rbee/ui/pages/{page_name}'\n"
    
    new_content = content[:insert_pos] + new_import + content[insert_pos:]
    stories_file.write_text(new_content)
    
    print(f"  ✓ Added import for {page_name}")
    return True


def add_story_to_template(stories_file: Path, page_name: str, template_name: str, 
                         props_name: str, container_props: str) -> bool:
    """Add story export to template stories file"""
    if not stories_file.exists():
        return False
    
    content = stories_file.read_text()
    story_name = get_story_name(page_name, template_name)
    
    # Check if story already exists
    if f"export const {story_name}" in content:
        print(f"  ✓ Story {story_name} already exists")
        return True
    
    description = get_story_description(page_name, template_name)
    page_base = page_name.replace("Page", "")
    
    story_code = f'''
/**
 * {template_name} as used on the {page_base} page
 * - {description}
 */
export const {story_name}: Story = {{
  render: (args) => (
    <TemplateContainer {{...{container_props}}}>
      <{template_name} {{...args}} />
    </TemplateContainer>
  ),
  args: {props_name},
}}
'''
    
    # Append to end of file
    new_content = content.rstrip() + "\n" + story_code
    stories_file.write_text(new_content)
    
    print(f"  ✓ Added story {story_name}")
    return True


def process_page(page_name: str):
    """Process a single page and add stories to all its templates"""
    print(f"\n{'='*60}")
    print(f"Processing {page_name}")
    print(f"{'='*60}")
    
    templates = extract_templates_from_page(page_name)
    
    if not templates:
        print(f"  No templates found in {page_name}")
        return
    
    print(f"  Found {len(templates)} templates:")
    for template_name, props_name, container_props in templates:
        print(f"    - {template_name} ({props_name}, {container_props})")
    
    for template_name, props_name, container_props in templates:
        print(f"\n  Processing {template_name}...")
        
        # Find template stories file
        template_dir = TEMPLATES_DIR / template_name
        stories_file = template_dir / f"{template_name}.stories.tsx"
        
        if not stories_file.exists():
            print(f"    ⚠️  No stories file found, skipping")
            continue
        
        # Add import
        add_import_to_stories(stories_file, page_name, props_name, container_props)
        
        # Add story
        add_story_to_template(stories_file, page_name, template_name, props_name, container_props)


def main():
    print("🚀 Template Stories Automation Tool")
    print("=" * 60)
    
    for page_name in NEW_PAGES:
        process_page(page_name)
    
    print(f"\n{'='*60}")
    print("✅ Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
