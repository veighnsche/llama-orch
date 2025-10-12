#!/usr/bin/env python3
"""
Analyze CSS tokens used across components to identify standardization opportunities.
"""

import os
import re
from collections import defaultdict, Counter
from pathlib import Path

# Patterns to extract
PATTERNS = {
    'spacing': r'\b(p-\d+|py-\d+|px-\d+|pt-\d+|pb-\d+|pl-\d+|pr-\d+|m-\d+|my-\d+|mx-\d+|mt-\d+|mb-\d+|ml-\d+|mr-\d+|space-[xy]-\d+|gap-\d+)\b',
    'sizing': r'\b(h-\d+|w-\d+|min-h-\d+|max-h-\d+|min-w-\d+|max-w-\d+)\b',
    'text': r'\b(text-(?:xs|sm|base|lg|xl|2xl|3xl|4xl|5xl|6xl|7xl))\b',
    'font_weight': r'\b(font-(?:thin|light|normal|medium|semibold|bold|extrabold|black))\b',
    'colors': r'\b(text-[a-z][\w-]*|bg-[a-z][\w-]*|border-[a-z][\w-]*)\b',
    'rounded': r'\b(rounded(?:-[a-z]+)?)\b',
    'shadow': r'\b(shadow(?:-[a-z0-9]+)?)\b',
    'opacity': r'\b(opacity-\d+)\b',
    'transitions': r'\b(transition(?:-[a-z]+)?|duration-\d+|ease-[a-z]+)\b',
    'hover': r'\bhover:([a-z][\w-]*)\b',
}

def extract_classnames(content):
    """Extract all className values from TSX content."""
    # Match className="..." or className={...}
    classname_pattern = r'className\s*=\s*(?:"([^"]*)"|{[^}]*cn\([^)]*\)})'
    matches = re.finditer(classname_pattern, content)
    
    classnames = []
    for match in matches:
        if match.group(1):
            classnames.append(match.group(1))
        else:
            # Extract from cn() calls
            cn_content = match.group(0)
            # Simple extraction of string literals within cn()
            string_literals = re.findall(r"'([^']*)'|\"([^']*)\"", cn_content)
            for lit in string_literals:
                classnames.append(lit[0] or lit[1])
    
    return ' '.join(classnames)

def analyze_directory(components_dir):
    """Analyze all TSX files in the components directory."""
    
    results = {
        'files_analyzed': 0,
        'total_lines': 0,
        'patterns': defaultdict(Counter),
        'file_patterns': defaultdict(lambda: defaultdict(set)),
    }
    
    for root, dirs, files in os.walk(components_dir):
        for file in files:
            if file.endswith('.tsx'):
                filepath = Path(root) / file
                relative_path = filepath.relative_to(components_dir)
                
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                        lines = content.count('\n')
                        
                        results['files_analyzed'] += 1
                        results['total_lines'] += lines
                        
                        # Extract all classnames
                        classnames = extract_classnames(content)
                        
                        # Apply patterns
                        for pattern_name, pattern_regex in PATTERNS.items():
                            matches = re.findall(pattern_regex, classnames)
                            for match in matches:
                                results['patterns'][pattern_name][match] += 1
                                results['file_patterns'][pattern_name][match].add(str(relative_path))
                
                except Exception as e:
                    print(f"Error processing {filepath}: {e}")
    
    return results

def generate_report(results, output_file):
    """Generate a comprehensive report."""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# CSS Token Standardization Analysis\n\n")
        f.write(f"**Files Analyzed:** {results['files_analyzed']}\n")
        f.write(f"**Total Lines:** {results['total_lines']:,}\n\n")
        
        f.write("---\n\n")
        
        # Analyze each pattern category
        for pattern_name in sorted(results['patterns'].keys()):
            counter = results['patterns'][pattern_name]
            file_patterns = results['file_patterns'][pattern_name]
            
            f.write(f"## {pattern_name.replace('_', ' ').title()}\n\n")
            f.write(f"**Unique tokens:** {len(counter)}\n")
            f.write(f"**Total usage:** {sum(counter.values())}\n\n")
            
            # Top tokens
            f.write("### Most Used Tokens\n\n")
            f.write("| Token | Count | Files |\n")
            f.write("|-------|-------|-------|\n")
            
            for token, count in counter.most_common(20):
                file_count = len(file_patterns[token])
                f.write(f"| `{token}` | {count} | {file_count} |\n")
            
            f.write("\n")
            
            # Identify standardization opportunities
            if pattern_name in ['spacing', 'sizing', 'text']:
                f.write("### Standardization Opportunities\n\n")
                
                # Group similar tokens
                token_groups = defaultdict(list)
                for token in counter.keys():
                    # Extract base pattern
                    base = re.sub(r'-\d+$', '-X', token)
                    token_groups[base].append(token)
                
                for base, tokens in sorted(token_groups.items()):
                    if len(tokens) > 1:
                        total = sum(counter[t] for t in tokens)
                        f.write(f"- **{base}**: {len(tokens)} variants, {total} uses\n")
                        for token in sorted(tokens):
                            f.write(f"  - `{token}`: {counter[token]} uses\n")
                
                f.write("\n")
            
            f.write("---\n\n")
        
        # Color analysis
        f.write("## Color Token Analysis\n\n")
        
        color_counter = results['patterns']['colors']
        
        # Group by prefix
        color_groups = defaultdict(Counter)
        for token, count in color_counter.items():
            if token.startswith('text-'):
                color_groups['text'][token] += count
            elif token.startswith('bg-'):
                color_groups['background'][token] += count
            elif token.startswith('border-'):
                color_groups['border'][token] += count
        
        for group_name, group_counter in sorted(color_groups.items()):
            f.write(f"### {group_name.title()} Colors\n\n")
            f.write("| Token | Count |\n")
            f.write("|-------|-------|\n")
            
            for token, count in group_counter.most_common(30):
                f.write(f"| `{token}` | {count} |\n")
            
            f.write("\n")

def generate_work_plan(results, output_file):
    """Generate a work plan for standardization."""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# CSS Token Standardization Work Plan\n\n")
        f.write("## Overview\n\n")
        f.write(f"- **Files to update:** {results['files_analyzed']}\n")
        f.write(f"- **Total lines of code:** {results['total_lines']:,}\n\n")
        
        f.write("## Work Breakdown\n\n")
        
        # Calculate work items
        work_items = []
        
        # Spacing tokens
        spacing_counter = results['patterns']['spacing']
        spacing_variants = len(spacing_counter)
        spacing_uses = sum(spacing_counter.values())
        work_items.append({
            'category': 'Spacing Tokens',
            'variants': spacing_variants,
            'uses': spacing_uses,
            'priority': 'HIGH',
            'effort': 'Large',
            'description': 'Standardize padding, margin, gap, and space utilities'
        })
        
        # Text sizing
        text_counter = results['patterns']['text']
        text_variants = len(text_counter)
        text_uses = sum(text_counter.values())
        work_items.append({
            'category': 'Text Sizing',
            'variants': text_variants,
            'uses': text_uses,
            'priority': 'HIGH',
            'effort': 'Medium',
            'description': 'Standardize font size scale'
        })
        
        # Colors
        color_counter = results['patterns']['colors']
        color_variants = len(color_counter)
        color_uses = sum(color_counter.values())
        work_items.append({
            'category': 'Color Tokens',
            'variants': color_variants,
            'uses': color_uses,
            'priority': 'CRITICAL',
            'effort': 'Very Large',
            'description': 'Standardize all color usage (text, background, border)'
        })
        
        # Rounded corners
        rounded_counter = results['patterns']['rounded']
        rounded_variants = len(rounded_counter)
        rounded_uses = sum(rounded_counter.values())
        work_items.append({
            'category': 'Border Radius',
            'variants': rounded_variants,
            'uses': rounded_uses,
            'priority': 'MEDIUM',
            'effort': 'Small',
            'description': 'Standardize border radius values'
        })
        
        # Shadows
        shadow_counter = results['patterns']['shadow']
        shadow_variants = len(shadow_counter)
        shadow_uses = sum(shadow_counter.values())
        work_items.append({
            'category': 'Shadows',
            'variants': shadow_variants,
            'uses': shadow_uses,
            'priority': 'MEDIUM',
            'effort': 'Small',
            'description': 'Standardize shadow utilities'
        })
        
        # Font weights
        font_counter = results['patterns']['font_weight']
        font_variants = len(font_counter)
        font_uses = sum(font_counter.values())
        work_items.append({
            'category': 'Font Weights',
            'variants': font_variants,
            'uses': font_uses,
            'priority': 'LOW',
            'effort': 'Small',
            'description': 'Standardize font weight scale'
        })
        
        # Sizing
        sizing_counter = results['patterns']['sizing']
        sizing_variants = len(sizing_counter)
        sizing_uses = sum(sizing_counter.values())
        work_items.append({
            'category': 'Sizing Tokens',
            'variants': sizing_variants,
            'uses': sizing_uses,
            'priority': 'MEDIUM',
            'effort': 'Large',
            'description': 'Standardize width and height utilities'
        })
        
        # Sort by priority
        priority_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
        work_items.sort(key=lambda x: priority_order[x['priority']])
        
        f.write("### Priority Work Items\n\n")
        f.write("| Priority | Category | Variants | Uses | Effort | Description |\n")
        f.write("|----------|----------|----------|------|--------|-------------|\n")
        
        for item in work_items:
            f.write(f"| **{item['priority']}** | {item['category']} | {item['variants']} | {item['uses']} | {item['effort']} | {item['description']} |\n")
        
        f.write("\n\n")
        
        # Detailed breakdown
        f.write("## Detailed Work Breakdown\n\n")
        
        for item in work_items:
            f.write(f"### {item['category']}\n\n")
            f.write(f"- **Priority:** {item['priority']}\n")
            f.write(f"- **Effort:** {item['effort']}\n")
            f.write(f"- **Unique variants:** {item['variants']}\n")
            f.write(f"- **Total uses:** {item['uses']}\n")
            f.write(f"- **Description:** {item['description']}\n\n")
            
            f.write("**Tasks:**\n\n")
            f.write(f"1. Audit all {item['variants']} variants\n")
            f.write(f"2. Define standard token set in `globals.css`\n")
            f.write(f"3. Update {results['files_analyzed']} component files\n")
            f.write(f"4. Test visual consistency\n")
            f.write(f"5. Document token usage guidelines\n\n")
            
            f.write("---\n\n")
        
        # Estimated effort
        f.write("## Estimated Effort\n\n")
        
        total_variants = sum(item['variants'] for item in work_items)
        total_uses = sum(item['uses'] for item in work_items)
        
        f.write(f"- **Total unique tokens to standardize:** {total_variants}\n")
        f.write(f"- **Total token uses to update:** {total_uses}\n")
        f.write(f"- **Files to modify:** {results['files_analyzed']}\n\n")
        
        f.write("### Time Estimates\n\n")
        f.write("- **Analysis & Planning:** 4-6 hours\n")
        f.write("- **Token Definition:** 8-12 hours\n")
        f.write("- **Component Updates:** 40-60 hours\n")
        f.write("- **Testing & QA:** 16-24 hours\n")
        f.write("- **Documentation:** 4-8 hours\n\n")
        f.write("**Total Estimated Time:** 72-110 hours (2-3 weeks)\n\n")

if __name__ == '__main__':
    components_dir = Path(__file__).parent / 'components'
    
    print(f"Analyzing components in: {components_dir}")
    print("This may take a minute...")
    
    results = analyze_directory(components_dir)
    
    print(f"\nAnalyzed {results['files_analyzed']} files ({results['total_lines']:,} lines)")
    
    # Generate reports
    report_file = Path(__file__).parent / 'CSS_TOKEN_ANALYSIS.md'
    generate_report(results, report_file)
    print(f"Generated analysis report: {report_file}")
    
    work_plan_file = Path(__file__).parent / 'CSS_STANDARDIZATION_WORK_PLAN.md'
    generate_work_plan(results, work_plan_file)
    print(f"Generated work plan: {work_plan_file}")
    
    print("\nDone!")
