#!/usr/bin/env python3
"""
WCAG Component Compliance Checker

Scans all TSX/Vue components and checks color contrast compliance against design tokens.
Generates a comprehensive report of all color combinations used.

Usage:
    python check_components.py
    python check_components.py --verbose
    python check_components.py --output report.md
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

from wcag_utils import (
    parse_color,
    get_contrast_ratio,
    check_wcag_compliance,
    format_hex,
)


# Design token color mappings (from theme-tokens.css)
LIGHT_MODE_COLORS = {
    "background": "#ffffff",
    "foreground": "#0f172a",
    "card": "#ffffff",
    "card-foreground": "#0f172a",
    "primary": "#f59e0b",
    "primary-foreground": "#ffffff",
    "secondary": "#f1f5f9",
    "secondary-foreground": "#0f172a",
    "muted": "#f1f5f9",
    "muted-foreground": "#5a6b7f",
    "accent": "#f59e0b",
    "accent-foreground": "#0f172a",
    "destructive": "#ef4444",
    "destructive-foreground": "#ffffff",
    "border": "#e2e8f0",
    "chart-1": "#f59e0b",
    "chart-2": "#3b82f6",
    "chart-3": "#10b981",
    "chart-4": "#8b5cf6",
    "chart-5": "#ef4444",
}

DARK_MODE_COLORS = {
    "background": "#0f172a",
    "foreground": "#f1f5f9",
    "card": "#1e293b",
    "card-foreground": "#f1f5f9",
    "primary": "#f59e0b",
    "primary-foreground": "#0f172a",
    "secondary": "#1e293b",
    "secondary-foreground": "#f1f5f9",
    "muted": "#1e293b",
    "muted-foreground": "#94a3b8",
    "accent": "#f59e0b",
    "accent-foreground": "#0f172a",
    "destructive": "#ef4444",
    "destructive-foreground": "#0f172a",
    "border": "#334155",
    "chart-1": "#f59e0b",
    "chart-2": "#3b82f6",
    "chart-3": "#10b981",
    "chart-4": "#8b5cf6",
    "chart-5": "#ef4444",
}

# Common background contexts
BACKGROUND_CONTEXTS = {
    "default": ["background", "card", "secondary", "muted"],
    "primary": ["primary"],
    "destructive": ["destructive"],
}


class ColorUsage:
    """Tracks color usage in components."""
    
    def __init__(self, text_class: str, bg_class: str, file_path: str, line_num: int, context: str = ""):
        self.text_class = text_class
        self.bg_class = bg_class
        self.file_path = file_path
        self.line_num = line_num
        self.context = context


def extract_color_classes(content: str) -> List[Tuple[str, str]]:
    """Extract text-* and bg-* class combinations from content."""
    combinations = []
    
    # Find all className attributes
    classname_pattern = r'className\s*=\s*(?:"([^"]*)"|{[^}]*})'
    matches = re.finditer(classname_pattern, content, re.MULTILINE)
    
    for match in matches:
        if match.group(1):
            classes = match.group(1).split()
        else:
            # Extract from cn() or template literals
            cn_content = match.group(0)
            string_literals = re.findall(r"['\"]([^'\"]*)['\"]", cn_content)
            classes = " ".join(string_literals).split()
        
        # Find text and bg classes
        text_classes = [c for c in classes if c.startswith("text-")]
        bg_classes = [c for c in classes if c.startswith("bg-")]
        
        # Create combinations
        for text_class in text_classes:
            for bg_class in bg_classes:
                combinations.append((text_class, bg_class))
    
    return combinations


def infer_background_context(content: str, line_num: int) -> str:
    """Infer the background context for a given line."""
    lines = content.split("\n")
    
    # Look backwards for bg- classes
    for i in range(max(0, line_num - 10), line_num):
        if i < len(lines):
            if "bg-secondary" in lines[i] or "bgVariant=\"secondary\"" in lines[i]:
                return "secondary"
            if "bg-card" in lines[i]:
                return "card"
            if "bg-muted" in lines[i]:
                return "muted"
            if "bg-primary" in lines[i]:
                return "primary"
    
    return "background"


def scan_component_file(file_path: Path, verbose: bool = False) -> List[ColorUsage]:
    """Scan a single component file for color usage."""
    usages = []
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            lines = content.split("\n")
            
            # Extract explicit combinations
            combinations = extract_color_classes(content)
            for text_class, bg_class in combinations:
                usages.append(ColorUsage(
                    text_class=text_class,
                    bg_class=bg_class,
                    file_path=str(file_path),
                    line_num=0,
                    context="explicit"
                ))
            
            # Find text-* classes without explicit bg
            for i, line in enumerate(lines, 1):
                text_matches = re.findall(r'\btext-([a-z][\w-]*)', line)
                for text_match in text_matches:
                    text_class = f"text-{text_match}"
                    bg_context = infer_background_context(content, i)
                    usages.append(ColorUsage(
                        text_class=text_class,
                        bg_class=f"bg-{bg_context}",
                        file_path=str(file_path),
                        line_num=i,
                        context="inferred"
                    ))
    
    except Exception as e:
        if verbose:
            print(f"Error scanning {file_path}: {e}", file=sys.stderr)
    
    return usages


def scan_components_directory(components_dir: Path, verbose: bool = False) -> List[ColorUsage]:
    """Scan all component files in directory."""
    all_usages = []
    
    for file_path in components_dir.rglob("*.tsx"):
        if verbose:
            print(f"Scanning: {file_path.relative_to(components_dir)}")
        usages = scan_component_file(file_path, verbose)
        all_usages.extend(usages)
    
    for file_path in components_dir.rglob("*.vue"):
        if verbose:
            print(f"Scanning: {file_path.relative_to(components_dir)}")
        usages = scan_component_file(file_path, verbose)
        all_usages.extend(usages)
    
    return all_usages


def resolve_color_token(token_class: str, mode: str = "light") -> str:
    """Resolve a Tailwind class to a hex color."""
    colors = LIGHT_MODE_COLORS if mode == "light" else DARK_MODE_COLORS
    
    # Remove prefix (text-, bg-, border-)
    for prefix in ["text-", "bg-", "border-"]:
        if token_class.startswith(prefix):
            token = token_class[len(prefix):]
            if token in colors:
                return colors[token]
    
    return None


def check_color_combination(text_class: str, bg_class: str, mode: str = "light") -> dict:
    """Check WCAG compliance for a color combination."""
    text_color = resolve_color_token(text_class, mode)
    bg_color = resolve_color_token(bg_class, mode)
    
    if not text_color or not bg_color:
        return None
    
    try:
        text_rgb = parse_color(text_color)
        bg_rgb = parse_color(bg_color)
        ratio = get_contrast_ratio(text_rgb, bg_rgb)
        compliance = check_wcag_compliance(ratio)
        
        return {
            "text_class": text_class,
            "bg_class": bg_class,
            "text_color": text_color,
            "bg_color": bg_color,
            "ratio": ratio,
            "compliance": compliance,
        }
    except Exception:
        return None


def generate_report(usages: List[ColorUsage], output_file: Path, verbose: bool = False):
    """Generate WCAG compliance report."""
    
    # Group by color combination
    combinations = defaultdict(list)
    for usage in usages:
        key = (usage.text_class, usage.bg_class)
        combinations[key].append(usage)
    
    # Check each combination
    results = {
        "pass": [],
        "fail": [],
        "unknown": [],
    }
    
    for (text_class, bg_class), usage_list in combinations.items():
        light_check = check_color_combination(text_class, bg_class, "light")
        dark_check = check_color_combination(text_class, bg_class, "dark")
        
        if light_check:
            result = {
                "text_class": text_class,
                "bg_class": bg_class,
                "light": light_check,
                "dark": dark_check,
                "usages": usage_list,
            }
            
            # Check if passes AA normal text in both modes
            if (light_check["compliance"]["AA_normal"] and 
                (not dark_check or dark_check["compliance"]["AA_normal"])):
                results["pass"].append(result)
            else:
                results["fail"].append(result)
        else:
            results["unknown"].append({
                "text_class": text_class,
                "bg_class": bg_class,
                "usages": usage_list,
            })
    
    # Write report
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# WCAG 2.1 Component Compliance Report\n\n")
        f.write(f"**Generated:** {Path.cwd()}\n\n")
        f.write("---\n\n")
        
        # Summary
        f.write("## Summary\n\n")
        f.write(f"- **Total Combinations:** {len(combinations)}\n")
        f.write(f"- **✅ Pass (AA Normal):** {len(results['pass'])}\n")
        f.write(f"- **❌ Fail (AA Normal):** {len(results['fail'])}\n")
        f.write(f"- **⚠️ Unknown:** {len(results['unknown'])}\n\n")
        f.write("---\n\n")
        
        # Failures
        if results["fail"]:
            f.write("## ❌ WCAG Failures\n\n")
            f.write("These color combinations **fail** WCAG AA standards for normal text.\n\n")
            
            for result in sorted(results["fail"], key=lambda x: x["light"]["ratio"]):
                f.write(f"### `{result['text_class']}` on `{result['bg_class']}`\n\n")
                
                # Light mode
                light = result["light"]
                f.write(f"**Light Mode:**\n")
                f.write(f"- Colors: `{light['text_color']}` on `{light['bg_color']}`\n")
                f.write(f"- Contrast Ratio: **{light['ratio']:.2f}:1**\n")
                f.write(f"- AA Normal: {'✅ PASS' if light['compliance']['AA_normal'] else '❌ FAIL'}\n")
                f.write(f"- AA Large: {'✅ PASS' if light['compliance']['AA_large'] else '❌ FAIL'}\n\n")
                
                # Dark mode
                if result["dark"]:
                    dark = result["dark"]
                    f.write(f"**Dark Mode:**\n")
                    f.write(f"- Colors: `{dark['text_color']}` on `{dark['bg_color']}`\n")
                    f.write(f"- Contrast Ratio: **{dark['ratio']:.2f}:1**\n")
                    f.write(f"- AA Normal: {'✅ PASS' if dark['compliance']['AA_normal'] else '❌ FAIL'}\n")
                    f.write(f"- AA Large: {'✅ PASS' if dark['compliance']['AA_large'] else '❌ FAIL'}\n\n")
                
                # Usages
                f.write(f"**Used in {len(result['usages'])} locations:**\n\n")
                for usage in result["usages"][:5]:  # Show first 5
                    rel_path = Path(usage.file_path).name
                    if usage.line_num > 0:
                        f.write(f"- `{rel_path}:{usage.line_num}` ({usage.context})\n")
                    else:
                        f.write(f"- `{rel_path}` ({usage.context})\n")
                
                if len(result["usages"]) > 5:
                    f.write(f"- ... and {len(result['usages']) - 5} more\n")
                
                f.write("\n---\n\n")
        
        # Passes
        if results["pass"]:
            f.write("## ✅ WCAG Compliant\n\n")
            f.write("These color combinations **pass** WCAG AA standards.\n\n")
            
            f.write("| Text Class | Background Class | Light Ratio | Dark Ratio | Status |\n")
            f.write("|------------|------------------|-------------|------------|--------|\n")
            
            for result in sorted(results["pass"], key=lambda x: -x["light"]["ratio"]):
                light_ratio = f"{result['light']['ratio']:.2f}:1"
                dark_ratio = f"{result['dark']['ratio']:.2f}:1" if result['dark'] else "N/A"
                
                # AAA badge
                status = "AAA" if result['light']['compliance']['AAA_normal'] else "AA"
                
                f.write(f"| `{result['text_class']}` | `{result['bg_class']}` | {light_ratio} | {dark_ratio} | {status} |\n")
            
            f.write("\n---\n\n")
        
        # Unknown
        if results["unknown"]:
            f.write("## ⚠️ Unknown Combinations\n\n")
            f.write("These combinations could not be resolved to design tokens.\n\n")
            
            for result in results["unknown"]:
                f.write(f"- `{result['text_class']}` on `{result['bg_class']}` ({len(result['usages'])} usages)\n")
            
            f.write("\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Check WCAG compliance for all components")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--output", "-o", default="WCAG_COMPLIANCE_REPORT.md", help="Output file")
    parser.add_argument("--components-dir", "-d", help="Components directory to scan")
    
    args = parser.parse_args()
    
    # Determine components directory
    if args.components_dir:
        components_dir = Path(args.components_dir)
    else:
        # Default to packages/rbee-ui/src
        script_dir = Path(__file__).parent
        components_dir = script_dir.parent.parent / "packages" / "rbee-ui" / "src"
    
    if not components_dir.exists():
        print(f"❌ Error: Components directory not found: {components_dir}", file=sys.stderr)
        sys.exit(1)
    
    print(f"🔍 Scanning components in: {components_dir}")
    
    # Scan components
    usages = scan_components_directory(components_dir, args.verbose)
    
    print(f"📊 Found {len(usages)} color usages")
    
    # Generate report
    output_file = Path(args.output)
    generate_report(usages, output_file, args.verbose)
    
    print(f"✅ Report generated: {output_file}")


if __name__ == "__main__":
    main()
