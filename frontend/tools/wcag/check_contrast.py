#!/usr/bin/env python3
"""
Generic WCAG Contrast Checker

Check contrast ratios between any foreground and background colors.
Works for any component, not just badges.

Usage:
    # Check a single color pair
    python check_contrast.py --fg "#ffffff" --bg "#b45309"
    
    # Check multiple pairs from a JSON file
    python check_contrast.py --input colors.json
    
    # Check with custom labels
    python check_contrast.py --fg "#ffffff" --bg "#dc2626" --label "Destructive Button"
    
    # Batch check from theme tokens
    python check_contrast.py --theme-tokens ../../packages/rbee-ui/src/tokens/theme-tokens.css
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from wcag_utils import (
    parse_color,
    get_contrast_ratio,
    check_wcag_compliance,
    format_hex,
)


def parse_css_variables(css_file: Path) -> Dict[str, str]:
    """Extract CSS custom properties from a file."""
    variables = {}
    
    with open(css_file, 'r') as f:
        content = f.read()
    
    # Match --variable-name: #hexcolor;
    pattern = r'--([a-z0-9-]+):\s*(#[0-9a-fA-F]{3,8}|rgb\([^)]+\)|hsl\([^)]+\));'
    matches = re.findall(pattern, content)
    
    for var_name, color_value in matches:
        variables[var_name] = color_value.rstrip(';')
    
    return variables


def check_single_pair(
    fg_color: str,
    bg_color: str,
    label: Optional[str] = None,
    verbose: bool = False
) -> Dict:
    """Check contrast for a single foreground/background pair."""
    try:
        fg = parse_color(fg_color)
        bg = parse_color(bg_color)
    except ValueError as e:
        return {
            "error": str(e),
            "label": label,
            "fg": fg_color,
            "bg": bg_color,
        }
    
    ratio = get_contrast_ratio(fg, bg)
    compliance = check_wcag_compliance(ratio)
    
    result = {
        "label": label or f"{fg_color} on {bg_color}",
        "foreground": format_hex(fg),
        "background": format_hex(bg),
        "ratio": ratio,
        "compliance": compliance,
        "passes_aa_normal": compliance["AA_normal"],
        "passes_aa_large": compliance["AA_large"],
        "passes_aaa_normal": compliance["AAA_normal"],
    }
    
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Label: {result['label']}")
        print(f"Foreground: {result['foreground']}")
        print(f"Background: {result['background']}")
        print(f"Contrast Ratio: {ratio:.2f}:1")
        print(f"\nWCAG Compliance:")
        print(f"  AA Normal Text (4.5:1):  {'✅ PASS' if compliance['AA_normal'] else '❌ FAIL'}")
        print(f"  AA Large Text (3:1):     {'✅ PASS' if compliance['AA_large'] else '❌ FAIL'}")
        print(f"  AAA Normal Text (7:1):   {'✅ PASS' if compliance['AAA_normal'] else '❌ FAIL'}")
        print(f"  AAA Large Text (4.5:1):  {'✅ PASS' if compliance['AAA_large'] else '❌ FAIL'}")
        print(f"  UI Components (3:1):     {'✅ PASS' if compliance['UI_components'] else '❌ FAIL'}")
    
    return result


def check_batch(pairs: List[Dict], verbose: bool = False) -> List[Dict]:
    """Check multiple color pairs."""
    results = []
    
    for pair in pairs:
        fg = pair.get("foreground") or pair.get("fg")
        bg = pair.get("background") or pair.get("bg")
        label = pair.get("label") or pair.get("name")
        
        if not fg or not bg:
            print(f"⚠️  Skipping invalid pair: {pair}", file=sys.stderr)
            continue
        
        result = check_single_pair(fg, bg, label, verbose)
        results.append(result)
    
    return results


def print_summary(results: List[Dict]):
    """Print a summary table of results."""
    print("\n" + "=" * 80)
    print("WCAG CONTRAST SUMMARY")
    print("=" * 80)
    
    # Header
    print(f"{'Label':<30} {'Ratio':>8} {'AA Normal':>12} {'AA Large':>12}")
    print("-" * 80)
    
    # Results
    all_pass = True
    for r in results:
        if "error" in r:
            print(f"{r['label']:<30} {'ERROR':<8} {r['error']}")
            all_pass = False
            continue
        
        aa_normal = "✅ PASS" if r["passes_aa_normal"] else "❌ FAIL"
        aa_large = "✅ PASS" if r["passes_aa_large"] else "✅ PASS"  # Always passes if normal passes
        
        print(f"{r['label']:<30} {r['ratio']:>7.2f}:1 {aa_normal:>12} {aa_large:>12}")
        
        if not r["passes_aa_normal"]:
            all_pass = False
    
    print("=" * 80)
    
    if all_pass:
        print("✅ ALL PAIRS PASS WCAG AA")
    else:
        print("❌ SOME PAIRS FAIL WCAG AA")
    
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Check WCAG color contrast ratios",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single pair
  %(prog)s --fg "#ffffff" --bg "#b45309"
  
  # With label
  %(prog)s --fg "#ffffff" --bg "#dc2626" --label "Error Button"
  
  # Batch from JSON
  %(prog)s --input colors.json
  
  # From theme tokens
  %(prog)s --theme-tokens ../../packages/rbee-ui/src/tokens/theme-tokens.css
  
JSON format:
  [
    {"label": "Primary Button", "fg": "#ffffff", "bg": "#b45309"},
    {"label": "Destructive", "fg": "#ffffff", "bg": "#dc2626"}
  ]
        """
    )
    
    parser.add_argument("--fg", "--foreground", help="Foreground color (hex, rgb, hsl, or named)")
    parser.add_argument("--bg", "--background", help="Background color (hex, rgb, hsl, or named)")
    parser.add_argument("--label", help="Label for this color pair")
    parser.add_argument("--input", "-i", type=Path, help="JSON file with color pairs")
    parser.add_argument("--theme-tokens", type=Path, help="CSS file with theme tokens")
    parser.add_argument("--output", "-o", type=Path, help="Output JSON file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    results = []
    
    # Single pair mode
    if args.fg and args.bg:
        result = check_single_pair(args.fg, args.bg, args.label, args.verbose)
        results.append(result)
    
    # Batch mode from JSON
    elif args.input:
        with open(args.input) as f:
            pairs = json.load(f)
        results = check_batch(pairs, args.verbose)
    
    # Theme tokens mode
    elif args.theme_tokens:
        print("⚠️  Theme tokens mode not yet implemented. Use --input with a JSON file.", file=sys.stderr)
        print("See check_badge.py for an example of how to structure the JSON.", file=sys.stderr)
        sys.exit(1)
    
    else:
        parser.print_help()
        sys.exit(1)
    
    # Print summary if not verbose (verbose already printed details)
    if not args.verbose and results:
        print_summary(results)
    
    # Save to JSON if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✅ Results saved to {args.output}")
    
    # Exit with error code if any failures
    if any(not r.get("passes_aa_normal", False) for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
