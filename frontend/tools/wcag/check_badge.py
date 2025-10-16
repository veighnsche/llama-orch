#!/usr/bin/env python3
"""
Check WCAG compliance for Badge component variants.

DEPRECATED: Use check_contrast.py instead:
    python check_contrast.py --input badge_colors.json

This script is kept for backward compatibility but should not be used as a template.
For new components, use the generic check_contrast.py tool.
"""

import sys
from wcag_utils import parse_color, get_contrast_ratio, check_wcag_compliance, format_hex

print("⚠️  DEPRECATED: Use 'python check_contrast.py --input badge_colors.json' instead", file=sys.stderr)
print()


def check_badge_variant(name: str, text_color: str, bg_color: str, mode: str = "light"):
    """Check WCAG compliance for a badge variant."""
    text = parse_color(text_color)
    bg = parse_color(bg_color)
    ratio = get_contrast_ratio(text, bg)
    compliance = check_wcag_compliance(ratio)
    
    return {
        "name": name,
        "mode": mode,
        "text": format_hex(text),
        "background": format_hex(bg),
        "ratio": ratio,
        "compliance": compliance,
    }


def main():
    """Check all Badge variants from theme-tokens.css."""
    print("=== Badge Component - WCAG Compliance Report ===\n")
    
    # Define badge variants from theme-tokens.css
    variants = {
        "default": {
            "light": ("#ffffff", "#b45309"),  # primary-foreground on primary
            "dark": ("#ffffff", "#b45309"),
        },
        "secondary": {
            "light": ("#0f172a", "#f1f5f9"),  # secondary-foreground on secondary
            "dark": ("#f1f5f9", "#0f172a"),
        },
        "destructive": {
            "light": ("#ffffff", "#dc2626"),  # destructive-foreground on destructive
            "dark": ("#ffffff", "#dc2626"),
        },
        "outline": {
            "light": ("#0f172a", "#ffffff"),  # foreground on background
            "dark": ("#f1f5f9", "#020617"),
        },
    }
    
    all_pass = True
    
    for variant_name, modes in variants.items():
        print(f"## {variant_name.upper()} Variant\n")
        
        for mode, (text_color, bg_color) in modes.items():
            result = check_badge_variant(variant_name, text_color, bg_color, mode)
            
            print(f"**{mode.upper()} MODE:**")
            print(f"  Text: {result['text']}")
            print(f"  Background: {result['background']}")
            print(f"  Contrast Ratio: {result['ratio']:.2f}:1")
            
            aa_normal = result['compliance']['AA_normal']
            aa_large = result['compliance']['AA_large']
            
            print(f"  AA Normal Text (4.5:1): {'✅ PASS' if aa_normal else '❌ FAIL'}")
            print(f"  AA Large Text (3:1): {'✅ PASS' if aa_large else '❌ FAIL'}")
            
            if not aa_normal:
                all_pass = False
                print(f"  ⚠️  WARNING: Fails WCAG AA for normal text")
            
            print()
        
        print()
    
    # Summary
    print("=" * 60)
    if all_pass:
        print("✅ ALL BADGE VARIANTS PASS WCAG AA")
    else:
        print("❌ SOME BADGE VARIANTS FAIL WCAG AA")
        print("\nRecommendations:")
        print("- Use only for large text (≥18pt or ≥14pt bold)")
        print("- Or adjust colors in theme-tokens.css")
    print("=" * 60)


if __name__ == "__main__":
    main()
