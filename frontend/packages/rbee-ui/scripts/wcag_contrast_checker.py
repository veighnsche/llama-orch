#!/usr/bin/env python3
"""
WCAG 2.1 Color Contrast Checker

Checks if two colors meet WCAG 2.1 contrast requirements.
Supports multiple input formats: hex, rgb, hsl, named colors.

Usage:
    python wcag_contrast_checker.py "#f59e0b" "#0f172a"
    python wcag_contrast_checker.py "rgb(245, 158, 11)" "rgb(15, 23, 42)"
    python wcag_contrast_checker.py "hsl(45, 96%, 53%)" "hsl(222, 47%, 11%)"
    python wcag_contrast_checker.py "orange" "navy"

WCAG 2.1 Requirements:
- AA Normal Text: 4.5:1
- AA Large Text: 3:1
- AAA Normal Text: 7:1
- AAA Large Text: 4.5:1
- UI Components: 3:1
"""

import re
import sys
from typing import Tuple


# Named CSS colors (subset - add more as needed)
NAMED_COLORS = {
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "red": (255, 0, 0),
    "green": (0, 128, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
    "orange": (255, 165, 0),
    "purple": (128, 0, 128),
    "pink": (255, 192, 203),
    "brown": (165, 42, 42),
    "gray": (128, 128, 128),
    "grey": (128, 128, 128),
    "navy": (0, 0, 128),
    "teal": (0, 128, 128),
    "olive": (128, 128, 0),
    "maroon": (128, 0, 0),
    "lime": (0, 255, 0),
    "aqua": (0, 255, 255),
    "fuchsia": (255, 0, 255),
    "silver": (192, 192, 192),
}


def parse_color(color_str: str) -> Tuple[int, int, int]:
    """
    Parse a color string and return RGB values (0-255).
    
    Supports:
    - Hex: #RGB, #RRGGBB, #RRGGBBAA
    - RGB: rgb(r, g, b), rgba(r, g, b, a)
    - HSL: hsl(h, s%, l%), hsla(h, s%, l%, a)
    - Named: white, black, red, etc.
    """
    color_str = color_str.strip().lower()
    
    # Named color
    if color_str in NAMED_COLORS:
        return NAMED_COLORS[color_str]
    
    # Hex color
    if color_str.startswith("#"):
        hex_str = color_str[1:]
        
        # #RGB -> #RRGGBB
        if len(hex_str) == 3:
            hex_str = "".join([c * 2 for c in hex_str])
        
        # #RRGGBB or #RRGGBBAA
        if len(hex_str) in (6, 8):
            r = int(hex_str[0:2], 16)
            g = int(hex_str[2:4], 16)
            b = int(hex_str[4:6], 16)
            return (r, g, b)
    
    # RGB/RGBA color
    rgb_match = re.match(r"rgba?\((\d+),\s*(\d+),\s*(\d+)", color_str)
    if rgb_match:
        r, g, b = map(int, rgb_match.groups())
        return (r, g, b)
    
    # HSL/HSLA color
    hsl_match = re.match(r"hsla?\((\d+),\s*(\d+)%,\s*(\d+)%", color_str)
    if hsl_match:
        h, s, l = map(int, hsl_match.groups())
        return hsl_to_rgb(h, s / 100, l / 100)
    
    raise ValueError(f"Invalid color format: {color_str}")


def hsl_to_rgb(h: int, s: float, l: float) -> Tuple[int, int, int]:
    """Convert HSL to RGB (0-255)."""
    h = h / 360
    
    if s == 0:
        r = g = b = l
    else:
        def hue_to_rgb(p: float, q: float, t: float) -> float:
            if t < 0:
                t += 1
            if t > 1:
                t -= 1
            if t < 1/6:
                return p + (q - p) * 6 * t
            if t < 1/2:
                return q
            if t < 2/3:
                return p + (q - p) * (2/3 - t) * 6
            return p
        
        q = l * (1 + s) if l < 0.5 else l + s - l * s
        p = 2 * l - q
        r = hue_to_rgb(p, q, h + 1/3)
        g = hue_to_rgb(p, q, h)
        b = hue_to_rgb(p, q, h - 1/3)
    
    return (int(r * 255), int(g * 255), int(b * 255))


def get_relative_luminance(rgb: Tuple[int, int, int]) -> float:
    """
    Calculate relative luminance according to WCAG 2.1.
    https://www.w3.org/TR/WCAG21/#dfn-relative-luminance
    """
    r, g, b = rgb
    
    # Convert to 0-1 range
    r = r / 255
    g = g / 255
    b = b / 255
    
    # Apply gamma correction
    def gamma_correct(c: float) -> float:
        if c <= 0.03928:
            return c / 12.92
        else:
            return ((c + 0.055) / 1.055) ** 2.4
    
    r = gamma_correct(r)
    g = gamma_correct(g)
    b = gamma_correct(b)
    
    # Calculate luminance
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def get_contrast_ratio(color1: Tuple[int, int, int], color2: Tuple[int, int, int]) -> float:
    """
    Calculate contrast ratio between two colors.
    https://www.w3.org/TR/WCAG21/#dfn-contrast-ratio
    """
    l1 = get_relative_luminance(color1)
    l2 = get_relative_luminance(color2)
    
    # Ensure l1 is the lighter color
    if l1 < l2:
        l1, l2 = l2, l1
    
    # Contrast ratio formula
    return (l1 + 0.05) / (l2 + 0.05)


def check_wcag_compliance(ratio: float) -> dict:
    """Check WCAG 2.1 compliance levels."""
    return {
        "AA_normal": ratio >= 4.5,      # Normal text (< 18pt or < 14pt bold)
        "AA_large": ratio >= 3.0,       # Large text (≥ 18pt or ≥ 14pt bold)
        "AAA_normal": ratio >= 7.0,     # Normal text (enhanced)
        "AAA_large": ratio >= 4.5,      # Large text (enhanced)
        "UI_components": ratio >= 3.0,  # UI components and graphical objects
    }


def format_rgb(rgb: Tuple[int, int, int]) -> str:
    """Format RGB tuple as string."""
    return f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})"


def format_hex(rgb: Tuple[int, int, int]) -> str:
    """Format RGB tuple as hex string."""
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def print_results(color1_str: str, color2_str: str, rgb1: Tuple[int, int, int], 
                  rgb2: Tuple[int, int, int], ratio: float, compliance: dict):
    """Print formatted results."""
    print("\n" + "=" * 70)
    print("WCAG 2.1 COLOR CONTRAST CHECKER")
    print("=" * 70)
    
    print(f"\n📊 INPUT COLORS:")
    print(f"  Color 1: {color1_str}")
    print(f"    → {format_rgb(rgb1)} / {format_hex(rgb1)}")
    print(f"  Color 2: {color2_str}")
    print(f"    → {format_rgb(rgb2)} / {format_hex(rgb2)}")
    
    print(f"\n📈 CONTRAST RATIO: {ratio:.2f}:1")
    
    print(f"\n✅ WCAG 2.1 COMPLIANCE:")
    
    # AA Level
    print(f"\n  AA Level (Minimum):")
    print(f"    Normal Text (< 18pt):  {'✅ PASS' if compliance['AA_normal'] else '❌ FAIL'} (requires 4.5:1)")
    print(f"    Large Text (≥ 18pt):   {'✅ PASS' if compliance['AA_large'] else '❌ FAIL'} (requires 3:1)")
    print(f"    UI Components:         {'✅ PASS' if compliance['UI_components'] else '❌ FAIL'} (requires 3:1)")
    
    # AAA Level
    print(f"\n  AAA Level (Enhanced):")
    print(f"    Normal Text (< 18pt):  {'✅ PASS' if compliance['AAA_normal'] else '❌ FAIL'} (requires 7:1)")
    print(f"    Large Text (≥ 18pt):   {'✅ PASS' if compliance['AAA_large'] else '❌ FAIL'} (requires 4.5:1)")
    
    # Overall recommendation
    print(f"\n💡 RECOMMENDATION:")
    if compliance['AAA_normal']:
        print("  🌟 Excellent! Meets AAA standard for all text sizes.")
    elif compliance['AA_normal']:
        print("  ✅ Good! Meets AA standard for all text sizes.")
    elif compliance['AA_large']:
        print("  ⚠️  Acceptable for large text only (≥ 18pt or ≥ 14pt bold).")
    else:
        print("  ❌ Poor contrast. Does not meet WCAG 2.1 standards.")
        print("     Consider using a darker/lighter color for better accessibility.")
    
    print("\n" + "=" * 70 + "\n")


def main():
    """Main entry point."""
    if len(sys.argv) != 3:
        print("Usage: python wcag_contrast_checker.py <color1> <color2>")
        print("\nExamples:")
        print('  python wcag_contrast_checker.py "#f59e0b" "#0f172a"')
        print('  python wcag_contrast_checker.py "rgb(245, 158, 11)" "rgb(15, 23, 42)"')
        print('  python wcag_contrast_checker.py "hsl(45, 96%, 53%)" "hsl(222, 47%, 11%)"')
        print('  python wcag_contrast_checker.py "orange" "navy"')
        sys.exit(1)
    
    color1_str = sys.argv[1]
    color2_str = sys.argv[2]
    
    try:
        # Parse colors
        rgb1 = parse_color(color1_str)
        rgb2 = parse_color(color2_str)
        
        # Calculate contrast ratio
        ratio = get_contrast_ratio(rgb1, rgb2)
        
        # Check compliance
        compliance = check_wcag_compliance(ratio)
        
        # Print results
        print_results(color1_str, color2_str, rgb1, rgb2, ratio, compliance)
        
        # Exit code: 0 if AA normal text passes, 1 otherwise
        sys.exit(0 if compliance['AA_normal'] else 1)
        
    except ValueError as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
