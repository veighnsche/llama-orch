#!/usr/bin/env python3
"""
WCAG 2.1 Color Contrast Utilities

Shared utilities for WCAG color contrast checking.
"""

import re
from typing import Tuple


# Named CSS colors (subset)
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
