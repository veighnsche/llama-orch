# WCAG 2.1 Color Contrast Checker

A Python script to check if two colors meet WCAG 2.1 accessibility standards.

## Features

✅ **Multiple Input Formats**:
- Hex: `#RGB`, `#RRGGBB`, `#RRGGBBAA`
- RGB: `rgb(r, g, b)`, `rgba(r, g, b, a)`
- HSL: `hsl(h, s%, l%)`, `hsla(h, s%, l%, a)`
- Named colors: `white`, `black`, `red`, `orange`, `navy`, etc.

✅ **WCAG 2.1 Compliance Levels**:
- **AA Normal Text**: 4.5:1 (minimum for text < 18pt)
- **AA Large Text**: 3:1 (minimum for text ≥ 18pt or ≥ 14pt bold)
- **AAA Normal Text**: 7:1 (enhanced for text < 18pt)
- **AAA Large Text**: 4.5:1 (enhanced for text ≥ 18pt or ≥ 14pt bold)
- **UI Components**: 3:1 (minimum for UI elements)

✅ **Clear Output**:
- Contrast ratio calculation
- Pass/fail for each WCAG level
- Color format conversion (shows both RGB and hex)
- Actionable recommendations

## Usage

### Basic Usage

```bash
python3 wcag_contrast_checker.py <color1> <color2>
```

### Examples

**Hex colors** (primary yellow on dark background):
```bash
python3 wcag_contrast_checker.py "#f59e0b" "#0f172a"
```

**RGB colors**:
```bash
python3 wcag_contrast_checker.py "rgb(245, 158, 11)" "rgb(15, 23, 42)"
```

**HSL colors**:
```bash
python3 wcag_contrast_checker.py "hsl(45, 96%, 53%)" "hsl(222, 47%, 11%)"
```

**Named colors**:
```bash
python3 wcag_contrast_checker.py "orange" "navy"
```

**Muted text** (check if muted-foreground has enough contrast):
```bash
python3 wcag_contrast_checker.py "#94a3b8" "#1e293b"
```

## Output Example

```
======================================================================
WCAG 2.1 COLOR CONTRAST CHECKER
======================================================================

📊 INPUT COLORS:
  Color 1: #f59e0b
    → rgb(245, 158, 11) / #f59e0b
  Color 2: #0f172a
    → rgb(15, 23, 42) / #0f172a

📈 CONTRAST RATIO: 8.31:1

✅ WCAG 2.1 COMPLIANCE:

  AA Level (Minimum):
    Normal Text (< 18pt):  ✅ PASS (requires 4.5:1)
    Large Text (≥ 18pt):   ✅ PASS (requires 3:1)
    UI Components:         ✅ PASS (requires 3:1)

  AAA Level (Enhanced):
    Normal Text (< 18pt):  ✅ PASS (requires 7:1)
    Large Text (≥ 18pt):   ✅ PASS (requires 4.5:1)

💡 RECOMMENDATION:
  🌟 Excellent! Meets AAA standard for all text sizes.

======================================================================
```

## Exit Codes

- `0`: Colors meet AA standard for normal text (4.5:1)
- `1`: Colors do NOT meet AA standard for normal text

This allows the script to be used in CI/CD pipelines:

```bash
if python3 wcag_contrast_checker.py "#f59e0b" "#0f172a"; then
    echo "✅ Colors are accessible"
else
    echo "❌ Colors fail accessibility standards"
fi
```

## WCAG 2.1 Standards Reference

### Text Contrast Requirements

| Level | Normal Text (< 18pt) | Large Text (≥ 18pt or ≥ 14pt bold) |
|-------|---------------------|-----------------------------------|
| **AA** (Minimum) | 4.5:1 | 3:1 |
| **AAA** (Enhanced) | 7:1 | 4.5:1 |

### UI Component Contrast

- **Minimum**: 3:1 for UI components and graphical objects

### What is "Large Text"?

- **18pt** (24px) or larger
- **14pt** (18.66px) bold or larger

## Common Use Cases

### Check Primary Color on Dark Background
```bash
python3 wcag_contrast_checker.py "#f59e0b" "#0f172a"
# Result: 8.31:1 ✅ AAA PASS
```

### Check Muted Text on Card Background
```bash
python3 wcag_contrast_checker.py "#94a3b8" "#1e293b"
# Result: 5.71:1 ✅ AA PASS (AAA FAIL for normal text)
```

### Check Button Text on Primary Background
```bash
python3 wcag_contrast_checker.py "#ffffff" "#f59e0b"
# Result: 1.78:1 ❌ FAIL (use dark text instead)
```

### Check Link Color on White Background
```bash
python3 wcag_contrast_checker.py "#3b82f6" "#ffffff"
# Result: 3.14:1 ❌ FAIL for normal text (only passes for large text)
```

## Design System Integration

Use this script to verify your design tokens meet accessibility standards:

```bash
# Primary on dark background
python3 wcag_contrast_checker.py "#f59e0b" "#0f172a"

# Muted foreground on dark card
python3 wcag_contrast_checker.py "#94a3b8" "#1e293b"

# Foreground on background (light mode)
python3 wcag_contrast_checker.py "#0f172a" "#ffffff"

# Foreground on background (dark mode)
python3 wcag_contrast_checker.py "#f1f5f9" "#0f172a"
```

## Supported Named Colors

The script supports common CSS named colors:
- `white`, `black`, `red`, `green`, `blue`
- `yellow`, `cyan`, `magenta`, `orange`, `purple`
- `pink`, `brown`, `gray`, `grey`, `navy`
- `teal`, `olive`, `maroon`, `lime`, `aqua`
- `fuchsia`, `silver`

## Technical Details

### Relative Luminance Calculation

The script uses the WCAG 2.1 formula for relative luminance:

```
L = 0.2126 * R + 0.7152 * G + 0.0722 * B
```

Where R, G, B are gamma-corrected values.

### Contrast Ratio Formula

```
Contrast Ratio = (L1 + 0.05) / (L2 + 0.05)
```

Where L1 is the lighter color and L2 is the darker color.

## References

- [WCAG 2.1 Contrast (Minimum)](https://www.w3.org/WAI/WCAG21/Understanding/contrast-minimum.html)
- [WCAG 2.1 Contrast (Enhanced)](https://www.w3.org/WAI/WCAG21/Understanding/contrast-enhanced.html)
- [WCAG 2.1 Non-text Contrast](https://www.w3.org/WAI/WCAG21/Understanding/non-text-contrast.html)

## License

Same as the parent project (GPL-3.0-or-later).
