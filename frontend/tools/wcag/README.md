# WCAG Compliance Tools

Tools for checking WCAG 2.1 color contrast compliance across the frontend codebase.

## Tools

### `wcag_utils.py`

Shared utilities for WCAG color contrast calculations.

**Functions:**
- `parse_color(color_str)` - Parse hex, rgb, hsl, or named colors
- `get_contrast_ratio(color1, color2)` - Calculate WCAG contrast ratio
- `check_wcag_compliance(ratio)` - Check against WCAG standards

### `check_components.py`

Scans all TSX/Vue components and generates a compliance report.

**Usage:**
```bash
# Scan default directory (packages/rbee-ui/src)
python check_components.py

# Scan specific directory
python check_components.py --components-dir /path/to/components

# Verbose output
python check_components.py --verbose

# Custom output file
python check_components.py --output my-report.md
```

**Output:**
- Generates `WCAG_COMPLIANCE_REPORT.md` with:
  - Summary of pass/fail combinations
  - Detailed failure analysis with file locations
  - List of compliant combinations
  - Unknown/unresolved combinations

## WCAG 2.1 Standards

### Contrast Ratios

| Level | Text Size | Required Ratio |
|-------|-----------|----------------|
| AA | Normal (< 18pt) | 4.5:1 |
| AA | Large (≥ 18pt) | 3:1 |
| AAA | Normal (< 18pt) | 7:1 |
| AAA | Large (≥ 18pt) | 4.5:1 |
| - | UI Components | 3:1 |

### Text Size Definitions

- **Normal text:** < 18pt (24px) or < 14pt (18.5px) bold
- **Large text:** ≥ 18pt (24px) or ≥ 14pt (18.5px) bold

## Design Token Colors

The checker uses colors from `packages/rbee-ui/src/tokens/theme-tokens.css`:

### Light Mode
- `foreground`: #0f172a (dark slate)
- `background`: #ffffff (white)
- `muted-foreground`: #64748b (slate)
- `secondary`: #f1f5f9 (light slate)
- `primary`: #f59e0b (amber)

### Dark Mode
- `foreground`: #f1f5f9 (light slate)
- `background`: #0f172a (dark slate)
- `muted-foreground`: #94a3b8 (lighter slate)
- `secondary`: #1e293b (slate)
- `primary`: #f59e0b (amber)

## Common Issues

### ❌ `text-muted-foreground` on `bg-secondary`

**Light Mode:** 4.34:1 (FAILS AA normal text)
- Only acceptable for large text (≥18pt)
- Use `text-foreground` for smaller text

**Fix:**
```tsx
// ❌ BAD: Small text with muted-foreground
<p className="text-base text-muted-foreground">...</p>

// ✅ GOOD: Use foreground for small text
<p className="text-base text-foreground">...</p>

// ✅ GOOD: Use muted-foreground only for large text
<p className="text-xl text-muted-foreground">...</p>
```

## Integration

### Pre-commit Hook

Add to `.git/hooks/pre-commit`:
```bash
#!/bin/bash
cd frontend/tools/wcag
python check_components.py --output /tmp/wcag-report.md
if grep -q "❌ Fail" /tmp/wcag-report.md; then
    echo "⚠️  WCAG compliance issues found. Run: python tools/wcag/check_components.py"
fi
```

### CI/CD

Add to GitHub Actions:
```yaml
- name: Check WCAG Compliance
  run: |
    cd frontend/tools/wcag
    python check_components.py
    if grep -q "❌ Fail" WCAG_COMPLIANCE_REPORT.md; then
      echo "::warning::WCAG compliance issues found"
    fi
```

## Development

### Adding New Color Tokens

Update `LIGHT_MODE_COLORS` and `DARK_MODE_COLORS` in `check_components.py` when adding new design tokens.

### Testing

```bash
# Test individual color combinations
cd frontend/tools/wcag
python -c "
from wcag_utils import parse_color, get_contrast_ratio, check_wcag_compliance
text = parse_color('#64748b')
bg = parse_color('#f1f5f9')
ratio = get_contrast_ratio(text, bg)
print(f'Ratio: {ratio:.2f}:1')
print(check_wcag_compliance(ratio))
"
```
