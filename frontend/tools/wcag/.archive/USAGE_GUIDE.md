# WCAG Tools - Usage Guide

## For Engineers Checking Color Contrast

### Quick Start

**Check a single color pair:**
```bash
cd /home/vince/Projects/llama-orch/frontend/tools/wcag
python check_contrast.py --fg "#ffffff" --bg "#b45309" --label "My Button"
```

**Check multiple colors from JSON:**
```bash
python check_contrast.py --input my_colors.json
```

### Creating a JSON File for Your Component

1. Create a JSON file (e.g., `button_colors.json`):
```json
[
  {
    "label": "Button Primary Light",
    "fg": "#ffffff",
    "bg": "#b45309"
  },
  {
    "label": "Button Primary Dark",
    "fg": "#ffffff",
    "bg": "#b45309"
  },
  {
    "label": "Button Secondary Light",
    "fg": "#0f172a",
    "bg": "#f1f5f9"
  }
]
```

2. Run the checker:
```bash
python check_contrast.py --input button_colors.json
```

3. Fix any failures by adjusting colors in `theme-tokens.css`

4. Re-run until all pass ✅

### Examples

**Alert Component:**
```bash
# Create alert_colors.json
cat > alert_colors.json << 'EOF'
[
  {"label": "Alert Info", "fg": "#0f172a", "bg": "#3b82f6"},
  {"label": "Alert Success", "fg": "#ffffff", "bg": "#10b981"},
  {"label": "Alert Warning", "fg": "#0f172a", "bg": "#f59e0b"},
  {"label": "Alert Error", "fg": "#ffffff", "bg": "#dc2626"}
]
EOF

# Check them
python check_contrast.py --input alert_colors.json
```

**Card Component:**
```bash
python check_contrast.py --fg "#0f172a" --bg "#ffffff" --label "Card Light Mode" --verbose
python check_contrast.py --fg "#f1f5f9" --bg "#1e293b" --label "Card Dark Mode" --verbose
```

### Understanding Results

```
Label                             Ratio    AA Normal     AA Large
--------------------------------------------------------------------------------
Button Primary                    5.02:1       ✅ PASS       ✅ PASS
Button Secondary                  3.19:1       ❌ FAIL       ✅ PASS
```

- **AA Normal (4.5:1)**: Required for text < 18pt (or < 14pt bold)
- **AA Large (3.0:1)**: Required for text ≥ 18pt (or ≥ 14pt bold)

If something **FAILS AA Normal**:
1. Use it only for large text (≥18pt)
2. OR change the color to be darker/lighter
3. OR use a different color combination

### Verbose Mode

Get detailed information:
```bash
python check_contrast.py --fg "#ffffff" --bg "#3b82f6" --verbose
```

Output:
```
============================================================
Label: Button Primary
Foreground: #ffffff
Background: #3b82f6
Contrast Ratio: 3.68:1

WCAG Compliance:
  AA Normal Text (4.5:1):  ❌ FAIL
  AA Large Text (3:1):     ✅ PASS
  AAA Normal Text (7:1):   ❌ FAIL
  AAA Large Text (4.5:1):  ❌ FAIL
  UI Components (3:1):     ✅ PASS
```

### Save Results to JSON

```bash
python check_contrast.py --input my_colors.json --output results.json
```

## DO NOT Create Component-Specific Scripts

❌ **WRONG:**
```bash
# Don't do this!
cat > check_my_component.py << 'EOF'
def check_my_component():
    # hardcoded colors...
EOF
```

✅ **CORRECT:**
```bash
# Do this instead!
cat > my_component_colors.json << 'EOF'
[
  {"label": "MyComponent Primary", "fg": "#fff", "bg": "#b45309"}
]
EOF

python check_contrast.py --input my_component_colors.json
```

## Common Color Fixes

### Making Colors Darker for Better Contrast

If white text on a color fails:
- Amber: `#f59e0b` → `#b45309` (amber-500 → amber-700)
- Red: `#ef4444` → `#dc2626` (red-500 → red-600)
- Blue: `#3b82f6` → `#2563eb` (blue-500 → blue-600)
- Green: `#10b981` → `#059669` (emerald-500 → emerald-600)

### Making Colors Lighter for Better Contrast

If dark text on a color fails:
- Gray: `#f1f5f9` → `#ffffff` (slate-100 → white)
- Blue: `#dbeafe` → `#eff6ff` (blue-100 → blue-50)

## Integration with CI/CD

Add to your workflow:
```yaml
- name: Check WCAG Compliance
  run: |
    cd frontend/tools/wcag
    python check_contrast.py --input badge_colors.json
    python check_contrast.py --input button_colors.json
    # Add more as needed
```

## Questions?

See `README.md` for full documentation or check existing JSON files:
- `badge_colors.json` - Example for Badge component
