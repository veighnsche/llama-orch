#!/bin/bash
# Test rbee Design System Color Combinations for WCAG 2.1 Compliance

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKER="$SCRIPT_DIR/wcag_contrast_checker.py"

echo "🎨 Testing rbee Design System Color Combinations"
echo "=================================================="
echo ""

# Primary on Dark Background
echo "1️⃣  PRIMARY TEXT ON DARK BACKGROUND"
python3 "$CHECKER" "#f59e0b" "#0f172a"

# Muted Text on Dark Card
echo -e "\n2️⃣  MUTED TEXT ON DARK CARD"
python3 "$CHECKER" "#94a3b8" "#1e293b"

# White Text on Primary Background
echo -e "\n3️⃣  WHITE TEXT ON PRIMARY BACKGROUND"
python3 "$CHECKER" "#ffffff" "#f59e0b"

# Dark Text on Primary Background
echo -e "\n4️⃣  DARK TEXT ON PRIMARY BACKGROUND"
python3 "$CHECKER" "#0f172a" "#f59e0b"

# Foreground on Background (Light Mode)
echo -e "\n5️⃣  FOREGROUND ON BACKGROUND (LIGHT MODE)"
python3 "$CHECKER" "#0f172a" "#ffffff"

# Foreground on Background (Dark Mode)
echo -e "\n6️⃣  FOREGROUND ON BACKGROUND (DARK MODE)"
python3 "$CHECKER" "#f1f5f9" "#0f172a"

# Border on Background (Light Mode)
echo -e "\n7️⃣  BORDER ON BACKGROUND (LIGHT MODE)"
python3 "$CHECKER" "#e2e8f0" "#ffffff"

# Border on Background (Dark Mode)
echo -e "\n8️⃣  BORDER ON BACKGROUND (DARK MODE)"
python3 "$CHECKER" "#334155" "#0f172a"

echo -e "\n=================================================="
echo "✅ Design System Color Testing Complete"
