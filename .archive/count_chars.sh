#!/bin/bash
# Count characters in MANDATORY_ENGINEERING_RULES.md

FILE=".business/stakeholders/MANDATORY_ENGINEERING_RULES.md"

if [ ! -f "$FILE" ]; then
    echo "Error: File not found: $FILE"
    exit 1
fi

CHAR_COUNT=$(wc -m < "$FILE")
LINE_COUNT=$(wc -l < "$FILE")
WORD_COUNT=$(wc -w < "$FILE")

echo "═══════════════════════════════════════════════════════"
echo "Character Count for MANDATORY_ENGINEERING_RULES.md"
echo "═══════════════════════════════════════════════════════"
echo "Characters: $CHAR_COUNT"
echo "Lines:      $LINE_COUNT"
echo "Words:      $WORD_COUNT"
echo "═══════════════════════════════════════════════════════"

if [ "$CHAR_COUNT" -gt 12000 ]; then
    EXCESS=$((CHAR_COUNT - 12000))
    echo "⚠️  OVER LIMIT by $EXCESS characters"
    echo "Target: 12,000 characters"
    echo "Current: $CHAR_COUNT characters"
else
    REMAINING=$((12000 - CHAR_COUNT))
    echo "✅ UNDER LIMIT by $REMAINING characters"
fi

echo "═══════════════════════════════════════════════════════"
