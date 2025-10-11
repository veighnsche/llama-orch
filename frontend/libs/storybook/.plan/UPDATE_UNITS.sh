#!/bin/bash
# Script to add "Required Reading" section to all unit files
# Created by: TEAM-FE-003

REQUIRED_READING='
---

## 📚 Required Reading

**BEFORE starting this unit, read:**

1. **Design Tokens (CRITICAL):** `00-DESIGN-TOKENS-CRITICAL.md`
   - DO NOT copy colors from React reference
   - Use design tokens: `bg-primary`, `text-foreground`, etc.
   - Translation guide: React colors → Vue tokens

2. **Engineering Rules:** `/frontend/FRONTEND_ENGINEERING_RULES.md`
   - Section 2: Design tokens requirement
   - Section 3: Histoire `.story.vue` format
   - Section 8: Port vs create distinction

3. **Examples:** Look at completed components
   - HeroSection: `/frontend/libs/storybook/stories/organisms/HeroSection/`
   - WhatIsRbee: `/frontend/libs/storybook/stories/organisms/WhatIsRbee/`
   - ProblemSection: `/frontend/libs/storybook/stories/organisms/ProblemSection/`

**Key Rules:**
- ✅ Use `.story.vue` format (NOT `.story.ts`)
- ✅ Use design tokens (NOT hardcoded colors like `bg-amber-500`)
- ✅ Import from workspace: `import { Button } from '\''rbee-storybook/stories'\''`
- ✅ Add team signature: `<!-- TEAM-FE-XXX: Implemented ComponentName -->`
- ✅ Export in `stories/index.ts`

---
'

# Find all unit files (01-XX through 08-XX)
for file in /home/vince/Projects/llama-orch/frontend/.plan/[0-9][0-9]-*.md; do
    if [ -f "$file" ]; then
        # Check if "Required Reading" section already exists
        if ! grep -q "## 📚 Required Reading" "$file"; then
            echo "Updating: $file"
            # Add the section before the last line (which is usually "Next:" or end)
            echo "$REQUIRED_READING" >> "$file"
        else
            echo "Skipping (already has section): $file"
        fi
    fi
done

echo "Done! All unit files updated."
