# DX Tool Feature Requests

**Purpose:** Document missing features or improvements needed for the `dx` CLI tool.

**Rules:**
1. DO NOT create ad-hoc scripts or workarounds
2. Document the feature request here
3. Notify the DX tool team
4. They will implement the feature in the tool

---

## How to Submit a Feature Request

Include the following information:

1. **Use Case:** What are you trying to do?
2. **Current Workflow:** How do you do it now (if possible)?
3. **Desired Command:** What command would you like to run?
4. **Expected Output:** What should the output look like?
5. **Why Needed:** Why is this important?

---

## Example Request

### Request: Batch Inspect Multiple Variants

**Use Case:** I need to verify 10 button variants quickly.

**Current Workflow:**
```bash
dx inspect button http://localhost:6006/story/...?variantId=0
dx inspect button http://localhost:6006/story/...?variantId=1
dx inspect button http://localhost:6006/story/...?variantId=2
# ... repeat 10 times
```

**Desired Command:**
```bash
dx inspect-batch button http://localhost:6006/story/stories-atoms-button-button-story-vue --variants 0,1,2,3,4,5,6,7,8,9
```

**Expected Output:**
```
Variant 0: ✅ 30 classes, button tag, data-slot="button"
Variant 1: ✅ 30 classes, button tag, data-slot="button"
Variant 2: ✅ 30 classes, button tag, data-slot="button"
...
Summary: 10/10 variants passed
```

**Why Needed:** Saves time when verifying many variants. Reduces repetitive commands.

---

## Active Requests

### Request: List Available Stories (Components)

**Requested by:** TEAM-DX-004  
**Date:** 2025-10-12

**Use Case:** Developers need to discover which stories (components) are available in Histoire/Storybook without manually browsing the UI.

**Current Workflow:**
1. Open browser at http://localhost:6006
2. Manually expand "atoms" and "molecules" folders in sidebar
3. Manually note down component names
4. Manually construct URLs

**Desired Command:**
```bash
dx list-stories http://localhost:6006
```

**Expected Output:**
```
✓ Found 15 stories in Histoire

atoms/
  Accordion (1 variant)
    http://localhost:6006/story/stories-atoms-accordion-accordion-story-vue
  Alert (1 variant)
    http://localhost:6006/story/stories-atoms-alert-alert-story-vue
  Button (14 variants)
    http://localhost:6006/story/stories-atoms-button-button-story-vue
  Badge (5 variants)
    http://localhost:6006/story/stories-atoms-badge-badge-story-vue
  ...

molecules/
  ...
```

**Why Needed:** 
- Developers working via SSH/remote have no browser access
- Manual URL construction is error-prone
- Need to discover what components exist
- Essential for navigation and verification workflows

---

### Request: List Story Variants with URLs

**Requested by:** TEAM-DX-004  
**Date:** 2025-10-12  
**Status:** ✅ IMPLEMENTED (with edge case handling)

**Use Case:** After finding a story, developers need to see all available variants and their specific URLs for verification.

**Edge Case Handled:** Single-variant stories auto-redirect in Histoire. The command now detects this and constructs the variant-0 URL automatically.

**Current Workflow:**
1. Open browser at http://localhost:6006/story/stories-atoms-button-button-story-vue
2. Manually click each variant in the sidebar
3. Copy URL from address bar
4. Repeat for all variants

**Desired Command:**
```bash
dx list-variants http://localhost:6006/story/stories-atoms-button-button-story-vue
```

**Expected Output:**
```
✓ Found 14 variants for Button

Variant 0: Small
  http://localhost:6006/story/stories-atoms-button-button-story-vue?variantId=stories-atoms-button-button-story-vue-0

Variant 1: Default Size
  http://localhost:6006/story/stories-atoms-button-button-story-vue?variantId=stories-atoms-button-button-story-vue-1

Variant 2: Large
  http://localhost:6006/story/stories-atoms-button-button-story-vue?variantId=stories-atoms-button-button-story-vue-2

Variant 3: Primary
  http://localhost:6006/story/stories-atoms-button-button-story-vue?variantId=stories-atoms-button-button-story-vue-3

... (10 more variants)

Total: 14 variants
```

**Why Needed:**
- Developers can't guess variant IDs (0, 1, 2, ...)
- Need to verify all variants systematically
- Manual URL construction is error-prone
- Essential for batch verification workflows

**Alternative Syntax (with --copy-pastable flag):**
```bash
dx list-variants http://localhost:6006/story/stories-atoms-button-button-story-vue --copy-pastable
```

**Output:**
```bash
# Small
http://localhost:6006/story/stories-atoms-button-button-story-vue?variantId=stories-atoms-button-button-story-vue-0

# Default Size
http://localhost:6006/story/stories-atoms-button-button-story-vue?variantId=stories-atoms-button-button-story-vue-1

# Large
http://localhost:6006/story/stories-atoms-button-button-story-vue?variantId=stories-atoms-button-button-story-vue-2
```

---

### Request: Combined Story Discovery + Variant Listing

**Requested by:** TEAM-DX-004  
**Date:** 2025-10-12

**Use Case:** Quick workflow to discover and verify a specific component.

**Desired Command:**
```bash
dx list-stories http://localhost:6006 --component Button
```

**Expected Output:**
```
✓ Found Button in atoms/

Story URL:
  http://localhost:6006/story/stories-atoms-button-button-story-vue

Variants (14):
  0: Small - http://localhost:6006/story/...?variantId=...0
  1: Default Size - http://localhost:6006/story/...?variantId=...1
  2: Large - http://localhost:6006/story/...?variantId=...2
  ... (11 more)

To inspect all variants:
  dx list-variants http://localhost:6006/story/stories-atoms-button-button-story-vue
```

**Why Needed:** One-command workflow to find a component and see all its variants.

<!-- Add your requests below -->

---

### 🐛 BUG: CSS Class Detection Not Working

**Reported by:** TEAM-FE-013  
**Date:** 2025-10-12  
**Severity:** CRITICAL - Blocks frontend verification workflow

**Problem:** The `dx css --class-exists` command reports that `cursor-pointer` class does NOT exist in the stylesheet, but visual inspection in the browser shows the class IS working and the cursor DOES change to pointer on hover.

**Evidence:**
1. User confirms `cursor-pointer` works on http://localhost:6006 (storybook) - cursor changes to pointer on button hover
2. DX tool reports: `✗ Error: Class 'cursor-pointer' not found in stylesheet` for http://localhost:6006
3. Source code confirms class exists in `libs/storybook/stories/atoms/Button/Button.vue` line 12
4. This means the DX tool is NOT correctly extracting CSS rules from the compiled stylesheet

**Impact:**
- Cannot trust `dx css --class-exists` for verification
- Cannot verify Tailwind CSS is working correctly
- Blocks the entire frontend verification workflow per FRONTEND_ENGINEERING_RULES.md

**Reproduction:**
```bash
# 1. Verify cursor-pointer exists in source
grep -r "cursor-pointer" frontend/libs/storybook/stories/atoms/Button/Button.vue
# Output: cursor-pointer IS in the source code

# 2. Check if DX tool finds it
cd frontend/.dx-tool
cargo run --release -- css --class-exists "cursor-pointer" "http://localhost:6006"
# Output: ✗ Error: Class 'cursor-pointer' not found in stylesheet

# 3. Open browser and visually verify
# Open http://localhost:6006 in browser
# Hover over button
# Result: Cursor DOES change to pointer (class IS working)
```

**Expected Behavior:**
```bash
dx css --class-exists "cursor-pointer" "http://localhost:6006"
# Should output: ✓ Class 'cursor-pointer' found in stylesheet
```

**Root Cause IDENTIFIED:**

The `check_class_exists` function in `src/commands/css.rs` only checks if a CSS rule exists (e.g., `.cursor-pointer { cursor: pointer; }`), but it does NOT check if the class is actually in the HTML.

**Evidence from HTML inspection:**
```html
<!-- Port 6006 (working) -->
<button class="... cursor-pointer ...">Default Button</button>

<!-- Port 6007 (Navigation bar) -->
<button class="... cursor-pointer ...">Toggle theme</button>
```

The class `cursor-pointer` IS in the HTML class attribute on BOTH servers. But the DX tool reports it's not found because it's looking in the wrong place.

**The Problem:**
- Tailwind v4 may compile `cursor-pointer` into the final CSS differently
- The CSS parser may not be finding the rule `.cursor-pointer { cursor: pointer; }`
- But the class IS being applied in the HTML and IS working

**Next Steps for DX Tool Team:**
1. Update `check_class_exists` to ALSO check if class exists in HTML (not just CSS)
2. Add `--check-html` flag to verify class is in use
3. Debug CSS extraction in `src/commands/css.rs` line 24-57
4. Verify CssParser::class_exists() is correctly parsing Tailwind v4 output
5. Add test case for Tailwind v4 CSS extraction
6. Consider: Maybe the tool should check HTML first, then CSS for verification

**Workaround:** NONE - This is a critical bug that blocks verification

---

## Completed Requests

### ✅ List Story Variants with URLs
**Requested by:** TEAM-DX-004  
**Implemented:** 2025-10-12  
**Command:** `dx list-variants <story-url> [--copy-pastable]`  
**Status:** Complete - Lists all variants with full URLs

### ✅ Inspect Command (HTML + CSS in one)
**Requested by:** TEAM-DX-003  
**Implemented:** 2025-10-12  
**Command:** `dx inspect <selector> <url>`  
**Status:** Complete

### ✅ Story File Locator
**Requested by:** TEAM-DX-003  
**Implemented:** 2025-10-12  
**Command:** `dx story-file <url>`  
**Status:** Complete

### ✅ Headless Browser Support for SPAs
**Requested by:** TEAM-DX-003  
**Implemented:** 2025-10-12  
**Status:** Complete - Automatic Chrome launch + iframe detection

---

## Guidelines

**Good Request:**
- Clear use case
- Specific command syntax
- Expected output format
- Explains why it's needed

**Bad Request:**
- Vague description
- No example command
- No expected output
- Just says "make it better"

---

**Remember:** The DX tool is the standard way to verify frontend work. If you need a feature, request it here. Don't create workarounds.
