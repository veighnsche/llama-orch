# TEAM-DX-004 HANDOFF

**From:** TEAM-DX-003  
**To:** TEAM-DX-004 (Frontend Atom Developers)  
**Date:** 2025-10-12  
**Status:** DX Tool Complete with Headless Browser Support  
**Priority:** P0 - Use DX tool for Button component story development

---

## Mission: Develop Button Component Stories Using DX Tool

**Goal:** Add comprehensive story variants for the Button atom component in Storybook, using the `dx` CLI tool to verify your work without needing browser access.

**Why:** The DX tool now has full SPA support via headless Chrome. You must use it to verify your stories render correctly and have proper Tailwind CSS applied.

---

## What TEAM-DX-003 Delivered

### ✅ Complete DX Tool with SPA Support

**Headless Chrome Integration:**
- Automatically launches Chrome to render SPAs
- Waits for JavaScript execution (configurable timeout)
- Detects and navigates into Histoire iframes
- Extracts fully rendered HTML + all Tailwind CSS

**Commands Available:**
1. `dx css --class-exists <class>` - Check if Tailwind class exists
2. `dx css --selector <selector>` - Get CSS rules for selector
3. `dx css --list-classes --list-selector <selector>` - List all classes
4. `dx html --selector <selector>` - Query DOM structure
5. `dx html --attributes <selector>` - Get element attributes
6. `dx html --tree <selector>` - Visualize DOM tree
7. `dx story-file <URL>` - Locate story file from Storybook URL
8. **`dx inspect <selector> <URL>`** - Get HTML + all CSS in one command (NEW)

**Performance:**
- Tuned for Intel i5-1240P, 62GB RAM, NVMe SSD
- Default timeout: 3 seconds
- Maximum timeout: 10 seconds
- Typical execution: 6-8 seconds per command

---

## Your Mission: Button Component Stories

### Current State

**Existing Button Stories:**
- Location: `frontend/libs/storybook/stories/atoms/Button/Button.story.vue`
- Current variants: 3 (variant-0, variant-1, variant-2)
- All variants tested and working ✅

### What You Need to Do

**Add more button variants to showcase:**
1. **Size variants:** Small, Medium, Large, Extra Large
2. **Color variants:** Primary, Secondary, Destructive, Ghost, Link
3. **State variants:** Default, Hover, Active, Disabled, Loading
4. **Icon variants:** Icon Left, Icon Right, Icon Only
5. **Width variants:** Auto, Full Width, Fixed Width

**Minimum:** Add 10 new story variants (total 13+)

---

## Workflow: Using DX Tool

### Step 1: Locate the Story File

```bash
# Find where to edit
dx story-file "http://localhost:6006/story/stories-atoms-button-button-story-vue"

# Output:
# ✓ Story file located
#   URL: http://localhost:6006/story/stories-atoms-button-button-story-vue
#   File: /home/vince/Projects/llama-orch/frontend/libs/storybook/stories/atoms/Button/Button.story.vue
#   Component: /home/vince/Projects/llama-orch/frontend/libs/storybook/stories/atoms/Button/Button.vue
#   
#   To add more stories, edit:
#     - Story file: stories/atoms/Button/Button.story.vue
#     - Component: Button.vue (if needed)
```

### Step 2: Add a New Story Variant

Edit `Button.story.vue`:

```vue
<script setup lang="ts">
// Existing imports...

// Add new variant
const variantLarge: Variant = {
  id: 'large',
  title: 'Large Button',
  template: `
    <Button size="lg">
      Large Button
    </Button>
  `,
}

// Add to story
defineStory({
  title: 'Button',
  variants: [
    variantDefault,
    variantPrimary,
    variantSecondary,
    variantLarge, // NEW
  ],
})
</script>
```

### Step 3: Verify with DX Tool

```bash
# Inspect the new button variant
dx inspect button "http://localhost:6006/story/stories-atoms-button-button-story-vue?variantId=stories-atoms-button-button-story-vue-3"

# Expected output:
# ✓ Inspected: button
# 
# Element:
#   Tag: button
#   Count: 1 element
# 
# Classes:
#   • inline-flex
#   • items-center
#   • justify-center
#   • gap-2
#   • ... (30+ classes)
# 
# Tailwind CSS:
#   .inline-flex { display: inline-flex; }
#   .items-center { align-items: center; }
#   ... (all CSS rules)
# 
# HTML:
#   <button class="..." data-slot="button" type="button">Large Button</button>
```

### Step 4: Verify Tailwind Classes

```bash
# Check specific classes exist
dx css --class-exists "h-11" http://localhost:6006/story/...

# Get all classes on the button
dx css --list-classes --list-selector button http://localhost:6006/story/...

# Get specific CSS rules
dx css --selector ".h-11" http://localhost:6006/story/...
```

### Step 5: Verify HTML Structure

```bash
# Check button attributes
dx html --attributes button http://localhost:6006/story/...

# Visualize DOM tree
dx html --tree button --depth 2 http://localhost:6006/story/...
```

---

## Required Story Variants

### Size Variants (4 new)

```vue
// Small
<Button size="sm">Small</Button>

// Medium (default)
<Button size="md">Medium</Button>

// Large
<Button size="lg">Large</Button>

// Extra Large
<Button size="xl">Extra Large</Button>
```

**Verify with:**
```bash
dx inspect button "http://localhost:6006/story/stories-atoms-button-button-story-vue?variantId=stories-atoms-button-button-story-vue-small"
# Should show: h-8 px-3 py-1.5 text-xs

dx inspect button "http://localhost:6006/story/stories-atoms-button-button-story-vue?variantId=stories-atoms-button-button-story-vue-large"
# Should show: h-11 px-6 py-3 text-base
```

### Color Variants (5 new)

```vue
// Primary (default)
<Button variant="primary">Primary</Button>

// Secondary
<Button variant="secondary">Secondary</Button>

// Destructive
<Button variant="destructive">Destructive</Button>

// Ghost
<Button variant="ghost">Ghost</Button>

// Link
<Button variant="link">Link</Button>
```

**Verify with:**
```bash
dx inspect button "http://localhost:6006/story/...?variantId=...destructive"
# Should show: bg-destructive text-destructive-foreground
```

### State Variants (3 new)

```vue
// Disabled
<Button disabled>Disabled</Button>

// Loading
<Button loading>Loading</Button>

// With Icon
<Button>
  <Icon name="check" />
  With Icon
</Button>
```

**Verify with:**
```bash
dx inspect button "http://localhost:6006/story/...?variantId=...disabled"
# Should show: disabled:opacity-50 disabled:pointer-events-none
```

---

## Acceptance Criteria

### Must Have (P0)

- [ ] **10+ new button variants** added to `Button.story.vue`
- [ ] **All variants verified** using `dx inspect` command
- [ ] **Tailwind classes confirmed** for each variant
- [ ] **HTML structure validated** for each variant
- [ ] **Screenshots** of `dx inspect` output for 3 variants (in handoff)
- [ ] **No browser usage** - all verification done via `dx` tool

### Verification Checklist

For each new variant, run:

```bash
# 1. Inspect full output
dx inspect button "http://localhost:6006/story/stories-atoms-button-button-story-vue?variantId=<variant-id>"

# 2. Verify classes exist
dx css --class-exists "<expected-class>" http://localhost:6006/story/...

# 3. Check HTML structure
dx html --selector button http://localhost:6006/story/...

# 4. Validate attributes
dx html --attributes button http://localhost:6006/story/...
```

**Document results** in your handoff with:
- Variant name
- Expected classes
- Actual classes (from `dx inspect`)
- ✅ or ❌ for each verification

---

## DX Tool Reference

### Quick Command Reference

```bash
# Locate story file
dx story-file <storybook-url>

# Inspect element (HTML + CSS in one)
dx inspect <selector> <url>

# Check class exists
dx css --class-exists <class> <url>

# Get CSS rules
dx css --selector <selector> <url>

# List all classes
dx css --list-classes --list-selector <selector> <url>

# Query DOM
dx html --selector <selector> <url>

# Get attributes
dx html --attributes <selector> <url>

# Visualize tree
dx html --tree <selector> --depth <n> <url>

# JSON output (for scripting)
dx --format json inspect <selector> <url>
```

### Project Shortcuts

```bash
# Use --project flag instead of full URL
dx --project storybook inspect button

# Configured in .llorch.toml:
# [dx.projects]
# storybook = "http://localhost:6006"
```

### Performance Tips

```bash
# Default: 3 seconds wait (good for most cases)
dx inspect button <url>

# If you get "Selector not found", the page may need more time
# The tool will retry automatically, but if it persists:
# - Check if Storybook is running: curl http://localhost:6006
# - Check if the variant ID is correct
# - Try a different selector: dx inspect '[data-slot="button"]' <url>
```

---

## Example: Complete Workflow

```bash
# 1. Start Storybook
cd frontend/libs/storybook
pnpm story:dev
# Wait for: "Local: http://localhost:6006/"

# 2. Find story file
dx story-file "http://localhost:6006/story/stories-atoms-button-button-story-vue"
# Output: stories/atoms/Button/Button.story.vue

# 3. Edit story file
vim stories/atoms/Button/Button.story.vue
# Add new variant: variantLarge

# 4. Verify new variant
dx inspect button "http://localhost:6006/story/stories-atoms-button-button-story-vue?variantId=stories-atoms-button-button-story-vue-3"

# 5. Check specific classes
dx css --class-exists "h-11" "http://localhost:6006/story/stories-atoms-button-button-story-vue?variantId=stories-atoms-button-button-story-vue-3"

# 6. Get all classes
dx css --list-classes --list-selector button "http://localhost:6006/story/stories-atoms-button-button-story-vue?variantId=stories-atoms-button-button-story-vue-3"

# 7. Validate HTML
dx html --attributes button "http://localhost:6006/story/stories-atoms-button-button-story-vue?variantId=stories-atoms-button-button-story-vue-3"
```

---

## Troubleshooting

### "Selector not found: button"

**Cause:** Page hasn't finished rendering, or selector is wrong.

**Solution:**
1. Check Storybook is running: `curl http://localhost:6006`
2. Try different selector: `dx inspect '[data-slot="button"]' <url>`
3. Verify variant ID is correct (check Storybook URL in browser)
4. Wait a few seconds and retry

### "Network error" or "Connection failed"

**Cause:** Storybook not running.

**Solution:**
```bash
cd frontend/libs/storybook
pnpm story:dev
```

### Command takes too long (>15 seconds)

**Cause:** Headless Chrome is slow or page is heavy.

**Solution:**
- This is normal for first run (Chrome startup)
- Subsequent runs should be faster (~8 seconds)
- If consistently slow, check system resources

### Classes not showing in output

**Cause:** Tailwind not generating classes, or wrong selector.

**Solution:**
1. Check component uses Tailwind classes
2. Verify Tailwind config includes component path
3. Try `dx css --list-classes --list-selector button <url>` to see all classes

---

## Handoff Requirements

### What to Include

1. **Summary:**
   - How many variants added
   - Which categories covered (size, color, state, etc.)
   - Total variants now in Button.story.vue

2. **Verification Results:**
   - Table with variant name, expected classes, actual classes, status
   - Example:
     ```
     | Variant | Expected Classes | Verified | Status |
     |---------|------------------|----------|--------|
     | Small   | h-8 px-3 text-xs | ✅       | Pass   |
     | Large   | h-11 px-6 text-base | ✅    | Pass   |
     ```

3. **DX Tool Usage:**
   - Commands used for verification
   - Any issues encountered
   - Screenshots of `dx inspect` output (3 examples)

4. **Next Steps:**
   - Recommendations for other components
   - Any missing DX tool features needed

---

## CRITICAL: Frontend Engineering Rules

**Per FRONTEND_ENGINEERING_RULES.md:**

### If DX Tool is Missing a Feature

**DO NOT create ad-hoc custom commands or scripts.**

**INSTEAD:**
1. Document the missing feature in `frontend/.dx-tool/FEATURE_REQUESTS.md`
2. Include:
   - Use case (what you're trying to do)
   - Expected command syntax
   - Expected output
   - Why existing commands don't work
3. Notify the DX tool team
4. They will add the feature to the tool

**Example:**

```markdown
# FEATURE_REQUESTS.md

## Request: Batch Inspect Multiple Variants

**Use Case:** I need to verify 10 button variants quickly.

**Current Workflow:**
```bash
dx inspect button <url-variant-0>
dx inspect button <url-variant-1>
dx inspect button <url-variant-2>
# ... 10 times
```

**Desired Command:**
```bash
dx inspect-batch button <base-url> --variants 0,1,2,3,4,5,6,7,8,9
```

**Expected Output:**
```
Variant 0: ✅ 30 classes, button tag
Variant 1: ✅ 30 classes, button tag
...
```

**Why Needed:** Saves time, reduces repetitive commands.
```

### Using DX Tool is Mandatory

**All frontend teams MUST use the DX tool for:**
- Verifying Tailwind classes exist
- Checking component HTML structure
- Validating story rendering
- Locating story files

**DO NOT:**
- Use `curl` + manual parsing
- Write custom scripts to inspect HTML
- Rely on browser DevTools alone
- Skip verification

**The DX tool is the standard way to verify frontend work without browser access.**

---

## Success Criteria

### Definition of Done

- [ ] 10+ button variants added
- [ ] All variants verified with `dx inspect`
- [ ] All Tailwind classes confirmed
- [ ] HTML structure validated
- [ ] Handoff document with verification table
- [ ] Screenshots of 3 `dx inspect` outputs
- [ ] No ad-hoc scripts created
- [ ] Feature requests documented (if any)

### Quality Gates

- [ ] Every variant has a unique `variantId`
- [ ] Every variant has a descriptive title
- [ ] Every variant demonstrates a specific use case
- [ ] Every variant uses proper Tailwind classes
- [ ] Every variant passes `dx inspect` verification

---

## Resources

### Documentation

- **DX Tool README:** `frontend/.dx-tool/README.md`
- **Performance Specs:** `frontend/.dx-tool/PERFORMANCE_SPECS.md`
- **Headless Browser Guide:** `frontend/.dx-tool/HEADLESS_BROWSER_COMPLETE.md`
- **Inspect Command Demo:** `frontend/.dx-tool/INSPECT_COMMAND_DEMO.md`
- **Frontend Rules:** `frontend/FRONTEND_ENGINEERING_RULES.md`

### Example Commands

See `frontend/.dx-tool/INSPECT_COMMAND_DEMO.md` for full examples.

### Getting Help

If you encounter issues:
1. Check `PERFORMANCE_SPECS.md` for timeout tuning
2. Check `HEADLESS_BROWSER_COMPLETE.md` for SPA-specific issues
3. Document feature requests in `FEATURE_REQUESTS.md`
4. Ask DX tool team for help (don't create workarounds)

---

## Timeline

**Estimated:** 2-3 hours

- 30 min: Setup and familiarize with DX tool
- 1.5 hours: Add 10 button variants
- 1 hour: Verify all variants with `dx` commands
- 30 min: Document results in handoff

---

## Final Notes

**The DX tool is production-ready and fully functional.**

- Headless Chrome support: ✅
- SPA rendering: ✅
- Histoire iframe detection: ✅
- Tailwind CSS extraction: ✅
- HTML structure inspection: ✅
- Story file locator: ✅

**Use it. Trust it. Report issues if found.**

**Do not work around it. Extend it.**

---

**TEAM-DX-003 OUT. Good luck, TEAM-DX-004!**
