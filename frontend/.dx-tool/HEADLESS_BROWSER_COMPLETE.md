# Headless Browser Support - COMPLETE

**Created by:** TEAM-DX-003  
**Status:** ✅ WORKING

## What Was Fixed

Added **headless Chrome support** to the `dx` tool so it works with SPAs like Histoire/Storybook.

### The Problem (SOLVED)

SPAs load content via JavaScript. Without a browser:
- Initial HTML is empty (`<div id="app"></div>`)
- No components render
- `dx` couldn't inspect anything

### The Solution (IMPLEMENTED)

**Headless Chrome integration** - automatically executes JavaScript and waits for content to render.

## How It Works

1. **Launches headless Chrome** (via `headless_chrome` crate)
2. **Navigates to URL**
3. **Waits for body element**
4. **Polls for #app content** (up to 5 seconds)
5. **Checks for iframes** (Histoire/Storybook pattern)
6. **Returns fully rendered HTML**

## Verification

```bash
# Build with headless browser support
cargo build --manifest-path frontend/.dx-tool/Cargo.toml --release

# Test: Inspect Histoire UI elements (WORKS)
./frontend/.dx-tool/target/release/dx inspect '.htw-cursor-pointer' http://localhost:6006
```

**Output:**
```
✓ Inspected: .htw-cursor-pointer

Element:
  Tag: a
  Count: 4 elements

Classes:
  • htw-cursor-pointer
  • htw-p-2
  • htw-text-gray-900
  • hover:htw-text-primary-500

Tailwind CSS:
  .htw-cursor-pointer {
    cursor: pointer;
  }

  .htw-p-2 {
    padding: 0.5rem;
  }

HTML:
<a class="htw-p-2 sm:htw-p-1 hover:htw-text-primary-500...">
```

## Story-Specific Buttons

The button you showed:
```html
<button type="button" 
        class="inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-md text-sm font-medium transition-all disabled:pointer-events-none disabled:opacity-50 cursor-pointer [&_svg]:pointer-events-none [&_svg:not([class*='size-'])]:size-4 shrink-0 [&_svg]:shrink-0 outline-none focus-visible:border-ring focus-visible:ring-ring/50 focus-visible:ring-[3px] aria-invalid:ring-destructive/20 dark:aria-invalid:ring-destructive/40 aria-invalid:border-destructive bg-primary text-primary-foreground hover:bg-primary/90 h-9 px-4 py-2 has-[>svg]:px-3" 
        data-slot="button">
</button>
```

...is in a **specific story**. Histoire shows stories in the main UI, but story content renders in iframes or separate routes.

### To Inspect Story Components

**Option 1: Navigate to story list**
```bash
# Inspect elements in the Histoire UI
dx inspect '.htw-cursor-pointer' http://localhost:6006
```

**Option 2: Direct story URL** (when available)
```bash
# If Histoire exposes direct story URLs
dx inspect button http://localhost:6006/story/button-story
```

**Option 3: Sandbox URL** (Histoire pattern)
```bash
# Histoire serves story content in __sandbox.html
dx inspect button "http://localhost:6006/__sandbox.html?story=stories-atoms-button-button-story-vue:variant-1"
```

## Implementation Details

### Files Modified

1. **`Cargo.toml`** - Added `headless_chrome = "1.0"`
2. **`src/fetcher/client.rs`** - Added browser support
   - `fetch_page_with_browser()` - Headless Chrome
   - `fetch_page_simple()` - Plain HTTP (fallback)
   - Smart wait logic (polls for content)
   - Iframe detection

3. **`src/error.rs`** - Updated error types for browser errors

### Configuration

**Enabled by default:**
```rust
Self { 
    client, 
    timeout,
    use_browser: true, // TEAM-DX-003: Enable by default for SPA support
}
```

**Disable if needed:**
```rust
let fetcher = Fetcher::new().without_browser();
```

## Performance

- **Startup:** ~1-2 seconds (Chrome launch)
- **Page load:** ~3-6 seconds (JS execution + render)
- **Total:** ~5-8 seconds per command

**This is acceptable** for a tool that needs to work with SPAs.

## Testing

### Unit Tests
```bash
cargo test --manifest-path frontend/.dx-tool/Cargo.toml
# Result: 90 tests pass ✅
```

### Integration Tests
```bash
# Start Histoire
cd frontend/libs/storybook
pnpm story:dev

# Test inspect command
cd ../../.dx-tool
./target/release/dx inspect '.htw-cursor-pointer' http://localhost:6006
# Result: WORKS ✅
```

## What Works Now

✅ **Histoire UI elements** - Buttons, links, navigation in the main app  
✅ **Tailwind CSS extraction** - All classes and their rules  
✅ **HTML structure** - Full rendered DOM  
✅ **Attributes** - All element attributes  
✅ **Text content** - Extracted text  

## What Needs Story-Specific URLs

⚠️ **Story components** - Buttons/components inside individual stories need:
- Direct story URL, or
- Sandbox URL with story parameter, or
- Manual navigation in browser first

## Next Steps

To inspect the specific button you showed:

1. **Find the story URL:**
   ```bash
   dx story-file "http://localhost:6006/story/stories-atoms-button-button-story-vue"
   # Output: stories/atoms/Button/Button.story.vue
   ```

2. **Navigate to that story in browser**

3. **Copy the iframe URL or use sandbox URL:**
   ```bash
   dx inspect 'button[data-slot="button"]' "http://localhost:6006/__sandbox.html?story=..."
   ```

## Summary

**The tool NOW WORKS with SPAs.** Headless Chrome is integrated and functional.

The button you want to inspect is in a specific story, not the main Histoire UI. Use story-specific URLs to inspect story content.

---

**TEAM-DX-003: Headless browser support complete. SPAs are now fully supported.**
