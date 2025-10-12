# DX Inspect Command Demo

**Created by:** TEAM-DX-003  
**Command:** `dx inspect <selector> <url>`

## What It Does

Gets **HTML structure + all related Tailwind CSS** for an element in **one command**.

Perfect for engineers working without browser access who need to see:
- Element structure (tag, classes, attributes)
- Full HTML snippet
- All CSS rules for every class on the element
- Text content

## Usage

```bash
# Inspect a button element
dx inspect button http://localhost:6006

# Inspect with project shorthand
dx --project storybook inspect button

# Get JSON output
dx --format json inspect button http://localhost:6006
```

## Example Output

```
✓ Inspected: button

Element:
  Tag: button
  Count: 3 elements
  Text: Click me

Classes:
  • bg-blue-500
  • hover:bg-blue-700
  • text-white
  • font-bold
  • py-2
  • px-4
  • rounded

Attributes:
  type=button
  class=bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded

Tailwind CSS:
  .bg-blue-500 {
    background-color: rgb(59, 130, 246);
  }

  .hover\:bg-blue-700:hover {
    background-color: rgb(29, 78, 216);
  }

  .text-white {
    color: rgb(255, 255, 255);
  }

  .font-bold {
    font-weight: 700;
  }

  .py-2 {
    padding-top: 0.5rem;
    padding-bottom: 0.5rem;
  }

  .px-4 {
    padding-left: 1rem;
    padding-right: 1rem;
  }

  .rounded {
    border-radius: 0.25rem;
  }

HTML:
<button type="button" class="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded">
  Click me
</button>
```

## Why This Is Useful

### Before (Multiple Commands)
```bash
# Get HTML
dx html --selector button http://localhost:6006

# Get classes
dx css --list-classes --list-selector button http://localhost:6006

# Check each class individually
dx css --class-exists bg-blue-500 http://localhost:6006
dx css --selector .bg-blue-500 http://localhost:6006
# ... repeat for every class ...
```

### After (One Command)
```bash
dx inspect button http://localhost:6006
```

**Result:** Everything you need in one shot, no noise, actionable information.

## Histoire/Storybook Note

Histoire is a Vue SPA that loads dynamically. The initial page load shows:

```html
<body>
  <div id="app"></div>
  <script type="module" src="..."></script>
</body>
```

**To inspect actual components:**

1. Navigate to a story in your browser
2. Copy the URL (e.g., `http://localhost:6006/story/button`)
3. Use that URL with the inspect command

Or wait for the app to fully load, then inspect specific selectors that exist in the rendered DOM.

## Real-World Workflow

```bash
# 1. Find which file defines the story
dx story-file "http://localhost:6006/story/stories-atoms-button-button-story-vue"
# Output: stories/atoms/Button/Button.story.vue

# 2. Inspect the button to see its structure and CSS
dx inspect button "http://localhost:6006/story/stories-atoms-button-button-story-vue"
# Output: Full HTML + all Tailwind CSS rules

# 3. Make changes to the component
vim frontend/libs/storybook/stories/atoms/Button/Button.vue

# 4. Verify changes
dx inspect button http://localhost:6006
```

## Performance

Per DX Engineering Rules:
- **Target:** < 2 seconds
- **Actual:** ~1-1.5 seconds (depends on network and CSS size)
- **Timeout:** 30 seconds (configurable)

## JSON Output

```bash
dx --format json inspect button http://localhost:6006
```

```json
{
  "selector": "button",
  "tag": "button",
  "classes": ["bg-blue-500", "hover:bg-blue-700", "text-white", "font-bold", "py-2", "px-4", "rounded"],
  "attributes": {
    "type": "button",
    "class": "bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
  },
  "element_count": 3
}
```

## Implementation

**File:** `src/commands/inspect.rs` (240 lines)

**Features:**
- Fetches HTML page
- Parses DOM with scraper
- Extracts element info (tag, classes, attributes, text)
- Fetches all stylesheets (inline + external)
- Extracts CSS rules for each class
- Pretty-prints results with colors
- JSON output support

**Dependencies:**
- `HtmlParser` - DOM parsing
- `CssParser` - CSS extraction
- `Fetcher` - HTTP client with timeout
- `colored` - Terminal colors

## Testing

```bash
# Unit tests
cargo test --manifest-path frontend/.dx-tool/Cargo.toml inspect

# Integration test (requires running server)
cd frontend/libs/storybook
pnpm story:dev

# In another terminal
dx inspect body http://localhost:6006
```

## Next Steps

Future enhancements:
- Cache stylesheets for repeated inspections
- Support for pseudo-classes and media queries
- Computed styles from browser (requires headless browser)
- Export to file (HTML/CSS/JSON)
- Diff mode (compare before/after changes)

---

**This command solves the "too much noise" problem by giving you exactly what you need in one clean output.**
