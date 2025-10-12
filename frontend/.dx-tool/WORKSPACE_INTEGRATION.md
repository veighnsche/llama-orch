# Workspace Integration Guide

**Created by:** TEAM-DX-001  
**For:** Nuxt + Tailwind projects in `bin/commercial` and `libs/storybook`

---

## Overview

The DX tool is **workspace-aware** and knows about your frontend projects:

- **Commercial** (`bin/commercial`) - Nuxt app with Tailwind on port 3000
- **Storybook** (`libs/storybook`) - Component library on port 6006

---

## Quick Reference

### Commercial Frontend

```bash
# Start dev server
cd frontend/bin/commercial
pnpm dev

# Verify Tailwind classes (in another terminal)
dx --project commercial css --class-exists "cursor-pointer"
dx --project commercial css --class-exists "hover:bg-blue-500"
dx --project commercial css --class-exists "text-foreground"
```

### Storybook

```bash
# Start storybook
cd frontend/libs/storybook
pnpm story:dev

# Verify component styles (in another terminal)
dx --project storybook css --class-exists "btn-primary"
dx --project storybook css --class-exists "text-muted-foreground"
```

---

## Configuration

The tool reads `.dxrc.json` from the `frontend/` directory:

```json
{
  "base_url": "http://localhost:3000",
  "timeout": 2,
  "workspace": {
    "commercial": {
      "dir": "bin/commercial",
      "url": "http://localhost:3000",
      "port": 3000,
      "component_paths": [
        "app/**/*.vue",
        "components/**/*.vue",
        "layouts/**/*.vue",
        "pages/**/*.vue"
      ]
    },
    "storybook": {
      "dir": "libs/storybook",
      "url": "http://localhost:6006",
      "port": 6006,
      "component_paths": [
        "stories/**/*.vue"
      ]
    }
  }
}
```

---

## Tailwind v4 Integration

Both projects use **Tailwind v4** with automatic content scanning:

### Commercial (`bin/commercial/tailwind.config.js`)
```javascript
export default {
  content: [
    './app/**/*.{vue,js,ts,jsx,tsx}',
    './components/**/*.{vue,js,ts,jsx,tsx}',
    './layouts/**/*.{vue,js,ts,jsx,tsx}',
    './pages/**/*.{vue,js,ts,jsx,tsx}',
    './plugins/**/*.{js,ts}',
    // Scans storybook package for classes
    '../../libs/storybook/stories/**/*.{vue,js,ts,jsx,tsx}',
  ],
}
```

### Storybook (`libs/storybook`)
Uses Tailwind v4 with PostCSS, scans all `.vue` files in `stories/`.

---

## Common Workflows

### After Adding Tailwind Classes to Commercial

```bash
# 1. Add class to component
# app/components/Button.vue
<button class="cursor-pointer hover:bg-blue-500">Click me</button>

# 2. Verify class is generated
dx --project commercial css --class-exists "cursor-pointer"
dx --project commercial css --class-exists "hover:bg-blue-500"
```

### After Creating Storybook Component

```bash
# 1. Create component with Tailwind classes
# libs/storybook/stories/Button.vue
<button class="btn-primary text-foreground">Button</button>

# 2. Verify classes
dx --project storybook css --class-exists "btn-primary"
dx --project storybook css --class-exists "text-foreground"
```

### CI/CD Integration

```yaml
# .github/workflows/frontend-verify.yml
- name: Start commercial dev server
  run: |
    cd frontend/bin/commercial
    pnpm dev &
    sleep 5

- name: Verify Tailwind classes
  run: |
    cd frontend/.dx-tool
    ./target/release/dx --project commercial css --class-exists "cursor-pointer"
    ./target/release/dx --project commercial css --class-exists "hover:bg-blue-500"
```

---

## Troubleshooting

### "Class not found" in Commercial

**Problem:** Tailwind not generating the class

**Check:**
1. Is the class used in any `.vue` file in `app/`, `components/`, `layouts/`, or `pages/`?
2. Is the class used in storybook components? (Commercial scans storybook too)
3. Is the dev server running? (`pnpm dev`)

**Solution:**
```bash
# Restart dev server
cd frontend/bin/commercial
pnpm dev

# Verify again
dx --project commercial css --class-exists "your-class"
```

### "Class not found" in Storybook

**Problem:** Tailwind not generating the class

**Check:**
1. Is the class used in any `.vue` file in `stories/`?
2. Is storybook dev server running? (`pnpm story:dev`)

**Solution:**
```bash
# Restart storybook
cd frontend/libs/storybook
pnpm story:dev

# Verify again
dx --project storybook css --class-exists "your-class"
```

### Wrong Port

**Problem:** Tool connecting to wrong port

**Check `.dxrc.json`:**
```json
{
  "workspace": {
    "commercial": {
      "url": "http://localhost:3000",  // Should match pnpm dev port
      "port": 3000
    }
  }
}
```

---

## Benefits of Workspace-Aware Commands

### Before (manual URL)
```bash
dx css --class-exists "cursor-pointer" http://localhost:3000
dx css --class-exists "btn-primary" http://localhost:6006
```

### After (workspace-aware)
```bash
dx --project commercial css --class-exists "cursor-pointer"
dx --project storybook css --class-exists "btn-primary"
```

**Advantages:**
- ✅ No need to remember ports
- ✅ Shorter commands
- ✅ Self-documenting (project name is explicit)
- ✅ Works in CI/CD without hardcoded URLs
- ✅ Easy to switch between projects

---

## Future Enhancements

When Phase 2 adds more commands, they'll all be workspace-aware:

```bash
# HTML queries
dx --project commercial html --selector ".theme-toggle"
dx --project storybook html --selector ".btn-primary"

# Snapshots
dx --project commercial snapshot --create --name "homepage"
dx --project storybook snapshot --create --name "button-variants"
```

---

**TEAM-DX-001 - Workspace integration complete**
